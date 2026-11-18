// intersect.h - path tracer: where does a ray hit a primitive or a box.
#ifndef LR_INTERSECT_H
#define LR_INTERSECT_H

#include "scene/camera.h"   // Ray
#include "scene/scene.h"    // Primitive, Triangle, Aabb

typedef struct {
    real t;          // distance along the ray: hit point = o + t * d
    Vec3 p;          // hit point
    Vec3 n;          // unit shading normal, flipped so it always faces the incoming ray
    Vec3 ng;         // unit geometric normal, on the same side as n (smooth meshes: n != ng)
    int front_face;  // 1 if the ray hit the side the geometric normal points to
    int prim;        // shape id of what was hit (scene/scene.h: primitives, then triangles);
                     // for an instance: the triangle's index in its mesh
    int inst;        // the instance hit, or -1 (scene_intersect sets it)
    Vec2 uv;         // where on the shape: a sphere's longitude / latitude (0..1), a quad's
                     // (a, b), a triangle's barycentrics (b1, b2) (hit_texcoords turns
                     // them into texture coordinates)
} Hit;

// the material of what was hit
static inline int hit_material(const Scene *s, const Hit *h){
    return h->inst >= 0 ? scene_instance_material(s, h->inst, h->prim) : scene_shape_material(s, h->prim);
}

// Where a hit is in its shape's textures (PBR5): texture coordinates (a sphere's longitude
// and latitude, a quad's (a, b), a triangle's from its corners' (Triangle.t; without them,
// its barycentrics)) and the tangent there in the world (xyz along +u; w the handedness:
// the bitangent w (n x t) points up the image)
void hit_texcoords(const Scene *s, const Hit *h, Vec2 *uv, Vec4 *tangent);

// Slab test, used by both acceleration structures. Along each axis the ray is between the
// box's two planes for t in [(lo - o) / d, (hi - o) / d] (swapped when d < 0); it is inside
// the box where all three intervals overlap. inv_d = 1 / d, computed once per ray (it may be
// +-inf; fmax / fmin quietly ignore the NaN that 0 * inf produces).
// Returns 1 and the entry distance when the ray touches the box inside (tmin, tmax).
static inline int aabb_hit(const Aabb *b, const Ray *r, Vec3 inv_d, real tmin, real tmax, real *t_enter){
    real t0 = (b->lo.x - r->o.x) * inv_d.x, t1 = (b->hi.x - r->o.x) * inv_d.x;
    if (inv_d.x < 0){ real t = t0; t0 = t1; t1 = t; }
    tmin = r_fmax(t0, tmin); tmax = r_fmin(t1, tmax);
    t0 = (b->lo.y - r->o.y) * inv_d.y; t1 = (b->hi.y - r->o.y) * inv_d.y;
    if (inv_d.y < 0){ real t = t0; t0 = t1; t1 = t; }
    tmin = r_fmax(t0, tmin); tmax = r_fmin(t1, tmax);
    t0 = (b->lo.z - r->o.z) * inv_d.z; t1 = (b->hi.z - r->o.z) * inv_d.z;
    if (inv_d.z < 0){ real t = t0; t0 = t1; t1 = t; }
    tmin = r_fmax(t0, tmin); tmax = r_fmin(t1, tmax);
    if (tmax < tmin) return 0;
    *t_enter = tmin;
    return 1;
}

// Closest hit with t in (tmin, tmax). Returns 1 and fills *h on a hit (h->prim is NOT set).
int intersect_primitive(const Primitive *prim, const Ray *ray, real tmin, real tmax, Hit *h);
// the same for shape `id` of the scene (a primitive or a triangle, see scene/scene.h), with
// h->prim = id
int intersect_shape(const Scene *s, int id, const Ray *ray, real tmin, real tmax, Hit *h);

// ---- the tests alone, for the traversals (docs/PERF_REVIEW.md, N6) ----
// A traversal tests many shapes and keeps the nearest: shape_test says only whether shape
// `id` is hit in (tmin, tmax), at what t, and where on it (*uv: a quad's (a, b), a
// triangle's barycentrics; nothing for a sphere); shape_hit then makes the Hit of the one
// kept (point, normals, a sphere's longitude and latitude, a triangle's shading normal), with
// h->prim = id. The same Hit, to the bit, as intersect_shape's; the tests are inline here, so
// they are not a call into another file for every candidate.
static inline int lr_hit_sphere_t(const Primitive *s, const Ray *ray, real tmin, real tmax, real *t_out){
    Vec3 oc = v_sub(s->center, ray->o);
    real a = v_dot(ray->d, ray->d);
    real hh = v_dot(ray->d, oc);
    real c = v_dot(oc, oc) - s->radius * s->radius;
    real disc = hh * hh - a * c;
    if (disc < 0) return 0;
    real sq = r_sqrt(disc);
    real t = (hh - sq) / a;                   // nearer root first
    if (t <= tmin || t >= tmax){
        t = (hh + sq) / a;                    // we may be inside the sphere (glass)
        if (t <= tmin || t >= tmax) return 0;
    }
    *t_out = t;
    return 1;
}
static inline int lr_hit_quad_t(const Primitive *s, const Ray *ray, real tmin, real tmax, real *t_out, Vec2 *uv){
    real denom = v_dot(s->normal, ray->d);
    if (r_fabs(denom) < 1e-12) return 0;      // ray parallel to the plane
    real t = (s->plane_d - v_dot(s->normal, ray->o)) / denom;
    if (t <= tmin || t >= tmax) return 0;
    Vec3 p = v_add(ray->o, v_scale(ray->d, t));
    Vec3 rel = v_sub(p, s->q);
    real a = v_dot(s->w, v_cross(rel, s->v));
    real b = v_dot(s->w, v_cross(s->u, rel));
    if (a < 0 || a > 1 || b < 0 || b > 1) return 0;
    *t_out = t;
    *uv = vec2(a, b);
    return 1;
}
static inline int lr_hit_triangle_t(const Triangle *s, const Ray *ray, real tmin, real tmax, real *t_out, Vec2 *uv){
    Vec3 pvec = v_cross(ray->d, s->v);
    real det = v_dot(s->u, pvec);
    if (r_fabs(det) < 1e-12) return 0;        // ray parallel to the triangle (or degenerate)
    real inv = 1.0 / det;
    Vec3 tvec = v_sub(ray->o, s->q);
    real b1 = v_dot(tvec, pvec) * inv;
    if (b1 < 0 || b1 > 1) return 0;
    Vec3 qvec = v_cross(tvec, s->u);
    real b2 = v_dot(ray->d, qvec) * inv;
    if (b2 < 0 || b1 + b2 > 1) return 0;
    real t = v_dot(s->v, qvec) * inv;
    if (t <= tmin || t >= tmax) return 0;
    *t_out = t;
    *uv = vec2(b1, b2);
    return 1;
}
static inline int shape_test(const Scene *s, int id, const Ray *ray, real tmin, real tmax, real *t, Vec2 *uv){
    if (id < s->nprims){
        const Primitive *p = &s->prims[id];
        return p->type == PRIM_SPHERE ? lr_hit_sphere_t(p, ray, tmin, tmax, t) : lr_hit_quad_t(p, ray, tmin, tmax, t, uv);
    }
    return lr_hit_triangle_t(&s->tris[id - s->nprims], ray, tmin, tmax, t, uv);
}
void shape_hit(const Scene *s, int id, const Ray *ray, real t, Vec2 uv, Hit *h);

#endif
