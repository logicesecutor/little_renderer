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
// the same for shape `id` of the scene (a primitive or a triangle, see scene/scene.h)
int intersect_shape(const Scene *s, int id, const Ray *ray, real tmin, real tmax, Hit *h);

#endif
