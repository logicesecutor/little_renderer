#include "rt/intersect.h"

// stores t, point and a normal facing the ray
static void set_hit(Hit *h, const Ray *ray, real t, Vec3 outward_normal){
    h->t = t;
    h->p = v_add(ray->o, v_scale(ray->d, t));
    h->front_face = v_dot(ray->d, outward_normal) < 0;
    h->n = h->front_face ? outward_normal : v_neg(outward_normal);
    h->ng = h->n;
}

// The tests are in intersect.h (lr_hit_*_t: |o + t d - c|^2 = r^2 as a quadratic in t; a
// quad's plane, then the point as q + a u + b v with a, b in [0, 1]; Moller-Trumbore for a
// triangle). Here the hit each makes once it is the one kept.

static void finish_sphere(const Primitive *s, const Ray *ray, real t, Hit *h){
    Vec3 p = v_add(ray->o, v_scale(ray->d, t));
    Vec3 n = v_scale(v_sub(p, s->center), 1.0 / s->radius);
    set_hit(h, ray, t, n);
    real ny = n.y < -1 ? -1 : n.y > 1 ? 1 : n.y;
    real u = 0.5f + r_atan2(n.x, n.z) * (real)(0.5 / PI);
    h->uv = vec2(u - r_floor(u), r_acos(ny) * (real)(1.0 / PI));
}

static void finish_quad(const Primitive *s, const Ray *ray, real t, Vec2 uv, Hit *h){
    set_hit(h, ray, t, s->normal);
    h->uv = uv;
}

static void finish_triangle(const Triangle *s, const Vec3 *normals, const Ray *ray, real t, Vec2 uv, Hit *h){
    const real b1 = uv.x, b2 = uv.y;
    // The geometric normal is not stored (it would cost 12-24 bytes per triangle); it is
    // only needed for hits that are kept, so it is computed here.
    set_hit(h, ray, t, v_norm(v_cross(s->u, s->v)));   // front_face from the real normal
    h->uv = uv;
    if (s->n[0] >= 0){
        // shading normal: blend the 3 vertex normals with the barycentric weights
        Vec3 ns = v_norm(v_add(v_add(v_scale(normals[s->n[0]], 1.0 - b1 - b2), v_scale(normals[s->n[1]], b1)),
                               v_scale(normals[s->n[2]], b2)));
        // Turn it to the same side as h->n (which faces the ray). This does not depend on
        // the triangle's winding, so duplicated triangles wound the other way shade the same.
        if (v_dot(ns, h->n) < 0) ns = v_neg(ns);
        if (v_dot(ns, ns) > 0.5) h->n = ns;   // (vertex normals that cancelled out: keep the flat one)
    }
}

int intersect_primitive(const Primitive *prim, const Ray *ray, real tmin, real tmax, Hit *h){
    real t;
    Vec2 uv;
    if (prim->type == PRIM_SPHERE){
        if (!lr_hit_sphere_t(prim, ray, tmin, tmax, &t)) return 0;
        finish_sphere(prim, ray, t, h);
        return 1;
    }
    if (!lr_hit_quad_t(prim, ray, tmin, tmax, &t, &uv)) return 0;
    finish_quad(prim, ray, t, uv, h);
    return 1;
}

void shape_hit(const Scene *s, int id, const Ray *ray, real t, Vec2 uv, Hit *h){
    if (id < s->nprims){
        const Primitive *p = &s->prims[id];
        if (p->type == PRIM_SPHERE) finish_sphere(p, ray, t, h);
        else finish_quad(p, ray, t, uv, h);
    } else finish_triangle(&s->tris[id - s->nprims], s->normals, ray, t, uv, h);
    h->prim = id;
}

int intersect_shape(const Scene *s, int id, const Ray *ray, real tmin, real tmax, Hit *h){
    real t;
    Vec2 uv;
    if (!shape_test(s, id, ray, tmin, tmax, &t, &uv)) return 0;
    shape_hit(s, id, ray, t, uv, h);
    return 1;
}

// a sphere's tangent at longitude u: east, (cos phi, 0, -sin phi); its bitangent n x t points
// north (up the image, the latitude v grows downwards): w = 1
static Vec4 sphere_tangent(Vec2 uv){
    real ph = (uv.x - 0.5f) * (real)(2 * PI);
    return (Vec4){r_cos(ph), 0, -r_sin(ph), 1};
}
// a quad's: along u; n = u x v, so n x u points along +v (down the image): w = -1
static Vec4 quad_tangent(Vec3 u){
    Vec3 t = v_norm(u);
    return (Vec4){t.x, t.y, t.z, -1};
}
// a triangle's: its corners' texture coordinates and tangents, interpolated
static void tri_texcoords(const Triangle *t, const Vec2 *uvs, const Vec4 *tangents, int nuvs, Vec2 b, Vec2 *uv, Vec4 *tg){
    real w0 = 1 - b.x - b.y;
    if (t->t[0] < 0 || t->t[0] >= nuvs || t->t[1] >= nuvs || t->t[2] >= nuvs || t->t[1] < 0 || t->t[2] < 0){
        *uv = b;
        *tg = quad_tangent(t->u);
        tg->w = 1;
        return;
    }
    Vec2 a0 = uvs[t->t[0]], a1 = uvs[t->t[1]], a2 = uvs[t->t[2]];
    *uv = vec2(w0 * a0.x + b.x * a1.x + b.y * a2.x, w0 * a0.y + b.x * a1.y + b.y * a2.y);
    Vec4 t0 = tangents[t->t[0]], t1 = tangents[t->t[1]], t2 = tangents[t->t[2]];
    *tg = (Vec4){w0 * t0.x + b.x * t1.x + b.y * t2.x, w0 * t0.y + b.x * t1.y + b.y * t2.y, w0 * t0.z + b.x * t1.z + b.y * t2.z, t0.w};
}

void hit_texcoords(const Scene *s, const Hit *h, Vec2 *uv, Vec4 *tg){
    if (h->inst < 0){
        if (h->prim < s->nprims){
            const Primitive *p = &s->prims[h->prim];
            *uv = h->uv;
            *tg = p->type == PRIM_SPHERE ? sphere_tangent(h->uv) : quad_tangent(p->u);
            return;
        }
        tri_texcoords(&s->tris[h->prim - s->nprims], s->uvs, s->tangents, s->nuvs, h->uv, uv, tg);
        return;
    }
    // an instance: in its own space, the tangent turned into the world
    const Instance *in = &s->insts[h->inst];
    Vec4 t;
    *uv = h->uv;
    if (in->mesh >= 0){
        const SceneMesh *m = &s->meshes[in->mesh];
        tri_texcoords(&m->tris[h->prim], m->uvs, m->tangents, m->nuvs, h->uv, uv, &t);
    } else if (in->mesh == INST_SPHERE) t = sphere_tangent(h->uv);
    else {
        // the box's face: the axis the hit point (in the box's space) is farthest along
        Vec3 lp = xform_point(&in->inv, h->p);
        real ax = r_fabs(lp.x), ay = r_fabs(lp.y), az = r_fabs(lp.z);
        int k = ax >= ay && ax >= az ? 0 : (ay >= az ? 1 : 2);
        real side = k == 0 ? lp.x : (k == 1 ? lp.y : lp.z);
        t = quad_tangent(unit_box_face(2 * k + (side > 0), 0).u);
    }
    Vec3 d = v_norm(xform_dir(&in->xf, vec3(t.x, t.y, t.z)));
    *tg = (Vec4){d.x, d.y, d.z, t.w};
}
