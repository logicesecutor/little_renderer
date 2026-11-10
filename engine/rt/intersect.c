#include "rt/intersect.h"

// stores t, point and a normal facing the ray
static void set_hit(Hit *h, const Ray *ray, real t, Vec3 outward_normal){
    h->t = t;
    h->p = v_add(ray->o, v_scale(ray->d, t));
    h->front_face = v_dot(ray->d, outward_normal) < 0;
    h->n = h->front_face ? outward_normal : v_neg(outward_normal);
    h->ng = h->n;
}

// |o + t d - c|^2 = r^2 is a quadratic in t. With h = d.(c - o) the roots are
// t = (h -+ sqrt(h^2 - a*c')) / a, where a = d.d and c' = |c - o|^2 - r^2.
static int hit_sphere(const Primitive *s, const Ray *ray, real tmin, real tmax, Hit *h){
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
    Vec3 p = v_add(ray->o, v_scale(ray->d, t));
    set_hit(h, ray, t, v_scale(v_sub(p, s->center), 1.0 / s->radius));
    return 1;
}

// 1) intersect the plane of the quad, 2) express the hit point as q + a*u + b*v,
// 3) accept it if a and b are both inside [0, 1].
static int hit_quad(const Primitive *s, const Ray *ray, real tmin, real tmax, Hit *h){
    real denom = v_dot(s->normal, ray->d);
    if (r_fabs(denom) < 1e-12) return 0;      // ray parallel to the plane

    real t = (s->plane_d - v_dot(s->normal, ray->o)) / denom;
    if (t <= tmin || t >= tmax) return 0;

    Vec3 p = v_add(ray->o, v_scale(ray->d, t));
    Vec3 rel = v_sub(p, s->q);
    real a = v_dot(s->w, v_cross(rel, s->v));
    real b = v_dot(s->w, v_cross(s->u, rel));
    if (a < 0 || a > 1 || b < 0 || b > 1) return 0;

    set_hit(h, ray, t, s->normal);
    return 1;
}

// Moller-Trumbore: solve  o + t*d = q + b1*u + b2*v  for (t, b1, b2) with Cramer's rule.
// Inside the triangle when b1 >= 0, b2 >= 0 and b1 + b2 <= 1.
static int hit_triangle(const Triangle *s, const Vec3 *normals, const Ray *ray, real tmin, real tmax, Hit *h){
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

    // The geometric normal is not stored (it would cost 12-24 bytes per triangle); it is
    // only needed for hits that are kept, so it is computed here.
    set_hit(h, ray, t, v_norm(v_cross(s->u, s->v)));   // front_face from the real normal
    if (s->n[0] >= 0){
        // shading normal: blend the 3 vertex normals with the barycentric weights
        Vec3 ns = v_norm(v_add(v_add(v_scale(normals[s->n[0]], 1.0 - b1 - b2), v_scale(normals[s->n[1]], b1)),
                               v_scale(normals[s->n[2]], b2)));
        // Turn it to the same side as h->n (which faces the ray). This does not depend on
        // the triangle's winding, so duplicated triangles wound the other way shade the same.
        if (v_dot(ns, h->n) < 0) ns = v_neg(ns);
        if (v_dot(ns, ns) > 0.5) h->n = ns;   // (vertex normals that cancelled out: keep the flat one)
    }
    return 1;
}

int intersect_primitive(const Primitive *prim, const Ray *ray, real tmin, real tmax, Hit *h){
    return prim->type == PRIM_SPHERE ? hit_sphere(prim, ray, tmin, tmax, h) : hit_quad(prim, ray, tmin, tmax, h);
}

int intersect_shape(const Scene *s, int id, const Ray *ray, real tmin, real tmax, Hit *h){
    if (id < s->nprims) return intersect_primitive(&s->prims[id], ray, tmin, tmax, h);
    return hit_triangle(&s->tris[id - s->nprims], s->normals, ray, tmin, tmax, h);
}
