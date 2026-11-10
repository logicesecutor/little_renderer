#include "geometry.h"

#include <string.h>

Primitive make_sphere(Vec3 center, double radius, int material){
    Primitive p;
    memset(&p, 0, sizeof(p));
    p.type = PRIM_SPHERE;
    p.material = material;
    p.center = center;
    p.radius = radius;
    return p;
}

Primitive make_quad(Vec3 q, Vec3 u, Vec3 v, int material){
    Primitive p;
    memset(&p, 0, sizeof(p));
    p.type = PRIM_QUAD;
    p.material = material;
    p.q = q;
    p.u = u;
    p.v = v;
    Vec3 n = v_cross(u, v);
    p.normal = v_norm(n);
    p.plane_d = v_dot(p.normal, q);
    p.w = v_scale(n, 1.0 / v_dot(n, n));
    p.area = v_len(n);
    return p;
}

Primitive make_triangle(Vec3 a, Vec3 b, Vec3 c, const Vec3 *normals, int material){
    Primitive p;
    memset(&p, 0, sizeof(p));
    p.type = PRIM_TRIANGLE;
    p.material = material;
    p.q = a;
    p.u = v_sub(b, a);
    p.v = v_sub(c, a);
    Vec3 n = v_cross(p.u, p.v);
    p.normal = v_norm(n);
    p.area = 0.5 * v_len(n);
    if (normals){
        p.n0 = v_norm(normals[0]);
        p.n1 = v_norm(normals[1]);
        p.n2 = v_norm(normals[2]);
        p.smooth = 1;
    }
    return p;
}

// stores t, point and a normal facing the ray
static void set_hit(Hit *h, const Ray *ray, double t, Vec3 outward_normal){
    h->t = t;
    h->p = v_add(ray->o, v_scale(ray->d, t));
    h->front_face = v_dot(ray->d, outward_normal) < 0;
    h->n = h->front_face ? outward_normal : v_neg(outward_normal);
}

// |o + t d - c|^2 = r^2 is a quadratic in t. With h = d.(c - o) the roots are
// t = (h -+ sqrt(h^2 - a*c')) / a, where a = d.d and c' = |c - o|^2 - r^2.
static int hit_sphere(const Primitive *s, const Ray *ray, double tmin, double tmax, Hit *h){
    Vec3 oc = v_sub(s->center, ray->o);
    double a = v_dot(ray->d, ray->d);
    double hh = v_dot(ray->d, oc);
    double c = v_dot(oc, oc) - s->radius * s->radius;
    double disc = hh * hh - a * c;
    if (disc < 0) return 0;

    double sq = sqrt(disc);
    double t = (hh - sq) / a;                 // nearer root first
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
static int hit_quad(const Primitive *s, const Ray *ray, double tmin, double tmax, Hit *h){
    double denom = v_dot(s->normal, ray->d);
    if (fabs(denom) < 1e-12) return 0;        // ray parallel to the plane

    double t = (s->plane_d - v_dot(s->normal, ray->o)) / denom;
    if (t <= tmin || t >= tmax) return 0;

    Vec3 p = v_add(ray->o, v_scale(ray->d, t));
    Vec3 rel = v_sub(p, s->q);
    double a = v_dot(s->w, v_cross(rel, s->v));
    double b = v_dot(s->w, v_cross(s->u, rel));
    if (a < 0 || a > 1 || b < 0 || b > 1) return 0;

    set_hit(h, ray, t, s->normal);
    return 1;
}

// Moller-Trumbore: solve  o + t*d = q + b1*u + b2*v  for (t, b1, b2) with Cramer's rule.
// Inside the triangle when b1 >= 0, b2 >= 0 and b1 + b2 <= 1.
static int hit_triangle(const Primitive *s, const Ray *ray, double tmin, double tmax, Hit *h){
    Vec3 pvec = v_cross(ray->d, s->v);
    double det = v_dot(s->u, pvec);
    if (fabs(det) < 1e-12) return 0;          // ray parallel to the triangle (or degenerate)
    double inv = 1.0 / det;

    Vec3 tvec = v_sub(ray->o, s->q);
    double b1 = v_dot(tvec, pvec) * inv;
    if (b1 < 0 || b1 > 1) return 0;

    Vec3 qvec = v_cross(tvec, s->u);
    double b2 = v_dot(ray->d, qvec) * inv;
    if (b2 < 0 || b1 + b2 > 1) return 0;

    double t = v_dot(s->v, qvec) * inv;
    if (t <= tmin || t >= tmax) return 0;

    set_hit(h, ray, t, s->normal);   // front_face comes from the real (geometric) normal
    if (s->smooth){
        // shading normal: blend the 3 vertex normals with the barycentric weights
        Vec3 ns = v_norm(v_add(v_add(v_scale(s->n0, 1.0 - b1 - b2), v_scale(s->n1, b1)), v_scale(s->n2, b2)));
        // Turn it to the same side as h->n (which faces the ray). This does not depend on
        // the triangle's winding, so duplicated triangles wound the other way shade the same.
        if (v_dot(ns, h->n) < 0) ns = v_neg(ns);
        if (v_dot(ns, ns) > 0.5) h->n = ns;   // (vertex normals that cancelled out: keep the flat one)
    }
    return 1;
}

int intersect_primitive(const Primitive *prim, const Ray *ray, double tmin, double tmax, Hit *h){
    switch (prim->type){
    case PRIM_SPHERE: return hit_sphere(prim, ray, tmin, tmax, h);
    case PRIM_QUAD:   return hit_quad(prim, ray, tmin, tmax, h);
    default:          return hit_triangle(prim, ray, tmin, tmax, h);
    }
}

static void grow_box(Aabb *b, Vec3 p){
    b->lo = vec3(fmin(b->lo.x, p.x), fmin(b->lo.y, p.y), fmin(b->lo.z, p.z));
    b->hi = vec3(fmax(b->hi.x, p.x), fmax(b->hi.y, p.y), fmax(b->hi.z, p.z));
}

Aabb primitive_bounds(const Primitive *prim){
    Aabb b;
    if (prim->type == PRIM_SPHERE){
        Vec3 r = vec3(prim->radius, prim->radius, prim->radius);
        b.lo = v_sub(prim->center, r);
        b.hi = v_add(prim->center, r);
    } else {
        b.lo = b.hi = prim->q;
        grow_box(&b, v_add(prim->q, prim->u));
        grow_box(&b, v_add(prim->q, prim->v));
        if (prim->type == PRIM_QUAD) grow_box(&b, v_add(prim->q, v_add(prim->u, prim->v)));
    }
    Vec3 pad = vec3(1e-3, 1e-3, 1e-3);   // a wall is flat: give its box some thickness
    b.lo = v_sub(b.lo, pad);
    b.hi = v_add(b.hi, pad);
    return b;
}
