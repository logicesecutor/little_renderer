// geometry.h - rays, bounding boxes and the shapes we can hit: spheres, quads, triangles.
#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "vec3.h"

typedef struct {
    Vec3 o;   // origin
    Vec3 d;   // direction (not necessarily unit length)
} Ray;

typedef struct {
    double t;        // distance along the ray: hit point = o + t * d
    Vec3 p;          // hit point
    Vec3 n;          // unit normal, flipped so it always faces the incoming ray
    int front_face;  // 1 if the ray hit the side the geometric normal points to
    int prim;        // index of the primitive that was hit
} Hit;

// axis-aligned bounding box
typedef struct { Vec3 lo, hi; } Aabb;

// Slab test, used by both acceleration structures. Along each axis the ray is between the
// box's two planes for t in [(lo - o) / d, (hi - o) / d] (swapped when d < 0); it is inside
// the box where all three intervals overlap. inv_d = 1 / d, computed once per ray (it may be
// +-inf; fmax / fmin quietly ignore the NaN that 0 * inf produces).
// Returns 1 and the entry distance when the ray touches the box inside (tmin, tmax).
static inline int aabb_hit(const Aabb *b, const Ray *r, Vec3 inv_d, double tmin, double tmax, double *t_enter){
    double t0 = (b->lo.x - r->o.x) * inv_d.x, t1 = (b->hi.x - r->o.x) * inv_d.x;
    if (inv_d.x < 0){ double t = t0; t0 = t1; t1 = t; }
    tmin = fmax(t0, tmin); tmax = fmin(t1, tmax);
    t0 = (b->lo.y - r->o.y) * inv_d.y; t1 = (b->hi.y - r->o.y) * inv_d.y;
    if (inv_d.y < 0){ double t = t0; t0 = t1; t1 = t; }
    tmin = fmax(t0, tmin); tmax = fmin(t1, tmax);
    t0 = (b->lo.z - r->o.z) * inv_d.z; t1 = (b->hi.z - r->o.z) * inv_d.z;
    if (inv_d.z < 0){ double t = t0; t0 = t1; t1 = t; }
    tmin = fmax(t0, tmin); tmax = fmin(t1, tmax);
    if (tmax < tmin) return 0;
    *t_enter = tmin;
    return 1;
}

typedef enum { PRIM_SPHERE, PRIM_QUAD, PRIM_TRIANGLE } PrimType;

typedef struct {
    PrimType type;
    int material;          // index into Scene.mats

    // sphere
    Vec3 center;
    double radius;

    // quad:     the points q + a*u + b*v with a, b in [0, 1]
    // triangle: corners q, q + u, q + v  (u, v are the two edges leaving q)
    Vec3 q, u, v;
    Vec3 normal;           // unit normal = normalize(u x v); for lights: the side that emits
    double plane_d;        // quad plane equation: dot(normal, p) = plane_d
    Vec3 w;                // quad: (u x v) / |u x v|^2, used to find a, b
    double area;

    // triangle: optional per-corner normals for smooth shading
    Vec3 n0, n1, n2;
    int smooth;
} Primitive;

Primitive make_sphere(Vec3 center, double radius, int material);
Primitive make_quad(Vec3 q, Vec3 u, Vec3 v, int material);
// corners a, b, c counter-clockwise seen from outside; normals = 3 vertex normals or NULL
Primitive make_triangle(Vec3 a, Vec3 b, Vec3 c, const Vec3 *normals, int material);

// Closest hit with t in (tmin, tmax). Returns 1 and fills *h on a hit (h->prim is NOT set).
int intersect_primitive(const Primitive *prim, const Ray *ray, double tmin, double tmax, Hit *h);

// box that contains the primitive (slightly padded so flat shapes have some thickness)
Aabb primitive_bounds(const Primitive *prim);

#endif
