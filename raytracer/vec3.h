// vec3.h - 3D vectors in double precision. Used for points, directions AND colors.
#ifndef VEC3_H
#define VEC3_H

#include <math.h>

typedef struct { double x, y, z; } Vec3;

static inline Vec3 vec3(double x, double y, double z){ Vec3 r = {x, y, z}; return r; }
static inline Vec3 v_add(Vec3 a, Vec3 b){ return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline Vec3 v_sub(Vec3 a, Vec3 b){ return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline Vec3 v_mul(Vec3 a, Vec3 b){ return vec3(a.x * b.x, a.y * b.y, a.z * b.z); } // per component (colors)
static inline Vec3 v_scale(Vec3 a, double s){ return vec3(a.x * s, a.y * s, a.z * s); }
static inline Vec3 v_neg(Vec3 a){ return vec3(-a.x, -a.y, -a.z); }
static inline double v_dot(Vec3 a, Vec3 b){ return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 v_cross(Vec3 a, Vec3 b){
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline double v_len(Vec3 a){ return sqrt(v_dot(a, a)); }
static inline Vec3 v_norm(Vec3 a){ double l = v_len(a); return l > 0 ? v_scale(a, 1.0 / l) : a; }
static inline double v_max(Vec3 a){ return fmax(a.x, fmax(a.y, a.z)); }

// mirror d around the normal n (both unit length)
static inline Vec3 v_reflect(Vec3 d, Vec3 n){ return v_sub(d, v_scale(n, 2.0 * v_dot(d, n))); }

// Snell's law. d = unit incoming direction, n = unit normal on the side d comes from,
// eta = n_from / n_to. Caller must check total internal reflection first.
static inline Vec3 v_refract(Vec3 d, Vec3 n, double eta){
    double cos_i = fmin(v_dot(v_neg(d), n), 1.0);
    Vec3 perp = v_scale(v_add(d, v_scale(n, cos_i)), eta);          // part parallel to the surface
    Vec3 para = v_scale(n, -sqrt(fabs(1.0 - v_dot(perp, perp))));   // part along -n
    return v_add(perp, para);
}

#endif
