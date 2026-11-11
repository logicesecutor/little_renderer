// lrmath.h - the engine's math, shared by both backends: one number type (`real`),
// vectors and 4x4 matrices. Header only.
//
// `real` is float by default: half the memory of double, so more of a big scene fits in
// the CPU cache. Build with -DLR_DOUBLE ("make DOUBLE=1") for double precision everywhere.
// The float build also uses -fsingle-precision-constant, so a literal like 0.5 is a float
// and the arithmetic stays in single precision.
//
// Conventions: Mat4 is row-major and vectors are columns (v' = M * v). Right-handed world,
// y up. View space looks down -Z; clip space is OpenGL style (-w <= x, y, z <= w).
#ifndef LR_MATH_H
#define LR_MATH_H

#include <float.h>
#include <math.h>

#ifdef LR_DOUBLE
typedef double real;
#define REAL_MAX DBL_MAX
#define r_sqrt sqrt
#define r_sin sin
#define r_cos cos
#define r_tan tan
#define r_asin asin
#define r_atan2 atan2
#define r_pow pow
#define r_fabs fabs
#define r_fmin fmin
#define r_fmax fmax
#define r_floor floor
#define r_ceil ceil
#define r_copysign copysign
#define r_lround lround
#define r_log log
#define r_exp exp
#else
typedef float real;
#define REAL_MAX FLT_MAX
#define r_sqrt sqrtf
#define r_sin sinf
#define r_cos cosf
#define r_tan tanf
#define r_asin asinf
#define r_atan2 atan2f
#define r_pow powf
#define r_fabs fabsf
#define r_fmin fminf
#define r_fmax fmaxf
#define r_floor floorf
#define r_ceil ceilf
#define r_copysign copysignf
#define r_lround lroundf
#define r_log logf
#define r_exp expf
#endif

#define PI 3.14159265358979323846

typedef struct { real x, y; } Vec2;
typedef struct { real x, y, z; } Vec3;        // points, directions and RGB colors
typedef struct { real x, y, z, w; } Vec4;
typedef struct { real m[4][4]; } Mat4;

static inline Vec2 vec2(real x, real y){ Vec2 r = {x, y}; return r; }
static inline Vec3 vec3(real x, real y, real z){ Vec3 r = {x, y, z}; return r; }
static inline Vec4 vec4(real x, real y, real z, real w){ Vec4 r = {x, y, z, w}; return r; }

// ---- 3D vectors ---------------------------------------------------------------------

static inline Vec3 v_add(Vec3 a, Vec3 b){ return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline Vec3 v_sub(Vec3 a, Vec3 b){ return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline Vec3 v_mul(Vec3 a, Vec3 b){ return vec3(a.x * b.x, a.y * b.y, a.z * b.z); } // per component (colors)
static inline Vec3 v_scale(Vec3 a, real s){ return vec3(a.x * s, a.y * s, a.z * s); }
static inline Vec3 v_neg(Vec3 a){ return vec3(-a.x, -a.y, -a.z); }
static inline real v_dot(Vec3 a, Vec3 b){ return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 v_cross(Vec3 a, Vec3 b){
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline real v_len(Vec3 a){ return r_sqrt(v_dot(a, a)); }
static inline Vec3 v_norm(Vec3 a){ real l = v_len(a); return l > 0 ? v_scale(a, 1.0 / l) : a; }
static inline real v_max(Vec3 a){ return r_fmax(a.x, r_fmax(a.y, a.z)); }

// mirror d around the normal n (both unit length)
static inline Vec3 v_reflect(Vec3 d, Vec3 n){ return v_sub(d, v_scale(n, 2.0 * v_dot(d, n))); }

// Snell's law. d = unit incoming direction, n = unit normal on the side d comes from,
// eta = n_from / n_to. Caller must check total internal reflection first.
static inline Vec3 v_refract(Vec3 d, Vec3 n, real eta){
    real cos_i = r_fmin(v_dot(v_neg(d), n), 1.0);
    Vec3 perp = v_scale(v_add(d, v_scale(n, cos_i)), eta);            // part parallel to the surface
    Vec3 para = v_scale(n, -r_sqrt(r_fabs((real)1 - v_dot(perp, perp))));  // part along -n
    return v_add(perp, para);
}

// ---- 4x4 matrices -------------------------------------------------------------------

static inline Mat4 mat4_identity(void){
    Mat4 r = {{{0}}};
    for (int i = 0; i < 4; i++) r.m[i][i] = 1.0f;
    return r;
}

static inline Mat4 mat4_mul(Mat4 a, Mat4 b){
    Mat4 r = {{{0}}};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                r.m[i][j] += a.m[i][k] * b.m[k][j];
    return r;
}

static inline Vec4 mat4_mul_vec4(Mat4 a, Vec4 v){
    Vec4 r;
    r.x = a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z + a.m[0][3] * v.w;
    r.y = a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z + a.m[1][3] * v.w;
    r.z = a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z + a.m[2][3] * v.w;
    r.w = a.m[3][0] * v.x + a.m[3][1] * v.y + a.m[3][2] * v.z + a.m[3][3] * v.w;
    return r;
}

// transform a point (w = 1) and drop w
static inline Vec3 mat4_mul_point(Mat4 a, Vec3 p){
    Vec4 r = mat4_mul_vec4(a, vec4(p.x, p.y, p.z, 1.0f));
    return vec3(r.x, r.y, r.z);
}

// transform a direction (w = 0): ignores translation. Fine for normals as long as
// the matrix is rotation + uniform scale.
static inline Vec3 mat4_mul_dir(Mat4 a, Vec3 d){
    Vec4 r = mat4_mul_vec4(a, vec4(d.x, d.y, d.z, 0.0f));
    return vec3(r.x, r.y, r.z);
}

static inline Mat4 mat4_rotate_x(real a){
    Mat4 r = mat4_identity();
    real c = r_cos(a), s = r_sin(a);
    r.m[1][1] = c; r.m[1][2] = -s;
    r.m[2][1] = s; r.m[2][2] = c;
    return r;
}

static inline Mat4 mat4_rotate_y(real a){
    Mat4 r = mat4_identity();
    real c = r_cos(a), s = r_sin(a);
    r.m[0][0] = c;  r.m[0][2] = s;
    r.m[2][0] = -s; r.m[2][2] = c;
    return r;
}

// view matrix: moves the world so the camera sits at the origin looking down -Z
static inline Mat4 mat4_look_at(Vec3 eye, Vec3 target, Vec3 up){
    Vec3 f = v_norm(v_sub(target, eye));   // forward
    Vec3 s = v_norm(v_cross(f, up));       // right
    Vec3 u = v_cross(s, f);                // true up
    Mat4 r = mat4_identity();
    r.m[0][0] = s.x;  r.m[0][1] = s.y;  r.m[0][2] = s.z;  r.m[0][3] = -v_dot(s, eye);
    r.m[1][0] = u.x;  r.m[1][1] = u.y;  r.m[1][2] = u.z;  r.m[1][3] = -v_dot(u, eye);
    r.m[2][0] = -f.x; r.m[2][1] = -f.y; r.m[2][2] = -f.z; r.m[2][3] = v_dot(f, eye);
    return r;
}

// perspective projection. After it, clip.w = -z_eye, and a point is inside the view
// volume when -w <= x,y,z <= w. Dividing by w gives NDC in [-1, 1].
static inline Mat4 mat4_perspective(real fov_y, real aspect, real near_z, real far_z){
    real f = 1.0f / r_tan(fov_y * 0.5f);
    Mat4 r = {{{0}}};
    r.m[0][0] = f / aspect;
    r.m[1][1] = f;
    r.m[2][2] = (far_z + near_z) / (near_z - far_z);
    r.m[2][3] = 2.0f * far_z * near_z / (near_z - far_z);
    r.m[3][2] = -1.0f;
    return r;
}


// ---- rotations: 3x3 matrices and quaternions (used by the physics engine) ------------------

typedef struct { real m[3][3]; } Mat3;
typedef struct { real w, x, y, z; } Quat;   // w + xi + yj + zk; unit length = a rotation

static inline Mat3 mat3_identity(void){
    Mat3 r = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    return r;
}
static inline Mat3 mat3_diag(Vec3 d){
    Mat3 r = {{{d.x, 0, 0}, {0, d.y, 0}, {0, 0, d.z}}};
    return r;
}
static inline Vec3 mat3_mul_vec(Mat3 a, Vec3 v){
    return vec3(a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
                a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
                a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z);
}
static inline Mat3 mat3_mul(Mat3 a, Mat3 b){
    Mat3 r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            r.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
    return r;
}
static inline Mat3 mat3_transpose(Mat3 a){
    Mat3 r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) r.m[i][j] = a.m[j][i];
    return r;
}
static inline Vec3 mat3_col(Mat3 a, int j){ return vec3(a.m[0][j], a.m[1][j], a.m[2][j]); }

static inline Quat quat(real w, real x, real y, real z){ Quat q = {w, x, y, z}; return q; }
static inline Quat quat_identity(void){ return quat(1, 0, 0, 0); }
// rotation by `angle` radians around the unit vector `axis`
static inline Quat quat_axis_angle(Vec3 axis, real angle){
    real s = r_sin(angle * 0.5);
    return quat(r_cos(angle * 0.5), axis.x * s, axis.y * s, axis.z * s);
}
static inline Quat quat_mul(Quat a, Quat b){   // first b, then a
    return quat(a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
}
static inline Quat quat_norm(Quat q){
    real l = r_sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    return l > 0 ? quat(q.w / l, q.x / l, q.y / l, q.z / l) : quat_identity();
}
// the rotation matrix of a unit quaternion
static inline Mat3 quat_to_mat3(Quat q){
    real w = q.w, x = q.x, y = q.y, z = q.z;
    Mat3 r = {{{1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)},
               {2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
               {2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)}}};
    return r;
}
static inline Vec3 quat_rotate(Quat q, Vec3 v){ return mat3_mul_vec(quat_to_mat3(q), v); }
// Integrate an angular velocity omega (radians / s, world frame) over dt:
//   dq/dt = 0.5 * (0, omega) * q
static inline Quat quat_integrate(Quat q, Vec3 omega, real dt){
    Quat w = quat(0, omega.x, omega.y, omega.z);
    Quat d = quat_mul(w, q);
    real h = 0.5 * dt;
    return quat_norm(quat(q.w + d.w * h, q.x + d.x * h, q.y + d.y * h, q.z + d.z * h));
}

#endif
