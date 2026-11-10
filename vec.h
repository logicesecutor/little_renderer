// vec.h - tiny vector / matrix math, header only.
// Conventions: mat4 is row-major and vectors are columns, so  v' = M * v.
// Right-handed world, camera looks down -Z, clip space is OpenGL style (z in [-w, w]).
#ifndef VEC_H
#define VEC_H

#include <math.h>

typedef struct { float x, y; } vec2;
typedef struct { float x, y, z; } vec3;
typedef struct { float x, y, z, w; } vec4;
typedef struct { float m[4][4]; } mat4;

static inline vec3 v3(float x, float y, float z){ vec3 r = {x, y, z}; return r; }
static inline vec3 v3_add(vec3 a, vec3 b){ return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline vec3 v3_sub(vec3 a, vec3 b){ return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline vec3 v3_scale(vec3 a, float s){ return v3(a.x * s, a.y * s, a.z * s); }
static inline float v3_dot(vec3 a, vec3 b){ return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline vec3 v3_cross(vec3 a, vec3 b){
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float v3_len(vec3 a){ return sqrtf(v3_dot(a, a)); }
static inline vec3 v3_normalize(vec3 a){
    float l = v3_len(a);
    return l > 1e-12f ? v3_scale(a, 1.0f / l) : a;
}

static inline mat4 mat4_identity(void){
    mat4 r = {{{0}}};
    for (int i = 0; i < 4; i++) r.m[i][i] = 1.0f;
    return r;
}

static inline mat4 mat4_mul(mat4 a, mat4 b){
    mat4 r = {{{0}}};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                r.m[i][j] += a.m[i][k] * b.m[k][j];
    return r;
}

static inline vec4 mat4_mul_vec4(mat4 a, vec4 v){
    vec4 r;
    r.x = a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z + a.m[0][3] * v.w;
    r.y = a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z + a.m[1][3] * v.w;
    r.z = a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z + a.m[2][3] * v.w;
    r.w = a.m[3][0] * v.x + a.m[3][1] * v.y + a.m[3][2] * v.z + a.m[3][3] * v.w;
    return r;
}

// transform a point (w = 1) and drop w
static inline vec3 mat4_mul_point(mat4 a, vec3 p){
    vec4 r = mat4_mul_vec4(a, (vec4){p.x, p.y, p.z, 1.0f});
    return v3(r.x, r.y, r.z);
}

// transform a direction (w = 0): ignores translation. Fine for normals as long as
// the matrix is rotation + uniform scale (which is all we use).
static inline vec3 mat4_mul_dir(mat4 a, vec3 d){
    vec4 r = mat4_mul_vec4(a, (vec4){d.x, d.y, d.z, 0.0f});
    return v3(r.x, r.y, r.z);
}

static inline mat4 mat4_translate(vec3 t){
    mat4 r = mat4_identity();
    r.m[0][3] = t.x; r.m[1][3] = t.y; r.m[2][3] = t.z;
    return r;
}

static inline mat4 mat4_scale(float s){
    mat4 r = mat4_identity();
    r.m[0][0] = r.m[1][1] = r.m[2][2] = s;
    return r;
}

static inline mat4 mat4_rotate_x(float a){
    mat4 r = mat4_identity();
    float c = cosf(a), s = sinf(a);
    r.m[1][1] = c; r.m[1][2] = -s;
    r.m[2][1] = s; r.m[2][2] = c;
    return r;
}

static inline mat4 mat4_rotate_y(float a){
    mat4 r = mat4_identity();
    float c = cosf(a), s = sinf(a);
    r.m[0][0] = c;  r.m[0][2] = s;
    r.m[2][0] = -s; r.m[2][2] = c;
    return r;
}

static inline mat4 mat4_rotate_z(float a){
    mat4 r = mat4_identity();
    float c = cosf(a), s = sinf(a);
    r.m[0][0] = c; r.m[0][1] = -s;
    r.m[1][0] = s; r.m[1][1] = c;
    return r;
}

// view matrix: moves the world so the camera sits at the origin looking down -Z
static inline mat4 mat4_look_at(vec3 eye, vec3 target, vec3 up){
    vec3 f = v3_normalize(v3_sub(target, eye));   // forward
    vec3 s = v3_normalize(v3_cross(f, up));       // right
    vec3 u = v3_cross(s, f);                      // true up
    mat4 r = mat4_identity();
    r.m[0][0] = s.x;  r.m[0][1] = s.y;  r.m[0][2] = s.z;  r.m[0][3] = -v3_dot(s, eye);
    r.m[1][0] = u.x;  r.m[1][1] = u.y;  r.m[1][2] = u.z;  r.m[1][3] = -v3_dot(u, eye);
    r.m[2][0] = -f.x; r.m[2][1] = -f.y; r.m[2][2] = -f.z; r.m[2][3] = v3_dot(f, eye);
    return r;
}

// perspective projection. After it, clip.w = -z_eye, and a point is inside the view
// volume when -w <= x,y,z <= w. Dividing by w gives NDC in [-1, 1].
static inline mat4 mat4_perspective(float fov_y, float aspect, float near_z, float far_z){
    float f = 1.0f / tanf(fov_y * 0.5f);
    mat4 r = {{{0}}};
    r.m[0][0] = f / aspect;
    r.m[1][1] = f;
    r.m[2][2] = (far_z + near_z) / (near_z - far_z);
    r.m[2][3] = 2.0f * far_z * near_z / (near_z - far_z);
    r.m[3][2] = -1.0f;
    return r;
}

#endif
