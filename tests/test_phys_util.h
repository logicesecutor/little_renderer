// test_phys_util.h - what test_physics.c and test_gpu_rigid.c share: meshes as triangles (a
// box, a torus) and the fastest point of a world.
#ifndef LR_TEST_PHYS_UTIL_H
#define LR_TEST_PHYS_UTIL_H

#include "lr.h"

// the 12 triangles of a box (half sizes h), turned by r and moved to c, in a growing array
static inline int box_tris(Triangle *t, Vec3 h, Mat3 r, Vec3 c){
    Vec3 p[8];
    for (int i = 0; i < 8; i++)
        p[i] = v_add(c, mat3_mul_vec(r, vec3((i & 1) ? h.x : -h.x, (i & 2) ? h.y : -h.y, (i & 4) ? h.z : -h.z)));
    static const int q[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
    int n = 0;
    for (int f = 0; f < 6; f++){
        Vec3 a = p[q[f][0]], b = p[q[f][1]], cc = p[q[f][2]], d = p[q[f][3]];
        t[n].q = a; t[n].u = v_sub(b, a); t[n].v = v_sub(cc, a); t[n].n[0] = -1; t[n].material = 0; n++;
        t[n].q = a; t[n].u = v_sub(cc, a); t[n].v = v_sub(d, a); t[n].n[0] = -1; t[n].material = 0; n++;
    }
    return n;
}

// a torus around the y axis (radii R, r), nu x nv quads
static inline int torus_tris(Triangle *t, real R, real r, int nu, int nv){
    int n = 0;
    for (int i = 0; i < nu; i++)
        for (int j = 0; j < nv; j++){
            Vec3 p[4];
            for (int k = 0; k < 4; k++){
                real u = (i + (k == 1 || k == 2)) * 2 * (real)PI / nu, v = (j + (k >= 2)) * 2 * (real)PI / nv;
                p[k] = vec3((R + r * r_cos(v)) * r_cos(u), r * r_sin(v), (R + r * r_cos(v)) * r_sin(u));
            }
            t[n].q = p[0]; t[n].u = v_sub(p[1], p[0]); t[n].v = v_sub(p[2], p[0]); t[n].n[0] = -1; t[n].material = 0; n++;
            t[n].q = p[0]; t[n].u = v_sub(p[2], p[0]); t[n].v = v_sub(p[3], p[0]); t[n].n[0] = -1; t[n].material = 0; n++;
        }
    return n;
}

// the fastest point of any body (center speed + spin * size), units / s
static inline double max_speed(const RigidWorld *w){
    double m = 0;
    for (int i = 0; i < w->nbodies; i++){
        const Body *b = &w->bodies[i];
        double s = v_len(b->vel) + v_len(b->angvel) * v_len(b->half);
        if (s > m) m = s;
    }
    return m;
}

#endif
