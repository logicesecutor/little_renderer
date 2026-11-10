#define _POSIX_C_SOURCE 200809L   // for getline
#include "mesh.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// append to a growing array
#define PUSH(arr, n, cap, val) do {                                          \
        if ((n) >= (cap)){                                                    \
            (cap) = (cap) ? (cap) * 2 : 64;                                   \
            (arr) = realloc((arr), (size_t)(cap) * sizeof(*(arr)));           \
            if (!(arr)){ perror("realloc"); exit(1); }                        \
        }                                                                     \
        (arr)[(n)++] = (val);                                                 \
    } while (0)

// OBJ indices start at 1; negative ones count back from the end; 0 / missing -> -1
static int fix_index(long i, int count){
    if (i > 0) return (int)(i - 1);
    if (i < 0) return count + (int)i;
    return -1;
}

// parses one "v", "v/t", "v//n" or "v/t/n" token, moves *sp past it. Returns 0 at end of line.
static int parse_corner(char **sp, int nv, int nvt, int nvn, int out[3]){
    char *p = *sp;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0' || *p == '\n' || *p == '\r') return 0;

    char *end;
    long v = strtol(p, &end, 10);
    if (end == p) return 0;
    p = end;
    long t = 0, n = 0;
    if (*p == '/'){
        p++;
        if (*p != '/') t = strtol(p, &p, 10);
        if (*p == '/'){ p++; n = strtol(p, &p, 10); }
    }
    out[0] = fix_index(v, nv);
    out[1] = fix_index(t, nvt);
    out[2] = fix_index(n, nvn);
    *sp = p;
    return 1;
}

Mesh *mesh_load_obj(const char *path){
    FILE *fp = fopen(path, "r");
    if (!fp) return NULL;

    Mesh *m = calloc(1, sizeof(*m));
    int capv = 0, capvt = 0, capvn = 0, capf = 0;
    char *line = NULL;
    size_t linecap = 0;

    while (getline(&line, &linecap, fp) > 0){
        if (line[0] == 'v' && line[1] == ' '){
            vec3 p;
            if (sscanf(line + 2, "%f %f %f", &p.x, &p.y, &p.z) == 3) PUSH(m->v, m->nv, capv, p);
        } else if (line[0] == 'v' && line[1] == 't'){
            vec2 t = {0, 0};
            if (sscanf(line + 2, "%f %f", &t.x, &t.y) >= 2) PUSH(m->vt, m->nvt, capvt, t);
        } else if (line[0] == 'v' && line[1] == 'n'){
            vec3 n;
            if (sscanf(line + 2, "%f %f %f", &n.x, &n.y, &n.z) == 3) PUSH(m->vn, m->nvn, capvn, n);
        } else if (line[0] == 'f' && isspace((unsigned char)line[1])){
            int corners[64][3];
            int nc = 0;
            char *p = line + 1;
            while (nc < 64 && parse_corner(&p, m->nv, m->nvt, m->nvn, corners[nc])) nc++;

            // a polygon with nc corners becomes nc-2 triangles (a fan around corner 0)
            for (int k = 1; k + 1 < nc; k++){
                int idx[3] = {0, k, k + 1};
                Face f;
                int ok = 1;
                for (int j = 0; j < 3; j++){
                    int *c = corners[idx[j]];
                    f.v[j] = c[0];
                    f.t[j] = (c[1] >= 0 && c[1] < m->nvt) ? c[1] : -1;
                    f.n[j] = (c[2] >= 0 && c[2] < m->nvn) ? c[2] : -1;
                    if (f.v[j] < 0 || f.v[j] >= m->nv) ok = 0;   // broken file: skip this face
                }
                if (ok) PUSH(m->f, m->nf, capf, f);
            }
        }
    }
    free(line);
    fclose(fp);

    if (m->nv == 0 || m->nf == 0){
        mesh_destroy(m);
        return NULL;
    }
    return m;
}

void mesh_destroy(Mesh *m){
    if (!m) return;
    free(m->v);
    free(m->vt);
    free(m->vn);
    free(m->f);
    free(m);
}

void mesh_normalize(Mesh *m){
    vec3 lo = m->v[0], hi = m->v[0];
    for (int i = 1; i < m->nv; i++){
        vec3 p = m->v[i];
        if (p.x < lo.x) lo.x = p.x;
        if (p.y < lo.y) lo.y = p.y;
        if (p.z < lo.z) lo.z = p.z;
        if (p.x > hi.x) hi.x = p.x;
        if (p.y > hi.y) hi.y = p.y;
        if (p.z > hi.z) hi.z = p.z;
    }
    vec3 center = v3_scale(v3_add(lo, hi), 0.5f);
    vec3 half = v3_scale(v3_sub(hi, lo), 0.5f);
    float biggest = half.x > half.y ? half.x : half.y;
    if (half.z > biggest) biggest = half.z;
    if (biggest < 1e-12f) biggest = 1.0f;
    for (int i = 0; i < m->nv; i++)
        m->v[i] = v3_scale(v3_sub(m->v[i], center), 1.0f / biggest);
}

// For files without normals: add up the (area-weighted) normals of all triangles touching
// each vertex, then normalize. The normal index of every corner becomes its position index.
void mesh_compute_normals(Mesh *m){
    free(m->vn);
    m->vn = calloc((size_t)m->nv, sizeof(vec3));
    m->nvn = m->nv;
    for (int i = 0; i < m->nf; i++){
        Face *f = &m->f[i];
        vec3 a = m->v[f->v[0]], b = m->v[f->v[1]], c = m->v[f->v[2]];
        vec3 n = v3_cross(v3_sub(b, a), v3_sub(c, a));   // length = 2 * triangle area
        for (int j = 0; j < 3; j++){
            m->vn[f->v[j]] = v3_add(m->vn[f->v[j]], n);
            f->n[j] = f->v[j];
        }
    }
    for (int i = 0; i < m->nv; i++) m->vn[i] = v3_normalize(m->vn[i]);
}
