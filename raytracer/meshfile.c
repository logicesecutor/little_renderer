#define _POSIX_C_SOURCE 200809L   // getline
#include "meshfile.h"

#include "rng.h"   // PI
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- growable arrays ----------------------------------------------------------------

typedef struct { Vec3 *a; int n, cap; } VecList;
typedef struct { int v[3], n[3]; } Tri;          // position / normal index per corner
typedef struct { Tri *a; int n, cap; } TriList;

static void push_vec(VecList *l, Vec3 v){
    if (l->n >= l->cap){
        l->cap = l->cap ? l->cap * 2 : 256;
        l->a = realloc(l->a, (size_t)l->cap * sizeof(Vec3));
        if (!l->a){ perror("realloc"); exit(1); }
    }
    l->a[l->n++] = v;
}

static void push_tri(TriList *l, Tri t){
    if (l->n >= l->cap){
        l->cap = l->cap ? l->cap * 2 : 256;
        l->a = realloc(l->a, (size_t)l->cap * sizeof(Tri));
        if (!l->a){ perror("realloc"); exit(1); }
    }
    l->a[l->n++] = t;
}

// ---- .obj ---------------------------------------------------------------------------

// OBJ indices start at 1; negative ones count back from the end; 0 / missing -> -1
static int fix_index(long i, int count){
    if (i > 0) return (int)(i - 1);
    if (i < 0) return count + (int)i;
    return -1;
}

// one "v", "v/t", "v//n" or "v/t/n" token. Returns 0 at the end of the line.
static int parse_corner(char **sp, int nv, int nn, int *vi, int *ni){
    char *p = *sp;
    while (*p == ' ' || *p == '\t') p++;
    char *end;
    long v = strtol(p, &end, 10);
    if (end == p) return 0;
    p = end;
    long n = 0;
    if (*p == '/'){
        p++;
        if (*p != '/') strtol(p, &p, 10);         // texture index: not needed here
        if (*p == '/'){ p++; n = strtol(p, &p, 10); }
    }
    *vi = fix_index(v, nv);
    *ni = fix_index(n, nn);
    *sp = p;
    return 1;
}

static int read_obj(FILE *f, VecList *pos, VecList *nor, TriList *tris){
    char *line = NULL;
    size_t cap = 0;
    while (getline(&line, &cap, f) > 0){
        Vec3 p;
        if (line[0] == 'v' && line[1] == ' '){
            if (sscanf(line + 2, "%lf %lf %lf", &p.x, &p.y, &p.z) == 3) push_vec(pos, p);
        } else if (line[0] == 'v' && line[1] == 'n'){
            if (sscanf(line + 2, "%lf %lf %lf", &p.x, &p.y, &p.z) == 3) push_vec(nor, p);
        } else if (line[0] == 'f' && isspace((unsigned char)line[1])){
            int cv[64], cn[64], nc = 0;
            char *q = line + 1;
            while (nc < 64 && parse_corner(&q, pos->n, nor->n, &cv[nc], &cn[nc])) nc++;
            for (int k = 1; k + 1 < nc; k++){            // polygon -> fan of triangles
                Tri t = {{cv[0], cv[k], cv[k + 1]}, {cn[0], cn[k], cn[k + 1]}};
                push_tri(tris, t);
            }
        }
    }
    free(line);
    return 0;
}

// ---- .ply (ASCII only) --------------------------------------------------------------
//
//   ply
//   format ascii 1.0
//   element vertex 437645          <- N lines of numbers
//   property float x               <- one per number on the line
//   property float y
//   property float z
//   element face 871414            <- M lines: "count i0 i1 i2 ..."
//   property list uchar int vertex_indices
//   end_header

static int read_ply(FILE *f, VecList *pos, TriList *tris){
    char *line = NULL;
    size_t cap = 0;
    long nverts = -1, nfaces = -1;
    int nprops = 0, ix = -1, iy = -1, iz = -1, ascii = 0;
    int in_vertex = 0;

    if (getline(&line, &cap, f) <= 0 || strncmp(line, "ply", 3) != 0){ free(line); return -1; }
    while (getline(&line, &cap, f) > 0){
        char name[64];
        long count;
        if (!strncmp(line, "format ascii", 12)) ascii = 1;
        else if (sscanf(line, "element %63s %ld", name, &count) == 2){
            in_vertex = !strcmp(name, "vertex");
            if (in_vertex) nverts = count;
            else if (!strcmp(name, "face")) nfaces = count;
            else { fprintf(stderr, "ply: element '%s' is not supported\n", name); free(line); return -1; }
        } else if (in_vertex && sscanf(line, "property %*s %63s", name) == 1){
            if (!strcmp(name, "x")) ix = nprops;
            if (!strcmp(name, "y")) iy = nprops;
            if (!strcmp(name, "z")) iz = nprops;
            nprops++;
        } else if (!strncmp(line, "end_header", 10)) break;
    }
    if (!ascii){ fprintf(stderr, "ply: only ASCII PLY files are supported\n"); free(line); return -1; }
    if (nverts < 0 || nfaces < 0 || ix < 0 || iy < 0 || iz < 0){
        fprintf(stderr, "ply: header without vertex x/y/z or faces\n");
        free(line);
        return -1;
    }

    for (long i = 0; i < nverts && getline(&line, &cap, f) > 0; i++){
        double val[3] = {0, 0, 0};
        char *p = line;
        for (int k = 0; k < nprops; k++){
            double d = strtod(p, &p);
            if (k == ix) val[0] = d;
            if (k == iy) val[1] = d;
            if (k == iz) val[2] = d;
        }
        push_vec(pos, vec3(val[0], val[1], val[2]));
    }
    for (long i = 0; i < nfaces && getline(&line, &cap, f) > 0; i++){
        char *p = line;
        long n = strtol(p, &p, 10);
        if (n < 3 || n > 64) continue;
        int idx[64];
        for (int k = 0; k < n; k++) idx[k] = (int)strtol(p, &p, 10);
        for (int k = 1; k + 1 < n; k++){
            Tri t = {{idx[0], idx[k], idx[k + 1]}, {-1, -1, -1}};
            push_tri(tris, t);
        }
    }
    free(line);
    return 0;
}

// ---- common -------------------------------------------------------------------------

static int ends_with(const char *s, const char *suffix){
    size_t a = strlen(s), b = strlen(suffix);
    if (a < b) return 0;
    for (size_t i = 0; i < b; i++)
        if (tolower((unsigned char)s[a - b + i]) != suffix[i]) return 0;
    return 1;
}

int mesh_load(Scene *s, const char *path, int material, double size, Vec3 base, double rot_y_deg){
    int is_ply = ends_with(path, ".ply");
    FILE *f = fopen(path, "r");
    if (!f){ perror(path); return -1; }

    VecList pos = {0}, nor = {0};
    TriList tris = {0};
    int err = is_ply ? read_ply(f, &pos, &tris) : read_obj(f, &pos, &nor, &tris);
    fclose(f);

    // drop triangles that point at vertices that do not exist (broken files)
    int kept = 0;
    for (int i = 0; i < tris.n; i++){
        Tri t = tris.a[i];
        int ok = 1;
        for (int j = 0; j < 3; j++){
            if (t.v[j] < 0 || t.v[j] >= pos.n) ok = 0;
            if (t.n[j] >= nor.n) t.n[j] = -1;
        }
        if (ok) tris.a[kept++] = t;
    }
    tris.n = kept;

    if (err || pos.n == 0 || tris.n == 0){
        if (!err) fprintf(stderr, "%s: no triangles found\n", path);
        free(pos.a); free(nor.a); free(tris.a);
        return -1;
    }

    // No normals in the file (the Stanford scans have none): compute smooth ones by adding up
    // the normals of every triangle around a vertex. The cross product is not normalized,
    // so big triangles count more than tiny ones (area weighting).
    if (nor.n == 0){
        nor.a = calloc((size_t)pos.n, sizeof(Vec3));
        if (!nor.a){ perror("calloc"); exit(1); }
        nor.n = nor.cap = pos.n;
        for (int i = 0; i < tris.n; i++){
            Tri *t = &tris.a[i];
            Vec3 a = pos.a[t->v[0]], b = pos.a[t->v[1]], c = pos.a[t->v[2]];
            Vec3 fn = v_cross(v_sub(b, a), v_sub(c, a));
            for (int j = 0; j < 3; j++){
                nor.a[t->v[j]] = v_add(nor.a[t->v[j]], fn);
                t->n[j] = t->v[j];
            }
        }
    }

    // turn around y, then fit into the box: largest side -> size, stand it on base
    double ang = rot_y_deg * PI / 180.0, c = cos(ang), sn = sin(ang);
    for (int i = 0; i < pos.n; i++){
        Vec3 p = pos.a[i];
        pos.a[i] = vec3(c * p.x + sn * p.z, p.y, -sn * p.x + c * p.z);
    }
    for (int i = 0; i < nor.n; i++){
        Vec3 p = nor.a[i];
        nor.a[i] = vec3(c * p.x + sn * p.z, p.y, -sn * p.x + c * p.z);
    }
    Vec3 lo = pos.a[0], hi = pos.a[0];
    for (int i = 1; i < pos.n; i++){
        lo = vec3(fmin(lo.x, pos.a[i].x), fmin(lo.y, pos.a[i].y), fmin(lo.z, pos.a[i].z));
        hi = vec3(fmax(hi.x, pos.a[i].x), fmax(hi.y, pos.a[i].y), fmax(hi.z, pos.a[i].z));
    }
    double biggest = v_max(v_sub(hi, lo));
    double scale = biggest > 0 ? size / biggest : 1.0;
    Vec3 anchor = vec3((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
    for (int i = 0; i < pos.n; i++) pos.a[i] = v_add(v_scale(v_sub(pos.a[i], anchor), scale), base);

    for (int i = 0; i < tris.n; i++){
        const Tri *t = &tris.a[i];
        Vec3 nrm[3];
        int has_n = t->n[0] >= 0 && t->n[1] >= 0 && t->n[2] >= 0;
        if (has_n) for (int j = 0; j < 3; j++) nrm[j] = nor.a[t->n[j]];
        scene_add_prim(s, make_triangle(pos.a[t->v[0]], pos.a[t->v[1]], pos.a[t->v[2]],
                                        has_n ? nrm : NULL, material));
    }

    int count = tris.n;
    free(pos.a); free(nor.a); free(tris.a);
    return count;
}
