#include "scene/mesh.h"

#include <ctype.h>
#include <stdint.h>
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

typedef struct { int v, vt, vn, f; } Caps;   // allocated sizes of the 4 arrays

// ---- .obj ---------------------------------------------------------------------------

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

// a polygon with nc corners becomes nc-2 triangles (a fan around corner 0)
static void add_polygon(Mesh *m, Caps *cap, int (*corners)[3], int nc){
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
        if (ok) PUSH(m->f, m->nf, cap->f, f);
    }
}

// One line of any length into *line (grown as needed), like POSIX getline (which Windows
// does not have). Returns its length, -1 at the end of the file, -2 (said on stderr) when
// memory runs out (the file is then not read as if it ended there).
static long read_line(char **line, size_t *cap, FILE *f){
    size_t n = 0;
    if (!*line || *cap < 128){
        char *p = realloc(*line, 128);
        if (!p){ fprintf(stderr, "mesh: out of memory reading a line\n"); return -2; }
        *line = p;
        *cap = 128;
    }
    for (;;){
        if (!fgets(*line + n, (int)(*cap - n), f)) break;
        n += strlen(*line + n);
        if (n > 0 && (*line)[n - 1] == '\n') break;
        if (n + 1 < *cap) continue;            // no newline, but the buffer is not full: end of file next
        char *p = realloc(*line, *cap * 2);
        if (!p){ fprintf(stderr, "mesh: out of memory reading a line of %zu bytes\n", *cap * 2); return -2; }
        *line = p;
        *cap *= 2;
    }
    return n > 0 ? (long)n : -1;
}

static int read_obj(FILE *fp, Mesh *m, Caps *cap){
    char *line = NULL;
    size_t linecap = 0;
    long len;
    int cut = 0;                 // faces cut to their first 64 corners
    while ((len = read_line(&line, &linecap, fp)) > 0){
        double x, y, z;   // parse in double, store as real
        if (line[0] == 'v' && line[1] == ' '){
            if (sscanf(line + 2, "%lf %lf %lf", &x, &y, &z) == 3) PUSH(m->v, m->nv, cap->v, vec3(x, y, z));
        } else if (line[0] == 'v' && line[1] == 't'){
            x = y = 0;
            if (sscanf(line + 2, "%lf %lf", &x, &y) >= 2) PUSH(m->vt, m->nvt, cap->vt, vec2(x, y));
        } else if (line[0] == 'v' && line[1] == 'n'){
            if (sscanf(line + 2, "%lf %lf %lf", &x, &y, &z) == 3) PUSH(m->vn, m->nvn, cap->vn, vec3(x, y, z));
        } else if (line[0] == 'f' && isspace((unsigned char)line[1])){
            int corners[64][3];
            int nc = 0;
            char *p = line + 1;
            while (nc < 64 && parse_corner(&p, m->nv, m->nvt, m->nvn, corners[nc])) nc++;
            if (nc == 64){ int more[3]; if (parse_corner(&p, m->nv, m->nvt, m->nvn, more)) cut++; }
            add_polygon(m, cap, corners, nc);
        }
    }
    free(line);
    if (cut) fprintf(stderr, "mesh: %d face(s) of more than 64 corners: their first 64 kept\n", cut);
    return len == -2;
}

// ---- a text's numbers, fast ---------------------------------------------------------

// the rest of a file in memory, a 0 after it (NULL: no memory)
static char *read_rest(FILE *f, size_t *len){
    size_t cap = (size_t)1 << 20, n = 0;
    char *buf = malloc(cap + 1);
    while (buf){
        n += fread(buf + n, 1, cap - n, f);
        if (n < cap) break;
        char *t = realloc(buf, cap * 2 + 1);
        if (!t){ free(buf); return NULL; }
        buf = t;
        cap *= 2;
    }
    if (buf) buf[n] = 0;
    *len = n;
    return buf;
}

// the next line of a text in memory (a 0 put at its end); NULL after the last
static char *next_line(char **at, char *end){
    char *s = *at;
    if (s >= end) return NULL;
    char *e = memchr(s, '\n', (size_t)(end - s));
    if (e){ *e = 0; *at = e + 1; } else *at = end;
    return s;
}

// The number at *p (moved past it), as strtod gives it: its digits as one integer and a
// power of ten; with at most 15 digits and a power within 10^+-22 both are exact in a
// double, so the one multiplication or division rounds once, to the same double a correct
// strtod returns (Clinger 1990). Anything else (more digits, inf, nan, hexadecimal) is
// strtod's.
static double parse_num(char **pp){
    static const double p10[23] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
                                   1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
    char *start = *pp, *s = start;
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n' || *s == '\v' || *s == '\f') s++;
    int neg = 0, nd = 0, e = 0, any = 0;
    if (*s == '-'){ neg = 1; s++; } else if (*s == '+') s++;
    uint64_t mant = 0;
    for (; *s >= '0' && *s <= '9'; s++, any = 1){
        if (nd < 19){ mant = mant * 10 + (uint64_t)(*s - '0'); nd += mant != 0; } else { e++; nd++; }
    }
    if (*s == '.'){
        for (s++; *s >= '0' && *s <= '9'; s++, any = 1){
            if (nd < 19){ mant = mant * 10 + (uint64_t)(*s - '0'); nd += mant != 0; e--; } else nd++;
        }
    }
    if (!any || *s == 'x' || *s == 'X') return strtod(start, pp);
    if (*s == 'e' || *s == 'E'){
        char *t = s + 1;
        int es = 1, ev = 0, digits = 0;
        if (*t == '-'){ es = -1; t++; } else if (*t == '+') t++;
        for (; *t >= '0' && *t <= '9'; t++, digits = 1) if (ev < 10000) ev = ev * 10 + (*t - '0');
        if (digits){ e += es * ev; s = t; }
    }
    if (nd > 15 || e < -22 || e > 22) return strtod(start, pp);
    double v = (double)mant;
    v = e < 0 ? v / p10[-e] : v * p10[e];
    *pp = s;
    return neg ? -v : v;
}

// the integer at *p (moved past it; not moved, and 0, where there is none), as strtol in base 10
static long parse_int(char **pp){
    char *s = *pp;
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n' || *s == '\v' || *s == '\f') s++;
    int neg = 0;
    if (*s == '-'){ neg = 1; s++; } else if (*s == '+') s++;
    if (*s < '0' || *s > '9') return 0;
    long v = 0;
    for (; *s >= '0' && *s <= '9'; s++) if (v < 100000000L) v = v * 10 + (*s - '0');
    *pp = s;
    return neg ? -v : v;
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

static int read_ply(FILE *f, Mesh *m, Caps *cap){
    char *line = NULL;
    size_t linecap = 0;
    long nverts = -1, nfaces = -1;
    int nprops = 0, ix = -1, iy = -1, iz = -1, ascii = 0;
    int in_vertex = 0;

    if (read_line(&line, &linecap, f) <= 0 || strncmp(line, "ply", 3) != 0){ free(line); return -1; }
    while (read_line(&line, &linecap, f) > 0){
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

    // The numbers: the rest of the file read at once and its lines taken from memory, the
    // numbers by parse_num and parse_int. (A line at a time through fgets, strtod and strtol
    // was 300 of the 590 ms the full dragon takes to load: 1.3 million lines, 4.8 million
    // numbers, each call with the stream's lock and the locale.)
    free(line);
    size_t len = 0;
    char *text = read_rest(f, &len), *at = text, *end = text ? text + len : NULL;
    if (!text) return -1;
    for (long i = 0; i < nverts && (line = next_line(&at, end)) != NULL; i++){
        double val[3] = {0, 0, 0};
        char *p = line;
        for (int k = 0; k < nprops; k++){
            double d = parse_num(&p);
            if (k == ix) val[0] = d;
            if (k == iy) val[1] = d;
            if (k == iz) val[2] = d;
        }
        PUSH(m->v, m->nv, cap->v, vec3(val[0], val[1], val[2]));
    }
    for (long i = 0; i < nfaces && (line = next_line(&at, end)) != NULL; i++){
        char *p = line;
        long n = parse_int(&p);
        if (n < 3 || n > 64) continue;
        int corners[64][3];
        for (int k = 0; k < n; k++){
            corners[k][0] = (int)parse_int(&p);
            corners[k][1] = corners[k][2] = -1;
        }
        add_polygon(m, cap, corners, (int)n);
    }
    free(text);
    return 0;
}

// ---- public -------------------------------------------------------------------------

static int ends_with(const char *s, const char *suffix){
    size_t a = strlen(s), b = strlen(suffix);
    if (a < b) return 0;
    for (size_t i = 0; i < b; i++)
        if (tolower((unsigned char)s[a - b + i]) != suffix[i]) return 0;
    return 1;
}

Mesh *mesh_load(const char *path){
    FILE *fp = fopen(path, "r");
    if (!fp) return NULL;

    Mesh *m = calloc(1, sizeof(*m));
    if (!m){ fclose(fp); return NULL; }
    Caps cap = {0, 0, 0, 0};
    int err = ends_with(path, ".ply") ? read_ply(fp, m, &cap) : read_obj(fp, m, &cap);
    fclose(fp);

    if (err || m->nv == 0 || m->nf == 0){
        mesh_destroy(m);
        return NULL;
    }
    if (m->nvn == 0) mesh_compute_normals(m);
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

void mesh_compute_normals(Mesh *m){
    free(m->vn);
    m->vn = calloc((size_t)m->nv, sizeof(Vec3));
    if (!m->vn){ perror("calloc"); exit(1); }
    m->nvn = m->nv;
    for (int i = 0; i < m->nf; i++){
        Face *f = &m->f[i];
        Vec3 a = m->v[f->v[0]], b = m->v[f->v[1]], c = m->v[f->v[2]];
        Vec3 n = v_cross(v_sub(b, a), v_sub(c, a));   // length = 2 * triangle area
        for (int j = 0; j < 3; j++){
            m->vn[f->v[j]] = v_add(m->vn[f->v[j]], n);
            f->n[j] = f->v[j];
        }
    }
}

void mesh_normalize(Mesh *m){
    Vec3 lo = m->v[0], hi = m->v[0];
    for (int i = 1; i < m->nv; i++){
        Vec3 p = m->v[i];
        if (p.x < lo.x) lo.x = p.x;
        if (p.y < lo.y) lo.y = p.y;
        if (p.z < lo.z) lo.z = p.z;
        if (p.x > hi.x) hi.x = p.x;
        if (p.y > hi.y) hi.y = p.y;
        if (p.z > hi.z) hi.z = p.z;
    }
    Vec3 center = v_scale(v_add(lo, hi), 0.5f);
    Vec3 half = v_scale(v_sub(hi, lo), 0.5f);
    real biggest = half.x > half.y ? half.x : half.y;
    if (half.z > biggest) biggest = half.z;
    if (biggest < 1e-12f) biggest = 1.0f;
    for (int i = 0; i < m->nv; i++)
        m->v[i] = v_scale(v_sub(m->v[i], center), 1.0f / biggest);
}

void mesh_place(Mesh *m, real size, Vec3 base, real rot_y_deg){
    real ang = rot_y_deg * PI / 180.0, c = r_cos(ang), sn = r_sin(ang);
    for (int i = 0; i < m->nv; i++){
        Vec3 p = m->v[i];
        m->v[i] = vec3(c * p.x + sn * p.z, p.y, -sn * p.x + c * p.z);
    }
    for (int i = 0; i < m->nvn; i++){
        Vec3 p = m->vn[i];
        m->vn[i] = vec3(c * p.x + sn * p.z, p.y, -sn * p.x + c * p.z);
    }
    Vec3 lo = m->v[0], hi = m->v[0];
    for (int i = 1; i < m->nv; i++){
        lo = vec3(r_fmin(lo.x, m->v[i].x), r_fmin(lo.y, m->v[i].y), r_fmin(lo.z, m->v[i].z));
        hi = vec3(r_fmax(hi.x, m->v[i].x), r_fmax(hi.y, m->v[i].y), r_fmax(hi.z, m->v[i].z));
    }
    real biggest = v_max(v_sub(hi, lo));
    real scale = biggest > 0 ? size / biggest : 1.0;
    Vec3 anchor = vec3((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
    for (int i = 0; i < m->nv; i++) m->v[i] = v_add(v_scale(v_sub(m->v[i], anchor), scale), base);
}
