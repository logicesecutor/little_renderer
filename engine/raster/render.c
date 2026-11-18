#include "raster/render.h"

#include "raster/raster.h"
#include "scene/brdf.h"
#include "scene/env.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define AMBIENT 0.12f

// A vertex after the projection matrix, before the divide by w.
typedef struct {
    Vec4 clip;
    real vary[V_COUNT];
} ClipVert;

// what the fragment shaders get as ctx
typedef struct {
    Vec3 base;            // surface color
    Vec3 light;           // unit vector towards the light
    Vec3 eye;             // camera position
    const Texture *tex;   // NULL = use base
} Shading;

static real lambert(Vec3 n, Vec3 l){
    real d = v_dot(n, l);
    return AMBIENT + (1.0f - AMBIENT) * (d > 0 ? d : 0);
}

// ---- fragment shaders (color of one pixel) ------------------------------------------

// flat and gouraud: the vertex stage already computed the brightness
static uint32_t shade_light(const real *v, const void *ctx){
    const Shading *s = ctx;
    real l = v[V_LIGHT];
    return rgbf(s->base.x * l, s->base.y * l, s->base.z * l);
}

// phong (+ optional texture): lighting evaluated per pixel from the interpolated normal
static uint32_t shade_phong(const real *v, const void *ctx){
    const Shading *s = ctx;
    Vec3 n = v_norm(vec3(v[V_NX], v[V_NY], v[V_NZ]));
    Vec3 p = vec3(v[V_PX], v[V_PY], v[V_PZ]);

    Vec3 base = s->tex ? texture_sample(s->tex, v[V_U], v[V_V]) : s->base;

    real diff = v_dot(n, s->light);
    real spec = 0.0f;
    if (diff > 0){
        Vec3 to_eye = v_norm(v_sub(s->eye, p));
        Vec3 h = v_norm(v_add(s->light, to_eye));   // Blinn-Phong half vector
        real nh = v_dot(n, h);
        spec = nh > 0 ? 0.4f * r_pow(nh, 40.0f) : 0.0f;
    } else {
        diff = 0;
    }
    real l = AMBIENT + (1.0f - AMBIENT) * diff;
    return rgbf(base.x * l + spec, base.y * l + spec, base.z * l + spec);
}

// ---- pipeline pieces ----------------------------------------------------------------

static ClipVert lerp_vert(const ClipVert *a, const ClipVert *b, real t){
    ClipVert r;
    r.clip.x = a->clip.x + (b->clip.x - a->clip.x) * t;
    r.clip.y = a->clip.y + (b->clip.y - a->clip.y) * t;
    r.clip.z = a->clip.z + (b->clip.z - a->clip.z) * t;
    r.clip.w = a->clip.w + (b->clip.w - a->clip.w) * t;
    for (int k = 0; k < V_COUNT; k++) r.vary[k] = a->vary[k] + (b->vary[k] - a->vary[k]) * t;
    return r;
}

// Sutherland-Hodgman against the near plane (z >= -w), and against a guard band GUARD
// times the screen's size around it (|x|, |y| <= GUARD w). Without the near plane, points
// behind the camera would get divided by a negative w and explode across the screen; without
// the guard band, a big triangle reaching far off screen (a floor seen to the horizon: its
// corner by the camera lands millions of pixels away) gets edge functions whose rounding
// leaves holes along its far edge. Each plane adds at most one vertex: 3 -> at most 8.
#define GUARD 2.0f
#define CLIP_MAX 8
static int clip_plane(const ClipVert *in, int n, ClipVert *out, int plane){
    int m = 0;
    for (int i = 0; i < n; i++){
        const ClipVert *a = &in[i], *b = &in[(i + 1) % n];
        real da, db;                        // >= 0 means inside
        switch (plane){
        case 0: da = a->clip.z + a->clip.w; db = b->clip.z + b->clip.w; break;
        case 1: da = GUARD * a->clip.w - a->clip.x; db = GUARD * b->clip.w - b->clip.x; break;
        case 2: da = GUARD * a->clip.w + a->clip.x; db = GUARD * b->clip.w + b->clip.x; break;
        case 3: da = GUARD * a->clip.w - a->clip.y; db = GUARD * b->clip.w - b->clip.y; break;
        default: da = GUARD * a->clip.w + a->clip.y; db = GUARD * b->clip.w + b->clip.y; break;
        }
        if (da >= 0) out[m++] = *a;
        if ((da >= 0) != (db >= 0)) out[m++] = lerp_vert(a, b, da / (da - db));
    }
    return m;
}

// the corner c needs clipping: behind the near plane or outside the guard band
static inline int clip_needed(Vec4 c){
    return c.z + c.w < 0 || c.x > GUARD * c.w || -c.x > GUARD * c.w || c.y > GUARD * c.w || -c.y > GUARD * c.w;
}

// the triangle clipped by all five planes (out: CLIP_MAX vertices); returns how many
static int clip_triangle(const ClipVert *in, ClipVert *out){
    ClipVert a[CLIP_MAX], b[CLIP_MAX];
    int n = clip_plane(in, 3, a, 0);
    for (int p = 1; p < 5 && n >= 3; p++){
        n = clip_plane(a, n, b, p);
        memcpy(a, b, sizeof(ClipVert) * (size_t)n);
    }
    if (n < 3) return 0;
    memcpy(out, a, sizeof(ClipVert) * (size_t)n);
    return n;
}

// perspective divide + viewport transform (NDC [-1,1] -> pixels, flipping Y)
static ScreenVert to_screen(const ClipVert *c, const Framebuffer *fb){
    ScreenVert s;
    real iw = 1.0f / c->clip.w;
    s.x = (c->clip.x * iw * 0.5f + 0.5f) * fb->w;
    s.y = (0.5f - c->clip.y * iw * 0.5f) * fb->h;
    s.z = c->clip.z * iw * 0.5f + 0.5f;
    s.inv_w = iw;
    memcpy(s.vary, c->vary, sizeof(s.vary));
    return s;
}

// The second half of the pipeline, for one triangle already in clip space: near clip,
// perspective divide, cull, rasterize. Only rows row_lo..row_hi are written (see
// draw_triangles below: every thread owns a band of rows).
static int poly_area_ok(const ScreenVert *sv, int np, int cull){
    // facing: signed area of the polygon (shoelace). Counter-clockwise on screen (y down)
    // is NEGATIVE, see draw_triangle.
    real area = 0;
    for (int k = 0; k < np; k++){
        const ScreenVert *a = &sv[k], *b = &sv[(k + 1) % np];
        area += a->x * b->y - b->x * a->y;
    }
    return !(cull && area >= 0);
}

// pass, ids, id: draw_triangle_rows_pass's (RASTER_DIRECT: as one pass, ids unused); the
// clipped polygon's fan triangle k - 1 gets the id id + k - 1 (at most CLIP_MAX - 2 of them).
// Returns the pixels a RASTER_DEPTH pass wrote.
static int draw_clip_triangle(Framebuffer *fb, const ClipVert *cv, RenderMode mode, int cull,
                              FragShader shader, const void *ctx, uint32_t wire_color,
                              int row_lo, int row_hi, int pass, int *ids, int id){
    // 3. clip against the near plane and the guard band (when a corner needs it)
    ClipVert poly[CLIP_MAX];
    int np = 3;
    if (clip_needed(cv[0].clip) || clip_needed(cv[1].clip) || clip_needed(cv[2].clip)){
        np = clip_triangle(cv, poly);
        if (np < 3) return 0;
    } else memcpy(poly, cv, sizeof(ClipVert) * 3);

    // 4. divide by w, map to pixels
    ScreenVert sv[CLIP_MAX];
    for (int k = 0; k < np; k++) sv[k] = to_screen(&poly[k], fb);

    // 5. back-face culling
    if (!poly_area_ok(sv, np, cull)) return 0;

    // 6. rasterize
    int written = 0;
    if (mode == MODE_WIREFRAME){
        for (int k = 0; k < np; k++){
            const ScreenVert *a = &sv[k], *b = &sv[(k + 1) % np];
            draw_line_rows(fb, (int)r_lround(a->x), (int)r_lround(a->y),
                               (int)r_lround(b->x), (int)r_lround(b->y), wire_color, row_lo, row_hi);
        }
    } else {
        for (int k = 1; k + 1 < np; k++)
            written += draw_triangle_rows_pass(fb, &sv[0], &sv[k], &sv[k + 1], shader, ctx, 0, row_lo, row_hi,
                                               pass, ids, id + k - 1);
    }
    return written;
}

// ---- drawing many triangles on many cores ---------------------------------------------
//
// render_mesh and render_scene both describe their triangles as a TriJob: how many there
// are, where the corners of triangle i are, and the full vertex stage of triangle i.
// draw_triangles then draws them in two parallel passes:
//
//   1. binning, in parallel over triangles: transform only the corners, clip, project, and
//      note which horizontal BANDS of the screen (RASTER_BAND_ROWS rows each) the triangle
//      can touch. Triangles behind the camera, off screen or culled are dropped here.
//   2. drawing, in parallel over bands: every band is drawn by one thread, which runs the
//      full vertex stage for the triangles binned to it and fills only its own rows.
//
// No two threads ever write the same pixel, so there are no locks. And every band draws its
// triangles in their original order, so the z-buffer resolves ties exactly as a single
// thread would: the image is bit-for-bit the same for any number of threads.
// A triangle that spans several bands runs its vertex stage once per band; with bands of
// 16 rows that costs little (big triangles are few, small ones touch one or two bands).

#define RASTER_BAND_ROWS 16

typedef struct TriJob TriJob;
struct TriJob {
    int ntri;
    RenderMode mode;
    int cull;
    Mat4 view_proj;
    const void *data;
    // world-space corners of triangle i
    void (*corners)(const TriJob *j, int i, Vec3 *wp);
    // the whole vertex stage of triangle i: clip-space vertices with their varyings, plus
    // the fragment shader, its context and the wireframe color
    void (*vertices)(const TriJob *j, int i, ClipVert *cv, FragShader *shader, const void **ctx, uint32_t *wire);
};

typedef struct {
    int *tri;    // triangle indices grouped by band, in their original order within a band
    int *off;    // band b owns tri[off[b] .. off[b + 1])
} Bins;

static void *xmalloc(size_t n){
    void *p = malloc(n ? n : 1);
    if (!p){ perror("malloc"); exit(1); }
    return p;
}

// Pass 1 for one triangle: the rows it may touch, or 0 if it draws nothing. Uses the same
// clip / projection / cull arithmetic as pass 2, so it never drops a triangle that pass 2
// would draw.
//   filled triangles: draw_triangle samples pixel centers, so only rows whose center
//                     y + 0.5 lies in [ymin, ymax];
//   wireframe:        the line ends are rounded, so rows lround(ymin) .. lround(ymax).
// ROWS_TOL (1/64 pixel) is a safety margin for rounding (it only matters for a triangle
// that ends exactly on a pixel center), so a wrong guess can never lose a pixel.
#define ROWS_TOL 0.015625f

// floor and ceil for values well inside the int range
static inline int floor_small(real x){ int t = (int)x; return t - (x < (real)t); }
static inline int ceil_small(real x){ int t = (int)x; return t + (x > (real)t); }

static int triangle_rows(const TriJob *j, const Framebuffer *fb, int i, int *r0, int *r1){
    Vec3 wp[3];
    j->corners(j, i, wp);
    ScreenVert sv[CLIP_MAX];
    int np = 3;
    Vec4 c[3];
    for (int k = 0; k < 3; k++) c[k] = mat4_mul_vec4(j->view_proj, (Vec4){wp[k].x, wp[k].y, wp[k].z, 1.0f});
    if (!clip_needed(c[0]) && !clip_needed(c[1]) && !clip_needed(c[2])){
        // fast path, the usual case: all 3 corners in front of the near plane and inside the
        // guard band. Same formulas as to_screen, without copying the varyings.
        for (int k = 0; k < 3; k++){
            real iw = 1.0f / c[k].w;
            sv[k].x = (c[k].x * iw * 0.5f + 0.5f) * fb->w;
            sv[k].y = (0.5f - c[k].y * iw * 0.5f) * fb->h;
        }
    } else {
        ClipVert cv[3], poly[CLIP_MAX];
        memset(cv, 0, sizeof(cv));
        for (int k = 0; k < 3; k++) cv[k].clip = c[k];
        np = clip_triangle(cv, poly);
        if (np < 3) return 0;                               // behind the camera, or off screen
        for (int k = 0; k < np; k++) sv[k] = to_screen(&poly[k], fb);
    }
    if (!poly_area_ok(sv, np, j->cull)) return 0;           // back face

    // (plain comparisons: fminf / floorf are library calls on a default build)
    real xmin = sv[0].x, xmax = sv[0].x, ymin = sv[0].y, ymax = sv[0].y;
    for (int k = 1; k < np; k++){
        if (sv[k].x < xmin) xmin = sv[k].x;
        if (sv[k].x > xmax) xmax = sv[k].x;
        if (sv[k].y < ymin) ymin = sv[k].y;
        if (sv[k].y > ymax) ymax = sv[k].y;
    }
    if (!(xmin <= xmax && ymin <= ymax)) return 0;          // NaN: nothing sensible to draw
    if (xmax < -1 || xmin > fb->w + 1 || ymax < -1 || ymin > fb->h + 1) return 0;   // off screen
    if (ymin < -1) ymin = -1;                               // now small enough for int casts
    if (ymax > fb->h + 1) ymax = fb->h + 1;
    int a, b;
    if (j->mode == MODE_WIREFRAME){
        a = floor_small(ymin + 0.5f - ROWS_TOL);
        b = floor_small(ymax + 0.5f + ROWS_TOL);
    } else {
        a = ceil_small(ymin - 0.5f - ROWS_TOL);
        b = floor_small(ymax - 0.5f + ROWS_TOL);
    }
    if (a < 0) a = 0;
    if (b > fb->h - 1) b = fb->h - 1;
    if (a > b) return 0;                                    // between two pixel centers
    *r0 = a;
    *r1 = b;
    return 1;
}

// The depth pass of triangle i from its corners alone: the clip and screen positions are the
// full vertex stage's to the bit (both are view_proj times the same world corners, and
// clipping interpolates the positions on their own), without its normals, texture
// coordinates and lighting.
static int depth_triangle(Framebuffer *fb, const TriJob *j, int i, int row_lo, int row_hi, int *ids, int id){
    Vec3 wp[3];
    j->corners(j, i, wp);
    ClipVert cv[3];
    memset(cv, 0, sizeof(cv));
    for (int k = 0; k < 3; k++) cv[k].clip = mat4_mul_vec4(j->view_proj, (Vec4){wp[k].x, wp[k].y, wp[k].z, 1.0f});
    return draw_clip_triangle(fb, cv, j->mode, j->cull, NULL, NULL, 0, row_lo, row_hi, RASTER_DEPTH, ids, id);
}

static void draw_triangles(Framebuffer *fb, const TriJob *j){
    if (j->ntri <= 0 || fb->w <= 0 || fb->h <= 0) return;
    const int nbands = (fb->h + RASTER_BAND_ROWS - 1) / RASTER_BAND_ROWS;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    // a few chunks per thread so that slow chunks (many big triangles) even out
    int nchunks = threads * 4;
    if (nchunks > j->ntri) nchunks = j->ntri;
    Bins *bins = xmalloc((size_t)nchunks * sizeof(Bins));

    // ---- pass 1: binning. Chunk c handles a contiguous range of triangles, so chunks
    // taken in order 0, 1, 2... list the triangles in their original order.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int c = 0; c < nchunks; c++){
        int start = (int)((long long)j->ntri * c / nchunks);
        int end = (int)((long long)j->ntri * (c + 1) / nchunks);
        int cap = (end - start) + (end - start) / 4 + 16, n = 0;
        int *pair = xmalloc((size_t)cap * 2 * sizeof(int));   // (band, triangle) pairs
        for (int i = start; i < end; i++){
            int r0, r1;
            if (!triangle_rows(j, fb, i, &r0, &r1)) continue;
            for (int b = r0 / RASTER_BAND_ROWS; b <= r1 / RASTER_BAND_ROWS; b++){
                if (n == cap){
                    cap *= 2;
                    pair = realloc(pair, (size_t)cap * 2 * sizeof(int));
                    if (!pair){ perror("realloc"); exit(1); }
                }
                pair[2 * n] = b;
                pair[2 * n + 1] = i;
                n++;
            }
        }
        // counting sort by band; it is stable, so each band keeps the original order
        int *off = calloc((size_t)nbands + 1, sizeof(int));
        int *pos = xmalloc((size_t)nbands * sizeof(int));
        int *tri = xmalloc((size_t)n * sizeof(int));
        if (!off){ perror("calloc"); exit(1); }
        for (int k = 0; k < n; k++) off[pair[2 * k] + 1]++;
        for (int b = 0; b < nbands; b++) off[b + 1] += off[b];
        memcpy(pos, off, (size_t)nbands * sizeof(int));
        for (int k = 0; k < n; k++) tri[pos[pair[2 * k]]++] = pair[2 * k + 1];
        free(pos);
        free(pair);
        bins[c].tri = tri;
        bins[c].off = off;
    }

    // ---- pass 2: every band is drawn by one thread, in two passes over its triangles (filled
    // triangles; lines in one): (a) the depth alone, from the corners, noting which triangle
    // is nearest at each pixel (ids: the band's n-th triangle, CLIP_MAX ids each); (b) the
    // full vertex stage, then each pixel shaded once, by the triangle (a) left there. Shading
    // in one pass shaded every fragment nearer than what was there yet, and most were drawn
    // over again (the CPU rasterizer's time was 75% shading: docs/PERF_REVIEW.md, N3). The
    // same image: (a) keeps the same fragment as one pass would (nearer only, the first of a
    // tie), and (b) computes it as one pass did. A triangle that won no pixel in (a) is hidden
    // and skips (b).
    int *ids = j->mode != MODE_WIREFRAME ? xmalloc((size_t)fb->w * fb->h * sizeof(int)) : NULL;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int b = 0; b < nbands; b++){
        int row_lo = b * RASTER_BAND_ROWS;
        int row_hi = row_lo + RASTER_BAND_ROWS - 1;
        if (row_hi > fb->h - 1) row_hi = fb->h - 1;
        int count = 0;
        for (int c = 0; c < nchunks; c++) count += bins[c].off[b + 1] - bins[c].off[b];
        unsigned char *won = NULL;
        if (ids && count > 0){
            won = xmalloc((size_t)count);
            for (int y = row_lo; y <= row_hi; y++) memset(ids + (size_t)y * fb->w, 0xff, (size_t)fb->w * sizeof(int));
            int n = 0;
            for (int c = 0; c < nchunks; c++)
                for (int k = bins[c].off[b]; k < bins[c].off[b + 1]; k++, n++)
                    won[n] = depth_triangle(fb, j, bins[c].tri[k], row_lo, row_hi, ids, n * CLIP_MAX) > 0;
        }
        int n = 0;
        for (int c = 0; c < nchunks; c++){
            for (int k = bins[c].off[b]; k < bins[c].off[b + 1]; k++, n++){
                if (won && !won[n]) continue;                // nearest nowhere: hidden
                ClipVert cv[3];
                FragShader shader;
                const void *ctx;
                uint32_t wire;
                j->vertices(j, bins[c].tri[k], cv, &shader, &ctx, &wire);
                draw_clip_triangle(fb, cv, j->mode, j->cull, shader, ctx, wire, row_lo, row_hi,
                                   ids ? RASTER_SHADE : RASTER_DIRECT, ids, n * CLIP_MAX);
            }
        }
        free(won);
    }
    free(ids);

    for (int c = 0; c < nchunks; c++){ free(bins[c].tri); free(bins[c].off); }
    free(bins);
}

// ---- public -------------------------------------------------------------------------

RenderParams render_default_params(int width, int height){
    RenderParams p;
    p.model = mat4_identity();
    p.eye = vec3(0.0f, 0.0f, 3.2f);
    p.view = mat4_look_at(p.eye, vec3(0, 0, 0), vec3(0, 1, 0));
    p.proj = mat4_perspective(45.0f * 3.14159265f / 180.0f, (real)width / (real)height, 0.1f, 20.0f);
    p.light_dir = v_norm(vec3(0.4f, 0.7f, 0.9f));
    p.base_color = vec3(0.85f, 0.70f, 0.50f);
    p.mode = MODE_PHONG;
    p.texture = NULL;
    p.cull_backfaces = 1;
    return p;
}

const char *render_mode_name(RenderMode mode){
    static const char *names[MODE_COUNT] = {"wireframe", "flat", "gouraud", "phong", "textured"};
    return (mode >= 0 && mode < MODE_COUNT) ? names[mode] : "?";
}

int render_mode_from_name(const char *name){
    for (int m = 0; m < MODE_COUNT; m++)
        if (strcmp(name, render_mode_name((RenderMode)m)) == 0) return m;
    return -1;
}

RenderParams render_params_camera(Vec3 from, Vec3 at, Vec3 up, real vfov_deg, int width, int height){
    RenderParams p = render_default_params(width, height);
    p.eye = from;
    p.view = mat4_look_at(from, at, up);
    // near / far planes sized for scenes of a few hundred / thousand units (the Cornell box)
    p.proj = mat4_perspective(vfov_deg * PI / 180.0, (real)width / (real)height, 1.0, 20000.0);
    p.cull_backfaces = 0;
    return p;
}

typedef struct {
    const Mesh *mesh;
    const RenderParams *p;
    Shading sh;
    FragShader shader;
    uint32_t wire_color;
} MeshJob;

// 1. model space -> world space
static void mesh_corners(const TriJob *j, int i, Vec3 *wp){
    const MeshJob *m = j->data;
    const Face *f = &m->mesh->f[i];
    for (int k = 0; k < 3; k++) wp[k] = mat4_mul_point(m->p->model, m->mesh->v[f->v[k]]);
}

// 2. per-vertex values + world -> clip space
static void mesh_vertices(const TriJob *j, int i, ClipVert *cv, FragShader *shader, const void **ctx, uint32_t *wire){
    const MeshJob *m = j->data;
    const Mesh *mesh = m->mesh;
    const RenderParams *p = m->p;
    const Face *f = &mesh->f[i];
    Vec3 wp[3];
    mesh_corners(j, i, wp);
    Vec3 face_n = v_norm(v_cross(v_sub(wp[1], wp[0]), v_sub(wp[2], wp[0])));

    for (int k = 0; k < 3; k++){
        Vec3 n = face_n;
        if (p->mode != MODE_FLAT && f->n[k] >= 0)
            n = v_norm(mat4_mul_dir(p->model, mesh->vn[f->n[k]]));
        Vec2 uv = f->t[k] >= 0 ? mesh->vt[f->t[k]] : (Vec2){0, 0};

        cv[k].clip = mat4_mul_vec4(j->view_proj, (Vec4){wp[k].x, wp[k].y, wp[k].z, 1.0f});
        real *vy = cv[k].vary;
        vy[V_PX] = wp[k].x; vy[V_PY] = wp[k].y; vy[V_PZ] = wp[k].z;
        vy[V_NX] = n.x;     vy[V_NY] = n.y;     vy[V_NZ] = n.z;
        vy[V_U] = uv.x;     vy[V_V] = uv.y;
        vy[V_LIGHT] = lambert(n, p->light_dir);  // flat: n is the face normal -> same for all 3
    }
    *shader = m->shader;
    *ctx = &m->sh;
    *wire = m->wire_color;
}

void render_mesh(Framebuffer *fb, const Mesh *mesh, const RenderParams *p){
    MeshJob m;
    m.mesh = mesh;
    m.p = p;
    m.sh = (Shading){p->base_color, p->light_dir, p->eye, NULL};
    m.shader = shade_light;
    if (p->mode == MODE_PHONG) m.shader = shade_phong;
    if (p->mode == MODE_TEXTURED){
        m.shader = shade_phong;
        m.sh.tex = p->texture;                       // NULL texture falls back to base color
        if (m.sh.tex) m.sh.base = vec3(1, 1, 1);
    }
    m.wire_color = rgbf(p->base_color.x, p->base_color.y, p->base_color.z);

    TriJob j;
    j.ntri = mesh->nf;
    j.mode = p->mode;
    j.cull = p->cull_backfaces;
    j.view_proj = mat4_mul(p->proj, p->view);
    j.data = &m;
    j.corners = mesh_corners;
    j.vertices = mesh_vertices;
    draw_triangles(fb, &j);
}

// ---- whole scenes (the same Scene the path tracer renders) ----------------------------
//
// A rasterizer has no global light transport: no shadows, no bounced light, no refraction.
// What it can do fast is local lighting: every surface is the material's BSDF (scene/brdf.h)
// lit by the scene's lamp (render_shade: the lamp's area, not a point), plus an ambient
// light standing in for the light that bounces around. Good enough to frame a shot before
// path tracing it; mirrors and glass reflect the lamp and the ambient light (the room
// itself comes with PBR4's environment).

#define SPHERE_STACKS RENDER_SPHERE_STACKS
#define SPHERE_SLICES RENDER_SPHERE_SLICES

// A material as render_shade lights it: its BSDF, and what depends on one cosine alone,
// tabulated once (SHADE_TAB + 1 cosines over 0..1, linear in between): the environment's
// three parts (bsdf_albedo_parts) and the diffuse-like lobes' two factors, 1 - the layer
// above them (bsdf_diffuse_lobes: f = base kd gd(cv) gd(cl) + ms kms gm(cv) gm(cl))
#define SHADE_TAB 32
typedef struct {
    Bsdf b;
    real kd, kms;
    Vec3 spec[SHADE_TAB + 1], diff[SHADE_TAB + 1], trans[SHADE_TAB + 1];
    real gd[SHADE_TAB + 1], gm[SHADE_TAB + 1];
    int through;          // it transmits: the environment through it is looked up too
    int direct;           // no tables: the BSDF evaluated as it is (a textured material, per pixel)
    Vec3 emit;            // its glow (Material.emission of a MAT_PBR: added as it is)
    real ao;              // its occlusion (1 = none): scales the light from around, not the lamp's
} ShadeMat;

// the BSDF and its constants, without the tables (sm->direct)
static void shade_mat_direct(ShadeMat *sm, const Material *m){
    sm->b = bsdf_make(m, 1);
    Bsdf *b = &sm->b;
    const real mt = b->metallic, t = b->transmission;
    sm->kd = (1 - mt) * (1 - t) > 0 ? (1 - mt) * (1 - t) * b->kd_norm / (real)PI : 0;
    sm->kms = !b->smooth && mt > 0 ? mt * b->kms_norm : 0;
    sm->through = (1 - mt) * t > 0;
    sm->direct = 1;
    sm->emit = material_glows(m) ? m->emission : vec3(0, 0, 0);
    sm->ao = 1;
}

static void shade_mat_init(ShadeMat *sm, const Material *m){
    shade_mat_direct(sm, m);
    sm->direct = 0;
    Bsdf *b = &sm->b;
    for (int i = 0; i <= SHADE_TAB; i++){
        real mu = r_fmax((real)i / SHADE_TAB, 1e-4f);
        bsdf_albedo_parts(b, mu, &sm->spec[i], &sm->diff[i], &sm->trans[i]);
        sm->gd[i] = sm->kd > 0 ? 1 - bsdf_spec_albedo(b, mu) : 0;
        sm->gm[i] = sm->kms > 0 ? 1 - bsdf_albedo(b, mu) : 0;
    }
}

// where cosine mu falls in the tables: entry *i, and the weight of entry *i + 1
static inline real shade_tab_at(real mu, int *i){
    real x = bsdf_clamp01(mu) * SHADE_TAB;
    int k = (int)x;
    if (k > SHADE_TAB - 1) k = SHADE_TAB - 1;
    *i = k;
    return x - k;
}

typedef struct {
    const Material *mat;
    ShadeMat sm;
    const Scene *scene;   // (its textures)
    int textured;         // the material has textures: evaluated per pixel (vertex), not tabulated
    const RenderLamp *lamp;
    Vec3 eye;             // camera position
    uint32_t flat_color;  // emitters and wireframe lines
} SceneShading;

// ∫ cos dω over the lamp as seen from p, above the plane of n (the lamp clipped to the
// hemisphere first): Lambert's formula for a polygon, the sum over its edges of the angle θ
// they span times the cosine of their plane's normal with n, halved. θ / sin θ from the
// edge's cosine by Heitz and Hill's rational fit (0.1%; theirs is of θ / (2 π sin θ)), not
// atan2.
static real edge_theta_sin(real x){
    real y = r_fabs(x);
    real t = (0.8543985f + (0.4965155f + 0.0145206f * y) * y) / (3.4175940f + (4.1616724f + y) * y);
    return x > 0 ? t : 0.5f / r_sqrt(r_fmax(1 - x * x, 1e-7f)) - t;
}

static real lamp_form_factor(const RenderLamp *L, Vec3 p, Vec3 n){
    Vec3 c[4] = {v_sub(L->q, p), v_sub(v_add(L->q, L->u), p), v_sub(v_add(v_add(L->q, L->u), L->v), p), v_sub(v_add(L->q, L->v), p)};
    real d[4];
    int below = 0;
    for (int i = 0; i < 4; i++){ d[i] = v_dot(c[i], n); below |= d[i] < 0; }
    Vec3 o[8];
    int k = 0;
    if (!below) for (; k < 4; k++) o[k] = v_norm(c[k]);
    else {
        for (int i = 0; i < 4; i++){
            int j = (i + 1) & 3;
            if (d[i] >= 0) o[k++] = v_norm(c[i]);
            if ((d[i] >= 0) != (d[j] >= 0)) o[k++] = v_norm(v_add(c[i], v_scale(v_sub(c[j], c[i]), d[i] / (d[i] - d[j]))));
        }
        if (k < 3) return 0;
    }
    real sum = 0;
    for (int i = 0; i < k; i++){
        Vec3 a = o[i], b = o[(i + 1) % k];
        sum += v_dot(v_cross(a, b), n) * edge_theta_sin(v_dot(a, b));
    }
    return r_fabs(sum) * (real)PI;
}

// the lamp's point nearest the ray from p along r: where the ray meets the lamp's plane, at
// most 4 times as far as the lamp's centre (a ray almost parallel to the plane, or going
// away from it, takes the point of the plane under that far point: no jump in between),
// kept inside the quad
static Vec3 lamp_nearest(const RenderLamp *L, Vec3 p, Vec3 r){
    real dn = v_dot(r, L->n), far = 4 * v_len(v_sub(L->center, p));
    real t = dn < 0 ? v_dot(v_sub(L->q, p), L->n) / dn : far;
    if (!(t <= far)) t = far;
    if (t < 0) t = 0;
    Vec3 y = v_add(p, v_scale(r, t));
    Vec3 x = v_sub(y, v_scale(L->n, v_dot(v_sub(y, L->q), L->n)));
    Vec3 qx = v_sub(x, L->q);
    real a = v_dot(L->w, v_cross(qx, L->v)), b = v_dot(L->w, v_cross(L->u, qx));
    a = a < 0 ? 0 : a > 1 ? 1 : a;
    b = b < 0 ? 0 : b > 1 ? 1 : b;
    return v_add(L->q, v_add(v_scale(L->u, a), v_scale(L->v, b)));
}

static inline Vec3 tab_lerp(const Vec3 *t, int i, real f){ return v_add(t[i], v_scale(v_sub(t[i + 1], t[i]), f)); }

// Where the light seen through a transmitting object comes from: refracted in at n, out
// where the ray leaves a sphere (the exit normal mirrors n about the plane across the
// inside ray: n2 = n - 2 (n.t) t), refracted out there; reflected inside instead where it
// cannot leave. Exact for a sphere, close for convex shapes.
static Vec3 through_dir(Vec3 v, Vec3 n, real eta){
    Vec3 d = v_neg(v);
    real c = v_dot(v, n), k = 1 - eta * eta * (1 - c * c);
    if (k <= 0) return v_reflect(d, n);
    Vec3 t = v_norm(v_refract(d, n, eta));
    Vec3 n2 = v_sub(n, v_scale(t, 2 * v_dot(n, t)));         // outward at the exit
    real c2 = -v_dot(t, n2), k2 = 1 - (1 / eta) * (1 / eta) * (1 - c2 * c2);
    if (k2 <= 0) return v_reflect(t, v_neg(n2));
    return v_norm(v_refract(t, v_neg(n2), 1 / eta));
}

// the environment's light (split-sum): the specular part from the prefiltered map around the
// lobe's direction (between n and the reflected ray, nearer n the rougher: Lagarde &
// de Rousiers 2014), the diffuse from the irradiance, the transmitted around the ray
// through; a probe's lookups corrected by its box (env_box_dir)
// the luminance of 9 harmonics (rgb, 27 numbers) in direction d, by the first 4 alone
static inline real sh_lum4(const float *c, Vec3 d){
    real y1 = 0.488603f * d.y, y2 = 0.488603f * d.z, y3 = 0.488603f * d.x;
    real r = 0.282095f * c[0] + y1 * c[3] + y2 * c[6] + y3 * c[9];
    real g = 0.282095f * c[1] + y1 * c[4] + y2 * c[7] + y3 * c[10];
    real b = 0.282095f * c[2] + y1 * c[5] + y2 * c[8] + y3 * c[11];
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}
// a direction in an environment's own frame (its turn: env_irradiance's)
static inline Vec3 env_dir_rot(const Env *e, Vec3 n){
    return e->rot != 0 ? vec3(n.x * e->rot_c - n.z * e->rot_s, n.y, n.x * e->rot_s + n.z * e->rot_c) : n;
}

static Vec3 shade_env(const Env *E, const IrrGrid *irr, const ShadeMat *sm, Vec3 sp, Vec3 df, Vec3 tr, Vec3 p, Vec3 n, Vec3 v){
    const Bsdf *b = &sm->b;
    real r = b->smooth ? 0 : b->rough;
    Vec3 rr = v_reflect(v_neg(v), n);
    real s = 1 - r, k = s * (r_sqrt(s) + r);
    Vec3 rd = v_norm(v_add(n, v_scale(v_sub(rr, n), k)));
    Vec3 c = v_mul(sp, env_level(E, env_box_dir(E, p, rd), r));
    float sh[27];
    int ok = irr ? irr_grid_blend(irr, p, sh) : 0;
    if (ok){
        // the probe was made somewhere else: what it shows along rd is scaled by how much
        // light arrives around rd here (the grid) against there (the probe's own harmonics),
        // never brightened. (A surface deep in a corner the probe sees into from the room
        // reflects that corner's darkness, not the room: "probe normalization".) Their
        // luminances by the first 4 harmonics: enough for a ratio, and cheap
        real here = sh_lum4(sh, rd), there = sh_lum4(E->sh[0], env_dir_rot(E, rd)) * E->scale;
        if (there > 1e-6f && here < there) c = v_scale(c, r_fmax(here, 0) / there);
    }
    c = v_add(c, v_mul(df, ok ? irr_sh_eval(sh, n) : env_irradiance(E, n)));
    if (sm->through) c = v_add(c, v_mul(tr, env_level(E, env_box_dir(E, p, through_dir(v, n, b->eta)), r)));
    return c;
}

// lamp_hz: the share of the lamp above the plane of ng, the surface's own normal (turned to
// the eye), for a normal bent by a normal map: the bent normal can face a lamp the surface
// itself is turned away from (the path tracer sees none of it there). Taken as the lamp's
// disc of the same area, seen from p, cut by that plane: linear across its angular radius
static real lamp_hz(const RenderLamp *L, Vec3 p, Vec3 ng){
    Vec3 tc = v_sub(L->center, p);
    real d = v_len(tc);
    real s = r_fmax(r_sqrt(L->area / (real)PI) / d, 1e-3f);
    return r_fmin(r_fmax(0.5f + v_dot(ng, tc) / (d * 2 * s), 0), 1);
}

static Vec3 shade_point_g(const RenderLamp *L, const ShadeMat *sm, Vec3 p, Vec3 n, const Vec3 *ng, Vec3 eye);

static Vec3 shade_point(const RenderLamp *L, const ShadeMat *sm, Vec3 p, Vec3 n, Vec3 eye){
    return shade_point_g(L, sm, p, n, NULL, eye);
}

// ng: the surface's own normal when n is bent (NULL: n is the surface's)
static Vec3 shade_point_g(const RenderLamp *L, const ShadeMat *sm, Vec3 p, Vec3 n, const Vec3 *ng, Vec3 eye){
    Vec3 v = v_norm(v_sub(eye, p));
    if (v_dot(n, v) < 0) n = v_neg(n);           // both sides lit (the walls have no "outside")
    real cv = r_fmax(v_dot(n, v), 1e-4f);
    const Bsdf *b = &sm->b;
    Bsdf bd = sm->b;                              // (direct: the BSDF seen from v)
    int iv = 0;
    real fv = 0;
    Vec3 sp, df, tr;
    if (sm->direct) bsdf_albedo_parts(&bd, cv, &sp, &df, &tr);
    else {
        fv = shade_tab_at(cv, &iv);
        sp = tab_lerp(sm->spec, iv, fv); df = tab_lerp(sm->diff, iv, fv); tr = tab_lerp(sm->trans, iv, fv);
    }
    Vec3 c = L->env ? shade_env(L->env, L->irr, sm, sp, df, tr, p, n, v)
                    : v_scale(v_add(v_add(sp, df), tr), L->ambient + L->fill * cv);
    c = v_add(v_scale(c, sm->ao), sm->emit);
    if (!L->on || v_dot(v_sub(p, L->center), L->n) <= 0) return c;   // behind the lamp
    real hz = ng ? lamp_hz(L, p, *ng) : 1;
    if (hz <= 0) return c;
    Vec3 base = c;
    // the diffuse-like lobes: the lamp's form factor
    real ff = sm->kd > 0 || sm->kms > 0 ? lamp_form_factor(L, p, n) : 0;
    if (ff > 0){
        Vec3 lc = v_norm(v_sub(L->center, p));
        Vec3 f;
        if (sm->direct) f = bsdf_diffuse_lobes(&bd, r_fmax(v_dot(n, lc), 0));
        else {
            int il;
            real fl = shade_tab_at(v_dot(n, lc), &il);
            real gd = (sm->gd[iv] + fv * (sm->gd[iv + 1] - sm->gd[iv])) * (sm->gd[il] + fl * (sm->gd[il + 1] - sm->gd[il]));
            real gm = (sm->gm[iv] + fv * (sm->gm[iv + 1] - sm->gm[iv])) * (sm->gm[il] + fl * (sm->gm[il + 1] - sm->gm[il]));
            f = v_add(v_scale(b->base, sm->kd * gd), v_scale(b->ms, sm->kms * gm));
        }
        c = v_add(c, v_scale(v_mul(f, L->le), ff));
    }
    // the microfacet lobe. Its solid angle (4 pi alpha^2 cos: where f cos is near its peak,
    // F / that) against the lamp's decides how: a lobe wider than the lamp sees all of it, so
    // f is averaged over the lamp's 4 quarters; a narrower one sees only the lamp's part
    // around the reflected ray, so f is taken at the lamp's point nearest that ray, over the
    // lobe's solid angle (a mirror shows the lamp itself, with its edges); in between, both
    // weighted by how much wider the lobe is
    Vec3 tc = v_sub(L->center, p);
    real dc2 = v_dot(tc, tc);
    real om = r_fmin(L->area * r_fmax(-v_dot(L->n, tc), 0) / (dc2 * r_sqrt(dc2)), 2 * (real)PI);
    real lobe = 4 * (real)PI * b->alpha * b->alpha * cv;
    real w = lobe / (lobe + om);
    Vec3 spec = vec3(0, 0, 0);
    if (w > 0.01f && lobe > 32 * om){
        // a lobe much wider than the lamp: f hardly changes over it, its centre will do
        Vec3 lc = v_scale(tc, 1 / r_sqrt(dc2));
        real ck = v_dot(n, lc);
        if (ck > 0) spec = v_scale(bsdf_spec_lobe(b, n, v, lc, b->alpha), w * ck * om);
    } else if (w > 0.01f){
        for (int k = 0; k < 4; k++){
            Vec3 xk = v_add(L->q, v_add(v_scale(L->u, (k & 1) ? 0.75f : 0.25f), v_scale(L->v, (k & 2) ? 0.75f : 0.25f)));
            Vec3 dk = v_sub(xk, p);
            real d2 = v_dot(dk, dk);
            Vec3 lk = v_scale(dk, 1 / r_sqrt(d2));
            real ck = v_dot(n, lk), cosl = -v_dot(L->n, lk);
            if (ck > 0 && cosl > 0) spec = v_add(spec, v_scale(bsdf_spec_lobe(b, n, v, lk, b->alpha), w * ck * cosl * L->area * 0.25f / d2));
        }
    }
    if (w < 0.99f){
        Vec3 l = v_norm(v_sub(lamp_nearest(L, p, v_reflect(v_neg(v), n)), p));
        real cl = v_dot(n, l);
        if (cl > 0) spec = v_add(spec, v_scale(bsdf_spec_lobe(b, n, v, l, b->alpha), (1 - w) * cl * om * lobe / (om + lobe)));
    }
    c = v_add(c, v_mul(spec, L->le));
    return hz < 1 ? v_add(base, v_scale(v_sub(c, base), hz)) : c;
}

Vec3 render_shade(const RenderLamp *L, const Material *m, Vec3 p, Vec3 n, Vec3 eye){
    ShadeMat sm;
    shade_mat_init(&sm, m);
    return shade_point(L, &sm, p, n, eye);
}

// flat and gouraud: the colour from the vertex stage
static uint32_t shade_scene_light(const real *v, const void *ctx){
    (void)ctx;
    return image_pack_srgb(vec3(v[V_LIGHT], v[V_LIGHT_G], v[V_LIGHT_B]));
}

// A textured material at one point (PBR5): the material there (its mip levels for a
// footprint of 2^uv_lod), the normal (turned to the eye) bent by its normal map and tilted
// back towards the eye if the map turned it away (as the path tracer does), shaded directly
static Vec3 shade_textured(const SceneShading *s, Vec3 p, Vec3 n, Vec2 uv, Vec4 tg, real uv_lod){
    Vec3 v = v_norm(v_sub(s->eye, p));
    if (v_dot(n, v) < 0) n = v_neg(n);
    Material mt = material_at(s->scene, s->mat, uv, uv_lod);
    Vec3 ng = n;
    if (s->mat->tex_normal){
        n = material_normal(s->scene, s->mat, uv, uv_lod, n, tg);
        real c = v_dot(n, v);
        if (c < 1e-3f) n = v_norm(v_add(n, v_scale(v, 1e-3f - c)));
    }
    ShadeMat sm;
    shade_mat_direct(&sm, &mt);
    sm.ao = material_occlusion(s->scene, s->mat, uv, uv_lod);
    return shade_point_g(s->lamp, &sm, p, n, s->mat->tex_normal ? &ng : NULL, s->eye);
}

// phong: render_shade at every pixel
static uint32_t shade_scene_phong(const real *v, const void *ctx){
    const SceneShading *s = ctx;
    Vec3 n = v_norm(vec3(v[V_NX], v[V_NY], v[V_NZ]));
    Vec3 p = vec3(v[V_PX], v[V_PY], v[V_PZ]);
    if (s->textured)
        return image_pack_srgb(shade_textured(s, p, n, vec2(v[V_U], v[V_V]), (Vec4){v[V_TX], v[V_TY], v[V_TZ], v[V_TW]},
                                              material_uv_lod(v[V_DUDX], v[V_DVDX], v[V_DUDY], v[V_DVDY])));
    return image_pack_srgb(shade_point(s->lamp, &s->sm, p, n, s->eye));
}

static uint32_t shade_constant(const real *v, const void *ctx){
    (void)v;
    return ((const SceneShading *)ctx)->flat_color;
}

static Vec3 sphere_dir(int i, int j){
    real th = PI * i / SPHERE_STACKS, ph = 2.0f * PI * j / SPHERE_SLICES;
    return vec3(r_sin(th) * r_cos(ph), r_cos(th), r_sin(th) * r_sin(ph));
}

// A sphere is a latitude / longitude grid of SPHERE_STACKS x SPHERE_SLICES cells. Every cell
// is 2 triangles, except at the poles, where one of the two would have zero area.
#define SPHERE_TRIS RENDER_SPHERE_TRIS

// triangle k of a sphere -> cell (y, x) and which half (upper = the one touching row y)
static void sphere_cell(int k, int *y, int *x, int *upper){
    if (k < SPHERE_SLICES){ *y = 0; *x = k; *upper = 0; return; }   // top row: lower halves only
    k -= SPHERE_SLICES;
    int mid = 2 * SPHERE_SLICES * (SPHERE_STACKS - 2);
    if (k < mid){
        *y = 1 + k / (2 * SPHERE_SLICES);
        int r = k % (2 * SPHERE_SLICES);
        *x = r / 2;
        *upper = r % 2 == 0;
        return;
    }
    *y = SPHERE_STACKS - 1; *x = k - mid; *upper = 1;               // bottom row: upper halves only
}

// Triangle numbering: shape by shape, 2 per quad, SPHERE_TRIS per sphere, 1 per mesh
// triangle. To find the primitive of triangle i without a table of one entry
// per triangle, consecutive primitives with the same triangle count form a RUN (a mesh is
// one long run of 1s, the Cornell walls a run of 2s): a few entries, searched by bisection.
typedef struct {
    int first_tri;   // first triangle of the run
    int prim;        // its first primitive
    int per_prim;    // triangles per primitive in this run
} TriRun;

typedef struct {
    const Scene *s;
    TriRun *runs;
    int nruns;
    int world_tris;                 // triangles of the primitives and the world mesh; then
    int *inst_first;                // the instances': instance i's first triangle
    SceneShading sh[MAX_MATS];      // one shading context per material
    FragShader shader[MAX_MATS];
    RenderLamp lamp;
} SceneJob;

int render_prim_tri_count(const Primitive *p){
    return p->type == PRIM_QUAD ? 2 : SPHERE_TRIS;
}
#define prim_tri_count render_prim_tri_count

// triangle i -> (primitive, which of its triangles)
static void scene_tri_of(const SceneJob *sj, int i, int *prim, int *sub){
    int lo = 0, hi = sj->nruns - 1;
    while (lo < hi){                         // last run with first_tri <= i
        int mid = (lo + hi + 1) / 2;
        if (sj->runs[mid].first_tri <= i) lo = mid; else hi = mid - 1;
    }
    const TriRun *r = &sj->runs[lo];
    int rel = i - r->first_tri;
    *prim = r->prim + rel / r->per_prim;
    *sub = rel % r->per_prim;
}

static int inst_tri_count(const Scene *s, const Instance *in){
    return in->mesh >= 0 ? s->meshes[in->mesh].ntris : (in->mesh == INST_SPHERE ? SPHERE_TRIS : 12);
}

// triangle i (past the world's) -> (instance, its mesh's triangle)
static void inst_tri_of(const SceneJob *sj, int i, int *inst, int *tri){
    int lo = 0, hi = sj->s->ninsts - 1;
    while (lo < hi){
        int mid = (lo + hi + 1) / 2;
        if (sj->inst_first[mid] <= i) lo = mid; else hi = mid - 1;
    }
    *inst = lo;
    *tri = i - sj->inst_first[lo];
}

static int scene_tri_material(const SceneJob *sj, int i){
    if (i >= sj->world_tris){
        int inst, tri;
        inst_tri_of(sj, i, &inst, &tri);
        return scene_instance_material(sj->s, inst, tri);
    }
    int prim, sub;
    scene_tri_of(sj, i, &prim, &sub);
    return scene_shape_material(sj->s, prim);
}

// Triangle i of the scene: world-space corners, and (if wn is not NULL) vertex normals
// (*has_n = 0 when it has none) and the point flat shading is evaluated at.
// triangle i: its corners and (wn not NULL) normals, flat shading's point, and (uv not NULL)
// its corners' texture coordinates and tangents
static void scene_tri(const SceneJob *sj, int i, Vec3 *wp, Vec3 *wn, int *has_n, Vec3 *flat_at, Vec2 *uv, Vec4 *tg){
    const Scene *s = sj->s;
    if (i >= sj->world_tris){                     // an instance's triangle, moved into the world
        int inst, tri;
        inst_tri_of(sj, i, &inst, &tri);
        const Instance *in = &s->insts[inst];
        if (in->mesh < 0){                        // a ball or a box: its primitives in the world
            Primitive pr = instance_shape_prim(in, in->mesh == INST_SPHERE ? 0 : tri / 2);
            render_prim_triangle(&pr, in->mesh == INST_SPHERE ? tri : tri % 2, wp, wn, has_n, flat_at);
            if (uv) render_prim_texcoords(&pr, in->mesh == INST_SPHERE ? tri : tri % 2, uv, tg);
            return;
        }
        const SceneMesh *m = &s->meshes[in->mesh];
        const Triangle *t = &m->tris[tri];
        wp[0] = xform_point(&in->xf, t->q);
        wp[1] = xform_point(&in->xf, v_add(t->q, t->u));
        wp[2] = xform_point(&in->xf, v_add(t->q, t->v));
        if (!wn) return;
        *has_n = t->n[0] >= 0;
        if (*has_n) for (int k = 0; k < 3; k++) wn[k] = xform_normal(&in->inv, m->normals[t->n[k]]);
        *flat_at = v_scale(v_add(v_add(wp[0], wp[1]), wp[2]), 1.0f / 3.0f);
        if (uv) render_tri_texcoords(t, m->uvs, m->tangents, m->nuvs, &in->xf, uv, tg);
        return;
    }
    int prim, sub;
    scene_tri_of(sj, i, &prim, &sub);
    if (prim >= s->nprims){                       // a mesh triangle
        const Triangle *t = &s->tris[prim - s->nprims];
        wp[0] = t->q; wp[1] = v_add(t->q, t->u); wp[2] = v_add(t->q, t->v);
        if (!wn) return;
        *has_n = t->n[0] >= 0;
        if (*has_n) for (int k = 0; k < 3; k++) wn[k] = s->normals[t->n[k]];
        *flat_at = v_scale(v_add(v_add(wp[0], wp[1]), wp[2]), 1.0f / 3.0f);
        if (uv) render_tri_texcoords(t, s->uvs, s->tangents, s->nuvs, NULL, uv, tg);
        return;
    }
    render_prim_triangle(&s->prims[prim], sub, wp, wn, has_n, flat_at);
    if (uv && wn) render_prim_texcoords(&s->prims[prim], sub, uv, tg);
}

void render_tri_texcoords(const Triangle *t, const Vec2 *uvs, const Vec4 *tangents, int nuvs, const Xform *xf, Vec2 *uv, Vec4 *tg){
    int has = t->t[0] >= 0 && t->t[1] >= 0 && t->t[2] >= 0 && t->t[0] < nuvs && t->t[1] < nuvs && t->t[2] < nuvs;
    Vec3 eu = v_norm(t->u);
    for (int k = 0; k < 3; k++){
        uv[k] = has ? uvs[t->t[k]] : vec2(k == 1, k == 2);
        Vec4 g = has ? tangents[t->t[k]] : (Vec4){eu.x, eu.y, eu.z, 1};
        if (xf){
            Vec3 d = v_norm(xform_dir(xf, vec3(g.x, g.y, g.z)));
            g = (Vec4){d.x, d.y, d.z, g.w};
        }
        tg[k] = g;
    }
}

void render_prim_texcoords(const Primitive *pr, int sub, Vec2 *uv, Vec4 *tg){
    if (pr->type == PRIM_QUAD){
        // corners a = (0, 0), b = (1, 0), c = (1, 1), d = (0, 1); tangent along u, n x u along +v
        Vec2 a = vec2(0, 0), b = vec2(1, 0), c = vec2(1, 1), d = vec2(0, 1);
        uv[0] = a; uv[1] = sub == 0 ? b : c; uv[2] = sub == 0 ? c : d;
        Vec3 t = v_norm(pr->u);
        for (int k = 0; k < 3; k++) tg[k] = (Vec4){t.x, t.y, t.z, -1};
        return;
    }
    // sphere_dir(i, j)'s longitude: u = 0.75 - j / slices (atan2(x, z) = pi / 2 - phi), kept
    // unwrapped; at a pole the cell's middle
    int y, x, upper;
    sphere_cell(sub, &y, &x, &upper);
    int ci[3], cj[3];
    if (upper){ ci[0] = y; cj[0] = x; ci[1] = y; cj[1] = x + 1; ci[2] = y + 1; cj[2] = x + 1; }
    else      { ci[0] = y; cj[0] = x; ci[1] = y + 1; cj[1] = x + 1; ci[2] = y + 1; cj[2] = x; }
    for (int k = 0; k < 3; k++){
        int pole = ci[k] == 0 || ci[k] == SPHERE_STACKS;
        real u = 0.75f - ((real)cj[k] + (pole ? 0.5f : 0)) / SPHERE_SLICES;
        uv[k] = vec2(u, (real)ci[k] / SPHERE_STACKS);
        real ph = (u - 0.5f) * (real)(2 * PI);
        tg[k] = (Vec4){r_cos(ph), 0, -r_sin(ph), 1};
    }
}

void render_prim_triangle(const Primitive *pr, int sub, Vec3 *wp, Vec3 *wn, int *has_n, Vec3 *flat_at){
    if (pr->type == PRIM_QUAD){
        Vec3 a = pr->q, b = v_add(pr->q, pr->u), c = v_add(b, pr->v), d = v_add(pr->q, pr->v);
        wp[0] = a;
        wp[1] = sub == 0 ? b : c;
        wp[2] = sub == 0 ? c : d;
        if (!wn) return;
        *has_n = 0;
        *flat_at = v_scale(v_add(a, c), 0.5f);   // the quad's center: both halves get one color
    } else {
        int y, x, upper;
        sphere_cell(sub, &y, &x, &upper);
        Vec3 d00 = sphere_dir(y, x), d01 = sphere_dir(y, x + 1);
        Vec3 d10 = sphere_dir(y + 1, x), d11 = sphere_dir(y + 1, x + 1);
        Vec3 n[3];
        if (upper){ n[0] = d00; n[1] = d01; n[2] = d11; }
        else      { n[0] = d00; n[1] = d11; n[2] = d10; }
        for (int k = 0; k < 3; k++) wp[k] = v_add(pr->center, v_scale(n[k], pr->radius));
        if (!wn) return;
        for (int k = 0; k < 3; k++) wn[k] = n[k];   // the exact sphere normal at each corner
        *has_n = 1;
        *flat_at = v_scale(v_add(v_add(wp[0], wp[1]), wp[2]), 1.0f / 3.0f);
    }
}

static void scene_corners(const TriJob *j, int i, Vec3 *wp){
    scene_tri(j->data, i, wp, NULL, NULL, NULL, NULL, NULL);
}

// world-space triangle (+ optional vertex normals) -> clip space
static void scene_vertices(const TriJob *j, int i, ClipVert *cv, FragShader *shader, const void **ctx, uint32_t *wire){
    const SceneJob *sj = j->data;
    int mat = scene_tri_material(sj, i);
    const SceneShading *sh = &sj->sh[mat];
    Vec3 wp[3], wn[3], flat_at;
    Vec2 uv[3] = {{0, 0}, {0, 0}, {0, 0}};
    Vec4 tg[3] = {{1, 0, 0, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}};
    int has_n;
    scene_tri(sj, i, wp, wn, &has_n, &flat_at, uv, tg);
    RenderMode mode = j->mode;

    Vec3 face_n = v_norm(v_cross(v_sub(wp[1], wp[0]), v_sub(wp[2], wp[0])));
    const int lit = sh->mat->type != MAT_LIGHT;
    Vec3 flat_c = vec3(0, 0, 0);
    if (mode == MODE_FLAT && lit){
        if (sh->textured){                         // at the triangle's middle
            Vec2 um = vec2((uv[0].x + uv[1].x + uv[2].x) / 3, (uv[0].y + uv[1].y + uv[2].y) / 3);
            flat_c = shade_textured(sh, flat_at, face_n, um, tg[0], MATERIAL_LOD0);
        } else flat_c = shade_point(sh->lamp, &sh->sm, flat_at, face_n, sh->eye);
    }
    for (int k = 0; k < 3; k++){
        Vec3 n = (has_n && mode != MODE_FLAT) ? wn[k] : face_n;
        cv[k].clip = mat4_mul_vec4(j->view_proj, (Vec4){wp[k].x, wp[k].y, wp[k].z, 1.0f});
        real *vy = cv[k].vary;
        vy[V_PX] = wp[k].x; vy[V_PY] = wp[k].y; vy[V_PZ] = wp[k].z;
        vy[V_NX] = n.x;     vy[V_NY] = n.y;     vy[V_NZ] = n.z;
        vy[V_U] = uv[k].x;  vy[V_V] = uv[k].y;
        vy[V_TX] = tg[k].x; vy[V_TY] = tg[k].y; vy[V_TZ] = tg[k].z; vy[V_TW] = tg[k].w;
        Vec3 c = vec3(0, 0, 0);
        if (mode == MODE_FLAT) c = flat_c;
        else if (mode == MODE_GOURAUD && lit)
            c = sh->textured ? shade_textured(sh, wp[k], n, uv[k], tg[k], MATERIAL_LOD0) : shade_point(sh->lamp, &sh->sm, wp[k], n, sh->eye);
        vy[V_LIGHT] = c.x;  vy[V_LIGHT_G] = c.y;  vy[V_LIGHT_B] = c.z;
    }
    *shader = sj->shader[mat];
    *ctx = sh;
    *wire = sh->flat_color;
}

// the point light: at the lamp's center, moved off the lamp's plane by half the lamp's
// size on the side it shines to (an area light lights the wall around it a little, a
// point light on that wall would not)
Vec3 render_scene_light(const Scene *s){
    Vec3 light_pos = scene_light_center(s);
    if (s->light >= 0 && s->light < s->nprims){
        const Primitive *l = &s->prims[s->light];
        light_pos = v_add(light_pos, v_scale(l->normal, 0.5f * r_sqrt(l->area)));
    }
    return light_pos;
}

RenderLamp render_scene_lamp(const Scene *s){
    RenderLamp L;
    memset(&L, 0, sizeof(L));
    const char *amb = getenv("LR_RASTER_AMBIENT");
    real k = amb ? (real)atof(amb) : 1;
    L.ambient = RENDER_SCENE_AMBIENT * k;
    L.fill = RENDER_SCENE_FILL * k;
    L.env = k > 0 ? (s->probe ? s->probe : s->env) : NULL;
    L.irr = L.env && L.env == s->probe ? s->irr : NULL;
    if (s->light < 0 || s->light >= s->nprims || s->prims[s->light].type != PRIM_QUAD) return L;
    const Primitive *l = &s->prims[s->light];
    L.on = 1;
    L.q = l->q; L.u = l->u; L.v = l->v; L.n = l->normal; L.w = l->w;
    L.center = v_add(l->q, v_scale(v_add(l->u, l->v), 0.5f));
    L.le = s->mats[l->material].emission;
    L.area = l->area;
    return L;
}

// the sky where nothing was drawn (the z-buffer still at its far value): each pixel's view
// direction from the camera's axes (the view matrix's rows) and its lens (the projection's
// scale)
static void draw_sky(Framebuffer *fb, const Env *env, const RenderParams *p){
    Vec3 right = vec3(p->view.m[0][0], p->view.m[0][1], p->view.m[0][2]);
    Vec3 up = vec3(p->view.m[1][0], p->view.m[1][1], p->view.m[1][2]);
    Vec3 back = vec3(p->view.m[2][0], p->view.m[2][1], p->view.m[2][2]);
    real sx = 1 / p->proj.m[0][0], sy = 1 / p->proj.m[1][1];
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < fb->h; y++){
        real ny = (1 - 2 * (y + 0.5f) / fb->h) * sy;
        for (int x = 0; x < fb->w; x++){
            size_t i = (size_t)y * fb->w + x;
            if (fb->depth[i] < 1) continue;
            real nx = (2 * (x + 0.5f) / fb->w - 1) * sx;
            Vec3 d = v_norm(v_sub(v_add(v_scale(right, nx), v_scale(up, ny)), back));
            fb->color[i] = image_pack_srgb(env_eval_fast(env, d));
        }
    }
}

RenderMaterial render_scene_material(const Material *mt){
    RenderMaterial rm = {mt->albedo, 0};
    if (mt->type == MAT_LIGHT){
        real mx = v_max(mt->emission);               // show the light's color at full brightness
        rm.base = mx > 0 ? v_scale(mt->emission, 1.0f / mx) : mt->emission;
        rm.constant = 1;
    }
    return rm;
}

void render_scene(Framebuffer *fb, const Scene *s, const RenderParams *p){
    srgb_lut_init();
    RenderMode mode = p->mode == MODE_TEXTURED ? MODE_PHONG : p->mode;

    SceneJob *sj = xmalloc(sizeof(*sj));
    sj->s = s;

    sj->lamp = render_scene_lamp(s);

    // one shading context per material
    for (int m = 0; m < s->nmats && m < MAX_MATS; m++){
        const Material *mt = &s->mats[m];
        SceneShading *sh = &sj->sh[m];
        RenderMaterial rm = render_scene_material(mt);
        sh->eye = p->eye;
        sh->mat = mt;
        sh->scene = s;
        sh->textured = !rm.constant && material_textured(mt);
        if (!rm.constant) shade_mat_init(&sh->sm, mt);
        sh->lamp = &sj->lamp;
        sj->shader[m] = rm.constant ? shade_constant : (mode == MODE_PHONG ? shade_scene_phong : shade_scene_light);
        sh->flat_color = image_pack_srgb(rm.constant ? rm.base : material_resolved(mt).albedo);
    }

    // number the triangles (see TriRun): the primitives' triangles, then the mesh
    // triangles as one last run of 1 triangle per shape
    int ntri = 0, nruns = s->ntris > 0;
    for (int i = 0; i < s->nprims; i++)
        if (i == 0 || prim_tri_count(&s->prims[i]) != prim_tri_count(&s->prims[i - 1])) nruns++;
    sj->runs = xmalloc((size_t)(nruns ? nruns : 1) * sizeof(TriRun));
    sj->nruns = 0;
    for (int i = 0; i < s->nprims; i++){
        int k = prim_tri_count(&s->prims[i]);
        if (i == 0 || k != prim_tri_count(&s->prims[i - 1]))
            sj->runs[sj->nruns++] = (TriRun){ntri, i, k};
        ntri += k;
    }
    if (s->ntris > 0){
        sj->runs[sj->nruns++] = (TriRun){ntri, s->nprims, 1};
        ntri += s->ntris;
    }
    sj->world_tris = ntri;
    sj->inst_first = xmalloc((size_t)(s->ninsts ? s->ninsts : 1) * sizeof(int));
    for (int i = 0; i < s->ninsts; i++){
        sj->inst_first[i] = ntri;
        ntri += inst_tri_count(s, &s->insts[i]);
    }

    TriJob j;
    j.ntri = ntri;
    j.mode = mode;
    j.cull = p->cull_backfaces;
    j.view_proj = mat4_mul(p->proj, p->view);
    j.data = sj;
    j.corners = scene_corners;
    j.vertices = scene_vertices;
    draw_triangles(fb, &j);
    if (s->env && mode != MODE_WIREFRAME) draw_sky(fb, s->env, p);   // (lines draw no depth)

    free(sj->runs);
    free(sj->inst_first);
    free(sj);
}
