#include "raster/render.h"

#include "raster/raster.h"
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

// Sutherland-Hodgman against the near plane (z >= -w). A triangle that crosses it becomes
// a triangle or a quad (3 -> at most 4 vertices). Without this, points behind the camera
// would get divided by a negative w and explode across the screen.
static int clip_near(const ClipVert *in, int n, ClipVert *out){
    int m = 0;
    for (int i = 0; i < n; i++){
        const ClipVert *a = &in[i], *b = &in[(i + 1) % n];
        real da = a->clip.z + a->clip.w;    // >= 0 means inside
        real db = b->clip.z + b->clip.w;
        if (da >= 0) out[m++] = *a;
        if ((da >= 0) != (db >= 0)) out[m++] = lerp_vert(a, b, da / (da - db));
    }
    return m;
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

static void draw_clip_triangle(Framebuffer *fb, const ClipVert *cv, RenderMode mode, int cull,
                               FragShader shader, const void *ctx, uint32_t wire_color,
                               int row_lo, int row_hi){
    // 3. clip against the near plane
    ClipVert poly[4];
    int np = clip_near(cv, 3, poly);
    if (np < 3) return;

    // 4. divide by w, map to pixels
    ScreenVert sv[4];
    for (int k = 0; k < np; k++) sv[k] = to_screen(&poly[k], fb);

    // 5. back-face culling
    if (!poly_area_ok(sv, np, cull)) return;

    // 6. rasterize
    if (mode == MODE_WIREFRAME){
        for (int k = 0; k < np; k++){
            const ScreenVert *a = &sv[k], *b = &sv[(k + 1) % np];
            draw_line_rows(fb, (int)r_lround(a->x), (int)r_lround(a->y),
                               (int)r_lround(b->x), (int)r_lround(b->y), wire_color, row_lo, row_hi);
        }
    } else {
        for (int k = 1; k + 1 < np; k++)
            draw_triangle_rows(fb, &sv[0], &sv[k], &sv[k + 1], shader, ctx, 0, row_lo, row_hi);
    }
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
    ScreenVert sv[4];
    int np = 3;
    Vec4 c[3];
    for (int k = 0; k < 3; k++) c[k] = mat4_mul_vec4(j->view_proj, (Vec4){wp[k].x, wp[k].y, wp[k].z, 1.0f});
    if (c[0].z + c[0].w >= 0 && c[1].z + c[1].w >= 0 && c[2].z + c[2].w >= 0){
        // fast path, the usual case: all 3 corners in front of the near plane. Same
        // formulas as to_screen, without copying the varyings.
        for (int k = 0; k < 3; k++){
            real iw = 1.0f / c[k].w;
            sv[k].x = (c[k].x * iw * 0.5f + 0.5f) * fb->w;
            sv[k].y = (0.5f - c[k].y * iw * 0.5f) * fb->h;
        }
    } else {
        ClipVert cv[3], poly[4];
        memset(cv, 0, sizeof(cv));
        for (int k = 0; k < 3; k++) cv[k].clip = c[k];
        np = clip_near(cv, 3, poly);
        if (np < 3) return 0;                               // behind the camera
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

    // ---- pass 2: every band is drawn by one thread
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int b = 0; b < nbands; b++){
        int row_lo = b * RASTER_BAND_ROWS;
        int row_hi = row_lo + RASTER_BAND_ROWS - 1;
        if (row_hi > fb->h - 1) row_hi = fb->h - 1;
        for (int c = 0; c < nchunks; c++){
            for (int k = bins[c].off[b]; k < bins[c].off[b + 1]; k++){
                ClipVert cv[3];
                FragShader shader;
                const void *ctx;
                uint32_t wire;
                j->vertices(j, bins[c].tri[k], cv, &shader, &ctx, &wire);
                draw_clip_triangle(fb, cv, j->mode, j->cull, shader, ctx, wire, row_lo, row_hi);
            }
        }
    }

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
// What it can do fast is local lighting, so every surface gets ambient + Lambert from a
// point light placed at the light quad + a weak "fill" light from the camera (it stands in
// for the bounced light that brightens the faces the lamp does not see), and mirrors /
// glass are faked with a darker base and a sharp highlight. Good enough to frame a shot
// before path tracing it.

#define SCENE_AMBIENT RENDER_SCENE_AMBIENT
#define SCENE_LAMP RENDER_SCENE_LAMP
#define SCENE_FILL RENDER_SCENE_FILL
#define SPHERE_STACKS RENDER_SPHERE_STACKS
#define SPHERE_SLICES RENDER_SPHERE_SLICES

typedef struct {
    Vec3 base;            // material color
    Vec3 light_pos;       // the point light
    Vec3 eye;             // camera position
    real spec_k, spec_exp;
    uint32_t flat_color;  // emitters and wireframe lines
} SceneShading;

// Brightness from the lamp (diff) and the camera fill light; n must face the eye.
static real scene_k(real diff, Vec3 n, Vec3 to_eye){
    return SCENE_AMBIENT + SCENE_LAMP * (diff > 0 ? diff : 0) + SCENE_FILL * v_dot(n, to_eye);
}

// Lambert brightness at p. Surfaces are lit from both sides: the normal is turned towards
// the eye first (scene quads have no "outside", e.g. the walls are seen from inside the box).
static real scene_diffuse(const SceneShading *s, Vec3 p, Vec3 n){
    Vec3 to_eye = v_norm(v_sub(s->eye, p));
    if (v_dot(n, to_eye) < 0) n = v_neg(n);
    return scene_k(v_dot(n, v_norm(v_sub(s->light_pos, p))), n, to_eye);
}

// flat and gouraud: brightness from the vertex stage
static uint32_t shade_scene_light(const real *v, const void *ctx){
    const SceneShading *s = ctx;
    return image_pack_srgb(v_scale(s->base, v[V_LIGHT]));
}

// phong: per-pixel diffuse + Blinn-Phong highlight
static uint32_t shade_scene_phong(const real *v, const void *ctx){
    const SceneShading *s = ctx;
    Vec3 n = v_norm(vec3(v[V_NX], v[V_NY], v[V_NZ]));
    Vec3 p = vec3(v[V_PX], v[V_PY], v[V_PZ]);
    Vec3 to_eye = v_norm(v_sub(s->eye, p));
    if (v_dot(n, to_eye) < 0) n = v_neg(n);
    Vec3 l = v_norm(v_sub(s->light_pos, p));
    real diff = v_dot(n, l), spec = 0;
    if (diff > 0){
        real nh = v_dot(n, v_norm(v_add(l, to_eye)));
        if (s->spec_k > 0 && nh > 0) spec = s->spec_k * r_pow(nh, s->spec_exp);
    } else {
        diff = 0;
    }
    real k = scene_k(diff, n, to_eye);
    return image_pack_srgb(vec3(s->base.x * k + spec, s->base.y * k + spec, s->base.z * k + spec));
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
static void scene_tri(const SceneJob *sj, int i, Vec3 *wp, Vec3 *wn, int *has_n, Vec3 *flat_at){
    const Scene *s = sj->s;
    if (i >= sj->world_tris){                     // an instance's triangle, moved into the world
        int inst, tri;
        inst_tri_of(sj, i, &inst, &tri);
        const Instance *in = &s->insts[inst];
        if (in->mesh < 0){                        // a ball or a box: its primitives in the world
            Primitive pr = instance_shape_prim(in, in->mesh == INST_SPHERE ? 0 : tri / 2);
            render_prim_triangle(&pr, in->mesh == INST_SPHERE ? tri : tri % 2, wp, wn, has_n, flat_at);
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
        return;
    }
    render_prim_triangle(&s->prims[prim], sub, wp, wn, has_n, flat_at);
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
    scene_tri(j->data, i, wp, NULL, NULL, NULL);
}

// world-space triangle (+ optional vertex normals) -> clip space
static void scene_vertices(const TriJob *j, int i, ClipVert *cv, FragShader *shader, const void **ctx, uint32_t *wire){
    const SceneJob *sj = j->data;
    int mat = scene_tri_material(sj, i);
    const SceneShading *sh = &sj->sh[mat];
    Vec3 wp[3], wn[3], flat_at;
    int has_n;
    scene_tri(sj, i, wp, wn, &has_n, &flat_at);
    RenderMode mode = j->mode;

    Vec3 face_n = v_norm(v_cross(v_sub(wp[1], wp[0]), v_sub(wp[2], wp[0])));
    real flat_k = mode == MODE_FLAT ? scene_diffuse(sh, flat_at, face_n) : 0;
    for (int k = 0; k < 3; k++){
        Vec3 n = (has_n && mode != MODE_FLAT) ? wn[k] : face_n;
        cv[k].clip = mat4_mul_vec4(j->view_proj, (Vec4){wp[k].x, wp[k].y, wp[k].z, 1.0f});
        real *vy = cv[k].vary;
        vy[V_PX] = wp[k].x; vy[V_PY] = wp[k].y; vy[V_PZ] = wp[k].z;
        vy[V_NX] = n.x;     vy[V_NY] = n.y;     vy[V_NZ] = n.z;
        vy[V_U] = 0;        vy[V_V] = 0;
        vy[V_LIGHT] = mode == MODE_FLAT ? flat_k : (mode == MODE_GOURAUD ? scene_diffuse(sh, wp[k], n) : 0);
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

RenderMaterial render_scene_material(const Material *mt){
    RenderMaterial rm = {mt->albedo, 0, 1, 0};
    if (mt->type == MAT_LIGHT){
        real mx = v_max(mt->emission);               // show the light's color at full brightness
        rm.base = mx > 0 ? v_scale(mt->emission, 1.0f / mx) : mt->emission;
        rm.constant = 1;
    }
    MatType kind = material_kind(mt);           // (until PBR3: the preset closest to it)
    if (kind == MAT_MIRROR){ rm.base = v_scale(mt->albedo, 0.45f); rm.spec_k = 0.7f; rm.spec_exp = 80; }
    if (kind == MAT_GLASS){  rm.base = v_scale(mt->albedo, 0.25f); rm.spec_k = 0.9f; rm.spec_exp = 120; }
    return rm;
}

void render_scene(Framebuffer *fb, const Scene *s, const RenderParams *p){
    srgb_lut_init();
    RenderMode mode = p->mode == MODE_TEXTURED ? MODE_PHONG : p->mode;

    SceneJob *sj = xmalloc(sizeof(*sj));
    sj->s = s;

    Vec3 light_pos = render_scene_light(s);

    // one shading context per material
    for (int m = 0; m < s->nmats && m < MAX_MATS; m++){
        const Material *mt = &s->mats[m];
        SceneShading *sh = &sj->sh[m];
        RenderMaterial rm = render_scene_material(mt);
        sh->eye = p->eye;
        sh->light_pos = light_pos;
        sh->base = rm.base;
        sh->spec_k = rm.spec_k;
        sh->spec_exp = rm.spec_exp;
        sj->shader[m] = rm.constant ? shade_constant : (mode == MODE_PHONG ? shade_scene_phong : shade_scene_light);
        sh->flat_color = image_pack_srgb(sh->base);
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

    free(sj->runs);
    free(sj->inst_first);
    free(sj);
}
