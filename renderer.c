#include "renderer.h"

#include "raster.h"
#include <math.h>
#include <string.h>

#define AMBIENT 0.12f

// A vertex after the projection matrix, before the divide by w.
typedef struct {
    vec4 clip;
    float vary[V_COUNT];
} ClipVert;

// what the fragment shaders get as ctx
typedef struct {
    vec3 base;            // surface color
    vec3 light;           // unit vector towards the light
    vec3 eye;             // camera position
    const Texture *tex;   // NULL = use base
} Shading;

static float lambert(vec3 n, vec3 l){
    float d = v3_dot(n, l);
    return AMBIENT + (1.0f - AMBIENT) * (d > 0 ? d : 0);
}

// ---- fragment shaders (color of one pixel) ------------------------------------------

// flat and gouraud: the vertex stage already computed the brightness
static uint32_t shade_light(const float *v, const void *ctx){
    const Shading *s = ctx;
    float l = v[V_LIGHT];
    return rgbf(s->base.x * l, s->base.y * l, s->base.z * l);
}

// phong (+ optional texture): lighting evaluated per pixel from the interpolated normal
static uint32_t shade_phong(const float *v, const void *ctx){
    const Shading *s = ctx;
    vec3 n = v3_normalize(v3(v[V_NX], v[V_NY], v[V_NZ]));
    vec3 p = v3(v[V_PX], v[V_PY], v[V_PZ]);

    vec3 base = s->tex ? texture_sample(s->tex, v[V_U], v[V_V]) : s->base;

    float diff = v3_dot(n, s->light);
    float spec = 0.0f;
    if (diff > 0){
        vec3 to_eye = v3_normalize(v3_sub(s->eye, p));
        vec3 h = v3_normalize(v3_add(s->light, to_eye));   // Blinn-Phong half vector
        float nh = v3_dot(n, h);
        spec = nh > 0 ? 0.4f * powf(nh, 40.0f) : 0.0f;
    } else {
        diff = 0;
    }
    float l = AMBIENT + (1.0f - AMBIENT) * diff;
    return rgbf(base.x * l + spec, base.y * l + spec, base.z * l + spec);
}

// ---- pipeline pieces ----------------------------------------------------------------

static ClipVert lerp_vert(const ClipVert *a, const ClipVert *b, float t){
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
        float da = a->clip.z + a->clip.w;    // >= 0 means inside
        float db = b->clip.z + b->clip.w;
        if (da >= 0) out[m++] = *a;
        if ((da >= 0) != (db >= 0)) out[m++] = lerp_vert(a, b, da / (da - db));
    }
    return m;
}

// perspective divide + viewport transform (NDC [-1,1] -> pixels, flipping Y)
static ScreenVert to_screen(const ClipVert *c, const Framebuffer *fb){
    ScreenVert s;
    float iw = 1.0f / c->clip.w;
    s.x = (c->clip.x * iw * 0.5f + 0.5f) * fb->w;
    s.y = (0.5f - c->clip.y * iw * 0.5f) * fb->h;
    s.z = c->clip.z * iw * 0.5f + 0.5f;
    s.inv_w = iw;
    memcpy(s.vary, c->vary, sizeof(s.vary));
    return s;
}

// ---- public -------------------------------------------------------------------------

RenderParams render_default_params(int width, int height){
    RenderParams p;
    p.model = mat4_identity();
    p.eye = v3(0.0f, 0.0f, 3.2f);
    p.view = mat4_look_at(p.eye, v3(0, 0, 0), v3(0, 1, 0));
    p.proj = mat4_perspective(45.0f * 3.14159265f / 180.0f, (float)width / (float)height, 0.1f, 20.0f);
    p.light_dir = v3_normalize(v3(0.4f, 0.7f, 0.9f));
    p.base_color = v3(0.85f, 0.70f, 0.50f);
    p.mode = MODE_PHONG;
    p.texture = NULL;
    p.cull_backfaces = 1;
    return p;
}

const char *render_mode_name(RenderMode mode){
    static const char *names[MODE_COUNT] = {"wireframe", "flat", "gouraud", "phong", "textured"};
    return (mode >= 0 && mode < MODE_COUNT) ? names[mode] : "?";
}

void render_mesh(Framebuffer *fb, const Mesh *mesh, const RenderParams *p){
    mat4 view_proj = mat4_mul(p->proj, p->view);

    Shading sh = {p->base_color, p->light_dir, p->eye, NULL};
    FragShader shader = shade_light;
    if (p->mode == MODE_PHONG)    shader = shade_phong;
    if (p->mode == MODE_TEXTURED){
        shader = shade_phong;
        sh.tex = p->texture;                         // NULL texture falls back to base color
        if (sh.tex) sh.base = v3(1, 1, 1);
    }
    uint32_t wire_color = rgbf(p->base_color.x, p->base_color.y, p->base_color.z);

    for (int i = 0; i < mesh->nf; i++){
        const Face *f = &mesh->f[i];

        // 1. model space -> world space
        vec3 wp[3];
        for (int k = 0; k < 3; k++) wp[k] = mat4_mul_point(p->model, mesh->v[f->v[k]]);
        vec3 face_n = v3_normalize(v3_cross(v3_sub(wp[1], wp[0]), v3_sub(wp[2], wp[0])));

        // 2. per-vertex values + world -> clip space
        ClipVert cv[3];
        for (int k = 0; k < 3; k++){
            vec3 n = face_n;
            if (p->mode != MODE_FLAT && f->n[k] >= 0)
                n = v3_normalize(mat4_mul_dir(p->model, mesh->vn[f->n[k]]));
            vec2 uv = f->t[k] >= 0 ? mesh->vt[f->t[k]] : (vec2){0, 0};

            cv[k].clip = mat4_mul_vec4(view_proj, (vec4){wp[k].x, wp[k].y, wp[k].z, 1.0f});
            float *vy = cv[k].vary;
            vy[V_PX] = wp[k].x; vy[V_PY] = wp[k].y; vy[V_PZ] = wp[k].z;
            vy[V_NX] = n.x;     vy[V_NY] = n.y;     vy[V_NZ] = n.z;
            vy[V_U] = uv.x;     vy[V_V] = uv.y;
            vy[V_LIGHT] = lambert(n, p->light_dir);  // flat: n is the face normal -> same for all 3
        }

        // 3. clip against the near plane
        ClipVert poly[4];
        int np = clip_near(cv, 3, poly);
        if (np < 3) continue;

        // 4. divide by w, map to pixels
        ScreenVert sv[4];
        for (int k = 0; k < np; k++) sv[k] = to_screen(&poly[k], fb);

        // 5. facing: signed area of the polygon (shoelace). Counter-clockwise on screen
        //    (y down) is NEGATIVE, see draw_triangle.
        float area = 0;
        for (int k = 0; k < np; k++){
            const ScreenVert *a = &sv[k], *b = &sv[(k + 1) % np];
            area += a->x * b->y - b->x * a->y;
        }
        if (p->cull_backfaces && area >= 0) continue;

        // 6. rasterize
        if (p->mode == MODE_WIREFRAME){
            for (int k = 0; k < np; k++){
                const ScreenVert *a = &sv[k], *b = &sv[(k + 1) % np];
                draw_line(fb, (int)lroundf(a->x), (int)lroundf(a->y),
                              (int)lroundf(b->x), (int)lroundf(b->y), wire_color);
            }
        } else {
            for (int k = 1; k + 1 < np; k++)
                draw_triangle(fb, &sv[0], &sv[k], &sv[k + 1], shader, &sh, 0);
        }
    }
}
