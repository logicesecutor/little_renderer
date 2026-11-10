// main.c - headless demo: renders one image per milestone into out/*.ppm.
//
//   ./main [model.obj] [texture.ppm]
//
// Defaults to models/torus.obj and a procedural checkerboard.
// For the real-time window see viewer.c ("make viewer").
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "framebuffer.h"
#include "mesh.h"
#include "raster.h"
#include "renderer.h"
#include "texture.h"

#define OUT_DIR "out"
#define PI 3.14159265f

static double now_ms(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void save(const Framebuffer *fb, const char *name){
    char path[256];
    snprintf(path, sizeof(path), OUT_DIR "/%s", name);
    if (fb_save_ppm(fb, path) != 0) fprintf(stderr, "could not write %s\n", path);
    else printf("  wrote %s\n", path);
}

// ---- milestone 0: hello pixels ------------------------------------------------------
static void demo_gradient(void){
    Framebuffer *fb = fb_create(256, 256);
    for (int y = 0; y < fb->h; y++)
        for (int x = 0; x < fb->w; x++)
            fb_set_pixel(fb, x, y, rgb8(x, y, 128));
    save(fb, "0_gradient.ppm");
    fb_destroy(fb);
}

// ---- milestone 1: lines in every direction ------------------------------------------
static void demo_lines(void){
    Framebuffer *fb = fb_create(256, 256);
    fb_clear(fb, rgb8(15, 15, 20));
    int cx = 128, cy = 128;
    for (int i = 0; i < 48; i++){                       // a star: steep, shallow, backwards...
        float a = i * 2.0f * PI / 48.0f;
        int x = cx + (int)lroundf(120.0f * cosf(a));
        int y = cy + (int)lroundf(120.0f * sinf(a));
        draw_line(fb, cx, cy, x, y, rgbf(0.5f + 0.5f * cosf(a), 0.5f + 0.5f * sinf(a), 0.8f));
    }
    draw_line(fb, 0, 0, 255, 255, rgb8(255, 255, 255));   // the original test: the diagonal
    draw_line(fb, 5, 250, 250, 250, rgb8(255, 255, 0));   // horizontal
    draw_line(fb, 5, 5, 5, 250, rgb8(255, 255, 0));       // vertical
    draw_line(fb, -50, 100, 400, 140, rgb8(0, 255, 0));   // partly off screen: must not crash
    save(fb, "1_lines.ppm");
    fb_destroy(fb);
}

// ---- milestones 3 + 5: filled triangles and the z-buffer ----------------------------
static uint32_t shade_rgb(const float *vary, const void *ctx){
    (void)ctx;
    return rgbf(vary[V_NX], vary[V_NY], vary[V_NZ]);   // we smuggle a color through the normal slots
}

static ScreenVert make_vert(float x, float y, float z, float r, float g, float b){
    ScreenVert v;
    memset(&v, 0, sizeof(v));
    v.x = x; v.y = y; v.z = z; v.inv_w = 1.0f;
    v.vary[V_NX] = r; v.vary[V_NY] = g; v.vary[V_NZ] = b;
    return v;
}

static void demo_triangles(void){
    Framebuffer *fb = fb_create(320, 240);
    fb_clear(fb, rgb8(15, 15, 20));

    // big rainbow triangle in the middle distance: colors interpolate via barycentric weights
    ScreenVert a = make_vert(30, 210, 0.5f, 1, 0, 0);
    ScreenVert b = make_vert(290, 210, 0.5f, 0, 1, 0);
    ScreenVert c = make_vert(160, 20, 0.5f, 0, 0, 1);
    draw_triangle(fb, &a, &b, &c, shade_rgb, NULL, 1);

    // an orange triangle IN FRONT (z = 0.2), drawn second
    ScreenVert d = make_vert(120, 230, 0.2f, 1, 0.6f, 0.1f);
    ScreenVert e = make_vert(310, 150, 0.2f, 1, 0.6f, 0.1f);
    ScreenVert f = make_vert(200, 60, 0.2f, 1, 0.6f, 0.1f);
    draw_triangle(fb, &d, &e, &f, shade_rgb, NULL, 1);

    // a white triangle BEHIND (z = 0.8), drawn last: the z-buffer must keep it hidden
    // wherever the others are already in front of it
    ScreenVert g = make_vert(5, 120, 0.8f, 1, 1, 1);
    ScreenVert h = make_vert(200, 235, 0.8f, 1, 1, 1);
    ScreenVert i = make_vert(150, 30, 0.8f, 1, 1, 1);
    draw_triangle(fb, &g, &h, &i, shade_rgb, NULL, 0);

    save(fb, "3_triangles_zbuffer.ppm");
    fb_destroy(fb);
}

// ---- milestones 2, 4, 5, 6, 8: the model --------------------------------------------
static void demo_model(const Mesh *mesh, const Texture *tex){
    const int W = 640, H = 480;
    Framebuffer *fb = fb_create(W, H);
    uint32_t bg = rgb8(24, 26, 32);

    RenderParams p = render_default_params(W, H);
    p.model = mat4_mul(mat4_rotate_x(0.45f), mat4_rotate_y(0.6f));   // so it isn't edge-on
    p.texture = tex;

    struct { RenderMode mode; const char *file; } shots[] = {
        {MODE_WIREFRAME, "2_wireframe.ppm"},
        {MODE_FLAT,      "4_flat.ppm"},
        {MODE_GOURAUD,   "8_gouraud.ppm"},
        {MODE_PHONG,     "8_phong.ppm"},
        {MODE_TEXTURED,  "8_textured.ppm"},
    };
    for (size_t s = 0; s < sizeof(shots) / sizeof(shots[0]); s++){
        p.mode = shots[s].mode;
        fb_clear(fb, bg);
        double t0 = now_ms();
        render_mesh(fb, mesh, &p);
        printf("  %-9s %6.2f ms\n", render_mode_name(p.mode), now_ms() - t0);
        save(fb, shots[s].file);
    }

    // z-buffer as an image
    p.mode = MODE_PHONG;
    fb_clear(fb, bg);
    render_mesh(fb, mesh, &p);
    fb_show_depth(fb);
    save(fb, "5_depth.ppm");

    // back faces visible: the inside of the model shows through where it is open
    p.cull_backfaces = 0;
    p.mode = MODE_FLAT;
    fb_clear(fb, bg);
    render_mesh(fb, mesh, &p);
    save(fb, "4_flat_no_culling.ppm");
    p.cull_backfaces = 1;

    // near-plane clipping: put the camera so close that the near plane cuts the model
    p.mode = MODE_PHONG;
    p.eye = v3(0.0f, 0.2f, 1.15f);
    p.view = mat4_look_at(p.eye, v3(0, 0, 0), v3(0, 1, 0));
    fb_clear(fb, bg);
    render_mesh(fb, mesh, &p);
    save(fb, "9_near_clipping.ppm");

    fb_destroy(fb);
}

int main(int argc, char **argv){
    const char *model_path = argc > 1 ? argv[1] : "models/torus.obj";
    const char *tex_path = argc > 2 ? argv[2] : NULL;

    mkdir(OUT_DIR, 0755);   // fine if it already exists

    printf("milestones 0, 1, 3, 5:\n");
    demo_gradient();
    demo_lines();
    demo_triangles();

    Mesh *mesh = mesh_load_obj(model_path);
    if (!mesh){
        fprintf(stderr, "could not load %s (run: make models)\n", model_path);
        return 1;
    }
    mesh_normalize(mesh);
    if (mesh->nvn == 0) mesh_compute_normals(mesh);
    printf("loaded %s: %d vertices, %d triangles\n", model_path, mesh->nv, mesh->nf);

    Texture *tex = tex_path ? texture_load_ppm(tex_path) : NULL;
    if (tex_path && !tex) fprintf(stderr, "could not read texture %s, using checkerboard\n", tex_path);
    if (!tex) tex = texture_checker(256, 16);

    printf("model (640x480):\n");
    demo_model(mesh, tex);

    texture_destroy(tex);
    mesh_destroy(mesh);
    return 0;
}
