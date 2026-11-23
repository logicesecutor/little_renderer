// test_gpu_trace.c - the GPU path tracer's offline image (app/gpu_frame.h): exactly the samples
// asked for, and the same image every time. Each scene is traced twice, on a GPU and a view of
// its own each time (as two runs of bin/render would), and the two images must be the same to
// the last bit: nothing may depend on how fast the GPU happened to be (how many passes a
// submission holds) or on where the allocator put a buffer. A model's triangles in the box
// make the tracer's buffers grow after the layout was first written: it must be written again.
//
// A glowing ball over a floor (its light sampled: Scene.emitters): the GPU's image, by the one
// kernel and by the wavefront's, has the CPU's mean (within 2%), and little noise where the
// ball lights the floor.
//
// Part of make test. Without a GPU it says so and passes.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "lr.h"
#include "test_util.h"
#include "app/gpu_frame.h"
#include "rt/tracer.h"

#define W 96
#define H 96

// the scene traced with spp samples per pixel into img, on a new GPU and view; the samples
// per pixel the view counted, -1 if there is no GPU or it cannot hold the scene
static int trace_once(Scene *s, int spp, Image *img){
    Gpu *g;
    GpuView *v;
    if (gpu_frame_open(W, H, &g, &v)) return -1;
    RenderParams rp = render_params_camera(s->cam_from, s->cam_at, s->cam_up, s->cam_vfov_deg, W, H);
    MarchParams mp = march_default_params();
    Camera cam = camera_make(s->cam_from, s->cam_at, s->cam_up, s->cam_vfov_deg, (real)W / H);
    int passes = -1;
    if (gpu_frame_trace(g, v, s, &cam, &rp, &mp, spp, 8, 1, 0, 1, 0, img, 0) == 0) passes = gpu_view_trace_passes(v);
    gpu_frame_close(g, v);
    return passes;
}

static void same_twice(const char *name, const SceneOptions *so, int spp){
    static Scene scene;
    memset(&scene, 0, sizeof(scene));
    char what[96];
    if (scene_cornell_box(&scene, so) != 0){
        snprintf(what, sizeof(what), "%s: the scene", name);
        check(what, 0);
        return;
    }
    AccelParams ap = accel_default_params();
    scene_set_accel(&scene, ACCEL_BVH, &ap);
    Image a = image_create(W, H), b = image_create(W, H);
    int pa = trace_once(&scene, spp, &a), pb = trace_once(&scene, spp, &b);
    snprintf(what, sizeof(what), "%s: %d samples per pixel, no more (%d, %d)", name, spp, pa, pb);
    check(what, pa == spp && pb == spp);
    double sum = 0;
    for (int i = 0; i < W * H; i++) sum += a.px[i].x + a.px[i].y + a.px[i].z;
    snprintf(what, sizeof(what), "%s: an image (mean %.3f)", name, sum / (3.0 * W * H));
    check(what, pa == spp && sum > 0);
    snprintf(what, sizeof(what), "%s: traced again, the same to the last bit", name);
    check(what, pa == spp && pb == spp && !memcmp(a.px, b.px, (size_t)W * H * sizeof(a.px[0])));
    image_free(&a);
    image_free(&b);
    scene_free(&scene);
}

// the image's mean and, over the floor's square under the ball, its pixels' spread around their
// own neighbours' (noise): on the GPU (wavefront: its kernels) or the CPU
static void emit_stats(Scene *s, int where, int spp, double *mean, double *noise){
    Image img = image_create(W, H);
    Camera cam = camera_make(s->cam_from, s->cam_at, s->cam_up, s->cam_vfov_deg, (real)W / H);
    int ok = 1;
    if (where < 2){
        Gpu *g;
        GpuView *v;
        ok = gpu_frame_open(W, H, &g, &v) == 0;
        if (ok){
            gpu_view_trace_wavefront(v, where == 1);
            RenderParams rp = render_params_camera(s->cam_from, s->cam_at, s->cam_up, s->cam_vfov_deg, W, H);
            MarchParams mp = march_default_params();
            ok = gpu_frame_trace(g, v, s, &cam, &rp, &mp, spp, 8, 1, 0, 0, 0, &img, 0) == 0;
            gpu_frame_close(g, v);
        }
    } else {
        TraceParams tp = trace_default_params();
        Vec3 *px = malloc(sizeof(Vec3) * W * H);
        trace_image(s, &cam, &tp, W, H, spp, 1, 0, px, NULL, NULL);
        for (int i = 0; i < W * H; i++) img.px[i] = px[i];
        free(px);
    }
    double sum = 0, d2 = 0;
    int n = 0;
    for (int i = 0; i < W * H; i++) sum += img.px[i].x;
    for (int y = H / 2; y < H - 8; y++)
        for (int x = W / 4; x < 3 * W / 4; x++){
            double c = img.px[y * W + x].x, nb = (img.px[y * W + x - 1].x + img.px[y * W + x + 1].x) / 2;
            d2 += (c - nb) * (c - nb);
            n++;
        }
    *mean = ok ? sum / (W * H) : -1;
    *noise = ok ? sqrt(d2 / n) : -1;
    image_free(&img);
}

static void glowing_ball(void){
    static Scene s;
    memset(&s, 0, sizeof(s));
    int floor = scene_add_material(&s, material_diffuse(vec3(0.5f, 0.5f, 0.5f)));
    Material glow = material_pbr(vec3(0, 0, 0), 0, 1, 1.5f, 0);
    glow.emission = vec3(10, 10, 10);
    int mg = scene_add_material(&s, glow);
    scene_add_prim(&s, make_quad(vec3(-2000, 0, -2000), vec3(0, 0, 4000), vec3(4000, 0, 0), floor));
    scene_add_instance(&s, INST_SPHERE, xform_axes(mat3_identity(), vec3(10, 10, 10), vec3(0, 50, 0)), mg);
    s.light = -1;
    s.cam_from = vec3(0, 160, -220);
    s.cam_at = vec3(0, 10, 0);
    s.cam_up = vec3(0, 1, 0);
    s.cam_vfov_deg = 40;
    scene_set_accel(&s, ACCEL_BVH, NULL);
    double m[3], nz[3];
    for (int k = 0; k < 3; k++) emit_stats(&s, k, 64, &m[k], &nz[k]);
    printf("  glowing ball: means %.4f %.4f %.4f (GPU, wavefront, CPU), noise %.4f %.4f %.4f\n", m[0], m[1], m[2], nz[0], nz[1], nz[2]);
    check("a glowing ball sampled as a light: the GPU's mean is the CPU's (2%)", m[0] > 0 && fabs(m[0] - m[2]) < 0.02 * m[2]);
    check("... the wavefront's too", m[1] > 0 && fabs(m[1] - m[2]) < 0.02 * m[2]);
    check("... with as little noise as the CPU's (under twice)", nz[0] < 2 * nz[2] && nz[1] < 2.5 * nz[2]);
    scene_free(&s);
}

int main(void){
    printf("test_gpu_trace\n");
    Gpu *g;
    GpuView *v;
    if (gpu_frame_open(W, H, &g, &v)){
        printf("  no GPU: skipped\nall GPU path tracer tests passed\n");
        return 0;
    }
    gpu_frame_close(g, v);
    SceneOptions boxes = {.kind = SCENE_BOXES};
    same_twice("boxes", &boxes, 20);          // (20: not a number of passes a fast GPU's submissions add up to)
    SceneOptions mesh = {.kind = SCENE_MESH, .obj_path = "models/torus.obj"};
    for (int k = 0; k < 3; k++) same_twice("a model in the box", &mesh, 5);
    glowing_ball();
    printf(failures ? "test_gpu_trace: %d FAILED\n" : "all GPU path tracer tests passed\n", failures);
    return failures ? 1 : 0;
}
