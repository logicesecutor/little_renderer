// test_gpu_trace.c - the GPU path tracer's offline image (app/gpu_frame.h): exactly the samples
// asked for, and the same image every time. Each scene is traced twice, on a GPU and a view of
// its own each time (as two runs of bin/render would), and the two images must be the same to
// the last bit: nothing may depend on how fast the GPU happened to be (how many passes a
// submission holds) or on where the allocator put a buffer. A model's triangles in the box
// make the tracer's buffers grow after the layout was first written: it must be written again.
//
// Part of make test. Without a GPU it says so and passes.
#include <stdio.h>
#include <string.h>
#include "lr.h"
#include "test_util.h"
#include "app/gpu_frame.h"

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
    printf(failures ? "test_gpu_trace: %d FAILED\n" : "all GPU path tracer tests passed\n", failures);
    return failures ? 1 : 0;
}
