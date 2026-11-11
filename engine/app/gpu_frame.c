#define _POSIX_C_SOURCE 200809L   // nanosleep
#include "app/gpu_frame.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

static void nap_ms(int ms){
    struct timespec ts = {0, (long)ms * 1000000L};
    nanosleep(&ts, NULL);
}

int gpu_frame_open(int w, int h, Gpu **gpu, GpuView **view){
    *gpu = NULL;
    *view = NULL;
    if (!gpu_available()) return 1;
    Gpu *g = gpu_create(NULL, w, h);
    if (!g) return 1;
    for (int k = 0; k < 20000 && gpu_state(g) == GPU_PENDING; k++) nap_ms(1);   // asked asynchronously
    GpuView *v = gpu_state(g) == GPU_READY ? gpu_view_create(g, w, h) : NULL;
    if (!v){
        fprintf(stderr, "GPU not used (%s): the CPU renders\n", gpu_message(g));
        gpu_destroy(g);
        return 1;
    }
    *gpu = g;
    *view = v;
    return 0;
}

void gpu_frame_close(Gpu *gpu, GpuView *view){
    gpu_view_destroy(view);
    gpu_destroy(gpu);
}

typedef struct { Image *img; int done; } TraceRead;

static void on_samples(void *user, const float *sc, int w, int h){
    TraceRead *r = user;
    for (int i = 0; i < w * h; i++){
        float n = sc[4 * i + 3];
        r->img->px[i] = n > 0 ? vec3(sc[4 * i] / n, sc[4 * i + 1] / n, sc[4 * i + 2] / n) : vec3(0, 0, 0);
    }
    r->done = 1;
}

int gpu_frame_trace(Gpu *gpu, GpuView *view, const Scene *s, const Camera *cam, const RenderParams *rp,
                    const MarchParams *mp, int spp, int max_depth, int mis, real clamp, Image *img, int progress){
    GpuTraceParams tp = {*cam, max_depth, mis, clamp};
    if (!gpu_view_trace_start(view, s, &tp)) return 1;
    int shown = -1;
    while (gpu_view_trace_passes(view) < spp && gpu_state(gpu) == GPU_READY){
        // one band (or, once the GPU proves fast, several whole passes) at a time; the
        // last submission may add a few passes more than spp, which only lowers the noise
        gpu_view_trace_more(view, s, rp, mp);
        gpu_wait(gpu);
        int p = gpu_view_trace_passes(view) * 100 / (spp > 0 ? spp : 1);
        if (progress && p != shown){
            fprintf(stderr, "\r  %3d%%", p > 100 ? 100 : p);
            fflush(stderr);
            shown = p;
        }
    }
    if (progress) fprintf(stderr, "\n");
    if (gpu_state(gpu) != GPU_READY) return 1;
    TraceRead r = {img, 0};
    if (!gpu_view_trace_read(view, on_samples, &r)) return 1;
    while (!r.done && gpu_state(gpu) == GPU_READY) gpu_wait(gpu);
    return r.done ? 0 : 1;
}

typedef struct { Framebuffer *fb; int done; } FrameRead;

static void on_frame(void *user, const uint32_t *px, int w, int h){
    FrameRead *r = user;
    if (w == r->fb->w && h == r->fb->h) memcpy(r->fb->color, px, (size_t)w * h * sizeof(uint32_t));
    r->done = 1;
}

int gpu_frame_raster(Gpu *gpu, GpuView *view, const Scene *s, const RenderParams *rp, const MarchParams *mp,
                     int volumes, Framebuffer *fb){
    FrameRead r = {fb, 0};
    if (!gpu_view_capture(view, on_frame, &r)) return 1;
    if (!gpu_view_present_scene(view, s, rp, mp, volumes)) return 1;
    while (!r.done && gpu_state(gpu) == GPU_READY) gpu_wait(gpu);
    return r.done ? 0 : 1;
}
