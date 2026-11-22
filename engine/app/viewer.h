// viewer.h - the real-time viewer as a part of the engine's app layer: a scene in a window
// (or a canvas), drawn every frame by the rasterizer or the path tracer, on the GPU when there
// is one; the camera flown with the keys and the mouse. bin/view is this and an options
// parser (apps/view.c):
//
//     options_parse(viewer_options(), PROG_VIEW, argc, argv);
//     if (viewer_setup_scene() || viewer_open("view")) return 1;
//     viewer_run();            // natively until Esc; in a page the browser calls the frames
//
// One viewer per program (its state is the module's own).
#ifndef LR_APP_VIEWER_H
#define LR_APP_VIEWER_H

#include "app/options.h"

// the camera as a position and two angles
typedef struct {
    Vec3 pos;
    double yaw, pitch;   // radians; yaw 0 / pitch 0 = looking down +z
    double fov;          // vertical field of view in degrees
} View;
Vec3 view_forward(const View *v);
Vec3 view_right(const View *v);
View view_from_scene(const Scene *s);   // the scene's own camera (cam_from, cam_at, cam_vfov_deg)

Options *viewer_options(void);   // the options it runs with: parse into these first
Scene *viewer_scene(void);       // the scene it shows: set it up before viewer_open
int viewer_setup_scene(void);    // ... as the options say (-sim on: the physics demo). 0 = done
int viewer_open(const char *title);   // the window, the GPU (asynchronously), the buffers. 0 = done
int viewer_frame(void);          // one frame: input, camera, physics, render, show. 0 = quit
void viewer_run(void);           // frames until quit, then viewer_close (in a page: the browser's loop)
void viewer_close(void);
void viewer_print_controls(void);   // the keys

// ---- for a program that drives the viewer itself (the web editor) ----
void viewer_set_input(int on);   // 0: the window's keys and mouse are left to the program (in a page: to its elements)
void viewer_set_title(int on);   // 0: the window's title is left alone
void viewer_set_hud(int on);     // 0: no frame rate over the image
void viewer_set_settle(double seconds);   // hybrid: path tracing after this long still (0: the viewer's 0.12 s)
// hybrid: the GPU's path traced image carried over a move (its samples reprojected, as in rt
// mode), not started over each time the rasterizer takes over
void viewer_set_keep_trace(int on);
void viewer_set_backend(Backend b);   // (to path tracing: its image starts over)
Backend viewer_backend(void);
View viewer_view(void);
void viewer_set_view(const View *v);  // the camera, as if moved by the keys
void viewer_set_home(const View *v);  // where R goes back to
// The program changed the scene (instances added, moved, their materials...): what shows
// follows. restart: the path traced image starts over; else the GPU's keeps the samples of
// what did not change (the CPU's always starts over).
void viewer_scene_changed(int restart);
void viewer_relight(void);       // the rasterizer's light probe made again (key L)
// The rasterizer's light changed (the program made the probe again itself): its image is drawn
// again; the path traced image carries on.
void viewer_redraw(void);
// The image's size: the buffers, the window (a canvas: its pixels) and the GPU's surface
// at it, the GPU's view made again; the image starts over. 0 = done, 1 = out of memory.
int viewer_resize(int w, int h);

typedef struct {
    int tracing;         // what shows is path traced (else rasterized)
    int on_gpu;          // the GPU draws (else the CPU)
    int gpu_starting;    // the GPU is still being set up
    int gpu_tracing;     // the path traced image is the GPU's
    int spp, spp_target; // its samples per pixel, and where it stops (-s)
    double fps;          // frames shown per second
    int width, height;
} ViewerStatus;
void viewer_status(ViewerStatus *st);

#endif
