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

#endif
