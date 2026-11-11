// options.h - command-line options shared by the three programs:
//   render  (offline: one image into a PPM, with the path tracer or the rasterizer)
//   view    (real-time window: rasterized preview while moving, path tracing when still)
//   sim     (offline: a physics simulation rendered frame by frame into an animation)
// Run either one with -h for the full list. Apps only, not needed to use the engine.
#ifndef LR_OPTIONS_H
#define LR_OPTIONS_H

#include <stdio.h>
#include "core/scene.h"
#include "core/volume.h"
#include "raster/render.h"
#include "raster/volume_march.h"
#include "rt/accel.h"

typedef enum { PROG_RENDER, PROG_VIEW, PROG_SIM } Program;

typedef enum {
    BACKEND_RT,       // path tracer
    BACKEND_RASTER,   // rasterizer
    BACKEND_HYBRID    // view only: rasterizer while the camera moves, path tracer when it stops
} Backend;

// the physics demos (app/simdemo.h): what is in the room and what happens
typedef enum {
    SIM_DROP,    // balls and boxes fall through the cloud, balloons rise slowly through it
    SIM_BLAST    // things rest on the floor under the cloud; an explosion at 1 s
} SimDemo;

typedef struct {
    int on;                // view: -sim on; always on for sim
    SimDemo demo;
    int frames;            // sim: frames to render
    double fps;            // sim: frames per second (the physics runs at the same rate, substepped)
    int balls, boxes, balloons;
    Vec3 wind;             // the breeze the air drifts towards, units / s
    int fluid_res;         // air cells along the room's side
    double buoyancy;       // the cloud rises with this acceleration per unit density
    double vorticity;      // vorticity confinement strength
    int seed;              // where things start
    int gif;               // sim: also make <out>.gif with tools/frames_to_gif.py (uv, or python3 + Pillow)
} SimOptions;

#define AUTO_ACCEL_MIN_PRIMS 64   // "-accel auto": below this, brute force is fastest

typedef struct {
    Program prog;
    Backend backend;

    // image and quality
    int width, height;     // pixels
    int spp;               // render: samples per pixel; view: stop refining after this many
    int max_depth;         // maximum bounces per path
    double clamp;          // firefly clamp, 0 = off
    int threads;           // 0 = all cores
    const char *out;       // render: output file; view: screenshot file prefix

    // path tracer estimators
    int mis;               // multiple importance sampling (NEE vs bounce), default on
    int caustics_lt;       // caustics by light tracing (render only), default on
    double photons;        // light paths per camera path for -caustics lt

    // rasterizer
    RenderMode shading;    // wireframe, flat, gouraud or phong
    int raster_clouds;     // draw the media by raymarching (raster/volume_march.h), default on
    int gpu;               // view: use the GPU (gpu/): 0 off, 1 on (default in a web browser), 2 offscreen
    int gpu_fluid;         // view -sim on with the GPU: step the air there too (default 1)
    int gpu_raster;        // view with the GPU: the rasterizer's triangles are drawn there (default 1)
    int gpu_trace;         // view with the GPU: it path traces too, while the scene stands still
                           // (1 / 0; -1 = automatic: yes unless the GPU is a software one)
    MarchParams march;     // -march (steps), -lightres

    // scene
    SceneOptions scene;
    int use_cloud;         // -cloud on (always on for -scene cloud)
    CloudParams cloud;     // -density, -albedo, -g, -seed, -detail, -time, -cloudres

    // camera overrides (otherwise the scene's default camera)
    int set_cam, set_look, set_fov;
    Vec3 cam, look;
    double fov;

    // acceleration structure (path tracer)
    int accel;             // -1 = auto, otherwise an AccelKind
    AccelParams ap;

    // view only
    int preview;           // path tracer at 1/preview resolution while moving (1 = full)
    int move_spp;          // samples per pixel for each low-resolution frame while moving
    double speed;          // movement speed in scene units per second
    double autoquit;       // quit after this many seconds (0 = never); for tests/benchmarks

    // physics (sim, and view -sim on)
    SimOptions sim;
} Options;

// Fills *o with the defaults for `prog`, then applies argv. Exits on -h or on bad input.
void options_parse(Options *o, Program prog, int argc, char **argv);

// Builds the scene, applies the camera overrides, chooses and builds the acceleration
// structure (not for `render -backend raster`, which does not need one), sets the thread
// count, and prints what it did. Returns 0 on success.
int options_setup_scene(const Options *o, Scene *s);

// The AccelKind that "-accel auto" (or the explicit choice) gives for this scene.
AccelKind options_resolve_accel(const Options *o, const Scene *s);

// Prints a `render` command line that path traces this camera with these options.
void options_print_command(const Options *o, Vec3 cam, Vec3 look, double fov, FILE *f);

const char *backend_name(Backend b);
void options_usage(Program prog, FILE *f);

#endif
