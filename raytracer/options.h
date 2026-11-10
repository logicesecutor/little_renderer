// options.h - command-line options shared by `raytracer` (offline, writes a PPM) and
// `rtview` (real-time window). Run either program with -h for the full list.
#ifndef OPTIONS_H
#define OPTIONS_H

#include <stdio.h>
#include "scene.h"

typedef enum { PROG_RAYTRACER, PROG_RTVIEW } Program;

#define AUTO_ACCEL_MIN_PRIMS 64   // "-accel auto": below this, brute force is fastest

typedef struct {
    Program prog;

    // image and quality
    int width, height;     // pixels
    int spp;               // raytracer: samples per pixel; rtview: stop refining after this many
    int max_depth;         // maximum bounces per path
    double clamp;          // firefly clamp, 0 = off
    int threads;           // 0 = all cores
    const char *out;       // raytracer: output file; rtview: screenshot file prefix

    // scene
    SceneOptions scene;

    // camera overrides (otherwise the scene's default camera)
    int set_cam, set_look, set_fov;
    Vec3 cam, look;
    double fov;

    // acceleration structure
    int accel;             // -1 = auto, otherwise an AccelKind
    AccelParams ap;

    // rtview only
    int preview;           // render at 1/preview resolution while moving (1 = off)
    int move_spp;          // samples per pixel for each low-resolution frame while moving
    double speed;          // movement speed in scene units per second
    double autoquit;       // quit after this many seconds (0 = never); for tests/benchmarks
} Options;

// Fills *o with the defaults for `prog`, then applies argv. Exits on -h or on bad input.
void options_parse(Options *o, Program prog, int argc, char **argv);

// Builds the scene, applies the camera overrides, chooses and builds the acceleration
// structure, sets the thread count, and prints what it did. Returns 0 on success.
int options_setup_scene(const Options *o, Scene *s);

// The AccelKind that "-accel auto" (or the explicit choice) gives for this scene.
AccelKind options_resolve_accel(const Options *o, const Scene *s);

// Prints a `raytracer` command line that renders this camera with these options.
void options_print_command(const Options *o, Vec3 cam, Vec3 look, double fov, FILE *f);

void options_usage(Program prog, FILE *f);

#endif
