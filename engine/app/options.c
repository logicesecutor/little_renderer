#include "app/options.h"
#include "base/clock.h"
#include "gpu/gpu.h"
#include "rt/probe.h"
#include "scene/env.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int app_run_command(const char *cmd){
#ifdef _WIN32
    size_t n = strlen(cmd) + 3;
    char *q = malloc(n);
    if (!q) return -1;
    snprintf(q, n, "\"%s\"", cmd);
    int r = system(q);
    free(q);
    return r;
#else
    return system(cmd);
#endif
}

const char *backend_name(Backend b){
    return b == BACKEND_RT ? "rt" : (b == BACKEND_RASTER ? "raster" : "hybrid");
}
// The options, one table that both the parser and -h read (it was a chain of strcmp and a help
// text typing every name and default again, which drifted apart: B1, B3). The defaults printed
// are the values options_parse starts from, but for a few that depend on other options
// (written in `def`).

typedef enum {
    K_SECTION,   // -h: a heading (name), no option
    K_TEXT,      // -h: a paragraph (name)
    K_INT, K_U32, K_DBL,
    K_REAL,      // a `real` field (float or double by the build)
    K_STR,       // the argument itself, kept (argv outlives the options)
    K_ONOFF,     // on | off: 1 | 0
    K_PICK,      // one of names[]: its index + base
    K_VEC3,      // x,y,z
    K_COLOR      // a or r,g,b
} Kind;

enum { P_R = 1 << PROG_RENDER, P_V = 1 << PROG_VIEW, P_S = 1 << PROG_SIM };

typedef struct {
    const char *name;           // "-n" (a section's heading, a paragraph's text)
    Kind kind;
    unsigned char progs;        // the programs whose -h shows it (0 = all); also the only ones
    unsigned char strict;       // that take it when strict (else all do: render takes -preview)
    unsigned char help_only;    // -h shows it, the parser uses another entry of the same name
    unsigned char join;         // -h: on the line of the entry before ("-march n / -lightres n")
    size_t off, off2;           // the field; off2 a second one set alike (-n), 0 = none
    const char *const *names;   // K_PICK
    int base;
    size_t set;                 // also *(int *)(o + set) |= setv (-cam: set_cam; -base: mesh_set)
    int setv;
    int (*ok)(double);          // a range check of the value read ...
    const char *bad;            // ... and its message
    const char *arg;            // -h: "size" in "-n size" (K_PICK, K_ONOFF: from the names)
    const char *text;           // -h: lines split at '\n'
    const char *def;            // -h: the default; NULL = the starting value, "" = none,
                                // "; ..." = the starting value and this, else this text
} Opt;

static const char *const SCENES[] = {"boxes", "spheres", "mesh", "cloud", "empty", "showcase", "sky", NULL};
// the names of -mat, -accel and -bvhsplit, read and printed (the order of MatType, AccelKind
// + 1, BvhSplit)
static const char *const MATS[] = {"diffuse", "mirror", "glass", NULL};
static const char *const ACCELS[] = {"auto", "none", "octree", "bvh", NULL};
static const char *const SPLITS[] = {"sah", "middle", NULL};
static const char *const DEMOS[] = {"drop", "blast", "sky", NULL};
static const char *const BACKENDS[] = {"rt", "raster", "hybrid", NULL};   // (Backend)
static const char *const BACKENDS_OFFLINE[] = {"rt", "raster", NULL};
static const char *const SHADINGS[] = {"wireframe", "flat", "gouraud", "phong", NULL};   // (RenderMode, not textured)
static const char *const GPUS[] = {"off", "on", "offscreen", NULL};
static const char *const AAS[] = {"off", "fxaa", NULL};
static const char *const SAMPLERS[] = {"random", "sobol", NULL};
static const char *const CAUSTICS[] = {"pt", "lt", NULL};

// the fields K_PICK writes through an int
_Static_assert(sizeof(Backend) == sizeof(int) && sizeof(RenderMode) == sizeof(int) && sizeof(SceneKind) == sizeof(int)
               && sizeof(MatType) == sizeof(int) && sizeof(BvhSplit) == sizeof(int) && sizeof(SimDemo) == sizeof(int),
               "enums read as int");

static int fov_ok(double x){ return x > 0 && x < 180; }
static int positive(double x){ return x > 0; }

#define F(field) offsetof(Options, field)
#define STR_(x) #x
#define STR(x) STR_(x)
#define INT(n, f, a, t)  {n, K_INT, .off = F(f), .arg = a, .text = t}
#define DBL(n, f, a, t)  {n, K_DBL, .off = F(f), .arg = a, .text = t}
#define REAL(n, f, a, t) {n, K_REAL, .off = F(f), .arg = a, .text = t}
#define ONOFF(n, f, t)   {n, K_ONOFF, .off = F(f), .text = t}
#define PICK(n, f, ns, t) {n, K_PICK, .off = F(f), .names = ns, .text = t}
#define SECTION(t, p)    {t, K_SECTION, .progs = p}

static const Opt OPTS[] = {
    {"physics simulation: renders every frame with the path tracer into <-o>/frame_NNNN.png", K_TEXT, .progs = P_S},
    SECTION("demo", P_S),
    SECTION("physics (a live simulation in the window)", P_V),
    {"-sim", K_ONOFF, P_V, 1, .off = F(sim.on), .text = "run the physics demo"},
    {"-demo", K_PICK, P_V, 1, .off = F(sim.demo), .names = DEMOS, .text = "which one"},
    {"-demo", K_PICK, P_S, 1, .off = F(sim.demo), .names = DEMOS,
     .text = "drop: balls and boxes fall through the cloud, balloons rise\n"
             "blast: an explosion in the cloud at 1 s\n"
             "sky: bodies rain onto a floor under the sky (-sky)"},
    {"-frames", K_INT, P_S, .off = F(sim.frames), .arg = "n", .text = "frames to render"},
    {"-fps", K_DBL, P_S, .off = F(sim.fps), .arg = "n", .text = "frames per second"},
    {"-gif", K_ONOFF, P_S, 1, .off = F(sim.gif), .text = "also write <-o>.gif (uv, or python3 + Pillow)"},
    {"-balls", K_INT, P_V | P_S, .off = F(sim.balls), .arg = "n", .text = "how many of each"},
    {"-boxes", K_INT, P_V | P_S, .join = 1, .off = F(sim.boxes), .arg = "n"},
    {"-balloons", K_INT, P_V | P_S, .join = 1, .off = F(sim.balloons), .arg = "n"},
    {"-meshes", K_INT, P_V | P_S, .off = F(sim.meshes), .arg = "n", .text = "n copies of the model (-obj) fall too, as convex hulls"},
    {"-collider", K_ONOFF, P_V | P_S, .off = F(sim.collider), .text = "the model stands in the room, its triangles collide"},
    {"-wind", K_VEC3, P_V | P_S, .off = F(sim.wind), .arg = "x,y,z",
     .text = "the air drifts towards this velocity, units / s\n(the room is closed but for its front: a wind mostly swirls)"},
    {"-fluidres", K_INT, P_V | P_S, .off = F(sim.fluid_res), .arg = "n", .text = "air cells along the room's side", .def = "; view: 32"},
    {"-buoyancy", K_DBL, P_V | P_S, .off = F(sim.buoyancy), .arg = "a", .text = "the cloud rises: acceleration per density"},
    {"-vorticity", K_DBL, P_V | P_S, .off = F(sim.vorticity), .arg = "e", .text = "keeps the swirls alive, 0 = off"},
    {"-physseed", K_INT, P_V | P_S, .off = F(sim.seed), .arg = "n", .text = "where things start"},
    {"-bodies", K_INT, P_V | P_S, .off = F(sim.bodies), .arg = "n", .text = "-demo sky: bodies falling / still things about"},
    {"-props", K_INT, P_V | P_S, .join = 1, .off = F(sim.props), .arg = "n"},

    SECTION("backend", 0),
    {"-backend", K_PICK, P_V, 1, .off = F(backend), .names = BACKENDS, .arg = "hybrid|rt|raster",
     .text = "hybrid = rasterizer while moving, path tracer when still"},
    {"-backend", K_PICK, P_R, 1, .off = F(backend), .names = BACKENDS_OFFLINE, .text = "path tracer or rasterizer"},
    {"-backend", K_PICK, P_S, 1, .off = F(backend), .names = BACKENDS_OFFLINE,
     .text = "path tracer, or rasterizer (raymarched cloud: fast)"},
    {"-shading", K_PICK, .off = F(shading), .names = SHADINGS, .arg = "phong|gouraud|flat|wireframe", .text = "rasterizer shading"},
    {"-rastercloud", K_ONOFF, P_R | P_V, 1, .off = F(raster_clouds), .text = "rasterizer: draw clouds and fog by raymarching"},
    {"-rastercloud", K_ONOFF, P_S, 1, .off = F(raster_clouds), .text = "rasterizer: draw the cloud by raymarching"},
    INT("-march", march.max_steps, "n", "raymarching: max steps per ray / light grid size"),
    {"-lightres", K_INT, .join = 1, .off = F(march.light_res), .arg = "n"},
    INT("-marchscale", march.scale, "1|2", "raymarching at full or half resolution"),
    {"-gpu", K_PICK, P_V, .off = F(gpu), .names = GPUS, .arg = "on|off|offscreen", .def = "",
     .text = "render (rasterizer, path tracer, clouds) and simulate the air on the GPU\n"
             "(WebGPU; on when the build has it: make web, or natively unless NOGPU=1);\n"
             "offscreen: the GPU draws into a texture read back to the canvas"},
    {"-gpufluid", K_ONOFF, P_V, .off = F(gpu_fluid), .text = "with -sim on and the GPU: the air and the cloud are simulated there too"},
    {"-gpurigid", K_ONOFF, P_V, .off = F(gpu_rigid), .text = "with -sim on and the GPU: the balls and boxes are simulated there too"},
    {"-gpuraster", K_ONOFF, P_V, .off = F(gpu_raster), .text = "with the GPU: the rasterizer's triangles are drawn there too"},
    {"-gputrace", K_ONOFF, P_V, .off = F(gpu_trace), .def = "",
     .text = "with the GPU: it path traces too (in rt mode every frame, while the\n"
             "camera or the physics move: each frame reprojected and denoised)\n"
             "(default: on, off for a software GPU, which is slower than the CPU at it)"},

    SECTION("image and quality", 0),
    {"-n", K_INT, .off = F(width), .off2 = F(height), .arg = "size", .text = "width = height in pixels"},
    {"-width", K_INT, .off = F(width), .arg = "px", .text = "set them separately", .def = ""},
    {"-height", K_INT, .join = 1, .off = F(height), .arg = "px", .def = ""},
    {"-s", K_INT, P_V, 1, .off = F(spp), .arg = "samples", .text = "stop refining after this many samples per pixel"},
    {"-s", K_INT, P_R | P_S, 1, .off = F(spp), .arg = "samples", .text = "samples (light paths) per pixel"},
    INT("-d", max_depth, "bounces", "maximum bounces per light path"),
    DBL("-clamp", clamp, "max", "firefly clamp, 0 = off (biased if on)"),
    DBL("-iclamp", iclamp, "max", "indirect clamp: each light a path brings after its first rough\nbounce, at most this much (the sun's disc found by chance:\nfireflies), 0 = off (biased; default: 4 in the viewer, else off)"),
    DBL("-iclampmove", iclamp_move, "max", "view, path tracing on the GPU: the indirect clamp while things move\n(when lower than -iclamp; 0 = as -iclamp; default 1): what moves\nkeeps 12 samples a pixel, and a firefly the denoiser spreads stays\na blotch for as long"),
    ONOFF("-denoise", denoise, "path tracing on the GPU: denoise the image"),
    ONOFF("-wavefront", wavefront, "path tracing on the GPU: small kernels, one step of all paths at a time"),
    INT("-threads", threads, "n", "CPU threads, 0 = all cores"),
    ONOFF("-mis", mis, "multiple importance sampling of the lamp"),
    {"-sampler", K_PICK, .off = F(sampler), .names = SAMPLERS, .arg = "sobol|random",
     .text = "a path's random numbers: Sobol with blue noise, or plain"},
    {"-caustics", K_PICK, P_V, 1, .off = F(caustics_lt), .names = CAUSTICS, .arg = "lt|pt",
     .text = "caustics by light tracing or by path tracing", .def = "; on the CPU: pt"},
    {"-caustics", K_PICK, P_R | P_S, 1, .off = F(caustics_lt), .names = CAUSTICS, .arg = "lt|pt",
     .text = "caustics by light tracing or by path tracing"},
    DBL("-photons", photons, "x", "light paths per camera path for -caustics lt"),
    ONOFF("-photoncache", photon_cache, "view, -caustics lt on the GPU: while moving, the caustics from a\n"
                                        "photon cache instead of traced again each frame"),
    {"-o", K_STR, P_R, 1, .off = F(out), .arg = "file", .text = "output PNG file", .def = "out/cornell_<scene>[_raster].png"},
    {"-o", K_STR, P_V, 1, .off = F(out), .arg = "file", .text = "screenshot prefix (key P)"},
    {"-o", K_STR, P_S, 1, .off = F(out), .arg = "file", .text = "folder for the frames"},

    SECTION("scene", 0),
    {"-scene", K_PICK, .off = F(scene.kind), .names = SCENES,
     .text = "what is in the Cornell box; showcase: a mirror dragon, a diffuse\n"
             "cube, a glass torus, a cloud; sky: no box, five spheres on a floor\n"
             "under the sky"},
    {"-sky", K_STR, .off = F(scene.sky), .arg = "file.hdr|procedural",
     .text = "the sky (a Radiance .hdr map; any scene: the box's open front\nlets it in)", .def = "; -scene sky: procedural"},
    {"-skyscale", K_REAL, .off = F(scene.sky_scale), .arg = "k", .ok = positive, .bad = "-skyscale must be > 0, got",
     .text = "its strength / turned about the vertical", .def = "1"},
    {"-skyrot", K_REAL, .join = 1, .off = F(scene.sky_rot), .arg = "deg"},
    ONOFF("-probe", probe, "rasterizer: light with a probe of the room (path traced)"),
    {"-proberes", K_INT, .off = F(probe_res), .arg = "n", .text = "the probe's width in texels / paths per texel",
     .def = STR(PROBE_RES) ", " STR(PROBE_RES_OPEN) " under a sky"},
    {"-probespp", K_INT, .join = 1, .off = F(probe_spp), .arg = "n"},
    {"-obj", K_STR, .off = F(scene.obj_path), .arg = "file.obj|file.ply", .text = "model for -scene mesh (ASCII .ply)"},
    {"-gltf", K_STR, .off = F(scene.gltf_path), .arg = "file.gltf|file.glb",
     .text = "a glTF 2.0 model instead (its own materials and textures), in\n"
             "the room (-scene mesh, the default with it) or under the sky\n"
             "(-scene sky, in place of the spheres)"},
    PICK("-mat", scene.mesh_mat, MATS, "material of the model"),
    {"-base", K_COLOR, .off = F(scene.mesh_base), .set = F(scene.mesh_set), .setv = MESH_SET_BASE, .arg = "r,g,b",
     .text = "its base colour; and the glTF parameters (docs/PBR.md):", .def = ""},
    {"-metal", K_REAL, .off = F(scene.mesh_metal), .set = F(scene.mesh_set), .setv = MESH_SET_METAL, .arg = "m",
     .text = "metallic 0..1, roughness 0 (smooth) ..1", .def = ""},
    {"-rough", K_REAL, .join = 1, .off = F(scene.mesh_rough), .set = F(scene.mesh_set), .setv = MESH_SET_ROUGH, .arg = "r", .def = ""},
    {"-ior", K_REAL, .off = F(scene.mesh_ior), .set = F(scene.mesh_set), .setv = MESH_SET_IOR, .arg = "x",
     .text = "index of refraction, transmission 0..1", .def = ""},
    {"-transmission", K_REAL, .join = 1, .off = F(scene.mesh_trans), .set = F(scene.mesh_set), .setv = MESH_SET_TRANS,
     .arg = "t", .def = "glass 1"},
    {"-tex", K_STR, .off = F(scene.mesh_tex), .arg = "file.png|name",
     .text = "its base colour texture; names made here: checker, bricks\n(bricks: with its normal and metal-roughness maps too)"},
    {"-mrmap", K_STR, .off = F(scene.mesh_mr), .def = "",
     .text = "its metal-roughness (glTF: G rough, B metal) and\nnormal maps; names made here: bricks_mr, plate_mr, bricks_normal"},
    {"-normalmap", K_STR, .join = 1, .off = F(scene.mesh_normal), .arg = "file.png|name"},
    {"-normalscale", K_REAL, .off = F(scene.mesh_normal_scale), .arg = "k", .ok = positive, .bad = "-normalscale must be > 0, got",
     .text = "the normal map's strength (its tilt times k)", .def = "1"},
    ONOFF("-normalflip", scene.mesh_normal_flip, "the normal map's green points down (DirectX's): turned over"),
    {"-bumpmap", K_STR, .off = F(scene.mesh_bump), .arg = "file.png",
     .text = "a bump (height: white high) map, made into a normal map\n(on top of -normalmap if there is one)", .def = ""},
    {"-bumpscale", K_REAL, .off = F(scene.mesh_bump_scale), .arg = "s", .ok = positive, .bad = "-bumpscale must be > 0, got",
     .text = "how steep: a step from black to white over 1/256 of the\nimage tilts it by atan(s / 2)", .def = "1"},
    {"-roughmap", K_STR, .off = F(scene.mesh_rough_map), .def = "",
     .text = "roughness and metallic maps apart (grey), packed into\nglTF's metal-roughness (in place of -mrmap)"},
    {"-metalmap", K_STR, .join = 1, .off = F(scene.mesh_metal_map), .arg = "file.png"},
    {"-aomap", K_STR, .off = F(scene.mesh_ao), .arg = "file.png", .text = "its ambient occlusion (red: how much light from around reaches)", .def = ""},
    {"-emissivemap", K_STR, .off = F(scene.mesh_emissive), .arg = "file.png", .text = "its glow's colour (times -emissive)", .def = ""},
    {"-emissive", K_REAL, .off = F(scene.mesh_emissive_strength), .arg = "k", .ok = positive, .bad = "-emissive must be > 0, got",
     .text = "how bright the glow is (radiance of white)", .def = "1"},
    {"-uvscale", K_REAL, .off = F(scene.mesh_uv_scale), .arg = "k", .ok = positive, .bad = "-uvscale must be > 0, got",
     .text = "its texture coordinates times k", .def = "1"},
    REAL("-rot", scene.rot_deg, "degrees", "turn the model around the vertical axis"),
    INT("-instances", scene.instances, "n", "n instances of the model (one mesh) in any scene"),

    SECTION("cloud (path tracer; -scene cloud, or -cloud on to add it to any scene)", 0),
    ONOFF("-cloud", use_cloud, "a volumetric cloud floating in the box"),
    REAL("-density", cloud.sigma_t, "sigma", "extinction at full density, per unit"),
    {"-albedo", K_COLOR, .off = F(cloud.albedo), .arg = "a | r,g,b", .text = "fraction scattered (rest absorbed)"},
    REAL("-g", cloud.g, "value", "anisotropy: 0 uniform, > 0 forward"),
    REAL("-detail", cloud.detail, "value", "0 smooth .. 1 wispy, eroded edges"),
    {"-seed", K_U32, .off = F(cloud.seed), .arg = "n", .text = "which cloud"},
    REAL("-time", cloud.time, "seconds", "animation time: the detail drifts with the wind"),
    {"-wind", K_VEC3, .help_only = 1, .off = F(sim.wind), .arg = "x,y,z",
     .text = "(render, view without -sim) the cloud leans downwind, units / s"},
    {"-cloudres", K_INT, .off = F(cloud.resolution), .arg = "n", .text = "voxels along the longest side", .def = "; view -sim on: 96"},
    REAL("-fog", scene.fog, "sigma", "fill the room with a uniform haze, 0 = off"),

    SECTION("camera (default: the classic Cornell view)", 0),
    {"-cam", K_VEC3, .off = F(cam), .set = F(set_cam), .setv = 1, .arg = "x,y,z", .text = "camera position", .def = "278,278,-800"},
    {"-look", K_VEC3, .off = F(look), .set = F(set_look), .setv = 1, .arg = "x,y,z", .text = "point the camera looks at",
     .def = "278,278,0"},
    {"-fov", K_DBL, .off = F(fov), .set = F(set_fov), .setv = 1, .arg = "degrees", .ok = fov_ok,
     .bad = "field of view must be between 0 and 180 degrees, got", .text = "vertical field of view", .def = "40"},

    SECTION("acceleration structure (path tracer)", 0),
    {"-accel", K_PICK, .off = F(accel), .names = ACCELS, .base = -1,
     .text = "auto = bvh from " STR(AUTO_ACCEL_MIN_PRIMS) " primitives up, else none"},
    INT("-octdepth", ap.oct_depth, "n", "octree max depth, 0 = automatic"),
    INT("-bvhleaf", ap.bvh_leaf, "n", "BVH: max primitives per leaf"),
    INT("-bvhbins", ap.bvh_bins, "n", "BVH: SAH buckets per axis"),
    INT("-bvhwidth", ap.bvh_width, "2|4", "BVH: children per node (4: half as deep, 4 boxes per step)"),
    PICK("-bvhsplit", ap.bvh_split, SPLITS, "BVH: how nodes are split"),

    SECTION("real-time viewer", P_V),
    {"-preview", K_INT, P_V, .off = F(preview), .arg = "n", .text = "-backend rt: 1/n resolution while moving, 1 = off",
     .def = "; -sim on: 2"},
    {"-movespp", K_INT, P_V, .off = F(move_spp), .arg = "n", .text = "samples per pixel of each preview frame"},
    {"-speed", K_DBL, P_V, .off = F(speed), .arg = "units", .text = "movement speed per second"},
    {"-autoquit", K_DBL, P_V, .off = F(autoquit), .arg = "seconds", .text = "quit (and save a screenshot) after this time",
     .def = "0 = never"},
    {"-history", K_INT, P_V, .off = F(history), .arg = "n",
     .text = "GPU path tracing while moving: samples per pixel carried from\n"
             "frame to frame (reprojected), 0 = start over on every move"},
    {"-orbit", K_DBL, P_V, .off = F(orbit), .arg = "deg/s",
     .text = "the camera swings by itself, +-25 degrees around the point it\n"
             "looks at, at this speed (benchmarks of the image while moving)"},
    {"-maxfps", K_ONOFF, P_V, .off = F(maxfps),
     .text = "a frame every time round, as many as the machine can; off: only\n"
             "when something changed, the path tracer stopping at -s samples\n"
             "(saves power)"},
    {"-vsync", K_ONOFF, P_V, .off = F(vsync),
     .text = "frames in step with the display; off: shown at once, past the\ndisplay's rate (it may tear)"},
    {"-band", K_DBL, P_V, .off = F(band_ms), .arg = "ms",
     .text = "GPU path tracer: GPU time per frame (0: 85% of the display's\n"
             "refresh period in step with it; else 60% but at least a whole\n"
             "pass, up to 33 ms; bigger: fewer frames, the image cleaner sooner)"},
    {"-stats", K_STR, P_V, .off = F(stats), .arg = "file", .text = "with -autoquit: the frame rate written to file (bin/bench)",
     .def = ""},
    {"-simhz", K_DBL, P_V, .off = F(sim_hz), .arg = "n",
     .text = "with -sim on: physics steps per second at most; frames between\nsteps refine the same image",
     .def = "; 0: a step every frame"},
    {"-simfixed", K_ONOFF, P_V, .off = F(sim_fixed),
     .text = "with -sim on: steps of exactly 1/simhz, as many as the time that\nwent by asks (at most 4 a frame): the same motion at any frame\nrate, the same seed the same run"},
    {"-denoisemove", K_INT, P_V, .off = F(denoise_move), .arg = "n",
     .text = "GPU denoiser while things move: its passes, 1-5 (a footprint of\n5 to 65 pixels), cheaper and less smooth",
     .def = "; 0: all 5, as still"},
    {"-aa", K_PICK, P_V, .off = F(aa), .names = AAS, .arg = "fxaa|off",
     .text = "GPU: anti-aliasing of the rasterized frames (not of the path\ntraced ones, already smooth): FXAA, one more pass"},
};
#define NOPTS ((int)(sizeof(OPTS) / sizeof(OPTS[0])))

// the values options_parse starts from (all but those decided after reading: B5)
static void defaults(Options *o, Program prog){
    memset(o, 0, sizeof(*o));
    o->prog = prog;
    o->backend = prog == PROG_VIEW ? BACKEND_HYBRID : BACKEND_RT;
    o->width = o->height = prog == PROG_VIEW ? 640 : (prog == PROG_SIM ? 320 : 400);
    o->spp = prog == PROG_VIEW ? 4096 : (prog == PROG_SIM ? 32 : 128);
    o->max_depth = 50;
    o->mis = 1;
    o->iclamp = prog == PROG_VIEW ? 4 : 0;
    o->iclamp_move = prog == PROG_VIEW ? 1 : 0;
    o->sampler = 1;
    o->caustics_lt = prog != PROG_SIM;     // (the CPU viewer path traces them whatever this says)
    o->photons = 1;
    o->shading = MODE_PHONG;
    o->raster_clouds = 1;
    o->gpu = gpu_available();              // the GPU whenever the build has one (natively, make web)
    o->gpu_fluid = 1;
    o->gpu_rigid = 1;
    o->gpu_raster = 1;
    o->gpu_trace = -1;                     // decided when the GPU is known
    o->denoise = prog != PROG_RENDER;      // a still image is the unbiased one unless asked
    o->maxfps = 1;
    o->vsync = 1;
    o->sim_hz = 60;
    o->denoise_move = 3;
    o->aa = 1;
    o->march = march_default_params();
    if (prog == PROG_VIEW) o->march.scale = 2;   // real time: half resolution
    o->probe = 1;
    o->probe_res = 0;                      // PROBE_RES in a room, PROBE_RES_OPEN under a sky
    o->probe_spp = PROBE_SPP;
    o->scene.kind = SCENE_BOXES;
    o->scene.obj_path = "models/torus.obj";
    o->scene.mesh_mat = MAT_DIFFUSE;
    o->accel = -1;
    o->ap = accel_default_params();
    o->cloud = cloud_default_params();
    o->preview = 4;
    o->move_spp = 1;
    o->history = 32;
    o->speed = 300;
    o->out = prog == PROG_VIEW ? "out/view" : (prog == PROG_SIM ? "out/sim" : NULL);
    o->sim.on = prog == PROG_SIM;
    o->sim.demo = SIM_DROP;
    o->sim.frames = 120;
    o->sim.fps = 30;
    o->sim.balls = 5;
    o->sim.boxes = 3;
    o->sim.balloons = 3;
    o->sim.bodies = 2000;
    o->sim.props = 3000;
    o->sim.fluid_res = 48;
    o->sim.vorticity = 0.25;
    o->sim.seed = 1;
    o->sim.gif = 1;
}

static int shown(const Opt *e, Program prog){
    return !e->progs || (e->progs & (1 << prog));
}

static int takes(const Opt *e, Program prog){
    return e->kind >= K_INT && !e->help_only && (!e->strict || (e->progs & (1 << prog)));
}

// "on|off", "a|b|c" (a K_PICK's names), or the entry's own words
static void arg_text(const Opt *e, char *buf, size_t n){
    buf[0] = 0;
    if (e->arg) snprintf(buf, n, "%s", e->arg);
    else if (e->kind == K_ONOFF) snprintf(buf, n, "on|off");
    else if (e->kind == K_PICK)
        for (int i = 0; e->names[i]; i++){
            size_t l = strlen(buf);
            snprintf(buf + l, n - l, "%s%s", i ? "|" : "", e->names[i]);
        }
}

// the value of e's field in o, as typed
static void value_text(const Opt *e, const Options *o, char *buf, size_t n){
    const void *p = (const char *)o + e->off;
    Vec3 v;
    switch (e->kind){
    case K_INT: snprintf(buf, n, "%d", *(const int *)p); break;
    case K_U32: snprintf(buf, n, "%u", (unsigned)*(const uint32_t *)p); break;
    case K_DBL: snprintf(buf, n, "%g", *(const double *)p); break;
    case K_REAL: snprintf(buf, n, "%g", (double)*(const real *)p); break;
    case K_STR: snprintf(buf, n, "%s", *(const char *const *)p ? *(const char *const *)p : "none"); break;
    case K_ONOFF: snprintf(buf, n, "%s", *(const int *)p ? "on" : "off"); break;
    case K_PICK: {
        int i = *(const int *)p - e->base, k = 0;
        while (e->names[k] && k < i) k++;
        snprintf(buf, n, "%s", i >= 0 && e->names[k] ? e->names[k] : "?");
        break;
    }
    case K_VEC3: case K_COLOR:
        memcpy(&v, p, sizeof(v));
        if (e->kind == K_COLOR && v.x == v.y && v.y == v.z) snprintf(buf, n, "%g", (double)v.x);
        else snprintf(buf, n, "%g,%g,%g", (double)v.x, (double)v.y, (double)v.z);
        break;
    default: buf[0] = 0;
    }
}

static void default_text(const Opt *e, const Options *d, char *buf, size_t n){
    if (e->def && e->def[0] != ';'){ snprintf(buf, n, "%s", e->def); return; }
    value_text(e, d, buf, n);
    if (e->def){
        size_t l = strlen(buf);
        snprintf(buf + l, n - l, "%s", e->def);
    }
}

#define HELP_TEXT_COL 29   // where the words start
#define HELP_DEF_COL 70    // and the default, where the line leaves room

void options_usage(Program prog, FILE *f){
    Options d;
    defaults(&d, prog);
    fprintf(f, "usage: %s [options]\n", prog == PROG_VIEW ? "view" : (prog == PROG_SIM ? "sim" : "render"));
    for (int i = 0; i < NOPTS; i++){
        const Opt *e = &OPTS[i];
        if (!shown(e, prog) || e->join) continue;
        if (e->kind == K_SECTION || e->kind == K_TEXT){ fprintf(f, "\n%s\n", e->name); continue; }
        // the label and the defaults of this entry and those joined to it
        char label[160] = "", def[160] = "", a[96], v[96];
        for (int j = i; j < NOPTS && (j == i || OPTS[j].join); j++){
            if (!shown(&OPTS[j], prog)) continue;
            arg_text(&OPTS[j], a, sizeof(a));
            size_t l = strlen(label);
            snprintf(label + l, sizeof(label) - l, "%s%s%s%s", j > i ? " / " : "", OPTS[j].name, a[0] ? " " : "", a);
            default_text(&OPTS[j], &d, v, sizeof(v));
            if (v[0]){
                l = strlen(def);
                snprintf(def + l, sizeof(def) - l, "%s%s", def[0] ? " / " : "", v);
            }
        }
        int col = fprintf(f, "  %s", label);
        if (col > HELP_DEF_COL - 30) col = fprintf(f, "\n%*s", HELP_TEXT_COL, "") - 1;   // a long label: the words below
        else col += fprintf(f, "%*s", col < HELP_TEXT_COL - 1 ? HELP_TEXT_COL - col : 2, "");
        for (const char *t = e->text ? e->text : ""; *t; ){
            const char *nl = strchr(t, '\n');
            int len = nl ? (int)(nl - t) : (int)strlen(t);
            col += fprintf(f, "%.*s", len, t);
            if (!nl) break;
            fprintf(f, "\n%*s", HELP_TEXT_COL, "");
            col = HELP_TEXT_COL;
            t = nl + 1;
        }
        if (def[0]) fprintf(f, "%*s(%s)", col < HELP_DEF_COL - 1 ? HELP_DEF_COL - col : 1, "", def);
        fputc('\n', f);
    }
    fprintf(f, "\n  -h, --help                 this text\n");
}

static void fail(Program prog, const char *msg, const char *arg){
    fprintf(stderr, "error: %s '%s'\n\n", msg, arg);
    options_usage(prog, stderr);
    exit(1);
}

static Vec3 parse_color(Program prog, const char *s){
    double x, y, z;
    int n = sscanf(s, "%lf,%lf,%lf", &x, &y, &z);
    if (n == 3) return vec3(x, y, z);
    if (n == 1) return vec3(x, x, x);
    fail(prog, "expected a or r,g,b but got", s);
    return vec3(0, 0, 0);
}

static Vec3 parse_vec3(Program prog, const char *s){
    double x, y, z;
    if (sscanf(s, "%lf,%lf,%lf", &x, &y, &z) != 3) fail(prog, "expected x,y,z but got", s);
    return vec3(x, y, z);
}

// The number after an option, all of it: "12x", "" or "1e" are refused (atoi and atof took
// "12x" as 12, and anything as 0)
static void bad_number(Program prog, const char *opt, const char *v){
    char msg[80];
    snprintf(msg, sizeof(msg), "expected a number after %s, got", opt);
    fail(prog, msg, v);
}

static long long to_integer(Program prog, const char *opt, const char *v, long long lo, long long hi){
    char *end;
    errno = 0;
    long long x = strtoll(v, &end, 10);
    if (end == v || *end || errno || x < lo || x > hi) bad_number(prog, opt, v);
    return x;
}

static double to_real(Program prog, const char *opt, const char *v){
    char *end;
    double x = strtod(v, &end);
    if (end == v || *end || !isfinite(x)) bad_number(prog, opt, v);
    return x;
}

// reads v into e's field(s)
static void apply(const Opt *e, Options *o, Program prog, const char *v){
    void *p = (char *)o + e->off;
    double x = 0;   // (the number checked by e->ok)
    switch (e->kind){
    case K_INT: {
        int n = (int)to_integer(prog, e->name, v, INT_MIN, INT_MAX);
        *(int *)p = n;
        if (e->off2) *(int *)((char *)o + e->off2) = n;
        x = n;
        break;
    }
    case K_U32: *(uint32_t *)p = (uint32_t)to_integer(prog, e->name, v, 0, UINT32_MAX); break;
    case K_DBL: x = *(double *)p = to_real(prog, e->name, v); break;
    case K_REAL: x = *(real *)p = (real)to_real(prog, e->name, v); break;
    case K_STR: *(const char **)p = v; break;
    case K_ONOFF:
        if (!strcmp(v, "on")) *(int *)p = 1;
        else if (!strcmp(v, "off")) *(int *)p = 0;
        else {
            char msg[64];
            snprintf(msg, sizeof(msg), "expected on or off after %s, got", e->name);
            fail(prog, msg, v);
        }
        break;
    case K_PICK: {
        int i = 0;
        while (e->names[i] && strcmp(v, e->names[i])) i++;
        if (!e->names[i]){
            char a[96], msg[160];
            arg_text(e, a, sizeof(a));
            snprintf(msg, sizeof(msg), "expected %s after %s, got", a, e->name);
            fail(prog, msg, v);
        }
        *(int *)p = i + e->base;
        break;
    }
    case K_VEC3: { Vec3 c = parse_vec3(prog, v); memcpy(p, &c, sizeof(c)); break; }
    case K_COLOR: { Vec3 c = parse_color(prog, v); memcpy(p, &c, sizeof(c)); break; }
    default: break;
    }
    if (e->ok && !e->ok(x)) fail(prog, e->bad, v);
    if (e->set) *(int *)((char *)o + e->set) |= e->setv;
}

static const Opt *find(const char *name, Program prog){
    for (int i = 0; i < NOPTS; i++)
        if (takes(&OPTS[i], prog) && !strcmp(name, OPTS[i].name)) return &OPTS[i];
    return NULL;
}

void options_parse(Options *o, Program prog, int argc, char **argv){
    defaults(o, prog);
    // the options given, for the defaults that depend on others (-sim on: -fluidres, ...)
    unsigned char given[NOPTS] = {0};
#define GIVEN(name) given[find(name, prog) - OPTS]

    for (int i = 1; i < argc; i++){
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")){ options_usage(prog, stdout); exit(0); }
        if (a[0] != '-') fail(prog, "unexpected argument", a);
        if (i + 1 >= argc) fail(prog, "missing value after", a);
        const char *v = argv[++i];
        const Opt *e = find(a, prog);
        if (!e) fail(prog, "unknown option", a);
        apply(e, o, prog, v);
        given[e - OPTS] = 1;
    }

    if (o->width <= 0 || o->height <= 0) fail(prog, "image size must be positive, got", "-n/-width/-height");
    if (o->spp <= 0) fail(prog, "samples must be positive", "-s");
    if (o->max_depth <= 0) fail(prog, "bounces must be positive", "-d");
    if (o->clamp < 0) fail(prog, "clamp must be >= 0", "-clamp");
    if (o->iclamp < 0) fail(prog, "iclamp must be >= 0", "-iclamp");
    if (o->iclamp_move < 0) fail(prog, "iclampmove must be >= 0", "-iclampmove");
    if (o->threads < 0) fail(prog, "threads must be >= 0 (0 = all cores)", "-threads");
    if (o->ap.oct_depth < 0) fail(prog, "octree depth must be >= 0 (0 = automatic)", "-octdepth");
    if (o->scene.instances < 0) fail(prog, "instances must be >= 0", "-instances");
    if (o->probe_res != 0 && (o->probe_res < 4 || o->probe_res > 2048)) fail(prog, "probe width must be 4..2048 (0 = automatic)", "-proberes");
    if (o->probe_spp < 1) fail(prog, "probe paths per texel must be >= 1", "-probespp");
    if (o->history < 0) fail(prog, "history must be >= 0", "-history");
    if (o->speed <= 0) fail(prog, "speed must be > 0", "-speed");
    if (o->sim_hz < 0) fail(prog, "physics steps per second must be >= 0 (0 = every frame)", "-simhz");
    if (o->sim_fixed && o->sim_hz <= 0) fail(prog, "-simfixed on needs a step: -simhz above 0", "-simfixed");
    if (o->autoquit < 0) fail(prog, "autoquit must be >= 0 (0 = never)", "-autoquit");
    if (o->band_ms < 0) fail(prog, "band must be >= 0 ms (0 = automatic)", "-band");
    if (o->denoise_move < 0 || o->denoise_move > 5) fail(prog, "denoiser passes while moving must be 0..5", "-denoisemove");
    if (o->sim.bodies < 0 || o->sim.bodies > 4096) fail(prog, "-bodies must be 0..4096 (SIM_MAX_SPAWNS)", "-bodies");
    if (o->ap.bvh_leaf < 1 || o->ap.bvh_bins < 2) fail(prog, "need -bvhleaf >= 1 and -bvhbins >= 2", "");
    if (o->ap.bvh_width != 2 && o->ap.bvh_width != 4) fail(prog, "-bvhwidth is 2 or 4", "");
    if (o->preview < 1) o->preview = 1;
    if (o->move_spp < 1) o->move_spp = 1;
    if (o->cloud.sigma_t < 0) fail(prog, "density must be >= 0", "-density");
    if (o->scene.fog < 0) fail(prog, "fog must be >= 0", "-fog");
    if (o->photons < 0) fail(prog, "photons must be >= 0", "-photons");
    if (o->march.max_steps < 4 || o->march.max_steps > 4096) fail(prog, "march steps must be 4..4096", "-march");
    if (o->march.light_res < 2 || o->march.light_res > 512) fail(prog, "light grid resolution must be 2..512", "-lightres");
    if (o->march.scale != 1 && o->march.scale != 2) fail(prog, "expected 1 or 2 after", "-marchscale");
    if (o->cloud.g <= -1 || o->cloud.g >= 1) fail(prog, "g must be between -1 and 1 (exclusive)", "-g");
    if (o->cloud.resolution < 4 || o->cloud.resolution > 1024) fail(prog, "cloud resolution must be 4..1024", "-cloudres");
    if (o->sim.on){
        if (o->sim.frames < 1 || o->sim.fps <= 0) fail(prog, "need -frames >= 1 and -fps > 0", "");
        if (o->sim.fluid_res < 8 || o->sim.fluid_res > 256) fail(prog, "fluid resolution must be 8..256", "-fluidres");
        if (o->sim.balls < 0 || o->sim.boxes < 0 || o->sim.balloons < 0 || o->sim.meshes < 0 || o->sim.props < 0)
            fail(prog, "counts must be >= 0", "-balls/-boxes/-balloons/-meshes/-props");
        // view -sim: hybrid (raster + raymarched cloud while it runs); live: coarser air and
        // cloud, so that a physics step fits in a frame, and 1/2 resolution while moving
        if (prog == PROG_VIEW && !GIVEN("-fluidres")) o->sim.fluid_res = 32;
        if (prog == PROG_VIEW && !GIVEN("-cloudres")) o->cloud.resolution = 96;
        if (prog == PROG_VIEW && !GIVEN("-preview")) o->preview = 2;
    }
    // a glTF model and no scene: the room with the model (SCENE_MESH)
    if (o->scene.gltf_path && !GIVEN("-scene")) o->scene.kind = SCENE_MESH;
#undef GIVEN
    if (!o->out){   // out/cornell_<scene>[_raster].png, out/sky[_raster].png
        static char out[64];
        snprintf(out, sizeof(out), "out/%s%s%s.png", o->scene.kind == SCENE_SKY ? "" : "cornell_", SCENES[o->scene.kind],
                 o->backend == BACKEND_RASTER ? "_raster" : "");
        o->out = out;
    }
}

AccelKind options_resolve_accel(const Options *o, const Scene *s){
    if (o->accel >= 0) return (AccelKind)o->accel;
    return scene_num_shapes(s) >= AUTO_ACCEL_MIN_PRIMS ? ACCEL_BVH : ACCEL_NONE;
}

int options_setup_scene(const Options *o, Scene *s){
#ifdef _OPENMP
    if (o->threads > 0) omp_set_num_threads(o->threads);
#endif
    SceneOptions so = o->scene;
    // -wind outside the physics: the cloud leans with it (its top pushed downwind, 0.25 of its
    // height per 100 units / s), its detail streaked along it
    CloudParams cp = o->cloud;
    if (!o->sim.on && o->prog != PROG_SIM) cp.lean = vec3(o->sim.wind.x / 400, 0, o->sim.wind.z / 400);
    so.cloud = (o->use_cloud || so.kind == SCENE_CLOUD || so.kind == SCENE_SHOWCASE) ? &cp : NULL;
    if (scene_cornell_box(s, &so) != 0) return 1;
    if (o->scene.kind == SCENE_SHOWCASE) ;   // (it says what it loaded)
    else if (o->scene.gltf_path && (o->scene.kind == SCENE_MESH || o->scene.kind == SCENE_SKY)){
        int nt = 0;
        for (int i = 0; i < s->nmeshes; i++) nt += s->meshes[i].ntris;
        printf("loaded %s: %d meshes, %d triangles, %d instances, %d textures\n", o->scene.gltf_path, s->nmeshes, nt, s->ninsts, s->ntexmaps);
    }
    else if (s->ninsts > 0 && s->nmeshes > 0) printf("loaded %s: %d triangles, %d instances of it\n", o->scene.obj_path, s->meshes[0].ntris, s->ninsts);
    else if (o->scene.kind == SCENE_MESH) printf("loaded %s: %d triangles\n", o->scene.obj_path, s->ntris);
    for (int i = 0; i < s->nvolumes; i++){
        const Volume *v = s->volumes[i];
        printf("medium %d: %dx%dx%d voxels, extinction %g, albedo %g,%g,%g, g %g%s\n", i, v->nx, v->ny, v->nz,
               (double)v->sigma_t, (double)v->albedo.x, (double)v->albedo.y, (double)v->albedo.z, (double)v->g,
               o->backend != BACKEND_RT && !o->raster_clouds ? " (not drawn by the rasterizer: -rastercloud off)" : "");
    }

    if (o->set_cam) s->cam_from = o->cam;
    if (o->set_look) s->cam_at = o->look;
    if (o->set_fov) s->cam_vfov_deg = o->fov;

    if (s->env) printf("sky: %d x %d%s, mean radiance %.3g\n", s->env->w, s->env->h,
                       o->scene.sky ? "" : " (procedural)", (double)(s->env->scale * (s->env->mean.x + s->env->mean.y + s->env->mean.z) / 3));
    int probe = o->probe && (s->light >= 0 || s->env) && (o->prog == PROG_VIEW || o->backend == BACKEND_RASTER);
    if (o->prog == PROG_RENDER && o->backend == BACKEND_RASTER && !probe) return 0;   // no rays, no accel
    scene_set_accel(s, options_resolve_accel(o, s), &o->ap);
    char buf[256];
    scene_accel_stats(s, buf, sizeof(buf));
    printf("%s\n", buf);
    options_bake_probe(o, s);
    return 0;
}

void options_bake_probe(const Options *o, Scene *s){
    if (!o->probe || (s->light < 0 && !s->env) || !(o->prog == PROG_VIEW || o->backend == BACKEND_RASTER)) return;
    double t0 = clock_s();
    Env *p = probe_bake_scene(s, o->probe_res, o->probe_spp);
    if (!p) return;
    env_free(s->probe);
    s->probe = p;
    irr_grid_free(s->irr);
    s->irr = probe_grid_bake_scene(s, PROBE_GRID_PATHS);
    printf("light probe: %d x %d, %d paths per texel, and %d x %d x %d for the diffuse light, %.0f ms (-probe off: none)\n",
           p->w, p->h, o->probe_spp, s->irr ? s->irr->n[0] : 0, s->irr ? s->irr->n[1] : 0, s->irr ? s->irr->n[2] : 0, 1000 * (clock_s() - t0));
}

// " -opt path", quoted when the path has a space
static void put_path(FILE *f, const char *opt, const char *path){
    fprintf(f, strchr(path, ' ') ? " %s \"%s\"" : " %s %s", opt, path);
}

void options_print_command(const Options *o, Vec3 cam, Vec3 look, double fov, FILE *f){
    fprintf(f, "bin/render -scene %s", SCENES[o->scene.kind]);
    if (o->scene.instances > 0) fprintf(f, " -instances %d", o->scene.instances);
    if (o->scene.gltf_path && (o->scene.kind == SCENE_MESH || o->scene.kind == SCENE_SKY)) put_path(f, "-gltf", o->scene.gltf_path);
    else if (o->scene.kind == SCENE_MESH || o->scene.instances > 0){
        put_path(f, "-obj", o->scene.obj_path);
        if (o->scene.mesh_mat != MAT_DIFFUSE && (int)o->scene.mesh_mat < 3) fprintf(f, " -mat %s", MATS[o->scene.mesh_mat]);
        const SceneOptions *so = &o->scene;
        if (so->mesh_set & MESH_SET_BASE) fprintf(f, " -base %g,%g,%g", (double)so->mesh_base.x, (double)so->mesh_base.y, (double)so->mesh_base.z);
        if (so->mesh_set & MESH_SET_METAL) fprintf(f, " -metal %g", (double)so->mesh_metal);
        if (so->mesh_set & MESH_SET_ROUGH) fprintf(f, " -rough %g", (double)so->mesh_rough);
        if (so->mesh_set & MESH_SET_IOR) fprintf(f, " -ior %g", (double)so->mesh_ior);
        if (so->mesh_set & MESH_SET_TRANS) fprintf(f, " -transmission %g", (double)so->mesh_trans);
        if (o->scene.rot_deg != 0) fprintf(f, " -rot %g", (double)o->scene.rot_deg);
        if (so->mesh_tex) put_path(f, "-tex", so->mesh_tex);
        if (so->mesh_mr) put_path(f, "-mrmap", so->mesh_mr);
        if (so->mesh_normal) put_path(f, "-normalmap", so->mesh_normal);
        if (so->mesh_normal_scale > 0) fprintf(f, " -normalscale %g", (double)so->mesh_normal_scale);
        if (so->mesh_normal_flip) fprintf(f, " -normalflip on");
        if (so->mesh_bump) put_path(f, "-bumpmap", so->mesh_bump);
        if (so->mesh_bump_scale > 0) fprintf(f, " -bumpscale %g", (double)so->mesh_bump_scale);
        if (so->mesh_rough_map) put_path(f, "-roughmap", so->mesh_rough_map);
        if (so->mesh_metal_map) put_path(f, "-metalmap", so->mesh_metal_map);
        if (so->mesh_ao) put_path(f, "-aomap", so->mesh_ao);
        if (so->mesh_emissive) put_path(f, "-emissivemap", so->mesh_emissive);
        if (so->mesh_emissive_strength > 0) fprintf(f, " -emissive %g", (double)so->mesh_emissive_strength);
        if (so->mesh_uv_scale > 0) fprintf(f, " -uvscale %g", (double)so->mesh_uv_scale);
    }
    if (o->use_cloud || o->scene.kind == SCENE_CLOUD || o->scene.kind == SCENE_SHOWCASE){
        const CloudParams *c = &o->cloud, d = cloud_default_params();
        if (o->scene.kind != SCENE_CLOUD && o->scene.kind != SCENE_SHOWCASE) fprintf(f, " -cloud on");
        if (o->sim.wind.x != 0 || o->sim.wind.z != 0) fprintf(f, " -wind %g,%g,%g", (double)o->sim.wind.x, (double)o->sim.wind.y, (double)o->sim.wind.z);
        if (c->sigma_t != d.sigma_t) fprintf(f, " -density %g", (double)c->sigma_t);
        if (c->albedo.x != d.albedo.x || c->albedo.y != d.albedo.y || c->albedo.z != d.albedo.z)
            fprintf(f, " -albedo %g,%g,%g", (double)c->albedo.x, (double)c->albedo.y, (double)c->albedo.z);
        if (c->g != d.g) fprintf(f, " -g %g", (double)c->g);
        if (c->detail != d.detail) fprintf(f, " -detail %g", (double)c->detail);
        if (c->seed != d.seed) fprintf(f, " -seed %u", (unsigned)c->seed);
        if (c->time != d.time) fprintf(f, " -time %g", (double)c->time);
        if (c->resolution != d.resolution) fprintf(f, " -cloudres %d", c->resolution);
    }
    if (o->scene.fog > 0) fprintf(f, " -fog %g", (double)o->scene.fog);
    if (o->scene.sky) put_path(f, "-sky", o->scene.sky);
    if (o->scene.sky_scale > 0 && o->scene.sky_scale != 1) fprintf(f, " -skyscale %g", (double)o->scene.sky_scale);
    if (o->scene.sky_rot != 0) fprintf(f, " -skyrot %g", (double)o->scene.sky_rot);
    fprintf(f, " -cam %.1f,%.1f,%.1f -look %.1f,%.1f,%.1f -fov %g",
            (double)cam.x, (double)cam.y, (double)cam.z, (double)look.x, (double)look.y, (double)look.z, fov);
    if (o->width == o->height) fprintf(f, " -n %d", o->width);
    else fprintf(f, " -width %d -height %d", o->width, o->height);
    // the samples asked for; the viewer's default (4096: refine until stopped) as 512, a render
    fprintf(f, " -s %d", o->prog == PROG_VIEW && o->spp == 4096 ? 512 : o->spp);
    if (o->max_depth != 50) fprintf(f, " -d %d", o->max_depth);
    if (!o->mis) fprintf(f, " -mis off");
    if (!o->sampler) fprintf(f, " -sampler random");
    if (o->clamp > 0) fprintf(f, " -clamp %g", o->clamp);
    if (o->iclamp > 0) fprintf(f, " -iclamp %g", o->iclamp);
    if (!o->caustics_lt) fprintf(f, " -caustics pt");
    if (o->photons != 1) fprintf(f, " -photons %g", (double)o->photons);
    if (o->accel >= 0) fprintf(f, " -accel %s", ACCELS[o->accel + 1]);
    const AccelParams ad = accel_default_params();
    if (o->ap.oct_depth != ad.oct_depth) fprintf(f, " -octdepth %d", o->ap.oct_depth);
    if (o->ap.bvh_leaf != ad.bvh_leaf) fprintf(f, " -bvhleaf %d", o->ap.bvh_leaf);
    if (o->ap.bvh_bins != ad.bvh_bins) fprintf(f, " -bvhbins %d", o->ap.bvh_bins);
    if (o->ap.bvh_width != ad.bvh_width) fprintf(f, " -bvhwidth %d", o->ap.bvh_width);
    if (o->ap.bvh_split != ad.bvh_split) fprintf(f, " -bvhsplit %s", SPLITS[o->ap.bvh_split]);
    fprintf(f, " -o out/from_viewer.png\n");
}
