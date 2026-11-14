#include "app/options.h"
#include "gpu/gpu.h"

#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

const char *backend_name(Backend b){
    return b == BACKEND_RT ? "rt" : (b == BACKEND_RASTER ? "raster" : "hybrid");
}

void options_usage(Program prog, FILE *f){
    int view = prog == PROG_VIEW, sim = prog == PROG_SIM;
    fprintf(f, "usage: %s [options]\n\n", view ? "view" : (sim ? "sim" : "render"));
    if (sim) fprintf(f,
        "physics simulation: renders every frame with the path tracer into <-o>/frame_NNNN.png\n\n"
        "demo\n"
        "  -demo drop|blast           drop: balls and boxes fall through the cloud, balloons rise\n"
        "                             blast: an explosion in the cloud at 1 s     (drop)\n"
        "  -frames n                  frames to render                          (120)\n"
        "  -fps n                     frames per second                         (30)\n"
        "  -gif on|off                also write <-o>.gif (uv, or python3 + Pillow) (on)\n");
    if (sim || view) fprintf(f,
        "%s"
        "  -balls n / -boxes n / -balloons n   how many of each                (5 / 3 / 3)\n"
        "  -meshes n                  n copies of the model (-obj) fall too, as convex hulls (0)\n"
        "  -collider on|off           the model stands in the room, its triangles collide (off)\n"
        "  -wind x,y,z                the air drifts towards this velocity, units / s (0,0,0)\n"
        "                             (the room is closed but for its front: a wind mostly swirls)\n"
        "  -fluidres n                air cells along the room's side           (48; view: 32)\n"
        "  -buoyancy a                the cloud rises: acceleration per density (0)\n"
        "  -vorticity e               keeps the swirls alive, 0 = off           (0.25)\n"
        "  -physseed n                where things start                        (1)\n\n",
        view ? "physics (a live simulation in the window)\n"
               "  -sim on|off                run the physics demo                       (off)\n"
               "  -demo drop|blast           which one                                  (drop)\n"
             : "");
    if (view) fprintf(f,
        "backend\n"
        "  -backend hybrid|rt|raster  hybrid = rasterizer while moving, path tracer when still (hybrid)\n"
        "  -shading phong|gouraud|flat|wireframe   rasterizer shading       (phong)\n"
        "  -rastercloud on|off        rasterizer: draw clouds and fog by raymarching (on)\n"
        "  -march n / -lightres n     raymarching: max steps per ray / light grid size (128 / 48)\n"
        "  -marchscale 1|2            raymarching at full or half resolution   (%s)\n"
        "  -gpu on|off|offscreen      render (rasterizer, path tracer, clouds) and simulate the air on the GPU\n"
        "                             (WebGPU; on when the build has it: make web, or make deps natively);\n"
        "                             offscreen: the GPU draws into a texture read back to the canvas\n"
        "  -gpufluid on|off           with -sim on and the GPU: the air and the cloud are simulated there too\n"
        "  -gpurigid on|off           with -sim on and the GPU: the balls and boxes are simulated there too\n"
        "  -gpuraster on|off          with the GPU: the rasterizer's triangles are drawn there too\n"
        "  -gputrace on|off           with the GPU: it path traces too (in rt mode every frame, while the\n"
        "                             camera or the physics move: each frame reprojected and denoised)\n"
        "                             (default: on, off for a software GPU, which is slower than the CPU at it)\n", view ? "2" : "1");
    else if (sim) fprintf(f,
        "backend\n"
        "  -backend rt|raster         path tracer, or rasterizer (raymarched cloud: fast) (rt)\n"
        "  -shading phong|gouraud|flat|wireframe   rasterizer shading       (phong)\n"
        "  -rastercloud on|off        rasterizer: draw the cloud by raymarching (on)\n"
        "  -march n / -lightres n     raymarching: max steps per ray / light grid size (128 / 48)\n"
        "  -marchscale 1|2            raymarching at full or half resolution   (%s)\n", view ? "2" : "1");
    else fprintf(f,
        "backend\n"
        "  -backend rt|raster         path tracer or rasterizer                (rt)\n"
        "  -shading phong|gouraud|flat|wireframe   rasterizer shading       (phong)\n"
        "  -rastercloud on|off        rasterizer: draw clouds and fog by raymarching (on)\n"
        "  -march n / -lightres n     raymarching: max steps per ray / light grid size (128 / 48)\n"
        "  -marchscale 1|2            raymarching at full or half resolution   (%s)\n", view ? "2" : "1");
    fprintf(f,
        "\n"
        "image and quality\n"
        "  -n size                    width = height in pixels                 (%s)\n"
        "  -width px / -height px     set them separately\n"
        "  -s samples                 %s (%s)\n"
        "  -d bounces                 maximum bounces per light path           (50)\n"
        "  -clamp max                 firefly clamp, 0 = off (biased if on)    (0)\n"
        "  -denoise on|off            path tracing on the GPU: denoise the image (%s)\n"
        "  -wavefront on|off          path tracing on the GPU: small kernels, one step of all paths at a time (off)\n"
        "  -threads n                 CPU threads, 0 = all cores               (0)\n"
        "  -mis on|off                multiple importance sampling of the lamp (on)\n"
        "  -sampler sobol|random      a path's random numbers: Sobol with blue noise, or plain (sobol)\n"
        "  -caustics lt|pt            caustics by light tracing or by path tracing (%s)\n"
        "  -photons x                 light paths per camera path for -caustics lt (1)\n"
        "  -o file                    %s (%s)\n"
        "\n"
        "scene\n"
        "  -scene boxes|spheres|mesh|cloud|empty|showcase  what is in the Cornell box (boxes);\n"
        "                             showcase: a mirror dragon, a diffuse cube, a glass torus, a cloud\n"
        "  -obj file.obj|file.ply     model for -scene mesh (ASCII .ply)       (models/torus.obj)\n"
        "  -mat diffuse|mirror|glass  material of the model                    (diffuse)\n"
        "  -base r,g,b                its base colour; and the glTF parameters (docs/PBR.md):\n"
        "  -metal m, -rough r         metallic 0..1, roughness 0 (smooth) ..1\n"
        "  -ior x, -transmission t    index of refraction, transmission 0..1 (glass 1)\n"
        "  -rot degrees               turn the model around the vertical axis  (0)\n"
        "  -instances n               n instances of the model (one mesh) in any scene (0)\n"
        "\n"
        "cloud (path tracer; -scene cloud, or -cloud on to add it to any scene)\n"
        "  -cloud on|off              a volumetric cloud floating in the box   (off)\n"
        "  -density sigma             extinction at full density, per unit     (0.05)\n"
        "  -albedo a | r,g,b          fraction scattered (rest absorbed)       (0.99)\n"
        "  -g value                   anisotropy: 0 uniform, > 0 forward       (0.6)\n"
        "  -detail value              0 smooth .. 1 wispy, eroded edges        (0.6)\n"
        "  -seed n                    which cloud                              (1)\n"
        "  -time seconds              animation time: the detail drifts with the wind (0)\n"
        "  -wind x,y,z                (render, view without -sim) the cloud leans downwind, units / s (0,0,0)\n"
        "  -cloudres n                voxels along the longest side            (128)\n"
        "  -fog sigma                 fill the room with a uniform haze, 0 = off (0)\n"
        "\n"
        "camera (default: the classic Cornell view)\n"
        "  -cam x,y,z                 camera position                          (278,278,-800)\n"
        "  -look x,y,z                point the camera looks at                (278,278,0)\n"
        "  -fov degrees               vertical field of view                   (40)\n"
        "\n"
        "acceleration structure (path tracer)\n"
        "  -accel auto|none|octree|bvh  auto = bvh from %d primitives up, else none   (auto)\n"
        "  -octdepth n                octree max depth, 0 = automatic          (0)\n"
        "  -bvhleaf n                 BVH: max primitives per leaf             (4)\n"
        "  -bvhbins n                 BVH: SAH buckets per axis                (16)\n"
        "  -bvhwidth 2|4              BVH: children per node (4: half as deep, 4 boxes per step) (4)\n"
        "  -bvhsplit sah|middle       BVH: how nodes are split                 (sah)\n",
        view ? "640" : (sim ? "320" : "400"),
        view ? "stop refining after this many samples per pixel" : "samples (light paths) per pixel  ",
        view ? "4096" : (sim ? "32" : "128"),
        prog == PROG_RENDER ? "off" : "on",
        view ? "lt; on the CPU: pt" : (sim ? "pt" : "lt"),
        view ? "screenshot prefix (key P)  " : (sim ? "folder for the frames      " : "output PNG file            "),
        view ? "out/view" : (sim ? "out/sim" : "out/cornell_<scene>[_raster].png"),
        AUTO_ACCEL_MIN_PRIMS);
    if (view) fprintf(f,
        "\n"
        "real-time viewer\n"
        "  -preview n                 -backend rt: 1/n resolution while moving, 1 = off (4)\n"
        "  -movespp n                 samples per pixel of each preview frame  (1)\n"
        "  -speed units               movement speed per second                (300)\n"
        "  -autoquit seconds          quit (and save a screenshot) after this time (0 = never)\n"
        "  -history n                 GPU path tracing while moving: samples per pixel carried from\n"
        "                             frame to frame (reprojected), 0 = start over on every move (32)\n"
        "  -orbit deg/s               the camera swings by itself, +-25 degrees around the point it\n"
        "                             looks at, at this speed (benchmarks of the image while moving) (0)\n");
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

void options_parse(Options *o, Program prog, int argc, char **argv){
    memset(o, 0, sizeof(*o));
    o->prog = prog;
    o->backend = prog == PROG_VIEW ? BACKEND_HYBRID : BACKEND_RT;
    o->width = o->height = prog == PROG_VIEW ? 640 : (prog == PROG_SIM ? 320 : 400);
    o->spp = prog == PROG_VIEW ? 4096 : (prog == PROG_SIM ? 32 : 128);
    o->max_depth = 50;
    o->mis = 1;
    o->sampler = 1;
    o->caustics_lt = prog != PROG_SIM;     // (the CPU viewer path traces them whatever this says)
    o->photons = 1;
    o->shading = MODE_PHONG;
    o->raster_clouds = 1;
    o->gpu = gpu_available();              // the GPU whenever the build has one (make deps, make web)
    o->gpu_fluid = 1;
    o->gpu_rigid = 1;
    o->gpu_raster = 1;
    o->gpu_trace = -1;                     // decided when the GPU is known
    o->denoise = prog != PROG_RENDER;      // a still image is the unbiased one unless asked
    o->march = march_default_params();
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
    o->sim.fluid_res = 48;
    o->sim.vorticity = 0.25;
    o->sim.seed = 1;
    o->sim.gif = 1;
    int backend_set = 0, march_scale_set = 0, fluidres_set = 0, cloudres_set = 0;

    for (int i = 1; i < argc; i++){
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")){ options_usage(prog, stdout); exit(0); }
        if (a[0] != '-') fail(prog, "unexpected argument", a);
        if (i + 1 >= argc) fail(prog, "missing value after", a);
        const char *v = argv[++i];

        if      (!strcmp(a, "-n"))       o->width = o->height = atoi(v);
        else if (!strcmp(a, "-width"))   o->width = atoi(v);
        else if (!strcmp(a, "-height"))  o->height = atoi(v);
        else if (!strcmp(a, "-s"))       o->spp = atoi(v);
        else if (!strcmp(a, "-d"))       o->max_depth = atoi(v);
        else if (!strcmp(a, "-clamp"))   o->clamp = atof(v);
        else if (!strcmp(a, "-threads")) o->threads = atoi(v);
        else if (!strcmp(a, "-o"))       o->out = v;
        else if (!strcmp(a, "-obj"))     o->scene.obj_path = v;
        else if (!strcmp(a, "-rot"))     o->scene.rot_deg = atof(v);
        else if (!strcmp(a, "-base"))    { o->scene.mesh_base = parse_color(prog, v); o->scene.mesh_set |= MESH_SET_BASE; }
        else if (!strcmp(a, "-metal"))   { o->scene.mesh_metal = atof(v); o->scene.mesh_set |= MESH_SET_METAL; }
        else if (!strcmp(a, "-rough"))   { o->scene.mesh_rough = atof(v); o->scene.mesh_set |= MESH_SET_ROUGH; }
        else if (!strcmp(a, "-ior"))     { o->scene.mesh_ior = atof(v); o->scene.mesh_set |= MESH_SET_IOR; }
        else if (!strcmp(a, "-transmission")) { o->scene.mesh_trans = atof(v); o->scene.mesh_set |= MESH_SET_TRANS; }
        else if (!strcmp(a, "-cam"))     { o->cam = parse_vec3(prog, v); o->set_cam = 1; }
        else if (!strcmp(a, "-look"))    { o->look = parse_vec3(prog, v); o->set_look = 1; }
        else if (!strcmp(a, "-fov"))     { o->fov = atof(v); o->set_fov = 1; }
        else if (!strcmp(a, "-octdepth")) o->ap.oct_depth = atoi(v);
        else if (!strcmp(a, "-bvhleaf")) o->ap.bvh_leaf = atoi(v);
        else if (!strcmp(a, "-bvhbins")) o->ap.bvh_bins = atoi(v);
        else if (!strcmp(a, "-bvhwidth")) o->ap.bvh_width = atoi(v);
        else if (!strcmp(a, "-preview")) o->preview = atoi(v);
        else if (!strcmp(a, "-movespp")) o->move_spp = atoi(v);
        else if (!strcmp(a, "-speed"))   o->speed = atof(v);
        else if (!strcmp(a, "-autoquit")) o->autoquit = atof(v);
        else if (!strcmp(a, "-orbit")) o->orbit = atof(v);
        else if (!strcmp(a, "-history")) o->history = atoi(v);
        else if (!strcmp(a, "-instances")) o->scene.instances = atoi(v);
        else if (!strcmp(a, "-photoncache")){
            if (!strcmp(v, "on")) o->photon_cache = 1;
            else if (!strcmp(v, "off")) o->photon_cache = 0;
            else fail(prog, "expected on or off after -photoncache, got", v);
        }
        else if (!strcmp(a, "-density")) o->cloud.sigma_t = atof(v);
        else if (!strcmp(a, "-albedo"))  o->cloud.albedo = parse_color(prog, v);
        else if (!strcmp(a, "-g"))       o->cloud.g = atof(v);
        else if (!strcmp(a, "-detail"))  o->cloud.detail = atof(v);
        else if (!strcmp(a, "-seed"))    o->cloud.seed = (uint32_t)strtoul(v, NULL, 10);
        else if (!strcmp(a, "-time"))    o->cloud.time = atof(v);
        else if (!strcmp(a, "-cloudres")){ o->cloud.resolution = atoi(v); cloudres_set = 1; }
        else if (!strcmp(a, "-fog"))     o->scene.fog = atof(v);
        else if (!strcmp(a, "-photons")) o->photons = atof(v);
        else if (!strcmp(a, "-march"))   o->march.max_steps = atoi(v);
        else if (!strcmp(a, "-lightres")) o->march.light_res = atoi(v);
        else if (!strcmp(a, "-marchscale")){ o->march.scale = atoi(v); march_scale_set = 1; }
        else if (!strcmp(a, "-gpu")){
            if (!strcmp(v, "on")) o->gpu = 1;
            else if (!strcmp(v, "off")) o->gpu = 0;
            else if (!strcmp(v, "offscreen")) o->gpu = 2;
            else fail(prog, "expected on, off or offscreen after -gpu, got", v);
        }
        else if (!strcmp(a, "-gpurigid")){
            if (!strcmp(v, "on")) o->gpu_rigid = 1;
            else if (!strcmp(v, "off")) o->gpu_rigid = 0;
            else fail(prog, "expected on or off after -gpurigid, got", v);
        }
        else if (!strcmp(a, "-gpufluid")){
            if (!strcmp(v, "on")) o->gpu_fluid = 1;
            else if (!strcmp(v, "off")) o->gpu_fluid = 0;
            else fail(prog, "expected on or off after -gpufluid, got", v);
        }
        else if (!strcmp(a, "-gpuraster")){
            if (!strcmp(v, "on")) o->gpu_raster = 1;
            else if (!strcmp(v, "off")) o->gpu_raster = 0;
            else fail(prog, "expected on or off after -gpuraster, got", v);
        }
        else if (!strcmp(a, "-gputrace")){
            if (!strcmp(v, "on")) o->gpu_trace = 1;
            else if (!strcmp(v, "off")) o->gpu_trace = 0;
            else fail(prog, "expected on or off after -gputrace, got", v);
        }
        else if (!strcmp(a, "-wavefront")){
            if (!strcmp(v, "on")) o->wavefront = 1;
            else if (!strcmp(v, "off")) o->wavefront = 0;
            else fail(prog, "expected on or off after -wavefront, got", v);
        }
        else if (!strcmp(a, "-denoise")){
            if (!strcmp(v, "on")) o->denoise = 1;
            else if (!strcmp(v, "off")) o->denoise = 0;
            else fail(prog, "expected on or off after -denoise, got", v);
        }
        else if (!strcmp(a, "-rastercloud")){
            if (!strcmp(v, "on")) o->raster_clouds = 1;
            else if (!strcmp(v, "off")) o->raster_clouds = 0;
            else fail(prog, "expected on or off after -rastercloud, got", v);
        }
        else if (!strcmp(a, "-frames"))  o->sim.frames = atoi(v);
        else if (!strcmp(a, "-fps"))     o->sim.fps = atof(v);
        else if (!strcmp(a, "-balls"))   o->sim.balls = atoi(v);
        else if (!strcmp(a, "-boxes"))   o->sim.boxes = atoi(v);
        else if (!strcmp(a, "-balloons")) o->sim.balloons = atoi(v);
        else if (!strcmp(a, "-wind"))    o->sim.wind = parse_vec3(prog, v);
        else if (!strcmp(a, "-fluidres")){ o->sim.fluid_res = atoi(v); fluidres_set = 1; }
        else if (!strcmp(a, "-buoyancy")) o->sim.buoyancy = atof(v);
        else if (!strcmp(a, "-vorticity")) o->sim.vorticity = atof(v);
        else if (!strcmp(a, "-physseed")) o->sim.seed = atoi(v);
        else if (!strcmp(a, "-meshes"))  o->sim.meshes = atoi(v);
        else if (!strcmp(a, "-collider")){
            if (!strcmp(v, "on")) o->sim.collider = 1;
            else if (!strcmp(v, "off")) o->sim.collider = 0;
            else fail(prog, "-collider takes on or off", v);
        }
        else if (!strcmp(a, "-sim") && prog == PROG_VIEW){
            if (!strcmp(v, "on")) o->sim.on = 1;
            else if (!strcmp(v, "off")) o->sim.on = 0;
            else fail(prog, "expected on or off after -sim, got", v);
        } else if (!strcmp(a, "-gif") && prog == PROG_SIM){
            if (!strcmp(v, "on")) o->sim.gif = 1;
            else if (!strcmp(v, "off")) o->sim.gif = 0;
            else fail(prog, "expected on or off after -gif, got", v);
        } else if (!strcmp(a, "-demo") && prog != PROG_RENDER){
            if (!strcmp(v, "drop")) o->sim.demo = SIM_DROP;
            else if (!strcmp(v, "blast")) o->sim.demo = SIM_BLAST;
            else fail(prog, "unknown demo", v);
        }
        else if (!strcmp(a, "-sampler")){
            if (!strcmp(v, "sobol")) o->sampler = 1;
            else if (!strcmp(v, "random")) o->sampler = 0;
            else fail(prog, "expected sobol or random after -sampler, got", v);
        }
        else if (!strcmp(a, "-mis")){
            if (!strcmp(v, "on")) o->mis = 1;
            else if (!strcmp(v, "off")) o->mis = 0;
            else fail(prog, "expected on or off after -mis, got", v);
        } else if (!strcmp(a, "-caustics")){
            if (!strcmp(v, "lt")) o->caustics_lt = 1;
            else if (!strcmp(v, "pt")) o->caustics_lt = 0;
            else fail(prog, "expected lt or pt after -caustics, got", v);
        }
        else if (!strcmp(a, "-cloud")){
            if (!strcmp(v, "on")) o->use_cloud = 1;
            else if (!strcmp(v, "off")) o->use_cloud = 0;
            else fail(prog, "expected on or off after -cloud, got", v);
        }
        else if (!strcmp(a, "-backend")){
            backend_set = 1;
            if (!strcmp(v, "rt")) o->backend = BACKEND_RT;
            else if (!strcmp(v, "raster")) o->backend = BACKEND_RASTER;
            else if (!strcmp(v, "hybrid") && prog == PROG_VIEW) o->backend = BACKEND_HYBRID;
            else fail(prog, "unknown backend", v);
        } else if (!strcmp(a, "-shading")){
            int m = render_mode_from_name(v);
            if (m < 0 || m == MODE_TEXTURED) fail(prog, "unknown shading", v);
            o->shading = (RenderMode)m;
        } else if (!strcmp(a, "-scene")){
            if (!strcmp(v, "boxes")) o->scene.kind = SCENE_BOXES;
            else if (!strcmp(v, "spheres")) o->scene.kind = SCENE_SPHERES;
            else if (!strcmp(v, "mesh")) o->scene.kind = SCENE_MESH;
            else if (!strcmp(v, "cloud")) o->scene.kind = SCENE_CLOUD;
            else if (!strcmp(v, "empty")) o->scene.kind = SCENE_EMPTY;
            else if (!strcmp(v, "showcase")) o->scene.kind = SCENE_SHOWCASE;
            else fail(prog, "unknown scene", v);
        } else if (!strcmp(a, "-mat")){
            if (!strcmp(v, "diffuse")) o->scene.mesh_mat = MAT_DIFFUSE;
            else if (!strcmp(v, "mirror")) o->scene.mesh_mat = MAT_MIRROR;
            else if (!strcmp(v, "glass")) o->scene.mesh_mat = MAT_GLASS;
            else fail(prog, "unknown material", v);
        } else if (!strcmp(a, "-accel")){
            if (!strcmp(v, "auto")) o->accel = -1;
            else if (!strcmp(v, "none")) o->accel = ACCEL_NONE;
            else if (!strcmp(v, "octree")) o->accel = ACCEL_OCTREE;
            else if (!strcmp(v, "bvh")) o->accel = ACCEL_BVH;
            else fail(prog, "unknown acceleration", v);
        } else if (!strcmp(a, "-bvhsplit")){
            if (!strcmp(v, "sah")) o->ap.bvh_split = BVH_SPLIT_SAH;
            else if (!strcmp(v, "middle")) o->ap.bvh_split = BVH_SPLIT_MIDDLE;
            else fail(prog, "unknown BVH split", v);
        } else fail(prog, "unknown option", a);
    }

    if (o->width <= 0 || o->height <= 0) fail(prog, "image size must be positive, got", "-n/-width/-height");
    if (o->spp <= 0) fail(prog, "samples must be positive", "-s");
    if (o->max_depth <= 0) fail(prog, "bounces must be positive", "-d");
    if (o->clamp < 0) fail(prog, "clamp must be >= 0", "-clamp");
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
    if (prog == PROG_VIEW && !march_scale_set) o->march.scale = 2;   // real time: half resolution
    if (o->cloud.g <= -1 || o->cloud.g >= 1) fail(prog, "g must be between -1 and 1 (exclusive)", "-g");
    if (o->cloud.resolution < 4 || o->cloud.resolution > 1024) fail(prog, "cloud resolution must be 4..1024", "-cloudres");
    if (o->sim.on){
        if (o->sim.frames < 1 || o->sim.fps <= 0) fail(prog, "need -frames >= 1 and -fps > 0", "");
        if (o->sim.fluid_res < 8 || o->sim.fluid_res > 256) fail(prog, "fluid resolution must be 8..256", "-fluidres");
        if (o->sim.balls < 0 || o->sim.boxes < 0 || o->sim.balloons < 0 || o->sim.meshes < 0)
            fail(prog, "counts must be >= 0", "-balls/-boxes/-balloons/-meshes");
        (void)backend_set;   // view -sim: hybrid (raster + raymarched cloud while it runs)
        // live: coarser air and cloud, so that a physics step fits in a frame
        if (prog == PROG_VIEW && !fluidres_set) o->sim.fluid_res = 32;
        if (prog == PROG_VIEW && !cloudres_set) o->cloud.resolution = 96;
    }
    if (!o->out){
        static const char *rt[] = {"out/cornell_boxes.png", "out/cornell_spheres.png", "out/cornell_mesh.png",
                                   "out/cornell_cloud.png", "out/cornell_empty.png", "out/cornell_showcase.png"};
        static const char *ra[] = {"out/cornell_boxes_raster.png", "out/cornell_spheres_raster.png",
                                   "out/cornell_mesh_raster.png", "out/cornell_cloud_raster.png", "out/cornell_empty_raster.png",
                                   "out/cornell_showcase_raster.png"};
        o->out = o->backend == BACKEND_RASTER ? ra[o->scene.kind] : rt[o->scene.kind];
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

    if (o->prog == PROG_RENDER && o->backend == BACKEND_RASTER) return 0;   // no rays, no accel
    scene_set_accel(s, options_resolve_accel(o, s), &o->ap);
    char buf[256];
    scene_accel_stats(s, buf, sizeof(buf));
    printf("%s\n", buf);
    return 0;
}

void options_print_command(const Options *o, Vec3 cam, Vec3 look, double fov, FILE *f){
    static const char *scenes[] = {"boxes", "spheres", "mesh", "cloud", "empty", "showcase"};
    static const char *mats[] = {"diffuse", "mirror", "glass", "diffuse"};
    fprintf(f, "bin/render -scene %s", scenes[o->scene.kind]);
    if (o->scene.instances > 0) fprintf(f, " -instances %d", o->scene.instances);
    if (o->scene.kind == SCENE_MESH || o->scene.instances > 0){
        fprintf(f, " -obj %s", o->scene.obj_path);
        if (o->scene.mesh_mat != MAT_DIFFUSE) fprintf(f, " -mat %s", mats[o->scene.mesh_mat]);
        const SceneOptions *so = &o->scene;
        if (so->mesh_set & MESH_SET_BASE) fprintf(f, " -base %g,%g,%g", (double)so->mesh_base.x, (double)so->mesh_base.y, (double)so->mesh_base.z);
        if (so->mesh_set & MESH_SET_METAL) fprintf(f, " -metal %g", (double)so->mesh_metal);
        if (so->mesh_set & MESH_SET_ROUGH) fprintf(f, " -rough %g", (double)so->mesh_rough);
        if (so->mesh_set & MESH_SET_IOR) fprintf(f, " -ior %g", (double)so->mesh_ior);
        if (so->mesh_set & MESH_SET_TRANS) fprintf(f, " -transmission %g", (double)so->mesh_trans);
        if (o->scene.rot_deg != 0) fprintf(f, " -rot %g", (double)o->scene.rot_deg);
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
    fprintf(f, " -cam %.1f,%.1f,%.1f -look %.1f,%.1f,%.1f -fov %g",
            (double)cam.x, (double)cam.y, (double)cam.z, (double)look.x, (double)look.y, (double)look.z, fov);
    if (o->width == o->height) fprintf(f, " -n %d", o->width);
    else fprintf(f, " -width %d -height %d", o->width, o->height);
    fprintf(f, " -s 512");
    if (o->max_depth != 50) fprintf(f, " -d %d", o->max_depth);
    if (!o->mis) fprintf(f, " -mis off");
    if (!o->sampler) fprintf(f, " -sampler random");
    if (o->clamp > 0) fprintf(f, " -clamp %g", o->clamp);
    fprintf(f, " -o out/from_viewer.png\n");
}
