#include "options.h"

#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void options_usage(Program prog, FILE *f){
    int view = prog == PROG_RTVIEW;
    fprintf(f, "usage: %s [options]\n\n", view ? "rtview" : "raytracer");
    fprintf(f,
        "image and quality\n"
        "  -n size                    width = height in pixels                 (%s)\n"
        "  -width px / -height px     set them separately\n"
        "  -s samples                 %s (%s)\n"
        "  -d bounces                 maximum bounces per light path           (50)\n"
        "  -clamp max                 firefly clamp, 0 = off (biased if on)    (0)\n"
        "  -threads n                 CPU threads, 0 = all cores               (0)\n"
        "  -o file                    %s (%s)\n"
        "\n"
        "scene\n"
        "  -scene boxes|spheres|mesh  what is inside the Cornell box           (boxes)\n"
        "  -obj file.obj|file.ply     model for -scene mesh (ASCII .ply)       (../models/torus.obj)\n"
        "  -mat diffuse|mirror|glass  material of the model                    (diffuse)\n"
        "  -rot degrees               turn the model around the vertical axis  (0)\n"
        "\n"
        "camera (default: the classic Cornell view)\n"
        "  -cam x,y,z                 camera position                          (278,278,-800)\n"
        "  -look x,y,z                point the camera looks at                (278,278,0)\n"
        "  -fov degrees               vertical field of view                   (40)\n"
        "\n"
        "acceleration structure\n"
        "  -accel auto|none|octree|bvh  auto = bvh from %d primitives up, else none   (auto)\n"
        "  -octdepth n                octree max depth, 0 = automatic          (0)\n"
        "  -bvhleaf n                 BVH: max primitives per leaf             (4)\n"
        "  -bvhbins n                 BVH: SAH buckets per axis                (16)\n"
        "  -bvhsplit sah|middle       BVH: how nodes are split                 (sah)\n",
        view ? "640" : "400",
        view ? "stop refining after this many samples per pixel" : "samples (light paths) per pixel  ",
        view ? "4096" : "128",
        view ? "screenshot prefix (key P)  " : "output PPM file            ",
        view ? "out/view" : "out/cornell_<scene>.ppm",
        AUTO_ACCEL_MIN_PRIMS);
    if (view) fprintf(f,
        "\n"
        "real-time viewer\n"
        "  -preview n                 render at 1/n resolution while moving, 1 = off (4)\n"
        "  -movespp n                 samples per pixel of each preview frame  (1)\n"
        "  -speed units               movement speed per second                (300)\n"
        "  -autoquit seconds          quit (and save a screenshot) after this time (0 = never)\n");
    fprintf(f, "\n  -h, --help                 this text\n");
}

static void fail(Program prog, const char *msg, const char *arg){
    fprintf(stderr, "error: %s '%s'\n\n", msg, arg);
    options_usage(prog, stderr);
    exit(1);
}

static Vec3 parse_vec3(Program prog, const char *s){
    Vec3 v;
    if (sscanf(s, "%lf,%lf,%lf", &v.x, &v.y, &v.z) != 3) fail(prog, "expected x,y,z but got", s);
    return v;
}

void options_parse(Options *o, Program prog, int argc, char **argv){
    memset(o, 0, sizeof(*o));
    o->prog = prog;
    o->width = o->height = prog == PROG_RTVIEW ? 640 : 400;
    o->spp = prog == PROG_RTVIEW ? 4096 : 128;
    o->max_depth = 50;
    o->scene.kind = SCENE_BOXES;
    o->scene.obj_path = "../models/torus.obj";
    o->scene.mesh_mat = MAT_DIFFUSE;
    o->accel = -1;
    o->ap.oct_depth = 0;
    o->ap.bvh_leaf = 4;
    o->ap.bvh_bins = 16;
    o->ap.bvh_split = BVH_SPLIT_SAH;
    o->preview = 4;
    o->move_spp = 1;
    o->speed = 300;
    o->out = prog == PROG_RTVIEW ? "out/view" : NULL;

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
        else if (!strcmp(a, "-cam"))     { o->cam = parse_vec3(prog, v); o->set_cam = 1; }
        else if (!strcmp(a, "-look"))    { o->look = parse_vec3(prog, v); o->set_look = 1; }
        else if (!strcmp(a, "-fov"))     { o->fov = atof(v); o->set_fov = 1; }
        else if (!strcmp(a, "-octdepth")) o->ap.oct_depth = atoi(v);
        else if (!strcmp(a, "-bvhleaf")) o->ap.bvh_leaf = atoi(v);
        else if (!strcmp(a, "-bvhbins")) o->ap.bvh_bins = atoi(v);
        else if (!strcmp(a, "-preview")) o->preview = atoi(v);
        else if (!strcmp(a, "-movespp")) o->move_spp = atoi(v);
        else if (!strcmp(a, "-speed"))   o->speed = atof(v);
        else if (!strcmp(a, "-autoquit")) o->autoquit = atof(v);
        else if (!strcmp(a, "-scene")){
            if (!strcmp(v, "boxes")) o->scene.kind = SCENE_BOXES;
            else if (!strcmp(v, "spheres")) o->scene.kind = SCENE_SPHERES;
            else if (!strcmp(v, "mesh")) o->scene.kind = SCENE_MESH;
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
    if (o->preview < 1) o->preview = 1;
    if (o->move_spp < 1) o->move_spp = 1;
    if (!o->out){
        static const char *names[] = {"out/cornell_boxes.ppm", "out/cornell_spheres.ppm", "out/cornell_mesh.ppm"};
        o->out = names[o->scene.kind];
    }
}

AccelKind options_resolve_accel(const Options *o, const Scene *s){
    if (o->accel >= 0) return (AccelKind)o->accel;
    return s->nprims >= AUTO_ACCEL_MIN_PRIMS ? ACCEL_BVH : ACCEL_NONE;
}

int options_setup_scene(const Options *o, Scene *s){
#ifdef _OPENMP
    if (o->threads > 0) omp_set_num_threads(o->threads);
#endif
    if (scene_cornell_box(s, &o->scene) != 0) return 1;
    if (o->scene.kind == SCENE_MESH) printf("loaded %s: %d primitives\n", o->scene.obj_path, s->nprims);

    if (o->set_cam) s->cam_from = o->cam;
    if (o->set_look) s->cam_at = o->look;
    if (o->set_fov) s->cam_vfov_deg = o->fov;

    scene_set_accel(s, options_resolve_accel(o, s), &o->ap);
    char buf[256];
    scene_accel_stats(s, buf, sizeof(buf));
    printf("%s\n", buf);
    return 0;
}

void options_print_command(const Options *o, Vec3 cam, Vec3 look, double fov, FILE *f){
    static const char *scenes[] = {"boxes", "spheres", "mesh"};
    static const char *mats[] = {"diffuse", "mirror", "glass", "diffuse"};
    fprintf(f, "./raytracer -scene %s", scenes[o->scene.kind]);
    if (o->scene.kind == SCENE_MESH){
        fprintf(f, " -obj %s", o->scene.obj_path);
        if (o->scene.mesh_mat != MAT_DIFFUSE) fprintf(f, " -mat %s", mats[o->scene.mesh_mat]);
        if (o->scene.rot_deg != 0) fprintf(f, " -rot %g", o->scene.rot_deg);
    }
    fprintf(f, " -cam %.1f,%.1f,%.1f -look %.1f,%.1f,%.1f -fov %g", cam.x, cam.y, cam.z, look.x, look.y, look.z, fov);
    if (o->width == o->height) fprintf(f, " -n %d", o->width);
    else fprintf(f, " -width %d -height %d", o->width, o->height);
    fprintf(f, " -s 512");
    if (o->max_depth != 50) fprintf(f, " -d %d", o->max_depth);
    if (o->clamp > 0) fprintf(f, " -clamp %g", o->clamp);
    fprintf(f, " -o out/from_viewer.ppm\n");
}
