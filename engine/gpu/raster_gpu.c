// raster_gpu.c - the rasterizer of raster/render.c (render_scene), on the GPU.
//
// The same triangles (raster/render.h hands them out: 2 per quad, a latitude / longitude
// grid per sphere, the mesh triangles), the same lamp and BSDF (render_shade, in WGSL with
// gpu/brdf_gpu.c's GPU_WGSL_BSDF), the same four shading modes, drawn by a WebGPU render pipeline into a color and a depth texture
// that the view then composites (gpu/view_gpu.c). What the CPU does per triangle on many
// threads (transform, clip, cull, fill, z-test) is the GPU's own machinery here.
//
// The vertices are not vertex buffers but two storage buffers the vertex shader reads by
// index ("vertex pulling"): a vertex can then look at the other two corners of its
// triangle, which flat shading needs (one brightness per triangle, from the face normal),
// and the wireframe draws the same data as lines.
//   pos[i] = position, w = material + 16 * flat_flag + 64 * has_normals
//   nrm[i] = normal (when has_normals)
// flat_flag says where flat shading is evaluated: 0 the centroid, 1 / 2 the middle of the
// edge v0-v2 / v0-v1 (a quad's center, for its two halves: both get one color).
#include "gpu/gpu_internal.h"

#ifdef LR_GPU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "scene/env.h"

GPU_WGSL(SHADER, raster_shader);

typedef struct {
    float vp[16], eye[4], lamp[32];
    float envp[4], envc[4], envlo[4], envhi[4];
    uint32_t lv[ENV_LEVELS][4];
    float sh[9][4];
    float sky[4];
    uint32_t skyd[4];
    float cam_r[4], cam_u[4], cam_b[4], vps[4];
    uint32_t irrn[4];
    float irrlo[4], irrhi[4];
    float mats[MAX_MATS][16];          // base, pbr, tex (uint32 bits), emission
} GRU;

// Levels of detail: from this many instances on, a mesh instance small on the screen is
// drawn with a simpler copy of its mesh (scene_mesh_simplify, a grid of LOD_GRID cells over
// it), level k while its projected radius is under LOD_PIXELS[k - 1] pixels. Fewer
// instances: always the mesh itself (so small scenes look exactly as before). LR_GPU_LOD=0
// turns them off, =1 on for any count.
#define LODS 3
#define LOD_MIN_INSTANCES 4096
static const int LOD_GRID[LODS] = {0, 16, 6};
static const float LOD_PIXELS[LODS - 1] = {40.0f, 12.0f};

// one set of triangles on the GPU
typedef struct {
    WGPUBuffer pos, nrm, tex;
    size_t pos_cap, nrm_cap, tex_cap;
    WGPUBindGroup bg;
    uint32_t nverts;
} Geo;

struct GpuRaster {
    Gpu *g;
    int w, h;
    int inst_valid, inst_count;            // the instances' transforms as uploaded (upload_instances)
    unsigned inst_version, insts_id;       // (Scene.inst_version, insts_id then)
    float *rec;                            // memory for packing them, kept
    size_t rec_cap;
    WGPUTexture color_tex, depth_tex;
    WGPUTextureView color, depth;
    WGPUBuffer uni, tab;                   // the uniforms; the BSDF's tables
    WGPUBuffer lv, bg;                     // the environment's levels, the sky behind (scene/env.h)
    size_t lv_cap, bg_cap;
    unsigned lv_id, bg_id;                 // the Env ids they hold (0: none)
    WGPUBuffer ig, txt;                    // the irradiance grid; the textures' table (uniform)
    size_t ig_cap;
    unsigned ig_id;
    GpuTextures gt;                        // the textures' atlas
    uint32_t lv_at[ENV_LEVELS];            // where each level starts in lv
    WGPUBindGroupLayout bgl;
    WGPURenderPipeline fill[2], wire, sky; // fill[1]: back faces culled
    Geo prims, mesh;                       // spheres and quads (they move); mesh triangles
    WGPUBuffer ix;                         // the draws' transforms: the world, the instances
    size_t ix_cap;
    // the scene's meshes (SceneMesh.id, version), each with simpler copies for far away
    // (lod[k].nverts 0: none, the one before is used)
    struct { Geo geo, lod[LODS - 1]; unsigned id; int ntris; unsigned version; } *mg;
    int nmg;
    Geo unit[2];                           // the unit sphere and box (shape instances: the physics' bodies)
    WGPUBuffer ord;                        // the draws' order: draw instance -> record (ORD)
    size_t ord_cap;
    uint32_t *ordv;                        // ... packed here: the records by mesh and level
    int *bucket;                           // (per mesh and level: its count, then where it starts)
    size_t ordv_cap, bucket_cap;
    // what is on the GPU now
    Primitive *prim_copy;
    int nprim_copy;
    unsigned mesh_version;                 // (the world's triangles: Scene.tris_version)
    int mesh_ntris;
    float *tmp_pos, *tmp_nrm, *tmp_tex;    // scratch for the uploads
    size_t tmp_cap;                        // vertices
};

#define CHUNK_TRIS 32768   // mesh triangles converted and uploaded at a time

// wire: 0 the triangles, 1 the wireframe, 2 the sky (a screen-filling triangle at the far
// depth, where the depth is still the far one)
static WGPURenderPipeline make_pipeline(Gpu *g, WGPUShaderModule m, WGPUBindGroupLayout bgl, int wire, int cull){
    const int sky = wire == 2;
    WGPUPipelineLayoutDescriptor ld;
    memset(&ld, 0, sizeof(ld));
    ld.bindGroupLayoutCount = 1;
    ld.bindGroupLayouts = &bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(g->device, &ld);
    WGPUColorTargetState target;
    memset(&target, 0, sizeof(target));
    target.format = WGPUTextureFormat_RGBA8Unorm;
    target.writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState fs;
    memset(&fs, 0, sizeof(fs));
    fs.module = m;
    fs.entryPoint = gpu_str(sky ? "fs_sky" : wire ? "fs_wire" : "fs");
    fs.targetCount = 1;
    fs.targets = &target;
    // the CPU's z-buffer: the nearest wins, the first drawn wins a tie. Its wireframe draws
    // lines over everything and leaves the z-buffer alone.
    WGPUDepthStencilState ds;
    memset(&ds, 0, sizeof(ds));
    ds.format = WGPUTextureFormat_Depth32Float;
    ds.depthWriteEnabled = wire ? WGPUOptionalBool_False : WGPUOptionalBool_True;
    ds.depthCompare = sky ? WGPUCompareFunction_LessEqual : wire ? WGPUCompareFunction_Always : WGPUCompareFunction_Less;
    ds.stencilFront.compare = ds.stencilBack.compare = WGPUCompareFunction_Always;
    ds.stencilFront.failOp = ds.stencilBack.failOp = WGPUStencilOperation_Keep;
    ds.stencilFront.depthFailOp = ds.stencilBack.depthFailOp = WGPUStencilOperation_Keep;
    ds.stencilFront.passOp = ds.stencilBack.passOp = WGPUStencilOperation_Keep;
    WGPURenderPipelineDescriptor rd;
    memset(&rd, 0, sizeof(rd));
    rd.label = gpu_str(sky ? "scene: the sky" : wire ? "scene wireframe" : "scene");
    rd.layout = pl;
    rd.vertex.module = m;
    rd.vertex.entryPoint = gpu_str(sky ? "vs_sky" : wire ? "vs_wire" : "vs");
    rd.primitive.topology = wire == 1 ? WGPUPrimitiveTopology_LineList : WGPUPrimitiveTopology_TriangleList;
    rd.primitive.frontFace = WGPUFrontFace_CCW;     // counter-clockwise as seen: the CPU's rule
    rd.primitive.cullMode = cull ? WGPUCullMode_Back : WGPUCullMode_None;
    rd.depthStencil = &ds;
    rd.multisample.count = 1;
    rd.multisample.mask = 0xFFFFFFFFu;
    rd.fragment = &fs;
    WGPURenderPipeline p = wgpuDeviceCreateRenderPipeline(g->device, &rd);
    wgpuPipelineLayoutRelease(pl);
    return p;
}

static WGPUTexture make_target(Gpu *g, const char *label, WGPUTextureFormat fmt, int w, int h){
    WGPUTextureDescriptor td;
    memset(&td, 0, sizeof(td));
    td.label = gpu_str(label);
    td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
    td.dimension = WGPUTextureDimension_2D;
    td.size = (WGPUExtent3D){(uint32_t)w, (uint32_t)h, 1};
    td.format = fmt;
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    return wgpuDeviceCreateTexture(g->device, &td);
}

GpuRaster *gpu_raster_create(Gpu *g, int width, int height){
    if (!g || g->state != GPU_READY) return NULL;
    GpuRaster *r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->g = g;
    r->w = width;
    r->h = height;
    r->color_tex = make_target(g, "scene color", WGPUTextureFormat_RGBA8Unorm, width, height);
    r->depth_tex = make_target(g, "scene depth", WGPUTextureFormat_Depth32Float, width, height);
    r->color = wgpuTextureCreateView(r->color_tex, NULL);
    r->depth = wgpuTextureCreateView(r->depth_tex, NULL);
    r->uni = gpu_buffer(g, "scene", sizeof(GRU), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    float tab[GPU_BSDF_TABLE_FLOATS];
    gpu_bsdf_tables(tab);
    r->tab = gpu_buffer(g, "scene: BSDF tables", sizeof(tab), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    wgpuQueueWriteBuffer(g->queue, r->tab, 0, tab, sizeof(tab));
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    r->lv = gpu_buffer(g, "scene: no environment", 16, st);
    r->bg = gpu_buffer(g, "scene: no sky", 16, st);
    r->ig = gpu_buffer(g, "scene: no irradiance grid", 112, st);
    r->ord_cap = 16;
    r->ord = gpu_buffer(g, "raster: the draws' order", r->ord_cap, st);
    r->txt = gpu_buffer(g, "scene: the textures' table", sizeof(r->gt.table), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    r->bgl = gpu_layout(g, WGPUShaderStage_Vertex | WGPUShaderStage_Fragment, "URRRURRRRTUR");
    char *src = malloc(strlen(GPU_WGSL_BSDF) + strlen(GPU_WGSL_TEX) + strlen(SHADER) + 3);
    if (!src){ gpu_raster_destroy(r); return NULL; }
    strcpy(src, GPU_WGSL_BSDF); strcat(src, "\n"); strcat(src, GPU_WGSL_TEX); strcat(src, "\n"); strcat(src, SHADER);
    WGPUShaderModule m = gpu_shader(g, "scene", src);
    free(src);
    r->fill[0] = make_pipeline(g, m, r->bgl, 0, 0);
    r->fill[1] = make_pipeline(g, m, r->bgl, 0, 1);
    r->wire = make_pipeline(g, m, r->bgl, 1, 0);
    r->sky = make_pipeline(g, m, r->bgl, 2, 0);
    wgpuShaderModuleRelease(m);
    return r;
}

WGPUTextureView gpu_raster_color(GpuRaster *r){ return r->color; }
WGPUTextureView gpu_raster_depth(GpuRaster *r){ return r->depth; }

static void put(float *d, Vec3 a, float w){ d[0] = (float)a.x; d[1] = (float)a.y; d[2] = (float)a.z; d[3] = w; }

static int tmp_reserve(GpuRaster *r, size_t verts){
    if (r->tmp_cap >= verts) return 1;
    float *a = realloc(r->tmp_pos, verts * 16), *b = realloc(r->tmp_nrm, verts * 16), *c = realloc(r->tmp_tex, verts * 16);
    if (a) r->tmp_pos = a;
    if (b) r->tmp_nrm = b;
    if (c) r->tmp_tex = c;
    if (!a || !b || !c) return 0;
    r->tmp_cap = verts;
    return 1;
}

static void geo_bind(GpuRaster *r, Geo *geo){
    if (geo->bg) wgpuBindGroupRelease(geo->bg);
    geo->bg = NULL;
    if (!geo->pos || !r->ix || !r->ord || !r->gt.view) return;
    WGPUBuffer b[12] = {r->uni, geo->pos, geo->nrm, r->ix, r->tab, r->lv, r->bg, geo->tex, r->ig, NULL, r->txt, r->ord};
    WGPUBindGroupEntry e[12];
    memset(e, 0, sizeof(e));
    for (int i = 0; i < 12; i++){
        e[i].binding = (uint32_t)i;
        if (i == 9) e[i].textureView = r->gt.view;
        else { e[i].buffer = b[i]; e[i].size = WGPU_WHOLE_SIZE; }
    }
    WGPUBindGroupDescriptor d;
    memset(&d, 0, sizeof(d));
    d.layout = r->bgl;
    d.entryCount = 12;
    d.entries = e;
    geo->bg = wgpuDeviceCreateBindGroup(r->g->device, &d);
}

// room for `verts` vertices in a set; its bind group follows the buffers
static void geo_reserve(GpuRaster *r, Geo *geo, size_t verts, const char *label){
    Gpu *g = r->g;
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    int grown = gpu_buffer_reserve(g, &geo->pos, &geo->pos_cap, verts * 16, st, label);
    grown |= gpu_buffer_reserve(g, &geo->nrm, &geo->nrm_cap, verts * 16, st, label);
    grown |= gpu_buffer_reserve(g, &geo->tex, &geo->tex_cap, verts * 16, st, label);
    if (!grown && geo->bg) return;
    geo_bind(r, geo);
}

// a vertex's texture coordinates and tangent (vtex: u, v, the tangent octahedral; its
// handedness into the normal's w)
static void put_tex(GpuRaster *r, size_t at, Vec2 uv, Vec4 tg){
    float *o = r->tmp_tex + 4 * at;
    o[0] = (float)uv.x; o[1] = (float)uv.y;
    gpu_oct_encode(vec3(tg.x, tg.y, tg.z), o + 2);
    r->tmp_nrm[4 * at + 3] = (float)tg.w;
}

// primitives as triangles into tmp_pos / tmp_nrm / tmp_tex (from vertex nv on); returns the new count
static size_t prims_to_tris(GpuRaster *r, const Primitive *pr, int n, size_t nv){
    for (int i = 0; i < n; i++){
        int nt = render_prim_tri_count(&pr[i]);
        for (int sub = 0; sub < nt; sub++){
            Vec3 wp[3], wn[3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, flat_at;
            int has_n = 0;
            render_prim_triangle(&pr[i], sub, wp, wn, &has_n, &flat_at);
            Vec2 uv[3];
            Vec4 tg[3];
            render_prim_texcoords(&pr[i], sub, uv, tg);
            int flag = pr[i].type == PRIM_QUAD ? (sub == 0 ? 1 : 2) : 0;
            float code = (float)((pr[i].material & 15) + 16 * flag + 64 * (has_n != 0));
            for (int k = 0; k < 3; k++, nv++){
                put(r->tmp_pos + 4 * nv, wp[k], code);
                put(r->tmp_nrm + 4 * nv, wn[k], 0);
                put_tex(r, nv, uv[k], tg[k]);
            }
        }
    }
    return nv;
}

// the unit sphere and the unit box, once: shape instances are drawn as them, moved
static int upload_unit_shapes(GpuRaster *r){
    if (r->unit[0].nverts && r->unit[1].nverts) return 1;
    Primitive faces[6];
    for (int f = 0; f < 6; f++) faces[f] = unit_box_face(f, 0);
    Primitive sphere = make_sphere(vec3(0, 0, 0), 1, 0);
    if (!tmp_reserve(r, (size_t)RENDER_SPHERE_TRIS * 3 + 64)) return 0;
    for (int k = 0; k < 2; k++){
        size_t nv = prims_to_tris(r, k == 0 ? &sphere : faces, k == 0 ? 1 : 6, 0);
        geo_reserve(r, &r->unit[k], nv, k == 0 ? "unit sphere" : "unit box");
        wgpuQueueWriteBuffer(r->g->queue, r->unit[k].pos, 0, r->tmp_pos, nv * 16);
        wgpuQueueWriteBuffer(r->g->queue, r->unit[k].nrm, 0, r->tmp_nrm, nv * 16);
        wgpuQueueWriteBuffer(r->g->queue, r->unit[k].tex, 0, r->tmp_tex, nv * 16);
        r->unit[k].nverts = (uint32_t)nv;
    }
    return 1;
}

// the spheres and quads: few, and they move (the physics), so they are compared with what
// was uploaded last and sent again when anything changed
static int upload_prims(GpuRaster *r, const Scene *s){
    if (r->nprim_copy == s->nprims && r->prim_copy &&
        (s->nprims == 0 || memcmp(r->prim_copy, s->prims, (size_t)s->nprims * sizeof(Primitive)) == 0)) return 1;
    size_t ntri = 0;
    for (int i = 0; i < s->nprims; i++) ntri += (size_t)render_prim_tri_count(&s->prims[i]);
    if (!tmp_reserve(r, ntri * 3 + 3)) return 0;
    size_t nv = prims_to_tris(r, s->prims, s->nprims, 0);
    geo_reserve(r, &r->prims, nv ? nv : 3, "scene primitives");
    if (nv){
        wgpuQueueWriteBuffer(r->g->queue, r->prims.pos, 0, r->tmp_pos, nv * 16);
        wgpuQueueWriteBuffer(r->g->queue, r->prims.nrm, 0, r->tmp_nrm, nv * 16);
        wgpuQueueWriteBuffer(r->g->queue, r->prims.tex, 0, r->tmp_tex, nv * 16);
    }
    r->prims.nverts = (uint32_t)nv;
    Primitive *c = realloc(r->prim_copy, (size_t)(s->nprims ? s->nprims : 1) * sizeof(Primitive));
    if (c){
        r->prim_copy = c;
        if (s->nprims) memcpy(c, s->prims, (size_t)s->nprims * sizeof(Primitive));
        r->nprim_copy = s->nprims;
    } else r->nprim_copy = -1;
    return 1;
}

// triangles into a set (as they are: the world's, or a mesh's in its own space)
static int upload_tris(GpuRaster *r, Geo *geo, const Triangle *tris, int ntris, const Vec3 *normals,
                       const Vec2 *uvs, const Vec4 *tangents, int nuvs, const char *label){
    size_t nv = (size_t)ntris * 3;
    if (nv * 16 > r->g->max_storage){
        fprintf(stderr, "GPU: the mesh needs a %.0f MB buffer, this GPU allows %.0f MB: the CPU rasterizes\n",
                (double)nv * 16 / 1048576, (double)r->g->max_storage / 1048576);
        return 0;
    }
    if (nv){
        if (!tmp_reserve(r, (size_t)CHUNK_TRIS * 3)) return 0;
        geo_reserve(r, geo, nv, label);
        for (int first = 0; first < ntris; first += CHUNK_TRIS){
            int n = ntris - first < CHUNK_TRIS ? ntris - first : CHUNK_TRIS;
            for (int i = 0; i < n; i++){
                const Triangle *t = &tris[first + i];
                int has_n = t->n[0] >= 0;
                float code = (float)((t->material & 15) + 64 * has_n);
                Vec3 wp[3] = {t->q, v_add(t->q, t->u), v_add(t->q, t->v)};
                for (int k = 0; k < 3; k++){
                    put(r->tmp_pos + 4 * (3 * (size_t)i + k), wp[k], code);
                    put(r->tmp_nrm + 4 * (3 * (size_t)i + k), has_n ? normals[t->n[k]] : vec3(0, 0, 0), 0);
                }
                {
                    Vec2 uv[3];
                    Vec4 tg[3];
                    render_tri_texcoords(t, uvs, tangents, nuvs, NULL, uv, tg);
                    for (int k = 0; k < 3; k++) put_tex(r, 3 * (size_t)i + k, uv[k], tg[k]);
                }
            }
            wgpuQueueWriteBuffer(r->g->queue, geo->pos, (uint64_t)first * 48, r->tmp_pos, (size_t)n * 48);
            wgpuQueueWriteBuffer(r->g->queue, geo->nrm, (uint64_t)first * 48, r->tmp_nrm, (size_t)n * 48);
            wgpuQueueWriteBuffer(r->g->queue, geo->tex, (uint64_t)first * 48, r->tmp_tex, (size_t)n * 48);
        }
    }
    geo->nverts = (uint32_t)nv;
    return 1;
}

// the world's mesh triangles: many, and they stay where they are: uploaded once
static int upload_mesh(GpuRaster *r, const Scene *s){
    if (r->mesh_version == s->tris_version && r->mesh_ntris == s->ntris) return 1;
    if (!upload_tris(r, &r->mesh, s->tris, s->ntris, s->normals, s->uvs, s->tangents, s->nuvs, "scene mesh")) return 0;
    r->mesh_version = s->tris_version;
    r->mesh_ntris = s->ntris;
    return 1;
}

static void rebind_all(GpuRaster *r){
    geo_bind(r, &r->prims);
    geo_bind(r, &r->mesh);
    for (int m = 0; m < r->nmg; m++){
        geo_bind(r, &r->mg[m].geo);
        for (int l = 0; l < LODS - 1; l++) geo_bind(r, &r->mg[m].lod[l]);
    }
    geo_bind(r, &r->unit[0]);
    geo_bind(r, &r->unit[1]);
}

// the transforms (record 0: the world as it is; then the instances), and the meshes' own
// triangles (again when a dynamic one changed)
// record i (0: the world; then the instances): object -> world rows, the inverse's, material
static void pack_record(float *o, const Scene *s, size_t i){
    memset(o, 0, 112);
    Xform xf = i ? s->insts[i - 1].xf : xform_identity(), inv = i ? s->insts[i - 1].inv : xform_identity();
    for (int a = 0; a < 3; a++)
        for (int k = 0; k < 4; k++){ o[4 * a + k] = (float)xf.m[a][k]; o[12 + 4 * a + k] = (float)inv.m[a][k]; }
    uint32_t m = i && s->insts[i - 1].material >= 0 ? (uint32_t)s->insts[i - 1].material : 0xffffffffu;
    memcpy(o + 24, &m, 4);
}

// Records closer than this are sent in one write with those between (the same as on the
// GPU: rec keeps what was sent)
#define REC_GAP 32

static int upload_instances(GpuRaster *r, const Scene *s){
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    size_t n = 1 + (size_t)s->ninsts;
    // (the same instances as last time, Scene.inst_version: they are there already; the same
    // scene's, as many or more: only those moved or added since, Instance.version)
    const int same = r->ix && r->inst_valid && r->inst_version == s->inst_version && r->inst_count == s->ninsts;
    const int some = !same && r->ix && r->inst_valid && r->insts_id == s->insts_id && r->inst_count <= s->ninsts &&
                     r->rec && r->rec_cap >= n * 112 && r->ix_cap >= n * 112;
    if (some){
        for (size_t i = 1; i < n;){
            if (s->insts[i - 1].version <= r->inst_version){ i++; continue; }
            // a run of changed records (gaps under REC_GAP joined in)
            size_t first = i, last = i;
            for (size_t j = i; j < n && j <= last + REC_GAP; j++)
                if (s->insts[j - 1].version > r->inst_version){ pack_record(r->rec + 28 * j, s, j); last = j; }
            wgpuQueueWriteBuffer(r->g->queue, r->ix, first * 112, r->rec + 28 * first, (last + 1 - first) * 112);
            i = last + 1;
        }
        r->inst_version = s->inst_version;
        r->inst_count = s->ninsts;
    } else if (!same){
        if (r->rec_cap < n * 112){
            free(r->rec);
            r->rec = malloc(n * 112 + n * 28 + 1024);
            r->rec_cap = r->rec ? n * 112 + n * 28 + 1024 : 0;
        }
        if (!r->rec) return 0;
        for (size_t i = 0; i < n; i++) pack_record(r->rec + 28 * i, s, i);
        if (gpu_buffer_reserve(r->g, &r->ix, &r->ix_cap, n * 112, st, "raster: transforms")) rebind_all(r);
        wgpuQueueWriteBuffer(r->g->queue, r->ix, 0, r->rec, n * 112);
        r->inst_valid = 1;
        r->inst_version = s->inst_version;
        r->insts_id = s->insts_id;
        r->inst_count = s->ninsts;
    }
    if (s->nmeshes > r->nmg){
        void *mg = realloc(r->mg, (size_t)s->nmeshes * sizeof(r->mg[0]));
        if (!mg) return 0;
        r->mg = mg;
        memset(r->mg + r->nmg, 0, (size_t)(s->nmeshes - r->nmg) * sizeof(r->mg[0]));
        r->nmg = s->nmeshes;
    }
    for (int m = 0; m < s->nmeshes; m++){
        const SceneMesh *sm = &s->meshes[m];
        if (r->mg[m].geo.nverts && r->mg[m].id == sm->id && r->mg[m].ntris == sm->ntris && r->mg[m].version == sm->version)
            continue;
        if (!upload_tris(r, &r->mg[m].geo, sm->tris, sm->ntris, sm->normals, sm->uvs, sm->tangents, sm->nuvs, "instanced mesh")) return 0;
        // its simpler copies (not of a mesh that changes: they would be made again each time),
        // kept only when much simpler than the level before
        int before = sm->ntris;
        for (int l = 0; l < LODS - 1; l++){
            r->mg[m].lod[l].nverts = 0;
            Triangle *lt = NULL;
            Vec3 *ln = NULL;
            int nlt = 0, nln = 0;
            if (sm->dynamic || sm->ntris < 64 || scene_mesh_simplify(sm, LOD_GRID[l + 1], &lt, &nlt, &ln, &nln) != 0) continue;
            if (nlt > 0 && nlt < before * 7 / 10){
                if (upload_tris(r, &r->mg[m].lod[l], lt, nlt, ln, NULL, NULL, 0, "instanced mesh, far")) before = nlt;
                else r->mg[m].lod[l].nverts = 0;
            }
            free(lt);
            free(ln);
        }
        r->mg[m].id = sm->id;
        r->mg[m].ntris = sm->ntris;
        r->mg[m].version = sm->version;
    }
    return 1;
}

// The instances' draw order (ORD: draw instance -> record; ORD[0] = 0, the world): those in
// view only (their world box against the frustum's sides and the eye's plane), bucketed by
// mesh and level of detail, so each bucket is one draw. Returns 0 when out of memory.
static int order_instances(GpuRaster *r, const Scene *s, const RenderParams *p, Mat4 vp){
    const int nk = (s->nmeshes + 2) * LODS;
    const size_t n = 1 + (size_t)s->ninsts;
    if (r->ordv_cap < n){
        uint32_t *o = realloc(r->ordv, 2 * n * sizeof(uint32_t));   // (then the instances' buckets)
        if (!o) return 0;
        r->ordv = o;
        r->ordv_cap = n;
    }
    if (r->bucket_cap < (size_t)nk + 1){
        int *b = realloc(r->bucket, ((size_t)nk + 1) * sizeof(int));
        if (!b) return 0;
        r->bucket = b;
        r->bucket_cap = (size_t)nk + 1;
    }
    // the frustum's planes (rows of vp: w + x, w - x, w + y, w - y; and w itself)
    real pl[5][4];
    for (int k = 0; k < 4; k++){
        pl[0][k] = vp.m[3][k] + vp.m[0][k];
        pl[1][k] = vp.m[3][k] - vp.m[0][k];
        pl[2][k] = vp.m[3][k] + vp.m[1][k];
        pl[3][k] = vp.m[3][k] - vp.m[1][k];
        pl[4][k] = vp.m[3][k];
    }
    static int lod_env = -2;               // LR_GPU_LOD=0: never, =1: whatever the count
    if (lod_env == -2){ const char *e = getenv("LR_GPU_LOD"); lod_env = e ? atoi(e) != 0 : -1; }
    const int lod = lod_env >= 0 ? lod_env : s->ninsts >= LOD_MIN_INSTANCES;
    const real px_per = r_fabs(p->proj.m[1][1]) * (real)r->h * (real)0.5;   // pixels per unit at depth 1
    int *cnt = r->bucket;
    memset(cnt, 0, ((size_t)nk + 1) * sizeof(int));
    // first pass: each instance's bucket (or none: out of view)
    uint32_t *keys = r->ordv + r->ordv_cap;
    for (int i = 0; i < s->ninsts; i++){
        const Instance *in = &s->insts[i];
        const Aabb b = in->box;
        int seen = 1;
        for (int q = 0; q < 5 && seen; q++){
            Vec3 pv = vec3(pl[q][0] >= 0 ? b.hi.x : b.lo.x, pl[q][1] >= 0 ? b.hi.y : b.lo.y, pl[q][2] >= 0 ? b.hi.z : b.lo.z);
            if (pl[q][0] * pv.x + pl[q][1] * pv.y + pl[q][2] * pv.z + pl[q][3] < 0) seen = 0;
        }
        if (!seen){ keys[i] = 0xffffffffu; continue; }
        int l = 0;
        if (lod && in->mesh >= 0 && in->mesh < r->nmg){
            Vec3 c = v_scale(v_add(b.lo, b.hi), (real)0.5);
            real rad = (real)0.5 * v_len(v_sub(b.hi, b.lo));
            real w = pl[4][0] * c.x + pl[4][1] * c.y + pl[4][2] * c.z + pl[4][3];
            if (w > rad){
                real px = rad * px_per / w;
                while (l < LODS - 1 && px < LOD_PIXELS[l]) l++;
                while (l > 0 && !r->mg[in->mesh].lod[l - 1].nverts) l--;
            }
        }
        int key = (in->mesh + 2) * LODS + l;
        keys[i] = (uint32_t)key;
        cnt[key + 1]++;
    }
    for (int k = 0; k < nk; k++) cnt[k + 1] += cnt[k];      // where each bucket starts (from 1)
    // second pass: the records into their buckets, in the instances' order; cnt[k] ends
    // where bucket k ends, so its start is cnt[k - 1] (0 for the first)
    const int total = cnt[nk];
    for (int i = 0; i < s->ninsts; i++)
        if (keys[i] != 0xffffffffu) r->ordv[1 + cnt[keys[i]]++] = (uint32_t)i + 1;
    r->ordv[0] = 0;
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    if (gpu_buffer_reserve(r->g, &r->ord, &r->ord_cap, n * sizeof(uint32_t), st, "raster: the draws' order")) rebind_all(r);
    wgpuQueueWriteBuffer(r->g->queue, r->ord, 0, r->ordv, (1 + (size_t)total) * sizeof(uint32_t));
    return 1;
}

// the environment's levels (all six one after the other) and the sky behind, when they change
static int upload_env(GpuRaster *r, const Env *env, const Env *sky){
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    int rebind = 0;
    unsigned id = env ? env->id : 0;
    if (id && id != r->lv_id){
        size_t n = 0;
        for (int k = 0; k < ENV_LEVELS; k++){ r->lv_at[k] = (uint32_t)n; n += (size_t)env->lw[k] * env->lh[k]; }
        float *d = malloc(n * 16);
        if (!d) return 0;
        for (int k = 0; k < ENV_LEVELS; k++)
            for (size_t i = 0; i < (size_t)env->lw[k] * env->lh[k]; i++){
                float *o = d + 4 * (r->lv_at[k] + i);
                const float *c = env->level[k] + 3 * i;
                o[0] = c[0]; o[1] = c[1]; o[2] = c[2]; o[3] = 0;
            }
        rebind |= gpu_buffer_reserve(r->g, &r->lv, &r->lv_cap, n * 16, st, "scene: the environment's levels");
        wgpuQueueWriteBuffer(r->g->queue, r->lv, 0, d, n * 16);
        free(d);
        r->lv_id = id;
    }
    id = sky ? sky->id : 0;
    if (id && id != r->bg_id){
        size_t n = (size_t)sky->w * sky->h;
        if (n * 16 > r->g->max_storage) return 0;
        float *d = malloc(n * 16);
        if (!d) return 0;
        for (size_t i = 0; i < n; i++){ d[4 * i] = sky->rgb[3 * i]; d[4 * i + 1] = sky->rgb[3 * i + 1]; d[4 * i + 2] = sky->rgb[3 * i + 2]; d[4 * i + 3] = 0; }
        rebind |= gpu_buffer_reserve(r->g, &r->bg, &r->bg_cap, n * 16, st, "scene: the sky");
        wgpuQueueWriteBuffer(r->g->queue, r->bg, 0, d, n * 16);
        free(d);
        r->bg_id = id;
    }
    if (rebind) rebind_all(r);
    return 1;
}

// the irradiance grid, when it changes (scene/env.h IrrGrid: per probe 28 floats = 7 vec4)
static int upload_irr(GpuRaster *r, const IrrGrid *ig){
    if (!ig || ig->id == r->ig_id) return 1;
    size_t bytes = (size_t)ig->n[0] * ig->n[1] * ig->n[2] * IRR_FLOATS * 4;
    if (gpu_buffer_reserve(r->g, &r->ig, &r->ig_cap, bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "scene: the irradiance grid"))
        rebind_all(r);
    wgpuQueueWriteBuffer(r->g->queue, r->ig, 0, ig->sh, bytes);
    r->ig_id = ig->id;
    return 1;
}

int gpu_raster_draw(GpuRaster *r, WGPUCommandEncoder enc, const Scene *s, const RenderParams *p){
    Gpu *g = r->g;
    if (gpu_textures_sync(g, &r->gt, s)){
        wgpuQueueWriteBuffer(g->queue, r->txt, 0, r->gt.table, sizeof(r->gt.table));
        rebind_all(r);
    }
    if (!upload_instances(r, s) || !upload_unit_shapes(r) || !upload_prims(r, s) || !upload_mesh(r, s)) return 0;
    RenderLamp L = render_scene_lamp(s);
    const Env *sky = p->mode == MODE_WIREFRAME ? NULL : s->env;
    if (!upload_env(r, L.env, sky) || !upload_irr(r, L.irr)) return 0;
    RenderMode mode = p->mode == MODE_TEXTURED ? MODE_PHONG : p->mode;

    GRU U;
    memset(&U, 0, sizeof(U));
    Mat4 vp = mat4_mul(p->proj, p->view);
    if (!order_instances(r, s, p, vp)) return 0;
    for (int c = 0; c < 4; c++)
        for (int k = 0; k < 4; k++) U.vp[c * 4 + k] = (float)vp.m[k][c];      // WGSL matrices are stored by columns
    put(U.eye, p->eye, mode == MODE_WIREFRAME ? 0.0f : (mode == MODE_FLAT ? 1.0f : (mode == MODE_GOURAUD ? 2.0f : 3.0f)));
    put(U.lamp, L.q, L.on ? 1.0f : 0.0f);
    put(U.lamp + 4, L.u, (float)L.area);
    put(U.lamp + 8, L.v, 0);
    put(U.lamp + 12, L.n, 0);
    put(U.lamp + 16, L.w, 0);
    put(U.lamp + 20, L.center, 0);
    put(U.lamp + 24, L.le, 0);
    U.lamp[28] = (float)L.ambient; U.lamp[29] = (float)L.fill;
    if (L.env){
        const Env *e = L.env;
        U.envp[0] = (float)e->scale; U.envp[1] = (float)r_cos(e->rot); U.envp[2] = (float)r_sin(e->rot); U.envp[3] = 1;
        put(U.envc, e->center, e->boxed ? 1.0f : 0.0f);
        put(U.envlo, e->bmin, 0);
        put(U.envhi, e->bmax, 0);
        for (int k = 0; k < ENV_LEVELS; k++){ U.lv[k][0] = r->lv_at[k]; U.lv[k][1] = (uint32_t)e->lw[k]; U.lv[k][2] = (uint32_t)e->lh[k]; }
        for (int i = 0; i < 9; i++) for (int k = 0; k < 3; k++) U.sh[i][k] = e->sh[i][k];
    } else U.envp[1] = 1;
    if (sky){
        U.sky[0] = (float)sky->scale; U.sky[1] = (float)r_cos(sky->rot); U.sky[2] = (float)r_sin(sky->rot); U.sky[3] = 1;
        U.skyd[0] = (uint32_t)sky->w; U.skyd[1] = (uint32_t)sky->h;
    }
    put(U.cam_r, vec3(p->view.m[0][0], p->view.m[0][1], p->view.m[0][2]), (float)(1 / p->proj.m[0][0]));
    put(U.cam_u, vec3(p->view.m[1][0], p->view.m[1][1], p->view.m[1][2]), (float)(1 / p->proj.m[1][1]));
    put(U.cam_b, vec3(p->view.m[2][0], p->view.m[2][1], p->view.m[2][2]), 0);
    U.vps[0] = (float)r->w; U.vps[1] = (float)r->h;
    if (L.irr){
        for (int k = 0; k < 3; k++) U.irrn[k] = (uint32_t)L.irr->n[k];
        U.irrn[3] = 1;
        put(U.irrlo, L.irr->lo, 0);
        put(U.irrhi, L.irr->hi, 0);
    }
    for (int m = 0; m < s->nmats && m < MAX_MATS; m++){
        RenderMaterial rm = render_scene_material(&s->mats[m]);
        Material res = material_resolved(&s->mats[m]);
        put(U.mats[m], rm.constant ? rm.base : res.albedo, rm.constant ? 1.0f : 0.0f);
        U.mats[m][4] = (float)res.metallic;
        U.mats[m][5] = (float)res.roughness;
        U.mats[m][6] = (float)res.ior;
        U.mats[m][7] = (float)res.transmission;
        // the textures the GPU has (the first GPU_TEX_MAX), as 1 + index; the scale's bits
        const Material *mm = &s->mats[m];
        uint32_t tx[4];
        gpu_tex_pack(mm, r->gt.ntex, tx);
        if (rm.constant) tx[0] = tx[1] = tx[2] = 0;
        memcpy(&U.mats[m][8], tx, 16);
        put(U.mats[m] + 12, material_glows(mm) && !rm.constant ? mm->emission : vec3(0, 0, 0), 0);
    }
    wgpuQueueWriteBuffer(g->queue, r->uni, 0, &U, sizeof(U));

    WGPURenderPassColorAttachment ca;
    memset(&ca, 0, sizeof(ca));
    ca.view = r->color;
    ca.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
    ca.loadOp = WGPULoadOp_Clear;
    ca.storeOp = WGPUStoreOp_Store;
    ca.clearValue = (WGPUColor){0, 0, 0, 1};
    WGPURenderPassDepthStencilAttachment da;
    memset(&da, 0, sizeof(da));
    da.view = r->depth;
    da.depthLoadOp = WGPULoadOp_Clear;
    da.depthStoreOp = WGPUStoreOp_Store;
    da.depthClearValue = 1.0f;
    WGPURenderPassDescriptor rp;
    memset(&rp, 0, sizeof(rp));
    rp.colorAttachmentCount = 1;
    rp.colorAttachments = &ca;
    rp.depthStencilAttachment = &da;
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rp);
    const int wire = mode == MODE_WIREFRAME;
    wgpuRenderPassEncoderSetPipeline(pass, wire ? r->wire : r->fill[p->cull_backfaces != 0]);
    Geo *sets[2] = {&r->prims, &r->mesh};
    for (int k = 0; k < 2; k++){
        if (!sets[k]->nverts || !sets[k]->bg) continue;
        wgpuRenderPassEncoderSetBindGroup(pass, 0, sets[k]->bg, 0, NULL);
        wgpuRenderPassEncoderDraw(pass, wire ? sets[k]->nverts * 2 : sets[k]->nverts, 1, 0, 0);
    }
    // the instances in view: their mesh's triangles (or a simpler copy's), moved by their
    // record; one draw per bucket (order_instances: draw instance 1 + j is ORD[1 + j]'s record)
    for (int k = 0, nk = (s->nmeshes + 2) * LODS; k < nk; k++){
        int first = k ? r->bucket[k - 1] : 0, count = r->bucket[k] - first;
        if (count <= 0) continue;
        int mesh = k / LODS - 2, l = k % LODS;
        Geo *geo = mesh >= 0 ? (l ? &r->mg[mesh].lod[l - 1] : &r->mg[mesh].geo) : &r->unit[mesh == INST_SPHERE ? 0 : 1];
        if (!geo->nverts || !geo->bg) continue;
        wgpuRenderPassEncoderSetBindGroup(pass, 0, geo->bg, 0, NULL);
        wgpuRenderPassEncoderDraw(pass, wire ? geo->nverts * 2 : geo->nverts, (uint32_t)count, 0, (uint32_t)first + 1);
    }
    // the sky behind (any set's bind group: the uniforms and buffers are the same in all)
    if (sky){
        Geo *any = r->prims.bg ? &r->prims : (r->mesh.bg ? &r->mesh : &r->unit[0]);
        if (any->bg){
            wgpuRenderPassEncoderSetPipeline(pass, r->sky);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, any->bg, 0, NULL);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        }
    }
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
    return 1;
}

void gpu_raster_destroy(GpuRaster *r){
    if (!r) return;
    int nsets = 4 + r->nmg * LODS;
    for (int k = 0; k < nsets; k++){
        int j = k - 4;
        Geo *geo = k == 0 ? &r->prims : (k == 1 ? &r->mesh : (k < 4 ? &r->unit[k - 2]
                 : (j % LODS ? &r->mg[j / LODS].lod[j % LODS - 1] : &r->mg[j / LODS].geo)));
        if (geo->bg) wgpuBindGroupRelease(geo->bg);
        if (geo->pos){ wgpuBufferDestroy(geo->pos); wgpuBufferRelease(geo->pos); }
        if (geo->nrm){ wgpuBufferDestroy(geo->nrm); wgpuBufferRelease(geo->nrm); }
        if (geo->tex){ wgpuBufferDestroy(geo->tex); wgpuBufferRelease(geo->tex); }
    }
    free(r->mg);
    free(r->rec);
    free(r->ordv);
    free(r->bucket);
    if (r->ord){ wgpuBufferDestroy(r->ord); wgpuBufferRelease(r->ord); }
    if (r->ix){ wgpuBufferDestroy(r->ix); wgpuBufferRelease(r->ix); }
    if (r->fill[0]) wgpuRenderPipelineRelease(r->fill[0]);
    if (r->fill[1]) wgpuRenderPipelineRelease(r->fill[1]);
    if (r->wire) wgpuRenderPipelineRelease(r->wire);
    if (r->sky) wgpuRenderPipelineRelease(r->sky);
    if (r->lv){ wgpuBufferDestroy(r->lv); wgpuBufferRelease(r->lv); }
    if (r->ig){ wgpuBufferDestroy(r->ig); wgpuBufferRelease(r->ig); }
    if (r->txt){ wgpuBufferDestroy(r->txt); wgpuBufferRelease(r->txt); }
    gpu_textures_free(&r->gt);
    free(r->tmp_tex);
    if (r->bg){ wgpuBufferDestroy(r->bg); wgpuBufferRelease(r->bg); }
    if (r->bgl) wgpuBindGroupLayoutRelease(r->bgl);
    if (r->uni){ wgpuBufferDestroy(r->uni); wgpuBufferRelease(r->uni); }
    if (r->tab){ wgpuBufferDestroy(r->tab); wgpuBufferRelease(r->tab); }
    if (r->color) wgpuTextureViewRelease(r->color);
    if (r->depth) wgpuTextureViewRelease(r->depth);
    if (r->color_tex){ wgpuTextureDestroy(r->color_tex); wgpuTextureRelease(r->color_tex); }
    if (r->depth_tex){ wgpuTextureDestroy(r->depth_tex); wgpuTextureRelease(r->depth_tex); }
    free(r->prim_copy);
    free(r->tmp_pos);
    free(r->tmp_nrm);
    free(r);
}

#else
// built without webgpu.h: nothing here (gpu_view_present_scene returns 0)
typedef int raster_gpu_unused;
#endif
