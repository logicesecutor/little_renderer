// raster_gpu.c - the rasterizer of raster/render.c (render_scene), on the GPU.
//
// The same triangles (raster/render.h hands them out: 2 per quad, a latitude / longitude
// grid per sphere, the mesh triangles), the same point light and material model, the same
// four shading modes, drawn by a WebGPU render pipeline into a color and a depth texture
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

static const char *SHADER = WGSL(
struct RMat {
    base: vec4f,     /* color; w = 1: drawn as it is (lamps) */
    spec: vec4f,     /* highlight: strength, exponent */
};
struct RU {
    vp: mat4x4f,     /* world -> clip, z in -w..w as on the CPU */
    eye: vec4f,      /* w = mode: 0 wireframe, 1 flat, 2 gouraud, 3 phong */
    lamp: vec4f,     /* the point light */
    k: vec4f,        /* weights: ambient, lamp, fill light from the camera */
    mats: array<RMat, 16>,
};
@group(0) @binding(0) var<uniform> U: RU;
@group(0) @binding(1) var<storage, read> vpos: array<vec4f>;
@group(0) @binding(2) var<storage, read> vnrm: array<vec4f>;

struct VO {
    @builtin(position) pos: vec4f,
    @location(0) wp: vec3f,
    @location(1) n: vec3f,
    @location(2) light: f32,
    @location(3) @interpolate(flat) mat: u32,
};

/* brightness from the lamp (diff) and the fill light; n faces the eye */
fn scene_k(diff: f32, n: vec3f, to_eye: vec3f) -> f32 {
    return U.k.x + U.k.y * max(diff, 0.0) + U.k.z * dot(n, to_eye);
}
/* surfaces are lit from both sides: the normal is turned towards the eye first */
fn scene_diffuse(p: vec3f, n0: vec3f) -> f32 {
    let to_eye = normalize(U.eye.xyz - p);
    var n = n0;
    if (dot(n, to_eye) < 0.0) { n = -n; }
    return scene_k(dot(n, normalize(U.lamp.xyz - p)), n, to_eye);
}
/* WebGPU keeps z in 0..w: (z + w) / 2 is the same depth the CPU z-buffer stores */
fn clip_of(p: vec3f) -> vec4f {
    let c = U.vp * vec4f(p, 1.0);
    return vec4f(c.x, c.y, (c.z + c.w) * 0.5, c.w);
}
fn to_srgb(c0: vec3f) -> vec3f {
    let c = clamp(c0, vec3f(0.0), vec3f(1.0));
    return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VO {
    let t = (vi / 3u) * 3u;
    let p0 = vpos[t].xyz;
    let p1 = vpos[t + 1u].xyz;
    let p2 = vpos[t + 2u].xyz;
    let code = u32(vpos[vi].w);
    let mode = u32(U.eye.w);
    let face_n = normalize(cross(p1 - p0, p2 - p0));
    var n = face_n;
    if ((code & 64u) != 0u && mode != 1u) { n = vnrm[vi].xyz; }
    let p = vpos[vi].xyz;
    var o: VO;
    o.pos = clip_of(p);
    o.wp = p;
    o.n = n;
    o.mat = code & 15u;
    o.light = 0.0;
    if (mode == 1u) {
        let flag = (code >> 4u) & 3u;
        var at = (p0 + p1 + p2) * (1.0 / 3.0);
        if (flag == 1u) { at = (p0 + p2) * 0.5; }
        if (flag == 2u) { at = (p0 + p1) * 0.5; }
        o.light = scene_diffuse(at, face_n);
    } else if (mode == 2u) {
        o.light = scene_diffuse(p, n);
    }
    return o;
}

@fragment
fn fs(i: VO) -> @location(0) vec4f {
    let m = U.mats[i.mat];
    if (m.base.w > 0.5) { return vec4f(to_srgb(m.base.rgb), 1.0); }
    /* flat and gouraud: the vertex stage computed the brightness */
    if (u32(U.eye.w) != 3u) { return vec4f(to_srgb(m.base.rgb * i.light), 1.0); }
    /* phong: per-pixel diffuse + Blinn-Phong highlight */
    var n = normalize(i.n);
    let to_eye = normalize(U.eye.xyz - i.wp);
    if (dot(n, to_eye) < 0.0) { n = -n; }
    let l = normalize(U.lamp.xyz - i.wp);
    var diff = dot(n, l);
    var spec = 0.0;
    if (diff > 0.0) {
        let nh = dot(n, normalize(l + to_eye));
        if (m.spec.x > 0.0 && nh > 0.0) { spec = m.spec.x * pow(nh, m.spec.y); }
    } else {
        diff = 0.0;
    }
    let k = scene_k(diff, n, to_eye);
    return vec4f(to_srgb(m.base.rgb * k + spec), 1.0);
}

/* wireframe: the 3 edges of every triangle as lines, 6 vertices per triangle */
struct WO {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) mat: u32,
};
@vertex
fn vs_wire(@builtin(vertex_index) vi: u32) -> WO {
    let tri = vi / 6u;
    let e = (vi % 6u) / 2u;
    let corner = (e + (vi & 1u)) % 3u;
    let q = vpos[tri * 3u + corner];
    var o: WO;
    o.pos = clip_of(q.xyz);
    o.mat = u32(q.w) & 15u;
    return o;
}
@fragment
fn fs_wire(i: WO) -> @location(0) vec4f { return vec4f(to_srgb(U.mats[i.mat].base.rgb), 1.0); }
);

typedef struct {
    float vp[16], eye[4], lamp[4], k[4];
    float mats[MAX_MATS][8];
} GRU;

// one set of triangles on the GPU
typedef struct {
    WGPUBuffer pos, nrm;
    size_t pos_cap, nrm_cap;
    WGPUBindGroup bg;
    uint32_t nverts;
} Geo;

struct GpuRaster {
    Gpu *g;
    int w, h;
    WGPUTexture color_tex, depth_tex;
    WGPUTextureView color, depth;
    WGPUBuffer uni;
    WGPUBindGroupLayout bgl;
    WGPURenderPipeline fill[2], wire;      // fill[1]: back faces culled
    Geo prims, mesh;                       // spheres and quads (they move); mesh triangles
    // what is on the GPU now
    Primitive *prim_copy;
    int nprim_copy;
    const Triangle *mesh_tris;
    const Vec3 *mesh_normals;
    int mesh_ntris;
    float *tmp_pos, *tmp_nrm;              // scratch for the uploads
    size_t tmp_cap;                        // vertices
};

#define CHUNK_TRIS 32768   // mesh triangles converted and uploaded at a time

static WGPURenderPipeline make_pipeline(Gpu *g, WGPUShaderModule m, WGPUBindGroupLayout bgl, int wire, int cull){
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
    fs.entryPoint = gpu_str(wire ? "fs_wire" : "fs");
    fs.targetCount = 1;
    fs.targets = &target;
    // the CPU's z-buffer: the nearest wins, the first drawn wins a tie. Its wireframe draws
    // lines over everything and leaves the z-buffer alone.
    WGPUDepthStencilState ds;
    memset(&ds, 0, sizeof(ds));
    ds.format = WGPUTextureFormat_Depth32Float;
    ds.depthWriteEnabled = wire ? WGPUOptionalBool_False : WGPUOptionalBool_True;
    ds.depthCompare = wire ? WGPUCompareFunction_Always : WGPUCompareFunction_Less;
    ds.stencilFront.compare = ds.stencilBack.compare = WGPUCompareFunction_Always;
    ds.stencilFront.failOp = ds.stencilBack.failOp = WGPUStencilOperation_Keep;
    ds.stencilFront.depthFailOp = ds.stencilBack.depthFailOp = WGPUStencilOperation_Keep;
    ds.stencilFront.passOp = ds.stencilBack.passOp = WGPUStencilOperation_Keep;
    WGPURenderPipelineDescriptor rd;
    memset(&rd, 0, sizeof(rd));
    rd.label = gpu_str(wire ? "scene wireframe" : "scene");
    rd.layout = pl;
    rd.vertex.module = m;
    rd.vertex.entryPoint = gpu_str(wire ? "vs_wire" : "vs");
    rd.primitive.topology = wire ? WGPUPrimitiveTopology_LineList : WGPUPrimitiveTopology_TriangleList;
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
    r->bgl = gpu_layout(g, WGPUShaderStage_Vertex | WGPUShaderStage_Fragment, "URR");
    WGPUShaderModule m = gpu_shader(g, "scene", SHADER);
    r->fill[0] = make_pipeline(g, m, r->bgl, 0, 0);
    r->fill[1] = make_pipeline(g, m, r->bgl, 0, 1);
    r->wire = make_pipeline(g, m, r->bgl, 1, 0);
    wgpuShaderModuleRelease(m);
    return r;
}

WGPUTextureView gpu_raster_color(GpuRaster *r){ return r->color; }
WGPUTextureView gpu_raster_depth(GpuRaster *r){ return r->depth; }

static void put(float *d, Vec3 a, float w){ d[0] = (float)a.x; d[1] = (float)a.y; d[2] = (float)a.z; d[3] = w; }

static int tmp_reserve(GpuRaster *r, size_t verts){
    if (r->tmp_cap >= verts) return 1;
    float *a = realloc(r->tmp_pos, verts * 16), *b = realloc(r->tmp_nrm, verts * 16);
    if (a) r->tmp_pos = a;
    if (b) r->tmp_nrm = b;
    if (!a || !b) return 0;
    r->tmp_cap = verts;
    return 1;
}

// room for `verts` vertices in a set; its bind group follows the buffers
static void geo_reserve(GpuRaster *r, Geo *geo, size_t verts, const char *label){
    Gpu *g = r->g;
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    int grown = gpu_buffer_reserve(g, &geo->pos, &geo->pos_cap, verts * 16, st, label);
    grown |= gpu_buffer_reserve(g, &geo->nrm, &geo->nrm_cap, verts * 16, st, label);
    if (!grown && geo->bg) return;
    if (geo->bg) wgpuBindGroupRelease(geo->bg);
    WGPUBuffer b[3] = {r->uni, geo->pos, geo->nrm};
    geo->bg = gpu_bind(g, r->bgl, b, 3);
}

// the spheres and quads: few, and they move (the physics), so they are compared with what
// was uploaded last and sent again when anything changed
static int upload_prims(GpuRaster *r, const Scene *s){
    if (r->nprim_copy == s->nprims && r->prim_copy &&
        (s->nprims == 0 || memcmp(r->prim_copy, s->prims, (size_t)s->nprims * sizeof(Primitive)) == 0)) return 1;
    size_t ntri = 0;
    for (int i = 0; i < s->nprims; i++) ntri += (size_t)render_prim_tri_count(&s->prims[i]);
    if (!tmp_reserve(r, ntri * 3 + 3)) return 0;
    size_t nv = 0;
    for (int i = 0; i < s->nprims; i++){
        const Primitive *pr = &s->prims[i];
        int n = render_prim_tri_count(pr);
        for (int sub = 0; sub < n; sub++){
            Vec3 wp[3], wn[3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, flat_at;
            int has_n = 0;
            render_prim_triangle(pr, sub, wp, wn, &has_n, &flat_at);
            int flag = pr->type == PRIM_QUAD ? (sub == 0 ? 1 : 2) : 0;
            float code = (float)((pr->material & 15) + 16 * flag + 64 * (has_n != 0));
            for (int k = 0; k < 3; k++, nv++){
                put(r->tmp_pos + 4 * nv, wp[k], code);
                put(r->tmp_nrm + 4 * nv, wn[k], 0);
            }
        }
    }
    geo_reserve(r, &r->prims, nv ? nv : 3, "scene primitives");
    if (nv){
        wgpuQueueWriteBuffer(r->g->queue, r->prims.pos, 0, r->tmp_pos, nv * 16);
        wgpuQueueWriteBuffer(r->g->queue, r->prims.nrm, 0, r->tmp_nrm, nv * 16);
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

// the mesh triangles: many, and they stay where they are: uploaded once
static int upload_mesh(GpuRaster *r, const Scene *s){
    if (r->mesh_tris == s->tris && r->mesh_ntris == s->ntris && r->mesh_normals == s->normals) return 1;
    size_t nv = (size_t)s->ntris * 3;
    if (nv * 16 > r->g->max_storage){
        fprintf(stderr, "GPU: the mesh needs a %.0f MB buffer, this GPU allows %.0f MB: the CPU rasterizes\n",
                (double)nv * 16 / 1048576, (double)r->g->max_storage / 1048576);
        return 0;
    }
    if (nv){
        if (!tmp_reserve(r, (size_t)CHUNK_TRIS * 3)) return 0;
        geo_reserve(r, &r->mesh, nv, "scene mesh");
        for (int first = 0; first < s->ntris; first += CHUNK_TRIS){
            int n = s->ntris - first < CHUNK_TRIS ? s->ntris - first : CHUNK_TRIS;
            for (int i = 0; i < n; i++){
                const Triangle *t = &s->tris[first + i];
                int has_n = t->n[0] >= 0;
                float code = (float)((t->material & 15) + 64 * has_n);
                Vec3 wp[3] = {t->q, v_add(t->q, t->u), v_add(t->q, t->v)};
                for (int k = 0; k < 3; k++){
                    put(r->tmp_pos + 4 * (3 * (size_t)i + k), wp[k], code);
                    put(r->tmp_nrm + 4 * (3 * (size_t)i + k), has_n ? s->normals[t->n[k]] : vec3(0, 0, 0), 0);
                }
            }
            wgpuQueueWriteBuffer(r->g->queue, r->mesh.pos, (uint64_t)first * 48, r->tmp_pos, (size_t)n * 48);
            wgpuQueueWriteBuffer(r->g->queue, r->mesh.nrm, (uint64_t)first * 48, r->tmp_nrm, (size_t)n * 48);
        }
    }
    r->mesh.nverts = (uint32_t)nv;
    r->mesh_tris = s->tris;
    r->mesh_normals = s->normals;
    r->mesh_ntris = s->ntris;
    return 1;
}

int gpu_raster_draw(GpuRaster *r, WGPUCommandEncoder enc, const Scene *s, const RenderParams *p){
    Gpu *g = r->g;
    if (!upload_prims(r, s) || !upload_mesh(r, s)) return 0;
    RenderMode mode = p->mode == MODE_TEXTURED ? MODE_PHONG : p->mode;

    GRU U;
    memset(&U, 0, sizeof(U));
    Mat4 vp = mat4_mul(p->proj, p->view);
    for (int c = 0; c < 4; c++)
        for (int k = 0; k < 4; k++) U.vp[c * 4 + k] = (float)vp.m[k][c];      // WGSL matrices are stored by columns
    put(U.eye, p->eye, mode == MODE_WIREFRAME ? 0.0f : (mode == MODE_FLAT ? 1.0f : (mode == MODE_GOURAUD ? 2.0f : 3.0f)));
    put(U.lamp, render_scene_light(s), 0);
    U.k[0] = RENDER_SCENE_AMBIENT; U.k[1] = RENDER_SCENE_LAMP; U.k[2] = RENDER_SCENE_FILL;
    for (int m = 0; m < s->nmats && m < MAX_MATS; m++){
        RenderMaterial rm = render_scene_material(&s->mats[m]);
        put(U.mats[m], rm.base, rm.constant ? 1.0f : 0.0f);
        U.mats[m][4] = (float)rm.spec_k;
        U.mats[m][5] = (float)rm.spec_exp;
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
        if (!sets[k]->nverts) continue;
        wgpuRenderPassEncoderSetBindGroup(pass, 0, sets[k]->bg, 0, NULL);
        wgpuRenderPassEncoderDraw(pass, wire ? sets[k]->nverts * 2 : sets[k]->nverts, 1, 0, 0);
    }
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
    return 1;
}

void gpu_raster_destroy(GpuRaster *r){
    if (!r) return;
    Geo *sets[2] = {&r->prims, &r->mesh};
    for (int k = 0; k < 2; k++){
        if (sets[k]->bg) wgpuBindGroupRelease(sets[k]->bg);
        if (sets[k]->pos){ wgpuBufferDestroy(sets[k]->pos); wgpuBufferRelease(sets[k]->pos); }
        if (sets[k]->nrm){ wgpuBufferDestroy(sets[k]->nrm); wgpuBufferRelease(sets[k]->nrm); }
    }
    if (r->fill[0]) wgpuRenderPipelineRelease(r->fill[0]);
    if (r->fill[1]) wgpuRenderPipelineRelease(r->fill[1]);
    if (r->wire) wgpuRenderPipelineRelease(r->wire);
    if (r->bgl) wgpuBindGroupLayoutRelease(r->bgl);
    if (r->uni){ wgpuBufferDestroy(r->uni); wgpuBufferRelease(r->uni); }
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
