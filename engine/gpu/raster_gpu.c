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

static const char *SHADER = WGSL(
struct RMat {
    base: vec4f,     /* base colour (a lamp's: its colour at full brightness); w = 1: drawn as it is */
    pbr: vec4f,      /* metallic, roughness, index of refraction, transmission (material_resolved) */
};
/* the lamp (RenderLamp): the quad q + a u + b v, the side it shines to, w = (u x v) / |u x v|^2 */
struct RLamp {
    q: vec4f,        /* w = 1: there is a lamp */
    u: vec4f,        /* w = its area */
    v: vec4f,
    n: vec4f,
    w: vec4f,
    center: vec4f,
    le: vec4f,       /* its radiance */
    k: vec4f,        /* the ambient light: uniform, towards the camera */
};
struct RU {
    vp: mat4x4f,     /* world -> clip, z in -w..w as on the CPU */
    eye: vec4f,      /* w = mode: 0 wireframe, 1 flat, 2 gouraud, 3 phong */
    lamp: RLamp,
    mats: array<RMat, 16>,
};
@group(0) @binding(0) var<uniform> U: RU;
@group(0) @binding(1) var<storage, read> vpos: array<vec4f>;
@group(0) @binding(2) var<storage, read> vnrm: array<vec4f>;
/* where a draw's triangles are: record 0 the world itself, then one per instance (the
   draw's first instance): object -> world rows, the inverse's rows (normals), material */
struct IX { r0: vec4f, r1: vec4f, r2: vec4f, i0: vec4f, i1: vec4f, i2: vec4f, m: vec4u };
@group(0) @binding(3) var<storage, read> ix: array<IX>;
/* the BSDF's tables (gpu_bsdf_tables) */
struct RT { t: array<vec4f, 528> };
@group(0) @binding(4) var<uniform> TB: RT;
fn bsdf_tab(k: u32) -> f32 { return TB.t[k >> 2u][k & 3u]; }
fn to_world(k: u32, p: vec3f) -> vec3f {
    let x = ix[k];
    let q = vec4f(p, 1.0);
    return vec3f(dot(x.r0, q), dot(x.r1, q), dot(x.r2, q));
}
fn normal_to_world(k: u32, n: vec3f) -> vec3f {
    let x = ix[k];
    return normalize(n.x * x.i0.xyz + n.y * x.i1.xyz + n.z * x.i2.xyz);
}
fn mat_of(k: u32, code: u32) -> u32 { return select(code & 15u, ix[k].m.x & 15u, ix[k].m.x != 0xffffffffu); }

struct VO {
    @builtin(position) pos: vec4f,
    @location(0) wp: vec3f,
    @location(1) n: vec3f,
    @location(2) light: vec3f,
    @location(3) @interpolate(flat) mat: u32,
};

/* render_shade (raster/render.c): the lamp's corner i (q, q + u, q + u + v, q + v) */
fn lamp_corner(i: u32) -> vec3f {
    return U.lamp.q.xyz + select(vec3f(0.0), U.lamp.u.xyz, i == 1u || i == 2u) + select(vec3f(0.0), U.lamp.v.xyz, i >= 2u);
}
/* the lamp's form factor from p (normal n), the lamp clipped at p's horizon; an edge's
   theta / sin theta by Heitz and Hill's fit */
fn edge_theta_sin(x: f32) -> f32 {
    let y = abs(x);
    let t = (0.8543985 + (0.4965155 + 0.0145206 * y) * y) / (3.4175940 + (4.1616724 + y) * y);
    return select(0.5 * inverseSqrt(max(1.0 - x * x, 1e-7)) - t, t, x > 0.0);
}
fn lamp_form_factor(p: vec3f, n: vec3f) -> f32 {
    var o: array<vec3f, 8>;
    var k = 0u;
    for (var i = 0u; i < 4u; i++) {
        let a = lamp_corner(i) - p;
        let b = lamp_corner((i + 1u) & 3u) - p;
        let da = dot(a, n);
        let db = dot(b, n);
        if (da >= 0.0) { o[k] = normalize(a); k++; }
        if ((da >= 0.0) != (db >= 0.0)) { o[k] = normalize(a + (b - a) * (da / (da - db))); k++; }
    }
    if (k < 3u) { return 0.0; }
    var sum = 0.0;
    for (var i = 0u; i < k; i++) {
        let a = o[i];
        let b = o[(i + 1u) % k];
        sum += dot(cross(a, b), n) * edge_theta_sin(dot(a, b));
    }
    return abs(sum) * BSDF_PI;
}
/* the lamp's point nearest the ray from p along r */
fn lamp_nearest(p: vec3f, r: vec3f) -> vec3f {
    let ln = U.lamp.n.xyz;
    let q = U.lamp.q.xyz;
    let dn = dot(r, ln);
    let far = 4.0 * length(U.lamp.center.xyz - p);
    var t = far;
    if (dn < 0.0) { t = min(dot(q - p, ln) / dn, far); }
    t = max(t, 0.0);
    let y = p + r * t;
    let qx = y - ln * dot(y - q, ln) - q;
    let a = clamp(dot(U.lamp.w.xyz, cross(qx, U.lamp.v.xyz)), 0.0, 1.0);
    let b = clamp(dot(U.lamp.w.xyz, cross(U.lamp.u.xyz, qx)), 0.0, 1.0);
    return q + U.lamp.u.xyz * a + U.lamp.v.xyz * b;
}
/* the radiance p (normal n0, turned to the eye), material mi, sends to the eye */
fn scene_shade(mi: u32, p: vec3f, n0: vec3f) -> vec3f {
    let m = U.mats[mi];
    let v = normalize(U.eye.xyz - p);
    var n = n0;
    if (dot(n, v) < 0.0) { n = -n; }
    let cv = max(dot(n, v), 1e-4);
    var b = bsdf_make(m.base.rgb, m.pbr.x, m.pbr.y, m.pbr.z, m.pbr.w, true);
    var c = bsdf_albedo_uniform(&b, cv) * (U.lamp.k.x + U.lamp.k.y * cv);
    let center = U.lamp.center.xyz;
    let ln = U.lamp.n.xyz;
    if (U.lamp.q.w < 0.5 || dot(p - center, ln) <= 0.0) { return c; }
    let ff = lamp_form_factor(p, n);
    if (ff > 0.0) {
        let lc = normalize(center - p);
        c += bsdf_diffuse_lobes(&b, max(dot(n, lc), 0.0)) * U.lamp.le.rgb * ff;
    }
    let area = U.lamp.u.w;
    let tc = center - p;
    let dc2 = dot(tc, tc);
    let om = min(area * max(-dot(ln, tc), 0.0) / (dc2 * sqrt(dc2)), 2.0 * BSDF_PI);
    let lobe = 4.0 * BSDF_PI * b.alpha * b.alpha * cv;
    let w = lobe / (lobe + om);
    var spec = vec3f(0.0);
    if (w > 0.01 && lobe > 32.0 * om) {
        let lc = tc * inverseSqrt(dc2);
        let ck = dot(n, lc);
        if (ck > 0.0) { spec = bsdf_spec_lobe(&b, n, v, lc, b.alpha) * (w * ck * om); }
    } else if (w > 0.01) {
        for (var k = 0u; k < 4u; k++) {
            let xk = U.lamp.q.xyz + U.lamp.u.xyz * select(0.25, 0.75, (k & 1u) != 0u) + U.lamp.v.xyz * select(0.25, 0.75, (k & 2u) != 0u);
            let dk = xk - p;
            let d2 = dot(dk, dk);
            let lk = dk * inverseSqrt(d2);
            let ck = dot(n, lk);
            let cosl = -dot(ln, lk);
            if (ck > 0.0 && cosl > 0.0) { spec += bsdf_spec_lobe(&b, n, v, lk, b.alpha) * (w * ck * cosl * area * 0.25 / d2); }
        }
    }
    if (w < 0.99) {
        let l = normalize(lamp_nearest(p, reflect(-v, n)) - p);
        let cl = dot(n, l);
        if (cl > 0.0) { spec += bsdf_spec_lobe(&b, n, v, l, b.alpha) * ((1.0 - w) * cl * om * lobe / (om + lobe)); }
    }
    return c + spec * U.lamp.le.rgb;
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
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) k: u32) -> VO {
    let t = (vi / 3u) * 3u;
    let p0 = to_world(k, vpos[t].xyz);
    let p1 = to_world(k, vpos[t + 1u].xyz);
    let p2 = to_world(k, vpos[t + 2u].xyz);
    let code = u32(vpos[vi].w);
    let mode = u32(U.eye.w);
    let face_n = normalize(cross(p1 - p0, p2 - p0));
    var n = face_n;
    if ((code & 64u) != 0u && mode != 1u) { n = normal_to_world(k, vnrm[vi].xyz); }
    let p = to_world(k, vpos[vi].xyz);
    var o: VO;
    o.pos = clip_of(p);
    o.wp = p;
    o.n = n;
    o.mat = mat_of(k, code);
    o.light = vec3f(0.0);
    if (U.mats[o.mat].base.w > 0.5) { return o; }
    if (mode == 1u) {
        let flag = (code >> 4u) & 3u;
        var at = (p0 + p1 + p2) * (1.0 / 3.0);
        if (flag == 1u) { at = (p0 + p2) * 0.5; }
        if (flag == 2u) { at = (p0 + p1) * 0.5; }
        o.light = scene_shade(o.mat, at, face_n);
    } else if (mode == 2u) {
        o.light = scene_shade(o.mat, p, n);
    }
    return o;
}

@fragment
fn fs(i: VO) -> @location(0) vec4f {
    let m = U.mats[i.mat];
    if (m.base.w > 0.5) { return vec4f(to_srgb(m.base.rgb), 1.0); }
    /* flat and gouraud: the vertex stage shaded */
    if (u32(U.eye.w) != 3u) { return vec4f(to_srgb(i.light), 1.0); }
    /* phong: render_shade at every pixel */
    return vec4f(to_srgb(scene_shade(i.mat, i.wp, normalize(i.n))), 1.0);
}

/* wireframe: the 3 edges of every triangle as lines, 6 vertices per triangle */
struct WO {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) mat: u32,
};
@vertex
fn vs_wire(@builtin(vertex_index) vi: u32, @builtin(instance_index) k: u32) -> WO {
    let tri = vi / 6u;
    let e = (vi % 6u) / 2u;
    let corner = (e + (vi & 1u)) % 3u;
    let q = vpos[tri * 3u + corner];
    var o: WO;
    o.pos = clip_of(to_world(k, q.xyz));
    o.mat = mat_of(k, u32(q.w));
    return o;
}
@fragment
fn fs_wire(i: WO) -> @location(0) vec4f { return vec4f(to_srgb(U.mats[i.mat].base.rgb), 1.0); }
);

typedef struct {
    float vp[16], eye[4], lamp[32];
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
    WGPUBuffer uni, tab;                   // the uniforms; the BSDF's tables
    WGPUBindGroupLayout bgl;
    WGPURenderPipeline fill[2], wire;      // fill[1]: back faces culled
    Geo prims, mesh;                       // spheres and quads (they move); mesh triangles
    WGPUBuffer ix;                         // the draws' transforms: the world, the instances
    size_t ix_cap;
    struct { Geo geo; const Triangle *tris; int ntris; unsigned version; } *mg;   // the scene's meshes
    int nmg;
    Geo unit[2];                           // the unit sphere and box (shape instances: the physics' bodies)
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
    float tab[GPU_BSDF_TABLE_FLOATS];
    gpu_bsdf_tables(tab);
    r->tab = gpu_buffer(g, "scene: BSDF tables", sizeof(tab), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    wgpuQueueWriteBuffer(g->queue, r->tab, 0, tab, sizeof(tab));
    r->bgl = gpu_layout(g, WGPUShaderStage_Vertex | WGPUShaderStage_Fragment, "URRRU");
    char *src = malloc(strlen(GPU_WGSL_BSDF) + strlen(SHADER) + 2);
    if (!src){ gpu_raster_destroy(r); return NULL; }
    strcpy(src, GPU_WGSL_BSDF); strcat(src, "\n"); strcat(src, SHADER);
    WGPUShaderModule m = gpu_shader(g, "scene", src);
    free(src);
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

static void geo_bind(GpuRaster *r, Geo *geo){
    if (geo->bg) wgpuBindGroupRelease(geo->bg);
    geo->bg = NULL;
    if (!geo->pos || !r->ix) return;
    WGPUBuffer b[5] = {r->uni, geo->pos, geo->nrm, r->ix, r->tab};
    geo->bg = gpu_bind(r->g, r->bgl, b, 5);
}

// room for `verts` vertices in a set; its bind group follows the buffers
static void geo_reserve(GpuRaster *r, Geo *geo, size_t verts, const char *label){
    Gpu *g = r->g;
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    int grown = gpu_buffer_reserve(g, &geo->pos, &geo->pos_cap, verts * 16, st, label);
    grown |= gpu_buffer_reserve(g, &geo->nrm, &geo->nrm_cap, verts * 16, st, label);
    if (!grown && geo->bg) return;
    geo_bind(r, geo);
}

// primitives as triangles into tmp_pos / tmp_nrm (from vertex nv on); returns the new count
static size_t prims_to_tris(GpuRaster *r, const Primitive *pr, int n, size_t nv){
    for (int i = 0; i < n; i++){
        int nt = render_prim_tri_count(&pr[i]);
        for (int sub = 0; sub < nt; sub++){
            Vec3 wp[3], wn[3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, flat_at;
            int has_n = 0;
            render_prim_triangle(&pr[i], sub, wp, wn, &has_n, &flat_at);
            int flag = pr[i].type == PRIM_QUAD ? (sub == 0 ? 1 : 2) : 0;
            float code = (float)((pr[i].material & 15) + 16 * flag + 64 * (has_n != 0));
            for (int k = 0; k < 3; k++, nv++){
                put(r->tmp_pos + 4 * nv, wp[k], code);
                put(r->tmp_nrm + 4 * nv, wn[k], 0);
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

// triangles into a set (as they are: the world's, or a mesh's in its own space)
static int upload_tris(GpuRaster *r, Geo *geo, const Triangle *tris, int ntris, const Vec3 *normals, const char *label){
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
            }
            wgpuQueueWriteBuffer(r->g->queue, geo->pos, (uint64_t)first * 48, r->tmp_pos, (size_t)n * 48);
            wgpuQueueWriteBuffer(r->g->queue, geo->nrm, (uint64_t)first * 48, r->tmp_nrm, (size_t)n * 48);
        }
    }
    geo->nverts = (uint32_t)nv;
    return 1;
}

// the world's mesh triangles: many, and they stay where they are: uploaded once
static int upload_mesh(GpuRaster *r, const Scene *s){
    if (r->mesh_tris == s->tris && r->mesh_ntris == s->ntris && r->mesh_normals == s->normals) return 1;
    if (!upload_tris(r, &r->mesh, s->tris, s->ntris, s->normals, "scene mesh")) return 0;
    r->mesh_tris = s->tris;
    r->mesh_normals = s->normals;
    r->mesh_ntris = s->ntris;
    return 1;
}

// the transforms (record 0: the world as it is; then the instances), and the meshes' own
// triangles (again when a dynamic one changed)
static int upload_instances(GpuRaster *r, const Scene *s){
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    size_t n = 1 + (size_t)s->ninsts;
    float *rec = calloc(n, 112);
    if (!rec) return 0;
    for (size_t i = 0; i < n; i++){
        Xform xf = i ? s->insts[i - 1].xf : xform_identity(), inv = i ? s->insts[i - 1].inv : xform_identity();
        float *o = rec + 28 * i;
        for (int a = 0; a < 3; a++)
            for (int k = 0; k < 4; k++){ o[4 * a + k] = (float)xf.m[a][k]; o[12 + 4 * a + k] = (float)inv.m[a][k]; }
        uint32_t m = i && s->insts[i - 1].material >= 0 ? (uint32_t)s->insts[i - 1].material : 0xffffffffu;
        memcpy(o + 24, &m, 4);
    }
    if (gpu_buffer_reserve(r->g, &r->ix, &r->ix_cap, n * 112, st, "raster: transforms")){
        geo_bind(r, &r->prims);
        geo_bind(r, &r->mesh);
        for (int m = 0; m < r->nmg; m++) geo_bind(r, &r->mg[m].geo);
        geo_bind(r, &r->unit[0]);
        geo_bind(r, &r->unit[1]);
    }
    wgpuQueueWriteBuffer(r->g->queue, r->ix, 0, rec, n * 112);
    free(rec);
    if (s->nmeshes > r->nmg){
        void *mg = realloc(r->mg, (size_t)s->nmeshes * sizeof(r->mg[0]));
        if (!mg) return 0;
        r->mg = mg;
        memset(r->mg + r->nmg, 0, (size_t)(s->nmeshes - r->nmg) * sizeof(r->mg[0]));
        r->nmg = s->nmeshes;
    }
    for (int m = 0; m < s->nmeshes; m++){
        const SceneMesh *sm = &s->meshes[m];
        if (r->mg[m].geo.nverts && r->mg[m].tris == sm->tris && r->mg[m].ntris == sm->ntris && r->mg[m].version == sm->version)
            continue;
        if (!upload_tris(r, &r->mg[m].geo, sm->tris, sm->ntris, sm->normals, "instanced mesh")) return 0;
        r->mg[m].tris = sm->tris;
        r->mg[m].ntris = sm->ntris;
        r->mg[m].version = sm->version;
    }
    return 1;
}

int gpu_raster_draw(GpuRaster *r, WGPUCommandEncoder enc, const Scene *s, const RenderParams *p){
    Gpu *g = r->g;
    if (!upload_instances(r, s) || !upload_unit_shapes(r) || !upload_prims(r, s) || !upload_mesh(r, s)) return 0;
    RenderMode mode = p->mode == MODE_TEXTURED ? MODE_PHONG : p->mode;

    GRU U;
    memset(&U, 0, sizeof(U));
    Mat4 vp = mat4_mul(p->proj, p->view);
    for (int c = 0; c < 4; c++)
        for (int k = 0; k < 4; k++) U.vp[c * 4 + k] = (float)vp.m[k][c];      // WGSL matrices are stored by columns
    put(U.eye, p->eye, mode == MODE_WIREFRAME ? 0.0f : (mode == MODE_FLAT ? 1.0f : (mode == MODE_GOURAUD ? 2.0f : 3.0f)));
    RenderLamp L = render_scene_lamp(s);
    put(U.lamp, L.q, L.on ? 1.0f : 0.0f);
    put(U.lamp + 4, L.u, (float)L.area);
    put(U.lamp + 8, L.v, 0);
    put(U.lamp + 12, L.n, 0);
    put(U.lamp + 16, L.w, 0);
    put(U.lamp + 20, L.center, 0);
    put(U.lamp + 24, L.le, 0);
    U.lamp[28] = (float)L.ambient; U.lamp[29] = (float)L.fill;
    for (int m = 0; m < s->nmats && m < MAX_MATS; m++){
        RenderMaterial rm = render_scene_material(&s->mats[m]);
        Material res = material_resolved(&s->mats[m]);
        put(U.mats[m], rm.constant ? rm.base : res.albedo, rm.constant ? 1.0f : 0.0f);
        U.mats[m][4] = (float)res.metallic;
        U.mats[m][5] = (float)res.roughness;
        U.mats[m][6] = (float)res.ior;
        U.mats[m][7] = (float)res.transmission;
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
    // the instances: their mesh's triangles, moved by their record (the draw's first instance)
    for (int i = 0; i < s->ninsts; i++){
        int mesh = s->insts[i].mesh;
        Geo *geo = mesh >= 0 ? &r->mg[mesh].geo : &r->unit[mesh == INST_SPHERE ? 0 : 1];
        if (!geo->nverts || !geo->bg) continue;
        wgpuRenderPassEncoderSetBindGroup(pass, 0, geo->bg, 0, NULL);
        wgpuRenderPassEncoderDraw(pass, wire ? geo->nverts * 2 : geo->nverts, 1, 0, (uint32_t)i + 1);
    }
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
    return 1;
}

void gpu_raster_destroy(GpuRaster *r){
    if (!r) return;
    int nsets = 4 + r->nmg;
    for (int k = 0; k < nsets; k++){
        Geo *geo = k == 0 ? &r->prims : (k == 1 ? &r->mesh : (k < 4 ? &r->unit[k - 2] : &r->mg[k - 4].geo));
        if (geo->bg) wgpuBindGroupRelease(geo->bg);
        if (geo->pos){ wgpuBufferDestroy(geo->pos); wgpuBufferRelease(geo->pos); }
        if (geo->nrm){ wgpuBufferDestroy(geo->nrm); wgpuBufferRelease(geo->nrm); }
    }
    free(r->mg);
    if (r->ix){ wgpuBufferDestroy(r->ix); wgpuBufferRelease(r->ix); }
    if (r->fill[0]) wgpuRenderPipelineRelease(r->fill[0]);
    if (r->fill[1]) wgpuRenderPipelineRelease(r->fill[1]);
    if (r->wire) wgpuRenderPipelineRelease(r->wire);
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
