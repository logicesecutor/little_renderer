#include "gpu/view_gpu.h"

#include <stdlib.h>

#ifndef LR_GPU
struct GpuView { int unused; };
GpuView *gpu_view_create(Gpu *g, int width, int height){ (void)g; (void)width; (void)height; return NULL; }
void gpu_view_present(GpuView *v, const Framebuffer *fb, const Scene *s, const RenderParams *p,
                      const MarchParams *mp, int volumes){ (void)v; (void)fb; (void)s; (void)p; (void)mp; (void)volumes; }
void gpu_view_destroy(GpuView *v){ (void)v; }
int gpu_view_capture(GpuView *v, GpuCaptureFn fn, void *user){ (void)v; (void)fn; (void)user; return 0; }
int gpu_view_present_scene(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp, int volumes){
    (void)v; (void)s; (void)p; (void)mp; (void)volumes;
    return 0;
}
int gpu_view_trace_start(GpuView *v, const Scene *s, const GpuTraceParams *tp){ (void)v; (void)s; (void)tp; return 0; }
int gpu_view_trace_busy(const GpuView *v){ (void)v; return 0; }
int gpu_view_trace_read(GpuView *v, GpuTraceReadFn fn, void *user){ (void)v; (void)fn; (void)user; return 0; }
void gpu_view_trace_more(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp){ (void)v; (void)s; (void)p; (void)mp; }
int gpu_view_trace_passes(const GpuView *v){ (void)v; return 0; }
double gpu_view_trace_pass_ms(const GpuView *v){ (void)v; return 0; }
int gpu_view_present_traced(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp, int volumes){
    (void)v; (void)s; (void)p; (void)mp; (void)volumes;
    return 0;
}
#else

#include <stdio.h>
#include <string.h>
#include "core/volume.h"
#include "gpu/gpu_internal.h"

// ---- the shaders -----------------------------------------------------------------------------
//
// The same model as raster/volume_march.c, line for line: density by trilinear interpolation
// (0 outside the grid), the majorant bricks to hurry through empty space, the light grid of
// optical depths to the lamp, the lamp's irradiance L A cos / (d^2 + A/pi), 3 octaves of
// multiple scattering, the walls' ambient light, energy-conserving steps.

#define TYPES GPU_WGSL_TYPES
#define COMMON GPU_WGSL_MEDIA
const char *const GPU_WGSL_TYPES = WGSL(
struct Vol {
    lo: vec4f, hi: vec4f,
    alo: vec4f, ahi: vec4f,
    albedo: vec4f,
    dims: vec4u,
    bdims: vec4u,
    ldims: vec4u,
    p: vec4f,
};
struct Frame {
    right: vec4f, up: vec4f, back: vec4f, eye: vec4f,
    proj: vec4f,
    size: vec4f,
    lamp_pos: vec4f,
    lamp_n: vec4f,
    lamp_L: vec4f,
    amb: vec4f,
    vols: array<Vol, 2>,
};
/* computed on the GPU from the densities (STATS): where the medium is, and its largest density */
struct VStat { alo: vec4f, ahi: vec4f, mx: vec4f, };
struct Sel { k: vec4u };
@group(0) @binding(0) var<uniform> F: Frame;
);

const char *const GPU_WGSL_MEDIA = WGSL(
@group(0) @binding(1) var<storage, read> dens: array<f32>;
@group(0) @binding(2) var<storage, read> bricks: array<f32>;

fn density(k: u32, p: vec3f) -> f32 {
    let v = F.vols[k];
    let n = vec3f(v.dims.xyz);
    let g = (p - v.lo.xyz) / (v.hi.xyz - v.lo.xyz) * n - 0.5;
    if (any(g <= vec3f(-1.0)) || any(g >= n)) { return 0.0; }
    let fl = floor(g);
    let i0 = vec3i(fl);
    let f = g - fl;
    let ni = vec3i(v.dims.xyz);
    var c: array<f32, 8>;
    for (var q = 0; q < 8; q++) {
        let i = i0 + vec3i(q & 1, (q >> 1) & 1, q >> 2);
        var val = 0.0;
        if (all(i >= vec3i(0)) && all(i < ni)) {
            val = dens[v.dims.w + (u32(i.z) * v.dims.y + u32(i.y)) * v.dims.x + u32(i.x)];
        }
        c[q] = val;
    }
    let c00 = mix(c[0], c[1], f.x); let c10 = mix(c[2], c[3], f.x);
    let c01 = mix(c[4], c[5], f.x); let c11 = mix(c[6], c[7], f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

fn brick_majorant(k: u32, p: vec3f) -> f32 {
    let v = F.vols[k];
    let g = (p - v.lo.xyz) / (v.hi.xyz - v.lo.xyz) * vec3f(v.dims.xyz);
    if (any(g < vec3f(0.0))) { return 0.0; }
    let b = vec3u(g / 8.0);
    if (any(b >= v.bdims.xyz)) { return 0.0; }
    return bricks[v.bdims.w + (b.z * v.bdims.y + b.y) * v.bdims.x + b.x];
}

fn ray_box(o: vec3f, inv: vec3f, lo: vec3f, hi: vec3f) -> vec2f {
    let t0 = (lo - o) * inv;
    let t1 = (hi - o) * inv;
    let a = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let b = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return vec2f(a, b);
}

fn safe_inv(d: vec3f) -> vec3f {
    return vec3f(select(1e30, 1.0 / d.x, abs(d.x) > 1e-12), select(1e30, 1.0 / d.y, abs(d.y) > 1e-12),
                 select(1e30, 1.0 / d.z, abs(d.z) > 1e-12));
}

fn vol_active(k: u32) -> bool { return VS[k].mx.x > 0.0; }
);

static const char *LIGHT = WGSL(
@group(0) @binding(3) var<storage, read_write> light: array<f32>;
@group(0) @binding(4) var<uniform> S: Sel;
@group(0) @binding(5) var<storage, read> VS: array<VStat, 2>;

/* optical depth of all media from p to the lamp (midpoint rule, about a voxel per step) */
fn optical_depth(a: vec3f, b: vec3f) -> f32 {
    let d = b - a;
    let len = length(d);
    if (len < 1e-6) { return 0.0; }
    let dir = d / len;
    let inv = safe_inv(dir);
    var tau = 0.0;
    for (var k = 0u; k < u32(F.size.w); k++) {
        if (!vol_active(k)) { continue; }
        let v = F.vols[k];
        let t = ray_box(a, inv, VS[k].alo.xyz, VS[k].ahi.xyz);
        let t0 = max(t.x, 0.0);
        let t1 = min(t.y, len);
        if (t1 <= t0) { continue; }
        var n = i32(ceil((t1 - t0) / v.p.w));
        n = clamp(n, 1, 256);
        let ds = (t1 - t0) / f32(n);
        var sum = 0.0;
        for (var i = 0; i < n; i++) { sum += density(k, a + dir * (t0 + (f32(i) + 0.5) * ds)); }
        tau += v.albedo.w * sum * ds;
    }
    return tau;
}

@compute @workgroup_size(4, 4, 4)
fn light_grid(@builtin(global_invocation_id) id: vec3u) {
    let k = S.k.x;
    let v = F.vols[k];
    if (any(id >= v.ldims.xyz)) { return; }
    let size = v.hi.xyz - v.lo.xyz;
    let cell = size / vec3f(v.ldims.xyz);
    let c = v.lo.xyz + (vec3f(id) + 0.5) * cell;
    let idx = v.ldims.w + (id.z * v.ldims.y + id.y) * v.ldims.x + id.x;
    if (!vol_active(k) || any(c < VS[k].alo.xyz - cell) || any(c > VS[k].ahi.xyz + cell)) { light[idx] = 0.0; return; }
    light[idx] = optical_depth(c, F.lamp_pos.xyz);
}
);

// the frame the media are drawn over: a framebuffer of the CPU renderers (two buffers)...
static const char *FRAME_CPU = WGSL(
@group(0) @binding(4) var<storage, read> color: array<u32>;
@group(0) @binding(5) var<storage, read> depthbuf: array<f32>;
fn frame_rgb(x: u32, y: u32) -> vec3f {
    let c = color[y * u32(F.size.x) + x];
    return vec3f(f32((c >> 16u) & 255u), f32((c >> 8u) & 255u), f32(c & 255u)) / 255.0;
}
fn frame_depth(x: u32, y: u32) -> f32 { return depthbuf[y * u32(F.size.x) + x]; }
fn traced(x: u32, y: u32) -> vec4f { return vec4f(0.0); }
);
// ...or what the GPU rasterizer drew (two textures)
static const char *FRAME_GPU = WGSL(
@group(0) @binding(4) var scene_color: texture_2d<f32>;
@group(0) @binding(5) var scene_depth: texture_depth_2d;
fn frame_rgb(x: u32, y: u32) -> vec3f { return textureLoad(scene_color, vec2u(x, y), 0).rgb; }
fn frame_depth(x: u32, y: u32) -> f32 { return textureLoad(scene_depth, vec2u(x, y), 0); }
/* the path tracer's samples so far (rgb = their sum, a = how many), when its image is shown */
@group(0) @binding(7) var<storage, read> accum: array<vec4f>;
fn traced(x: u32, y: u32) -> vec4f {
    if (F.right.w < 0.5) { return vec4f(0.0); }
    return accum[y * u32(F.size.x) + x];
}
);

static const char *COMPOSITE = WGSL(
@group(0) @binding(3) var<storage, read> light: array<f32>;
@group(0) @binding(6) var<storage, read> VS: array<VStat, 2>;

fn grid_tau(k: u32, p: vec3f) -> f32 {
    let v = F.vols[k];
    let n = vec3f(v.ldims.xyz);
    var g = (p - v.lo.xyz) / (v.hi.xyz - v.lo.xyz) * n - 0.5;
    g = clamp(g, vec3f(0.0), n - 1.001);
    let i = vec3u(g);
    let f = g - vec3f(i);
    let sy = v.ldims.x;
    let sz = v.ldims.x * v.ldims.y;
    let o = v.ldims.w + i.z * sz + i.y * sy + i.x;
    let c00 = mix(light[o], light[o + 1u], f.x);
    let c10 = mix(light[o + sy], light[o + sy + 1u], f.x);
    let c01 = mix(light[o + sz], light[o + sz + 1u], f.x);
    let c11 = mix(light[o + sz + sy], light[o + sz + sy + 1u], f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

fn phase_hg(c: f32, g: f32) -> f32 {
    let d = 1.0 + g * g - 2.0 * g * c;
    return (1.0 - g * g) / (4.0 * 3.14159265 * d * sqrt(d));
}

fn pixel_jitter(x: u32, y: u32) -> f32 {
    var h = (x * 0x8da6b343u) ^ (y * 0xd8163841u);
    h ^= h >> 13u; h *= 0x5bd1e995u; h ^= h >> 15u;
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

fn srgb_to_linear(c: vec3f) -> vec3f {
    return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}
fn linear_to_srgb(c0: vec3f) -> vec3f {
    let c = clamp(c0, vec3f(0.0), vec3f(1.0));
    return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

/* the ray from the eye along unit dir, stopped at tmax: the light it collects, and T */
fn march(dir: vec3f, tmax: f32, jit: f32) -> vec4f {
    let eye = F.eye.xyz;
    let inv = safe_inv(dir);
    let nv = u32(F.size.w);
    var seg0 = array<f32, 2>(1.0, 1.0);
    var seg1 = array<f32, 2>(0.0, 0.0);
    var ta = 1e30;
    var tb = 0.0;
    for (var k = 0u; k < nv; k++) {
        if (!vol_active(k)) { continue; }
        let t = ray_box(eye, inv, VS[k].alo.xyz, VS[k].ahi.xyz);
        let a = max(t.x, 0.0);
        let b = min(t.y, tmax);
        if (t.y < t.x || t.y <= 0.0 || b <= a) { continue; }
        seg0[k] = a; seg1[k] = b;
        ta = min(ta, a); tb = max(tb, b);
    }
    var L = vec3f(0.0);
    var T = 1.0;
    if (tb <= ta) { return vec4f(L, T); }
    let max_steps = F.size.z;
    let ds_floor = (tb - ta) / max_steps;
    var t = ta;
    var is_first = true;
    var steps = 0.0;
    let octaves = i32(F.amb.w);
    loop {
        if (t >= tb || T <= 0.003 || steps >= 4.0 * max_steps) { break; }
        steps += 1.0;
        var ds = 1e30;
        for (var k = 0u; k < nv; k++) { if (t >= seg0[k] && t <= seg1[k]) { ds = min(ds, F.vols[k].p.y); } }
        if (ds >= 1e30) { ds = select(1.0, ds_floor, ds_floor > 0.0); }
        ds = max(ds, ds_floor);
        if (is_first) { t += ds * jit; is_first = false; if (t >= tb) { break; } }
        let pos = eye + dir * t;
        var sig = 0.0;
        var src = vec3f(0.0);
        var any_here = false;
        var lamp_done = false;
        var E = vec3f(0.0);
        var cosv = 0.0;
        for (var k = 0u; k < nv; k++) {
            if (t < seg0[k] || t > seg1[k]) { continue; }
            if (brick_majorant(k, pos) <= 0.0) { continue; }
            any_here = true;
            let rho = density(k, pos);
            if (rho <= 0.0) { continue; }
            let v = F.vols[k];
            let sg = v.albedo.w * rho;
            if (!lamp_done) {
                let d = pos - F.lamp_pos.xyz;
                let d2 = dot(d, d);
                let len = sqrt(d2);
                let w = select(vec3f(0.0, -1.0, 0.0), d / len, len > 1e-6);
                var c = 1.0;
                if (F.lamp_n.w > 0.5) { c = dot(F.lamp_n.xyz, w); }
                if (c > 0.0) { E = F.lamp_L.xyz * (F.lamp_pos.w * c / (d2 + F.lamp_pos.w / 3.14159265)); }
                cosv = -dot(w, dir);
                lamp_done = true;
            }
            let tau = grid_tau(k, pos);
            var lamp_term = 0.0;
            var a = 1.0; var b = 1.0; var cc = 1.0;
            for (var o = 0; o < octaves; o++) {
                lamp_term += a * exp(-b * tau) * phase_hg(cosv, v.p.x * cc);
                a *= 0.7; b *= 0.3; cc *= 0.5;
            }
            let amb_t = 0.35 + 0.65 * exp(-0.25 * tau);
            let S = E * (lamp_term * F.lamp_L.w) + F.amb.xyz * amb_t;
            src += v.albedo.xyz * S * sg;
            sig += sg;
        }
        var stp = select(ds * 4.0, ds, any_here);
        for (var k = 0u; k < nv; k++) {
            if (seg0[k] > t && seg0[k] < t + stp && F.vols[k].p.y < stp) { stp = seg0[k] - t + 1e-3; }
        }
        if (sig > 0.0) {
            let st = exp(-sig * stp);
            L += src * (T * (1.0 - st) / sig);
            T *= st;
        }
        t += stp;
    }
    return vec4f(L, T);
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let x = u32(pos.x);
    let y = u32(pos.y);
    let pt = traced(x, y);
    if (pt.w > 0.0) { return vec4f(linear_to_srgb(pt.rgb / pt.w), 1.0); }
    var rgb = frame_rgb(x, y);
    if (F.size.w > 0.5) {
        /* the ray through the pixel center; at view depth D it is at eye + D * du */
        let nx = (2.0 * (f32(x) + 0.5) / F.size.x - 1.0) * F.proj.x;
        let ny = (1.0 - 2.0 * (f32(y) + 0.5) / F.size.y) * F.proj.y;
        let du = F.right.xyz * nx + F.up.xyz * ny - F.back.xyz;
        let dl = length(du);
        var tmax = 1e30;
        let d = frame_depth(x, y);
        if (d < 1.0) {
            let vd = F.proj.w / (F.proj.z + 2.0 * d - 1.0);
            if (vd > 0.0) { tmax = vd * dl; }
        }
        let r = march(du / dl, tmax, pixel_jitter(x, y));
        if (r.w < 1.0) { rgb = linear_to_srgb(r.xyz + srgb_to_linear(rgb) * r.w); }
    }
    return vec4f(rgb, 1.0);
}
);

/* What volume_update does on the CPU, for media whose densities live on the GPU (and, the
   same way, for all the others): the majorant of every brick (with the voxel of interpolation
   around it), and, with atomics, the box of the voxels with density > 0 and their maximum. */
static const char *STATS = WGSL(
@group(0) @binding(1) var<storage, read> dens: array<f32>;
@group(0) @binding(2) var<storage, read_write> bricks: array<f32>;
@group(0) @binding(3) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> S: Sel;

@compute @workgroup_size(64)
fn brick_stats(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let k = S.k.x;
    let v = F.vols[k];
    let id = gid.x + gid.y * nw.x * 64u;
    if (id >= v.bdims.x * v.bdims.y * v.bdims.z) { return; }
    let b = vec3i(i32(id % v.bdims.x), i32((id / v.bdims.x) % v.bdims.y), i32(id / (v.bdims.x * v.bdims.y)));
    let n = vec3i(v.dims.xyz);
    let lo = max(b * 8 - 1, vec3i(0));
    let hi = min(b * 8 + 8, n - 1);
    let own_lo = b * 8;
    let own_hi = min(b * 8 + 7, n - 1);
    var m = 0.0;
    var own_max = 0.0;
    var amin = n;
    var amax = vec3i(-1);
    for (var z = lo.z; z <= hi.z; z++) {
        for (var y = lo.y; y <= hi.y; y++) {
            for (var x = lo.x; x <= hi.x; x++) {
                let d = dens[v.dims.w + (u32(z) * v.dims.y + u32(y)) * v.dims.x + u32(x)];
                m = max(m, d);
                let q = vec3i(x, y, z);
                if (d > 0.0 && all(q >= own_lo) && all(q <= own_hi)) {
                    amin = min(amin, q);
                    amax = max(amax, q);
                    own_max = max(own_max, d);
                }
            }
        }
    }
    bricks[v.bdims.w + id] = m;
    if (amax.x >= 0) {
        let o = k * 8u;
        atomicMin(&acc[o], u32(amin.x)); atomicMin(&acc[o + 1u], u32(amin.y)); atomicMin(&acc[o + 2u], u32(amin.z));
        atomicMax(&acc[o + 3u], u32(amax.x + 1)); atomicMax(&acc[o + 4u], u32(amax.y + 1)); atomicMax(&acc[o + 5u], u32(amax.z + 1));
        atomicMax(&acc[o + 6u], bitcast<u32>(own_max));   /* >= 0: the bits order like the values */
    }
}
);

static const char *FINISH = WGSL(
@group(0) @binding(1) var<storage, read> acc: array<u32>;
@group(0) @binding(2) var<storage, read_write> VS: array<VStat, 2>;
@group(0) @binding(3) var<uniform> S: Sel;

@compute @workgroup_size(1)
fn vol_finish() {
    let k = S.k.x;
    let v = F.vols[k];
    let o = k * 8u;
    let mx = bitcast<f32>(acc[o + 6u]);
    if (acc[o + 3u] == 0u || !(mx > 0.0)) {
        VS[k].alo = vec4f(1.0); VS[k].ahi = vec4f(-1.0); VS[k].mx = vec4f(0.0);
        return;
    }
    let vox = (v.hi.xyz - v.lo.xyz) / vec3f(v.dims.xyz);
    let c0 = v.lo.xyz + (vec3f(f32(acc[o]), f32(acc[o + 1u]), f32(acc[o + 2u])) + 0.5) * vox;
    let c1 = v.lo.xyz + (vec3f(f32(acc[o + 3u]), f32(acc[o + 4u]), f32(acc[o + 5u])) - 0.5) * vox;
    /* trilinear interpolation spreads a voxel's density up to one voxel around its center */
    VS[k].alo = vec4f(max(c0 - vox, v.lo.xyz), 0.0);
    VS[k].ahi = vec4f(min(c1 + vox, v.hi.xyz), 0.0);
    VS[k].mx = vec4f(mx, 0.0, 0.0, 0.0);
}
);

// ---- the CPU side ---------------------------------------------------------------------------

// the uniform block, laid out as WGSL lays out struct Frame (16-byte rows)
typedef struct {
    float lo[4], hi[4], alo[4], ahi[4], albedo[4];
    uint32_t dims[4], bdims[4], ldims[4];
    float p[4];
} GVol;
typedef struct {
    float right[4], up[4], back[4], eye[4], proj[4], size[4];
    float lamp_pos[4], lamp_n[4], lamp_L[4], amb[4];
    GVol vols[GPU_MAX_VOLUMES];
} GFrame;

struct GpuView {
    Gpu *g;
    int w, h;
    WGPUBuffer frame_u, sel_u[GPU_MAX_VOLUMES];
    WGPUBuffer dens, bricks, light, color, depth;
    size_t dens_cap, bricks_cap, light_cap;
    WGPUBuffer vstat, acc;      // per medium: active box and max (VStat), and the atomics that make it
    WGPUComputePipeline light_pipe, stats_pipe, finish_pipe;
    WGPURenderPipeline comp_pipe, comp_gpu_pipe;   // over a CPU framebuffer / over the GPU rasterizer's frame
    // explicit: "auto" would drop the bindings a shader does not read
    WGPUBindGroupLayout light_bgl, comp_bgl, comp_gpu_bgl, stats_bgl, finish_bgl;
    GpuRaster *raster;          // the scene's surfaces drawn on the GPU (created when first used)
    int raster_failed;
    // the path tracer (created when first used) and the image it is making
    GpuTrace *trace;
    int trace_failed, show_traced;
    GpuTraceParams tparams;
    int t_clear;                // the accumulation buffer must be cleared before its next use
    int t_y, t_rows, t_reps;    // next row; rows per band and passes per submit (both adapt to the GPU)
    int t_passes;               // samples per pixel submitted
    int t_done, t_done_next;    // finished; and what that becomes when the running band ends
    int t_inflight;             // a band is running
    double t_submit_ms, t_pass_t0, t_pass_ms;
    // reading the samples back
    WGPUBuffer t_rb;
    int t_rb_pending;
    GpuTraceReadFn t_read_fn;
    void *t_read_user;
    // media whose densities are already on the GPU (a GPU fluid's cloud): copied, not uploaded
    const Volume *src_vol[GPU_MAX_SOURCES];
    WGPUBuffer src_buf[GPU_MAX_SOURCES];
    int nsrc;
    // what is on the GPU now
    const Volume *vols[GPU_MAX_VOLUMES];
    uint32_t versions[GPU_MAX_VOLUMES];
    int nvols, light_res;
    float lamp[3];
    GVol gv[GPU_MAX_VOLUMES];
    int light_valid;
    float *depth_f;     // the z-buffer as 32-bit floats (the double build stores doubles)
    // screenshot: the next frame is also drawn into tex, copied into readback, mapped
    GpuCaptureFn cap_fn;
    void *cap_user;
    int cap_pending;    // 1 = draw it next frame, 2 = waiting for the map
    WGPUTexture cap_tex;
    WGPUBuffer cap_buf;
    uint32_t cap_row;   // bytes per row in cap_buf (a multiple of 256)
};

// the views alive: a "work done" callback can arrive after gpu_view_destroy
static GpuView *live_views[4];
static int view_is_live(const GpuView *v){
    for (int i = 0; i < 4; i++) if (v && live_views[i] == v) return 1;
    return 0;
}

static void v4(float *d, Vec3 a, float w){ d[0] = (float)a.x; d[1] = (float)a.y; d[2] = (float)a.z; d[3] = w; }

// a bind group layout of n buffers, binding i of type types[i], seen by the given stage
static WGPUBindGroupLayout make_layout(Gpu *g, WGPUShaderStage stage, const WGPUBufferBindingType *types, int n){
    WGPUBindGroupLayoutEntry e[8];
    memset(e, 0, sizeof(e));
    for (int i = 0; i < n; i++){
        e[i].binding = (uint32_t)i;
        e[i].visibility = stage;
        e[i].buffer.type = types[i];
    }
    WGPUBindGroupLayoutDescriptor d;
    memset(&d, 0, sizeof(d));
    d.entryCount = (size_t)n;
    d.entries = e;
    return wgpuDeviceCreateBindGroupLayout(g->device, &d);
}

static WGPUPipelineLayout make_pipeline_layout(Gpu *g, WGPUBindGroupLayout bgl){
    WGPUPipelineLayoutDescriptor d;
    memset(&d, 0, sizeof(d));
    d.bindGroupLayoutCount = 1;
    d.bindGroupLayouts = &bgl;
    return wgpuDeviceCreatePipelineLayout(g->device, &d);
}

GpuView *gpu_view_create(Gpu *g, int width, int height){
    if (!g || g->state != GPU_READY || (!g->surface && !g->offscreen)) return NULL;
    GpuView *v = calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->g = g;
    v->w = width;
    v->h = height;
    for (int i = 0; i < 4; i++) if (!live_views[i]){ live_views[i] = v; break; }
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    v->frame_u = gpu_buffer(g, "frame", sizeof(GFrame), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    for (int k = 0; k < GPU_MAX_VOLUMES; k++){
        v->sel_u[k] = gpu_buffer(g, "light volume", 16, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
        uint32_t sel[4] = {(uint32_t)k, 0, 0, 0};
        wgpuQueueWriteBuffer(g->queue, v->sel_u[k], 0, sel, sizeof(sel));
    }
    v->color = gpu_buffer(g, "color", (size_t)width * height * 4, st);
    v->depth = gpu_buffer(g, "depth", (size_t)width * height * 4, st);
    gpu_buffer_reserve(g, &v->dens, &v->dens_cap, 1024, st, "density");
    gpu_buffer_reserve(g, &v->bricks, &v->bricks_cap, 1024, st, "bricks");
    gpu_buffer_reserve(g, &v->light, &v->light_cap, 1024, WGPUBufferUsage_Storage, "light grid");
    v->vstat = gpu_buffer(g, "media boxes", GPU_MAX_VOLUMES * 48, WGPUBufferUsage_Storage);
    v->acc = gpu_buffer(g, "media box atomics", GPU_MAX_VOLUMES * 32, st);
    v->depth_f = malloc((size_t)width * height * sizeof(float));
    if (!v->depth_f){ free(v); return NULL; }

    char *src = malloc(strlen(TYPES) + strlen(COMMON) + strlen(LIGHT) + strlen(COMPOSITE) + strlen(STATS) + strlen(FINISH) +
                       strlen(FRAME_CPU) + strlen(FRAME_GPU) + 16);
    if (!src){ free(v->depth_f); free(v); return NULL; }
    strcpy(src, TYPES); strcat(src, "\n"); strcat(src, COMMON); strcat(src, "\n"); strcat(src, LIGHT);
    WGPUShaderModule ml = gpu_shader(g, "light grid", src);
    strcpy(src, TYPES); strcat(src, "\n"); strcat(src, COMMON); strcat(src, "\n"); strcat(src, FRAME_CPU); strcat(src, "\n");
    strcat(src, COMPOSITE);
    WGPUShaderModule mc = gpu_shader(g, "composite", src);
    strcpy(src, TYPES); strcat(src, "\n"); strcat(src, COMMON); strcat(src, "\n"); strcat(src, FRAME_GPU); strcat(src, "\n");
    strcat(src, COMPOSITE);
    WGPUShaderModule mcg = gpu_shader(g, "composite (GPU frame)", src);
    strcpy(src, TYPES); strcat(src, "\n"); strcat(src, STATS);
    WGPUShaderModule ms = gpu_shader(g, "media stats", src);
    strcpy(src, TYPES); strcat(src, "\n"); strcat(src, FINISH);
    WGPUShaderModule mf = gpu_shader(g, "media boxes", src);
    free(src);

    const WGPUBufferBindingType U = WGPUBufferBindingType_Uniform, R = WGPUBufferBindingType_ReadOnlyStorage,
                                W = WGPUBufferBindingType_Storage;
    const WGPUBufferBindingType light_b[6] = {U, R, R, W, U, R};
    const WGPUBufferBindingType comp_b[7] = {U, R, R, R, R, R, R};
    const WGPUBufferBindingType stats_b[5] = {U, R, W, W, U};
    const WGPUBufferBindingType finish_b[4] = {U, R, W, U};
    v->light_bgl = make_layout(g, WGPUShaderStage_Compute, light_b, 6);
    v->comp_bgl = make_layout(g, WGPUShaderStage_Fragment, comp_b, 7);
    {   // the same, with bindings 4 and 5 as the rasterizer's color and depth textures
        WGPUBindGroupLayoutEntry e[8];
        memset(e, 0, sizeof(e));
        for (int i = 0; i < 8; i++){
            e[i].binding = (uint32_t)i;
            e[i].visibility = WGPUShaderStage_Fragment;
            if (i == 4 || i == 5){
                e[i].texture.sampleType = i == 4 ? WGPUTextureSampleType_Float : WGPUTextureSampleType_Depth;
                e[i].texture.viewDimension = WGPUTextureViewDimension_2D;
            } else e[i].buffer.type = i < 7 ? comp_b[i] : R;      // 7: the path tracer's samples
        }
        WGPUBindGroupLayoutDescriptor d;
        memset(&d, 0, sizeof(d));
        d.entryCount = 8;
        d.entries = e;
        v->comp_gpu_bgl = wgpuDeviceCreateBindGroupLayout(g->device, &d);
    }
    WGPUPipelineLayout comp_gpu_pl = make_pipeline_layout(g, v->comp_gpu_bgl);
    v->stats_bgl = make_layout(g, WGPUShaderStage_Compute, stats_b, 5);
    v->finish_bgl = make_layout(g, WGPUShaderStage_Compute, finish_b, 4);
    WGPUPipelineLayout light_pl = make_pipeline_layout(g, v->light_bgl);
    WGPUPipelineLayout comp_pl = make_pipeline_layout(g, v->comp_bgl);
    WGPUPipelineLayout stats_pl = make_pipeline_layout(g, v->stats_bgl);
    WGPUPipelineLayout finish_pl = make_pipeline_layout(g, v->finish_bgl);

    WGPUComputePipelineDescriptor cd;
    memset(&cd, 0, sizeof(cd));
    cd.label = gpu_str("light grid");
    cd.layout = light_pl;
    cd.compute.module = ml;
    cd.compute.entryPoint = gpu_str("light_grid");
    v->light_pipe = wgpuDeviceCreateComputePipeline(g->device, &cd);
    cd.label = gpu_str("media stats");
    cd.layout = stats_pl;
    cd.compute.module = ms;
    cd.compute.entryPoint = gpu_str("brick_stats");
    v->stats_pipe = wgpuDeviceCreateComputePipeline(g->device, &cd);
    cd.label = gpu_str("media boxes");
    cd.layout = finish_pl;
    cd.compute.module = mf;
    cd.compute.entryPoint = gpu_str("vol_finish");
    v->finish_pipe = wgpuDeviceCreateComputePipeline(g->device, &cd);

    WGPUColorTargetState target;
    memset(&target, 0, sizeof(target));
    target.format = g->surface_format;
    target.writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState fs;
    memset(&fs, 0, sizeof(fs));
    fs.module = mc;
    fs.entryPoint = gpu_str("fs");
    fs.targetCount = 1;
    fs.targets = &target;
    WGPURenderPipelineDescriptor rd;
    memset(&rd, 0, sizeof(rd));
    rd.label = gpu_str("composite");
    rd.layout = comp_pl;
    rd.vertex.module = mc;
    rd.vertex.entryPoint = gpu_str("vs");
    rd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    rd.multisample.count = 1;
    rd.multisample.mask = 0xFFFFFFFFu;
    rd.fragment = &fs;
    v->comp_pipe = wgpuDeviceCreateRenderPipeline(g->device, &rd);
    rd.label = gpu_str("composite (GPU frame)");
    rd.layout = comp_gpu_pl;
    rd.vertex.module = mcg;
    fs.module = mcg;
    v->comp_gpu_pipe = wgpuDeviceCreateRenderPipeline(g->device, &rd);
    wgpuShaderModuleRelease(mcg);
    wgpuPipelineLayoutRelease(comp_gpu_pl);
    wgpuShaderModuleRelease(ml);
    wgpuShaderModuleRelease(mc);
    wgpuShaderModuleRelease(ms);
    wgpuShaderModuleRelease(mf);
    wgpuPipelineLayoutRelease(light_pl);
    wgpuPipelineLayoutRelease(comp_pl);
    wgpuPipelineLayoutRelease(stats_pl);
    wgpuPipelineLayoutRelease(finish_pl);
    return v;
}

static WGPUBindGroupEntry entry(uint32_t binding, WGPUBuffer b){
    WGPUBindGroupEntry e;
    memset(&e, 0, sizeof(e));
    e.binding = binding;
    e.buffer = b;
    e.size = WGPU_WHOLE_SIZE;
    return e;
}

static WGPUBindGroup bind(Gpu *g, WGPUBindGroupLayout layout, const WGPUBindGroupEntry *e, size_t n){
    WGPUBindGroupDescriptor d;
    memset(&d, 0, sizeof(d));
    d.layout = layout;
    d.entryCount = n;
    d.entries = e;
    return wgpuDeviceCreateBindGroup(g->device, &d);
}

// densities, bricks and light grid layout of the scene's media; uploads what changed.
// Returns 1 if the light grids must be recomputed.
static WGPUBuffer gpu_source(const GpuView *v, const Volume *vol){
    for (int i = 0; i < v->nsrc; i++) if (v->src_vol[i] == vol) return v->src_buf[i];
    return NULL;
}

static int upload_volumes(GpuView *v, const Scene *s, const MarchParams *mp, MarchLamp *lamp, WGPUCommandEncoder enc){
    Gpu *g = v->g;
    int n = s->nvolumes < GPU_MAX_VOLUMES ? s->nvolumes : GPU_MAX_VOLUMES;
    int changed = n != v->nvols || mp->light_res != v->light_res ||
                  lamp->pos.x != v->lamp[0] || lamp->pos.y != v->lamp[1] || lamp->pos.z != v->lamp[2];
    for (int k = 0; k < n && !changed; k++)
        if (v->vols[k] != s->volumes[k] || v->versions[k] != s->volumes[k]->version) changed = 1;
    if (!changed) return !v->light_valid;
    size_t nd = 0, nb = 0, nl = 0;
    for (int k = 0; k < n; k++){
        const Volume *vol = s->volumes[k];
        int ld[3];
        march_light_dims(vol, mp->light_res, ld);
        nd += (size_t)vol->nx * vol->ny * vol->nz;
        nb += (size_t)vol->bnx * vol->bny * vol->bnz;
        nl += (size_t)ld[0] * ld[1] * ld[2];
    }
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    int fresh = gpu_buffer_reserve(g, &v->dens, &v->dens_cap, nd * 4, st, "density");
    gpu_buffer_reserve(g, &v->bricks, &v->bricks_cap, nb * 4, st, "bricks");
    gpu_buffer_reserve(g, &v->light, &v->light_cap, nl * 4, WGPUBufferUsage_Storage, "light grid");
    uint32_t od = 0, ob = 0, ol = 0;
    for (int k = 0; k < n; k++){
        const Volume *vol = s->volumes[k];
        GVol *gv = &v->gv[k];
        int ld[3];
        march_light_dims(vol, mp->light_res, ld);
        Vec3 size = v_sub(vol->bounds.hi, vol->bounds.lo);
        real voxel = r_fmin(size.x / vol->nx, r_fmin(size.y / vol->ny, size.z / vol->nz));
        v4(gv->lo, vol->bounds.lo, 0);
        v4(gv->hi, vol->bounds.hi, 0);
        v4(gv->alo, vol->active.lo, 0);
        v4(gv->ahi, vol->active.hi, 0);
        v4(gv->albedo, vol->albedo, (float)vol->sigma_t);
        gv->dims[0] = (uint32_t)vol->nx; gv->dims[1] = (uint32_t)vol->ny; gv->dims[2] = (uint32_t)vol->nz; gv->dims[3] = od;
        gv->bdims[0] = (uint32_t)vol->bnx; gv->bdims[1] = (uint32_t)vol->bny; gv->bdims[2] = (uint32_t)vol->bnz; gv->bdims[3] = ob;
        gv->ldims[0] = (uint32_t)ld[0]; gv->ldims[1] = (uint32_t)ld[1]; gv->ldims[2] = (uint32_t)ld[2]; gv->ldims[3] = ol;
        int empty = vol->active.hi.x < vol->active.lo.x || vol->max_density <= 0;
        gv->p[0] = (float)vol->g;
        gv->p[1] = (float)(voxel * (mp->step_voxels > 0 ? mp->step_voxels : 1));
        gv->p[2] = empty ? 0.0f : (float)vol->max_density;
        gv->p[3] = (float)voxel;
        size_t cnt = (size_t)vol->nx * vol->ny * vol->nz;
        if (fresh || v->vols[k] != vol || v->versions[k] != vol->version || n != v->nvols){
            WGPUBuffer from = gpu_source(v, vol);
            if (from) wgpuCommandEncoderCopyBufferToBuffer(enc, from, 0, v->dens, (uint64_t)od * 4, cnt * 4);
            else wgpuQueueWriteBuffer(g->queue, v->dens, (uint64_t)od * 4, vol->density, cnt * 4);
        }
        // (the bricks, the active box and the maximum are computed on the GPU: STATS)
        od += (uint32_t)cnt;
        ob += (uint32_t)(vol->bnx * vol->bny * vol->bnz);
        ol += (uint32_t)(ld[0] * ld[1] * ld[2]);
        v->vols[k] = vol;
        v->versions[k] = vol->version;
    }
    v->nvols = n;
    v->light_res = mp->light_res;
    v->lamp[0] = (float)lamp->pos.x; v->lamp[1] = (float)lamp->pos.y; v->lamp[2] = (float)lamp->pos.z;
    v->light_valid = 0;
    return 1;
}

static void ensure_capture(GpuView *v){
    if (v->cap_tex) return;
    Gpu *g = v->g;
    WGPUTextureDescriptor td;
    memset(&td, 0, sizeof(td));
    td.label = gpu_str("screenshot");
    td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
    td.dimension = WGPUTextureDimension_2D;
    td.size = (WGPUExtent3D){(uint32_t)v->w, (uint32_t)v->h, 1};
    td.format = g->surface_format;
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    v->cap_tex = wgpuDeviceCreateTexture(g->device, &td);
    v->cap_row = ((uint32_t)v->w * 4 + 255) & ~255u;
    v->cap_buf = gpu_buffer(g, "screenshot readback", (size_t)v->cap_row * v->h, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
}

static void on_capture_mapped(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)u2;
    GpuView *v = u1;
    if (status != WGPUMapAsyncStatus_Success){
        // cancelled or aborted: the view or the device is going away, v may be gone already
        if (status == WGPUMapAsyncStatus_Error){
            fprintf(stderr, "GPU: reading the frame back failed: %.*s\n", (int)msg.length, msg.data ? msg.data : "");
            v->cap_pending = 0;
        }
        return;
    }
    v->cap_pending = 0;
    const uint8_t *src = wgpuBufferGetConstMappedRange(v->cap_buf, 0, (size_t)v->cap_row * v->h);
    uint32_t *px = malloc((size_t)v->w * v->h * sizeof(uint32_t));
    if (src && px){
        int bgra = v->g->surface_format == WGPUTextureFormat_BGRA8Unorm;
        for (int y = 0; y < v->h; y++)
            for (int x = 0; x < v->w; x++){
                const uint8_t *q = src + (size_t)y * v->cap_row + 4 * x;
                uint32_t r = bgra ? q[2] : q[0], gg = q[1], b = bgra ? q[0] : q[2];
                px[(size_t)y * v->w + x] = (r << 16) | (gg << 8) | b;
            }
        if (v->cap_fn) v->cap_fn(v->cap_user, px, v->w, v->h);
    }
    free(px);
    wgpuBufferUnmap(v->cap_buf);
}

void gpu_view_volume_on_gpu(GpuView *v, const Volume *vol, WGPUBuffer density){
    if (!v || !vol) return;
    v->nvols = -1;   // copy everything again
    for (int i = 0; i < v->nsrc; i++)
        if (v->src_vol[i] == vol){
            if (density) v->src_buf[i] = density;
            else {                                   // back to the CPU's densities
                v->nsrc--;
                v->src_vol[i] = v->src_vol[v->nsrc];
                v->src_buf[i] = v->src_buf[v->nsrc];
            }
            return;
        }
    if (density && v->nsrc < GPU_MAX_SOURCES){
        v->src_vol[v->nsrc] = vol;
        v->src_buf[v->nsrc++] = density;
    }
}

int gpu_view_capture(GpuView *v, GpuCaptureFn fn, void *user){
    if (!v || v->cap_pending) return 0;
    v->cap_fn = fn;
    v->cap_user = user;
    v->cap_pending = 1;
    return 1;
}

// The first half of a frame: the camera and the lamp into the uniforms, and the media made
// ready (densities, bricks and boxes, light grids), for the composite pass that follows.
static void frame_begin(GpuView *v, WGPUCommandEncoder enc, const Scene *s, const RenderParams *p,
                        const MarchParams *mp, int volumes){
    Gpu *g = v->g;
    MarchParams def = march_default_params();
    if (!mp) mp = &def;
    const int n = volumes ? (s->nvolumes < GPU_MAX_VOLUMES ? s->nvolumes : GPU_MAX_VOLUMES) : 0;
    MarchLamp lamp = march_lamp(s);

    int relight = n ? upload_volumes(v, s, mp, &lamp, enc) : 0;

    GFrame F;
    memset(&F, 0, sizeof(F));
    v4(F.right, vec3(p->view.m[0][0], p->view.m[0][1], p->view.m[0][2]), v->show_traced ? 1.0f : 0.0f);
    v4(F.up, vec3(p->view.m[1][0], p->view.m[1][1], p->view.m[1][2]), 0);
    v4(F.back, vec3(p->view.m[2][0], p->view.m[2][1], p->view.m[2][2]), 0);
    v4(F.eye, p->eye, 0);
    F.proj[0] = (float)(1.0 / p->proj.m[0][0]);
    F.proj[1] = (float)(1.0 / p->proj.m[1][1]);
    F.proj[2] = (float)p->proj.m[2][2];
    F.proj[3] = (float)p->proj.m[2][3];
    F.size[0] = (float)v->w; F.size[1] = (float)v->h; F.size[2] = (float)mp->max_steps; F.size[3] = (float)n;
    v4(F.lamp_pos, lamp.pos, (float)lamp.area);
    v4(F.lamp_n, lamp.normal, v_dot(lamp.normal, lamp.normal) > 0 ? 1.0f : 0.0f);
    v4(F.lamp_L, lamp.radiance, (float)mp->brightness);
    v4(F.amb, march_ambient(s, mp), (float)(mp->octaves < 1 ? 1 : (mp->octaves > 4 ? 4 : mp->octaves)));
    for (int k = 0; k < n; k++) F.vols[k] = v->gv[k];
    wgpuQueueWriteBuffer(g->queue, v->frame_u, 0, &F, sizeof(F));

    if (relight){
        // the media's bricks, boxes and maxima, from the densities now on the GPU
        uint32_t init[GPU_MAX_VOLUMES * 8];
        for (int k = 0; k < GPU_MAX_VOLUMES; k++){
            uint32_t *a = init + 8 * k;
            a[0] = a[1] = a[2] = 0xFFFFFFFFu;
            a[3] = a[4] = a[5] = a[6] = a[7] = 0;
        }
        wgpuQueueWriteBuffer(g->queue, v->acc, 0, init, sizeof(init));
        WGPUComputePassEncoder sp = wgpuCommandEncoderBeginComputePass(enc, NULL);
        for (int k = 0; k < n; k++){
            WGPUBindGroupEntry e[5] = {entry(0, v->frame_u), entry(1, v->dens), entry(2, v->bricks),
                                       entry(3, v->acc), entry(4, v->sel_u[k])};
            WGPUBindGroup bg = bind(g, v->stats_bgl, e, 5);
            uint32_t groups = (v->gv[k].bdims[0] * v->gv[k].bdims[1] * v->gv[k].bdims[2] + 63) / 64;
            uint32_t gx = groups < 65535 ? groups : 65535;
            wgpuComputePassEncoderSetPipeline(sp, v->stats_pipe);
            wgpuComputePassEncoderSetBindGroup(sp, 0, bg, 0, NULL);
            wgpuComputePassEncoderDispatchWorkgroups(sp, gx, (groups + gx - 1) / gx, 1);
            wgpuBindGroupRelease(bg);
            WGPUBindGroupEntry f[4] = {entry(0, v->frame_u), entry(1, v->acc), entry(2, v->vstat), entry(3, v->sel_u[k])};
            bg = bind(g, v->finish_bgl, f, 4);
            wgpuComputePassEncoderSetPipeline(sp, v->finish_pipe);
            wgpuComputePassEncoderSetBindGroup(sp, 0, bg, 0, NULL);
            wgpuComputePassEncoderDispatchWorkgroups(sp, 1, 1, 1);
            wgpuBindGroupRelease(bg);
        }
        wgpuComputePassEncoderEnd(sp);
        wgpuComputePassEncoderRelease(sp);
        // one compute pass per medium: every light grid cell marches to the lamp
        for (int k = 0; k < n; k++){
            WGPUBindGroupEntry e[6] = {entry(0, v->frame_u), entry(1, v->dens), entry(2, v->bricks),
                                       entry(3, v->light), entry(4, v->sel_u[k]), entry(5, v->vstat)};
            WGPUBindGroup bg = bind(g, v->light_bgl, e, 6);
            WGPUComputePassEncoder cp = wgpuCommandEncoderBeginComputePass(enc, NULL);
            wgpuComputePassEncoderSetPipeline(cp, v->light_pipe);
            wgpuComputePassEncoderSetBindGroup(cp, 0, bg, 0, NULL);
            wgpuComputePassEncoderDispatchWorkgroups(cp, (v->gv[k].ldims[0] + 3) / 4, (v->gv[k].ldims[1] + 3) / 4,
                                                     (v->gv[k].ldims[2] + 3) / 4);
            wgpuComputePassEncoderEnd(cp);
            wgpuComputePassEncoderRelease(cp);
            wgpuBindGroupRelease(bg);
        }
        v->light_valid = 1;
    }
}

// The second half: the composite pass (pipe, with the frame's inputs in bg, which is
// released here) onto the canvas (unless offscreen) and the readback texture if a capture
// was asked for; then everything is submitted.
static void frame_end(GpuView *v, WGPUCommandEncoder enc, WGPURenderPipeline pipe, WGPUBindGroup bg){
    Gpu *g = v->g;
    int capture = v->cap_pending == 1;
    WGPUSurfaceTexture st;
    memset(&st, 0, sizeof(st));
    WGPUTextureView view = NULL;
    if (g->surface){
        wgpuSurfaceGetCurrentTexture(g->surface, &st);
        if (st.status == WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal ||
            st.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)
            view = wgpuTextureCreateView(st.texture, NULL);
        else {
            static int told = 0;
            if (!told++) fprintf(stderr, "GPU: no canvas texture to draw on (status %d)\n", (int)st.status);
        }
    }
    WGPURenderPassColorAttachment ca;
    memset(&ca, 0, sizeof(ca));
    ca.view = view;
    ca.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
    ca.loadOp = WGPULoadOp_Clear;
    ca.storeOp = WGPUStoreOp_Store;
    WGPURenderPassDescriptor rp;
    memset(&rp, 0, sizeof(rp));
    rp.colorAttachmentCount = 1;
    rp.colorAttachments = &ca;
    for (int target = view ? 0 : 1; target < 1 + capture; target++){
        // target 0: the canvas; target 1: the screenshot texture (same pass, same pixels)
        WGPUTextureView tv = view;
        if (target == 1){
            ensure_capture(v);
            tv = wgpuTextureCreateView(v->cap_tex, NULL);
        }
        ca.view = tv;
        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rp);
        wgpuRenderPassEncoderSetPipeline(pass, pipe);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
        if (target == 1){
            wgpuTextureViewRelease(tv);
            WGPUTexelCopyTextureInfo src;
            memset(&src, 0, sizeof(src));
            src.texture = v->cap_tex;
            WGPUTexelCopyBufferInfo dst;
            memset(&dst, 0, sizeof(dst));
            dst.buffer = v->cap_buf;
            dst.layout.bytesPerRow = v->cap_row;
            dst.layout.rowsPerImage = (uint32_t)v->h;
            WGPUExtent3D ext = {(uint32_t)v->w, (uint32_t)v->h, 1};
            wgpuCommandEncoderCopyTextureToBuffer(enc, &src, &dst, &ext);
        }
    }
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    wgpuBindGroupRelease(bg);
    if (capture){
        v->cap_pending = 2;
        WGPUBufferMapCallbackInfo mi;
        memset(&mi, 0, sizeof(mi));
        mi.mode = WGPUCallbackMode_AllowSpontaneous;
        mi.callback = on_capture_mapped;
        mi.userdata1 = v;
        wgpuBufferMapAsync(v->cap_buf, WGPUMapMode_Read, 0, (size_t)v->cap_row * v->h, mi);
    }
    if (view){
        // present first, then let go of the texture: natively, a surface texture released
        // before it is presented is thrown away, and the window's images never come back
#ifndef __EMSCRIPTEN__
        wgpuSurfacePresent(g->surface);   // in a browser the canvas shows the frame by itself
#endif
        wgpuTextureViewRelease(view);
        wgpuTextureRelease(st.texture);
    }
}

void gpu_view_present(GpuView *v, const Framebuffer *fb, const Scene *s, const RenderParams *p,
                      const MarchParams *mp, int volumes){
    if (!v || fb->w != v->w || fb->h != v->h) return;
    Gpu *g = v->g;
    // the frame: colors and depths of the CPU renderer
    const size_t np = (size_t)v->w * v->h;
    wgpuQueueWriteBuffer(g->queue, v->color, 0, fb->color, np * 4);
    for (size_t i = 0; i < np; i++) v->depth_f[i] = (float)fb->depth[i];
    wgpuQueueWriteBuffer(g->queue, v->depth, 0, v->depth_f, np * 4);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    frame_begin(v, enc, s, p, mp, volumes);
    WGPUBindGroupEntry e[7] = {entry(0, v->frame_u), entry(1, v->dens), entry(2, v->bricks),
                               entry(3, v->light), entry(4, v->color), entry(5, v->depth), entry(6, v->vstat)};
    frame_end(v, enc, v->comp_pipe, bind(g, v->comp_bgl, e, 7));
}

// the GPU rasterizer's frame (and, with traced = 1, the path tracer's samples over it), composited
static int present_gpu_frame(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp, int volumes,
                             int traced, int draw_scene){
    if (!v || v->raster_failed) return 0;
    Gpu *g = v->g;
    if (!v->raster && !(v->raster = gpu_raster_create(g, v->w, v->h))){ v->raster_failed = 1; return 0; }
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    if (draw_scene && !gpu_raster_draw(v->raster, enc, s, p)){   // e.g. a mesh too big for this GPU
        wgpuCommandEncoderRelease(enc);
        v->raster_failed = 1;
        return 0;
    }
    if (traced && v->t_clear){ gpu_trace_clear(v->trace, enc); v->t_clear = 0; }
    v->show_traced = traced;
    frame_begin(v, enc, s, p, mp, volumes);
    v->show_traced = 0;
    WGPUBindGroupEntry e[8] = {entry(0, v->frame_u), entry(1, v->dens), entry(2, v->bricks), entry(3, v->light),
                               entry(4, NULL), entry(5, NULL), entry(6, v->vstat),
                               entry(7, v->trace ? gpu_trace_accum(v->trace) : v->color)};
    e[4].size = e[5].size = 0;
    e[4].textureView = gpu_raster_color(v->raster);
    e[5].textureView = gpu_raster_depth(v->raster);
    frame_end(v, enc, v->comp_gpu_pipe, bind(g, v->comp_gpu_bgl, e, 8));
    return 1;
}

int gpu_view_present_scene(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp, int volumes){
    return present_gpu_frame(v, s, p, mp, volumes, 0, 1);
}

// ---- path tracing ----------------------------------------------------------------------------

int gpu_view_trace_start(GpuView *v, const Scene *s, const GpuTraceParams *tp){
    if (!v || v->trace_failed) return 0;
    if (!v->trace && !(v->trace = gpu_trace_create(v->g, v->w, v->h, GPU_WGSL_TYPES, GPU_WGSL_MEDIA))){
        v->trace_failed = 1;
        return 0;
    }
    if (!gpu_trace_sync(v->trace, s)){ v->trace_failed = 1; return 0; }
    v->tparams = *tp;
    v->t_clear = 1;
    v->t_y = 0;
    v->t_passes = v->t_done = v->t_done_next = 0;
    if (v->t_rows < 1) v->t_rows = 16;
    if (v->t_reps < 1) v->t_reps = 1;
    v->t_pass_t0 = gpu_now_ms();
    return 1;
}

// A band finished: how long it took decides the size of the next ones. Small bands keep a
// slow GPU responsive (and under the browser's watchdog); a fast GPU gets the whole image,
// then several passes, per submit.
static void on_band_done(WGPUQueueWorkDoneStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)status; (void)msg; (void)u2;
    GpuView *v = u1;
    if (!view_is_live(v)) return;
    v->t_inflight = 0;
    v->t_done = v->t_done_next;
    double ms = gpu_now_ms() - v->t_submit_ms;
    if (ms < 12){
        if (v->t_rows < v->h) v->t_rows = v->t_rows * 2 < v->h ? v->t_rows * 2 : v->h;
        else if (v->t_reps < 16) v->t_reps *= 2;
    } else if (ms > 45){
        if (v->t_reps > 1) v->t_reps /= 2;
        else if (v->t_rows > 4) v->t_rows /= 2;
    }
}

void gpu_view_trace_more(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp){
    if (!v || !v->trace || v->t_inflight) return;
    Gpu *g = v->g;
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    if (v->t_clear){ gpu_trace_clear(v->trace, enc); v->t_clear = 0; }
    frame_begin(v, enc, s, p, mp, 1);         // the media: densities, bricks and boxes up to date
    const int nvol = s->nvolumes < GPU_MAX_VOLUMES ? s->nvolumes : GPU_MAX_VOLUMES;
    // a band of rows, or (the band being the whole image) several passes
    int whole = v->t_y == 0 && v->t_rows >= v->h;
    int rows = v->t_rows < v->h - v->t_y ? v->t_rows : v->h - v->t_y;
    int reps = whole ? v->t_reps : 1;
    gpu_trace_encode(v->trace, enc, s, &v->tparams, v->t_y, rows, reps, nvol, v->frame_u, v->dens, v->bricks, v->vstat);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    v->t_inflight = 1;
    v->t_submit_ms = gpu_now_ms();
    WGPUQueueWorkDoneCallbackInfo wd;
    memset(&wd, 0, sizeof(wd));
    wd.mode = WGPUCallbackMode_AllowSpontaneous;
    wd.callback = on_band_done;
    wd.userdata1 = v;
    wgpuQueueOnSubmittedWorkDone(g->queue, wd);
    v->t_y += rows;
    if (v->t_y >= v->h){
        v->t_y = 0;
        v->t_passes += reps;
        double now = gpu_now_ms();
        v->t_pass_ms = (now - v->t_pass_t0) / reps;
        v->t_pass_t0 = now;
    }
    v->t_done_next = v->t_passes;
}

int gpu_view_trace_passes(const GpuView *v){ return v ? v->t_done : 0; }
int gpu_view_trace_busy(const GpuView *v){ return v && v->t_inflight; }

static void on_trace_mapped(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)msg; (void)u2;
    GpuView *v = u1;
    if (status != WGPUMapAsyncStatus_Success || !view_is_live(v)) return;
    v->t_rb_pending = 0;
    size_t bytes = (size_t)v->w * v->h * 16;
    const float *p = wgpuBufferGetConstMappedRange(v->t_rb, 0, bytes);
    if (p && v->t_read_fn) v->t_read_fn(v->t_read_user, p, v->w, v->h);
    wgpuBufferUnmap(v->t_rb);
}

int gpu_view_trace_read(GpuView *v, GpuTraceReadFn fn, void *user){
    if (!v || !v->trace || v->t_rb_pending) return 0;
    Gpu *g = v->g;
    size_t bytes = (size_t)v->w * v->h * 16;
    if (!v->t_rb) v->t_rb = gpu_buffer(g, "path tracer readback", bytes, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    if (v->t_clear){ gpu_trace_clear(v->trace, enc); v->t_clear = 0; }
    wgpuCommandEncoderCopyBufferToBuffer(enc, gpu_trace_accum(v->trace), 0, v->t_rb, 0, bytes);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    v->t_read_fn = fn;
    v->t_read_user = user;
    v->t_rb_pending = 1;
    WGPUBufferMapCallbackInfo mi;
    memset(&mi, 0, sizeof(mi));
    mi.mode = WGPUCallbackMode_AllowSpontaneous;
    mi.callback = on_trace_mapped;
    mi.userdata1 = v;
    wgpuBufferMapAsync(v->t_rb, WGPUMapMode_Read, 0, bytes, mi);
    return 1;
}
double gpu_view_trace_pass_ms(const GpuView *v){ return v ? v->t_pass_ms : 0; }

int gpu_view_present_traced(GpuView *v, const Scene *s, const RenderParams *p, const MarchParams *mp, int volumes){
    if (!v || !v->trace) return 0;
    // the rasterized scene shows through where there is no sample yet: only during the first pass
    return present_gpu_frame(v, s, p, mp, volumes, 1, v->t_done == 0);
}

void gpu_view_destroy(GpuView *v){
    if (!v) return;
    WGPUBuffer bufs[] = {v->frame_u, v->sel_u[0], v->sel_u[1], v->dens, v->bricks, v->light, v->color, v->depth,
                         v->vstat, v->acc};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) if (bufs[i]){ wgpuBufferDestroy(bufs[i]); wgpuBufferRelease(bufs[i]); }
    if (v->light_pipe) wgpuComputePipelineRelease(v->light_pipe);
    if (v->comp_pipe) wgpuRenderPipelineRelease(v->comp_pipe);
    if (v->comp_gpu_pipe) wgpuRenderPipelineRelease(v->comp_gpu_pipe);
    gpu_raster_destroy(v->raster);
    gpu_trace_destroy(v->trace);
    for (int i = 0; i < 4; i++) if (live_views[i] == v) live_views[i] = NULL;
    if (v->stats_pipe) wgpuComputePipelineRelease(v->stats_pipe);
    if (v->finish_pipe) wgpuComputePipelineRelease(v->finish_pipe);
    WGPUBindGroupLayout bgls[] = {v->light_bgl, v->comp_bgl, v->comp_gpu_bgl, v->stats_bgl, v->finish_bgl};
    for (size_t i = 0; i < sizeof(bgls) / sizeof(bgls[0]); i++) if (bgls[i]) wgpuBindGroupLayoutRelease(bgls[i]);
    if (v->cap_buf){ wgpuBufferDestroy(v->cap_buf); wgpuBufferRelease(v->cap_buf); }
    if (v->t_rb){ wgpuBufferDestroy(v->t_rb); wgpuBufferRelease(v->t_rb); }
    if (v->cap_tex){ wgpuTextureDestroy(v->cap_tex); wgpuTextureRelease(v->cap_tex); }
    free(v->depth_f);
    free(v);
}

#endif   // LR_GPU
