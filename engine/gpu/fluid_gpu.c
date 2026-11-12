#include "gpu/fluid_gpu.h"

#include <stdlib.h>

#ifndef LR_GPU
// ---- built without webgpu.h: the air stays on the CPU ------------------------------------
GpuFluid *gpu_fluid_create(Gpu *g, PhysWorld *pw){ (void)g; (void)pw; return NULL; }
void gpu_fluid_attach_view(GpuFluid *gf, GpuView *v){ (void)gf; (void)v; }
void gpu_fluid_sync_cloud(GpuFluid *gf){ (void)gf; }
int gpu_fluid_pending(const GpuFluid *gf){ (void)gf; return 0; }
void gpu_fluid_set_wait(GpuFluid *gf, int wait){ (void)gf; (void)wait; }
void gpu_fluid_destroy(GpuFluid *gf){ (void)gf; }
#else

#include <stdio.h>
#include <string.h>
#include "core/volume.h"
#include "gpu/gpu_internal.h"
#include "phys/fluid.h"
#include "phys/rigid.h"

// ---- the kernels ------------------------------------------------------------------------------
//
// Every kernel is one @compute entry "main" of 64 threads per workgroup, over cells (nx ny nz),
// over grid points ((nx+1)(ny+1)(nz+1): the u, v and w faces at once, as the CPU loops do), over
// voxels of the cloud, or over bodies. The comments of phys/fluid.c explain the method.

// the step's parameters (struct GFP on the C side)
static const char *FHEAD = WGSL(
struct FP {
    n: vec4u,      /* nx, ny, nz, open faces (FLUID_OPEN_* bits) */
    lo: vec4f,     /* the grid's corner, w = h */
    a: vec4f,      /* 1 / h, dt, the wind's rate * dt (<= 1), buoyancy * dt */
    wind: vec4f,   /* the wind, w = vorticity * h */
    dn: vec4u,     /* the cloud's voxels per axis, w = 1 if there is a cloud */
    dlo: vec4f,    /* the cloud's box, w = what the dissipation keeps */
    dhi: vec4f,    /* w = number of expansion sources */
    m: vec4u,      /* bodies */
    src: array<vec4f, 8>,   /* sources: center, radius */
    rate: array<vec4f, 2>,  /* and their rates */
};
@group(0) @binding(0) var<uniform> P: FP;

fn NX() -> i32 { return i32(P.n.x); }
fn NY() -> i32 { return i32(P.n.y); }
fn NZ() -> i32 { return i32(P.n.z); }
fn H() -> f32 { return P.lo.w; }
fn tid(g: vec3u, nw: vec3u) -> u32 { return g.x + g.y * nw.x * 64u; }
fn ncells() -> u32 { return P.n.x * P.n.y * P.n.z; }
fn npoints() -> u32 { return (P.n.x + 1u) * (P.n.y + 1u) * (P.n.z + 1u); }
fn nvoxels() -> u32 { return P.dn.x * P.dn.y * P.dn.z; }
fn cell_of(id: u32) -> vec3i { let i = i32(id); return vec3i(i % NX(), (i / NX()) % NY(), i / (NX() * NY())); }
fn point_of(id: u32) -> vec3i { let i = i32(id); let a = NX() + 1; let b = NY() + 1; return vec3i(i % a, (i / a) % b, i / (a * b)); }
fn cidx(x: i32, y: i32, z: i32) -> u32 { return u32((z * NY() + y) * NX() + x); }
fn iu(x: i32, y: i32, z: i32) -> u32 { return u32((z * NY() + y) * (NX() + 1) + x); }
fn iv(x: i32, y: i32, z: i32) -> u32 { return u32((z * (NY() + 1) + y) * NX() + x); }
fn iw(x: i32, y: i32, z: i32) -> u32 { return u32((z * NY() + y) * NX() + x); }
fn center(c: vec3i) -> vec3f { return P.lo.xyz + (vec3f(c) + 0.5) * H(); }
fn pos_u(c: vec3i) -> vec3f { return P.lo.xyz + (vec3f(c) + vec3f(0.0, 0.5, 0.5)) * H(); }
fn pos_v(c: vec3i) -> vec3f { return P.lo.xyz + (vec3f(c) + vec3f(0.5, 0.0, 0.5)) * H(); }
fn pos_w(c: vec3i) -> vec3f { return P.lo.xyz + (vec3f(c) + vec3f(0.5, 0.5, 0.0)) * H(); }
fn SU() -> vec3i { return vec3i(NX() + 1, NY(), NZ()); }
fn SV() -> vec3i { return vec3i(NX(), NY() + 1, NZ()); }
fn SW() -> vec3i { return vec3i(NX(), NY(), NZ() + 1); }
fn DN() -> vec3i { return vec3i(P.dn.xyz); }
fn voxel_of(id: u32) -> vec3i { let i = i32(id); let n = DN(); return vec3i(i % n.x, (i / n.x) % n.y, i / (n.x * n.y)); }
fn dcenter(c: vec3i) -> vec3f { return P.dlo.xyz + (vec3f(c) + 0.5) * (P.dhi.xyz - P.dlo.xyz) / vec3f(P.dn.xyz); }
fn dgrid(p: vec3f) -> vec3f { return (p - P.dlo.xyz) * (vec3f(P.dn.xyz) / (P.dhi.xyz - P.dlo.xyz)) - 0.5; }

/* the divergence the expansion sources impose at a point */
fn src_div(p: vec3f) -> f32 {
    var d = 0.0;
    for (var s = 0u; s < u32(P.dhi.w); s++) {
        let c = P.src[s];
        let l = length(p - c.xyz) / c.w;
        if (l < 1.0) { d += P.rate[s / 4u][s % 4u] * (1.0 - l * l); }
    }
    return d;
}
);

// trilinear sample of array QA (dims S, x fastest), clamped to the grid (sample3 in fluid.c);
// QN names the function. QM: the same, also returning the min and max of the 8 values.
static const char *SAMPLE = WGSL(
fn s_QN(g0: vec3f, S: vec3i) -> f32 {
    let g = clamp(g0, vec3f(0.0), vec3f(S - 1));
    let i0 = min(vec3i(g), max(S - 2, vec3i(0)));
    let i1 = min(i0 + 1, S - 1);
    let f = g - vec3f(i0);
    let a = i0.z * S.x * S.y; let b = i1.z * S.x * S.y; let c = i0.y * S.x; let d = i1.y * S.x;
    let c00 = mix(QA[u32(a + c + i0.x)], QA[u32(a + c + i1.x)], f.x);
    let c10 = mix(QA[u32(a + d + i0.x)], QA[u32(a + d + i1.x)], f.x);
    let c01 = mix(QA[u32(b + c + i0.x)], QA[u32(b + c + i1.x)], f.x);
    let c11 = mix(QA[u32(b + d + i0.x)], QA[u32(b + d + i1.x)], f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}
);
static const char *SAMPLE_MM = WGSL(
fn sm_QN(g0: vec3f, S: vec3i) -> vec3f {
    let g = clamp(g0, vec3f(0.0), vec3f(S - 1));
    let i0 = min(vec3i(g), max(S - 2, vec3i(0)));
    let i1 = min(i0 + 1, S - 1);
    let f = g - vec3f(i0);
    let a = i0.z * S.x * S.y; let b = i1.z * S.x * S.y; let c = i0.y * S.x; let d = i1.y * S.x;
    let c000 = QA[u32(a + c + i0.x)]; let c100 = QA[u32(a + c + i1.x)];
    let c010 = QA[u32(a + d + i0.x)]; let c110 = QA[u32(a + d + i1.x)];
    let c001 = QA[u32(b + c + i0.x)]; let c101 = QA[u32(b + c + i1.x)];
    let c011 = QA[u32(b + d + i0.x)]; let c111 = QA[u32(b + d + i1.x)];
    let lo = min(min(min(c000, c100), min(c010, c110)), min(min(c001, c101), min(c011, c111)));
    let hi = max(max(max(c000, c100), max(c010, c110)), max(max(c001, c101), max(c011, c111)));
    let c00 = mix(c000, c100, f.x); let c10 = mix(c010, c110, f.x);
    let c01 = mix(c001, c101, f.x); let c11 = mix(c011, c111, f.x);
    return vec3f(mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z), lo, hi);
}
);

// the air velocity at a world point (needs u, v, w and their samplers)
static const char *VEL = WGSL(
fn vel_at(p: vec3f) -> vec3f {
    let g = (p - P.lo.xyz) * P.a.x;
    return vec3f(s_u(g - vec3f(0.0, 0.5, 0.5), SU()), s_v(g - vec3f(0.5, 0.0, 0.5), SV()), s_w(g - vec3f(0.5, 0.5, 0.0), SW()));
}
);

// obstacles (needs solid)
static const char *SOLID = WGSL(
fn solid_at(x: i32, y: i32, z: i32) -> bool {
    return x >= 0 && y >= 0 && z >= 0 && x < NX() && y < NY() && z < NZ() && solid[cidx(x, y, z)] != 0u;
}
);

// 1 if the face's value is fixed by a wall or a body, and the value (needs solid, svel)
static const char *FIXED = WGSL(
fn fixed_face(ax: i32, q: vec3i) -> vec2f {
    var qq = q;
    var nn = vec3i(NX(), NY(), NZ());
    let c = qq[ax];
    let bit = 1u << u32(2 * ax);
    if (c == 0 && (P.n.w & bit) == 0u) { return vec2f(1.0, 0.0); }
    if (c == nn[ax] && (P.n.w & (bit << 1u)) == 0u) { return vec2f(1.0, 0.0); }
    var o = vec3i(0);
    o[ax] = 1;
    let pa = q - o;
    let a = solid_at(pa.x, pa.y, pa.z);
    let b = solid_at(q.x, q.y, q.z);
    if (!a && !b) { return vec2f(0.0, 0.0); }
    var s = 0.0;
    var k = 0.0;
    if (a) { s += svel[cidx(pa.x, pa.y, pa.z)][ax]; k += 1.0; }
    if (b) { s += svel[cidx(q.x, q.y, q.z)][ax]; k += 1.0; }
    return vec2f(1.0, s / k);
}
);

/* the cells inside each body become obstacles moving with it (mark_solids in world.c);
   a body is 6 vec4: position (w: 0 sphere, 1 box, 2 + k a mesh body whose cells are bits
   k.. of occ: 48^3 over [-half, half] of its own frame, phys/convex.h), its 3 axes (w: half
   sizes), velocity (w: bounding radius), angular velocity (w: radius of the drag samples) */
static const char *K_MARK = WGSL(
@group(0) @binding(1) var<storage, read> bodies: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> solid: array<u32>;
@group(0) @binding(3) var<storage, read_write> svel: array<vec4f>;
@group(0) @binding(4) var<storage, read> occ: array<u32>;
const OCC = 48;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ncells()) { return; }
    let c = center(cell_of(id));
    var s = 0u;
    var vel = vec3f(0.0);
    for (var b = 0u; b < P.m.x; b++) {
        let o = 6u * b;
        let pos = bodies[o];
        let d = c - pos.xyz;
        let r = bodies[o + 4u].w;
        if (dot(d, d) > r * r) { continue; }
        var inside = false;
        if (pos.w < 0.5) {
            inside = dot(d, d) <= bodies[o + 1u].w * bodies[o + 1u].w;
        } else if (pos.w > 1.5) {
            let half = vec3f(bodies[o + 1u].w, bodies[o + 2u].w, bodies[o + 3u].w);
            let l = vec3f(dot(bodies[o + 1u].xyz, d), dot(bodies[o + 2u].xyz, d), dot(bodies[o + 3u].xyz, d));
            let q = vec3i(floor((l + half) / (2.0 * half) * f32(OCC)));
            if (all(q >= vec3i(0)) && all(q < vec3i(OCC))) {
                let i = u32((q.z * OCC + q.y) * OCC + q.x) + u32(pos.w - 2.0 + 0.5) * 1024u;   /* in units of 32 words */
                inside = ((occ[i >> 5u] >> (i & 31u)) & 1u) != 0u;
            }
        } else {
            inside = abs(dot(bodies[o + 1u].xyz, d)) <= bodies[o + 1u].w && abs(dot(bodies[o + 2u].xyz, d)) <= bodies[o + 2u].w &&
                     abs(dot(bodies[o + 3u].xyz, d)) <= bodies[o + 3u].w;
        }
        if (inside) { s = 1u; vel = bodies[o + 4u].xyz + cross(bodies[o + 5u].xyz, d); }
    }
    solid[id] = s;
    svel[id] = vec4f(vel, 0.0);
}
);

/* walls and bodies fix the faces they touch (enforce_boundaries) */
static const char *K_BOUND = WGSL(
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;
@group(0) @binding(4) var<storage, read> solid: array<u32>;
@group(0) @binding(5) var<storage, read> svel: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= npoints()) { return; }
    let q = point_of(id);
    if (q.y < NY() && q.z < NZ()) { let r = fixed_face(0, q); if (r.x > 0.5) { u[iu(q.x, q.y, q.z)] = r.y; } }
    if (q.x < NX() && q.z < NZ()) { let r = fixed_face(1, q); if (r.x > 0.5) { v[iv(q.x, q.y, q.z)] = r.y; } }
    if (q.x < NX() && q.y < NY()) { let r = fixed_face(2, q); if (r.x > 0.5) { w[iw(q.x, q.y, q.z)] = r.y; } }
}
);

/* wind and buoyancy (add_forces) */
static const char *K_FORCES = WGSL(
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;
@group(0) @binding(4) var<storage, read> dens: array<f32>;
/* volume_density: trilinear, 0 outside the grid */
fn vol_density(p: vec3f) -> f32 {
    let n = vec3f(P.dn.xyz);
    let g = (p - P.dlo.xyz) / (P.dhi.xyz - P.dlo.xyz) * n - 0.5;
    if (!(all(g > vec3f(-1.0)) && all(g < n))) { return 0.0; }
    let i0 = vec3i(floor(g));
    let f = g - vec3f(i0);
    let ni = DN();
    var c: array<f32, 8>;
    for (var k = 0; k < 8; k++) {
        let i = i0 + vec3i(k & 1, (k >> 1) & 1, k >> 2);
        var val = 0.0;
        if (all(i >= vec3i(0)) && all(i < ni)) { val = dens[u32((i.z * ni.y + i.y) * ni.x + i.x)]; }
        c[k] = val;
    }
    let c00 = mix(c[0], c[1], f.x); let c10 = mix(c[2], c[3], f.x);
    let c01 = mix(c[4], c[5], f.x); let c11 = mix(c[6], c[7], f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= npoints()) { return; }
    let q = point_of(id);
    let a = P.a.z;
    if (q.y < NY() && q.z < NZ()) { let i = iu(q.x, q.y, q.z); u[i] += (P.wind.x - u[i]) * a; }
    if (q.x < NX() && q.z < NZ()) {
        let i = iv(q.x, q.y, q.z);
        var val = v[i];
        val += (P.wind.y - val) * a;
        if (P.dn.w != 0u && P.a.w != 0.0) { val += P.a.w * vol_density(pos_v(q)); }
        v[i] = val;
    }
    if (q.x < NX() && q.y < NY()) { let i = iw(q.x, q.y, q.z); w[i] += (P.wind.z - w[i]) * a; }
}
);

/* vorticity confinement in 4 passes: velocity at the cell centers; vorticity and its length;
   the force N x omega; spread to the faces */
static const char *K_VA = WGSL(
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> cvel: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ncells()) { return; }
    let c = cell_of(id);
    cvel[id] = vec4f(0.5 * (u[iu(c.x, c.y, c.z)] + u[iu(c.x + 1, c.y, c.z)]),
                     0.5 * (v[iv(c.x, c.y, c.z)] + v[iv(c.x, c.y + 1, c.z)]),
                     0.5 * (w[iw(c.x, c.y, c.z)] + w[iw(c.x, c.y, c.z + 1)]), 0.0);
}
);
static const char *K_VB = WGSL(
@group(0) @binding(1) var<storage, read> cvel: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> omg: array<vec4f>;
fn at(x: i32, y: i32, z: i32) -> vec4f { return cvel[cidx(clamp(x, 0, NX() - 1), clamp(y, 0, NY() - 1), clamp(z, 0, NZ() - 1))]; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ncells()) { return; }
    let c = cell_of(id);
    let k = 0.5 / H();
    let dwdy = (at(c.x, c.y + 1, c.z).z - at(c.x, c.y - 1, c.z).z) * k;
    let dvdz = (at(c.x, c.y, c.z + 1).y - at(c.x, c.y, c.z - 1).y) * k;
    let dudz = (at(c.x, c.y, c.z + 1).x - at(c.x, c.y, c.z - 1).x) * k;
    let dwdx = (at(c.x + 1, c.y, c.z).z - at(c.x - 1, c.y, c.z).z) * k;
    let dvdx = (at(c.x + 1, c.y, c.z).y - at(c.x - 1, c.y, c.z).y) * k;
    let dudy = (at(c.x, c.y + 1, c.z).x - at(c.x, c.y - 1, c.z).x) * k;
    let o = vec3f(dwdy - dvdz, dudz - dwdx, dvdx - dudy);
    omg[id] = vec4f(o, length(o));
}
);
static const char *K_VC = WGSL(
@group(0) @binding(1) var<storage, read> omg: array<vec4f>;
@group(0) @binding(2) var<storage, read> solid: array<u32>;
@group(0) @binding(3) var<storage, read_write> frc: array<vec4f>;
fn mag(x: i32, y: i32, z: i32) -> f32 { return omg[cidx(clamp(x, 0, NX() - 1), clamp(y, 0, NY() - 1), clamp(z, 0, NZ() - 1))].w; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ncells()) { return; }
    let c = cell_of(id);
    let k = 0.5 / H();
    let g = vec3f(mag(c.x + 1, c.y, c.z) - mag(c.x - 1, c.y, c.z), mag(c.x, c.y + 1, c.z) - mag(c.x, c.y - 1, c.z),
                  mag(c.x, c.y, c.z + 1) - mag(c.x, c.y, c.z - 1)) * k;
    let gl = length(g);
    var force = vec3f(0.0);
    if (gl > 1e-6 && solid[id] == 0u) { force = cross(g / gl, omg[id].xyz) * P.wind.w; }
    frc[id] = vec4f(force, 0.0);
}
);
static const char *K_VD = WGSL(
@group(0) @binding(1) var<storage, read> frc: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> u: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> w: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ncells()) { return; }
    let c = cell_of(id);
    let dt = P.a.y;
    let f = frc[id];
    if (c.x > 0) { u[iu(c.x, c.y, c.z)] += 0.5 * (f.x + frc[cidx(c.x - 1, c.y, c.z)].x) * dt; }
    if (c.y > 0) { v[iv(c.x, c.y, c.z)] += 0.5 * (f.y + frc[cidx(c.x, c.y - 1, c.z)].y) * dt; }
    if (c.z > 0) { w[iw(c.x, c.y, c.z)] += 0.5 * (f.z + frc[cidx(c.x, c.y, c.z - 1)].z) * dt; }
}
);

/* semi-Lagrangian advection of the velocity into tu, tv, tw */
static const char *K_ADV = WGSL(
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> tu: array<f32>;
@group(0) @binding(5) var<storage, read_write> tv: array<f32>;
@group(0) @binding(6) var<storage, read_write> tw: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= npoints()) { return; }
    let q = point_of(id);
    let dt = P.a.y;
    let s = P.a.x;
    if (q.y < NY() && q.z < NZ()) {
        let c = pos_u(q);
        tu[iu(q.x, q.y, q.z)] = s_u((c - vel_at(c) * dt - P.lo.xyz) * s - vec3f(0.0, 0.5, 0.5), SU());
    }
    if (q.x < NX() && q.z < NZ()) {
        let c = pos_v(q);
        tv[iv(q.x, q.y, q.z)] = s_v((c - vel_at(c) * dt - P.lo.xyz) * s - vec3f(0.5, 0.0, 0.5), SV());
    }
    if (q.x < NX() && q.y < NY()) {
        let c = pos_w(q);
        tw[iw(q.x, q.y, q.z)] = s_w((c - vel_at(c) * dt - P.lo.xyz) * s - vec3f(0.5, 0.5, 0.0), SW());
    }
}
);

/* the divergence, minus what the expansion sources ask for */
static const char *K_DIV = WGSL(
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> solid: array<u32>;
@group(0) @binding(5) var<storage, read_write> div: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ncells()) { return; }
    if (solid[id] != 0u) { div[id] = 0.0; return; }
    let c = cell_of(id);
    let d = (u[iu(c.x + 1, c.y, c.z)] - u[iu(c.x, c.y, c.z)]) + (v[iv(c.x, c.y + 1, c.z)] - v[iv(c.x, c.y, c.z)]) +
            (w[iw(c.x, c.y, c.z + 1)] - w[iw(c.x, c.y, c.z)]);
    div[id] = d * P.a.x - src_div(center(c));
}
);

/* u -= grad p across the faces the pressure acts on */
static const char *K_GRAD = WGSL(
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<f32>;
@group(0) @binding(5) var<storage, read> solid: array<u32>;
@group(0) @binding(6) var<storage, read> svel: array<vec4f>;
fn pc(x: i32, y: i32, z: i32) -> f32 {
    if (x < 0 || y < 0 || z < 0 || x >= NX() || y >= NY() || z >= NZ()) { return 0.0; }
    return p[cidx(x, y, z)];
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= npoints()) { return; }
    let q = point_of(id);
    let s = P.a.x;
    let here = pc(q.x, q.y, q.z);
    if (q.y < NY() && q.z < NZ() && fixed_face(0, q).x < 0.5) { u[iu(q.x, q.y, q.z)] -= (here - pc(q.x - 1, q.y, q.z)) * s; }
    if (q.x < NX() && q.z < NZ() && fixed_face(1, q).x < 0.5) { v[iv(q.x, q.y, q.z)] -= (here - pc(q.x, q.y - 1, q.z)) * s; }
    if (q.x < NX() && q.y < NY() && fixed_face(2, q).x < 0.5) { w[iw(q.x, q.y, q.z)] -= (here - pc(q.x, q.y, q.z - 1)) * s; }
}
);

/* the cloud, MacCormack: one step backwards into d1, then the correction */
static const char *K_D1 = WGSL(
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> d0: array<f32>;
@group(0) @binding(5) var<storage, read_write> d1: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= nvoxels()) { return; }
    let c = dcenter(voxel_of(id));
    d1[id] = s_d0(dgrid(c - vel_at(c) * P.a.y), DN());
}
);
static const char *K_D2 = WGSL(
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> d0: array<f32>;
@group(0) @binding(5) var<storage, read> d1: array<f32>;
@group(0) @binding(6) var<storage, read_write> dens: array<f32>;
@group(0) @binding(7) var<storage, read> solid: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= nvoxels()) { return; }
    let c = dcenter(voxel_of(id));
    let fc = vec3i(floor((c - P.lo.xyz) * P.a.x));
    if (solid_at(fc.x, fc.y, fc.z)) { dens[id] = 0.0; return; }
    let dt = P.a.y;
    let vel = vel_at(c);
    let mm = sm_d0(dgrid(c - vel * dt), DN());
    let back = s_d1(dgrid(c + vel * dt), DN());
    var r = d1[id] + 0.5 * (d0[id] - back);
    r = clamp(r, mm.y, mm.z);
    r *= P.dlo.w;
    if (P.dhi.w > 0.5) { r *= exp(-src_div(c) * dt); }
    dens[id] = select(r, 0.0, r < 1e-4);
}
);

/* the air around each body: the mean over 14 points on a shell just outside it */
static const char *K_SAMP = WGSL(
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> bodies: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> around: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= P.m.x) { return; }
    let pos = bodies[6u * id].xyz;
    let r = bodies[6u * id + 5u].w;
    let k = 0.57735027;
    var dirs = array<vec3f, 14>(vec3f(1.0, 0.0, 0.0), vec3f(-1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, -1.0, 0.0),
        vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0), vec3f(k, k, k), vec3f(k, k, -k), vec3f(k, -k, k), vec3f(k, -k, -k),
        vec3f(-k, k, k), vec3f(-k, k, -k), vec3f(-k, -k, k), vec3f(-k, -k, -k));
    var sum = vec3f(0.0);
    for (var d = 0; d < 14; d++) { sum += vel_at(pos + dirs[d] * r); }
    around[id] = vec4f(sum / 14.0, 0.0);
}
);

// ---- the multigrid kernels (phys/poisson.c), on the levels' own parameters ----------------------

static const char *LHEAD = WGSL(
struct LU {
    d: vec4u,    /* this level: nx, ny, nz, open faces */
    c: vec4u,    /* the next coarser level: nx, ny, nz; w = the color of a red-black sweep */
    h: vec4f,    /* h^2 */
};
@group(0) @binding(0) var<uniform> L: LU;
fn tid(g: vec3u, nw: vec3u) -> u32 { return g.x + g.y * nw.x * 64u; }
fn lcount() -> u32 { return L.d.x * L.d.y * L.d.z; }
fn ccount() -> u32 { return L.c.x * L.c.y * L.c.z; }
fn lcell(id: u32) -> vec3i { let i = i32(id); let n = vec3i(L.d.xyz); return vec3i(i % n.x, (i / n.x) % n.y, i / (n.x * n.y)); }
fn ccell(id: u32) -> vec3i { let i = i32(id); let n = vec3i(L.c.xyz); return vec3i(i % n.x, (i / n.x) % n.y, i / (n.x * n.y)); }
fn lidx(q: vec3i) -> u32 { return u32((q.z * i32(L.d.y) + q.y) * i32(L.d.x) + q.x); }
fn cidx(q: vec3i) -> u32 { return u32((q.z * i32(L.c.y) + q.y) * i32(L.c.x) + q.x); }
);

// neighbors of a cell through its links (needs p, links)
static const char *NSUM = WGSL(
fn nsum(i: u32, m: u32) -> f32 {
    let sy = L.d.x;
    let sz = L.d.x * L.d.y;
    var s = 0.0;
    if ((m & 1u) != 0u) { s += p[i - 1u]; }
    if ((m & 2u) != 0u) { s += p[i + 1u]; }
    if ((m & 4u) != 0u) { s += p[i - sy]; }
    if ((m & 8u) != 0u) { s += p[i + sy]; }
    if ((m & 16u) != 0u) { s += p[i - sz]; }
    if ((m & 32u) != 0u) { s += p[i + sz]; }
    return s;
}
fn ncount(m: u32) -> f32 { return f32(countOneBits(m & 63u) + (m >> 6u)); }
);

static const char *K_LINKS = WGSL(
@group(0) @binding(1) var<storage, read> solid: array<u32>;
@group(0) @binding(2) var<storage, read_write> links: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= lcount()) { return; }
    var m = 0u;
    var o = 0u;
    if (solid[id] == 0u) {
        let q = lcell(id);
        let n = vec3i(L.d.xyz);
        var nb = array<vec3i, 6>(vec3i(-1, 0, 0), vec3i(1, 0, 0), vec3i(0, -1, 0), vec3i(0, 1, 0), vec3i(0, 0, -1), vec3i(0, 0, 1));
        for (var k = 0u; k < 6u; k++) {
            let a = q + nb[k];
            if (any(a < vec3i(0)) || any(a >= n)) {
                if (((L.d.w >> k) & 1u) != 0u) { o++; }
                continue;
            }
            if (solid[lidx(a)] == 0u) { m |= 1u << k; }
        }
    }
    links[id] = m | (o << 6u);
}
);
/* a coarse cell is solid only if all its fine cells are (L: the fine level) */
static const char *K_CSOLID = WGSL(
@group(0) @binding(1) var<storage, read> fine: array<u32>;
@group(0) @binding(2) var<storage, read_write> coarse: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ccount()) { return; }
    let q = ccell(id);
    var all_solid = 1u;
    for (var k = 0; k < 8; k++) {
        let a = 2 * q + vec3i(k & 1, (k >> 1) & 1, k >> 2);
        if (any(a >= vec3i(L.d.xyz))) { continue; }
        if (fine[lidx(a)] == 0u) { all_solid = 0u; }
    }
    coarse[id] = all_solid;
}
);
static const char *K_SMOOTH = WGSL(
@group(0) @binding(1) var<storage, read> links: array<u32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@group(0) @binding(3) var<storage, read> rhs: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= lcount()) { return; }
    let q = lcell(id);
    if (((u32(q.x + q.y + q.z) + L.c.w) & 1u) != 0u) { return; }
    let m = links[id];
    if (m == 0u) { return; }
    p[id] = (nsum(id, m) - L.h.x * rhs[id]) / ncount(m);
}
);
static const char *K_RESID = WGSL(
@group(0) @binding(1) var<storage, read> links: array<u32>;
@group(0) @binding(2) var<storage, read> p: array<f32>;
@group(0) @binding(3) var<storage, read> rhs: array<f32>;
@group(0) @binding(4) var<storage, read_write> res: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= lcount()) { return; }
    let m = links[id];
    if (m == 0u) { res[id] = 0.0; return; }
    res[id] = rhs[id] - (nsum(id, m) - ncount(m) * p[id]) / L.h.x;
}
);
/* the coarse right-hand side: the mean residual of the fine fluid cells (L: the fine level) */
static const char *K_RESTRICT = WGSL(
@group(0) @binding(1) var<storage, read> links: array<u32>;
@group(0) @binding(2) var<storage, read> res: array<f32>;
@group(0) @binding(3) var<storage, read_write> crhs: array<f32>;
@group(0) @binding(4) var<storage, read_write> cp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= ccount()) { return; }
    let q = ccell(id);
    var s = 0.0;
    var n = 0;
    for (var k = 0; k < 8; k++) {
        let a = 2 * q + vec3i(k & 1, (k >> 1) & 1, k >> 2);
        if (any(a >= vec3i(L.d.xyz))) { continue; }
        let fi = lidx(a);
        if (links[fi] == 0u) { continue; }
        s += res[fi];
        n++;
    }
    crhs[id] = select(0.0, s / f32(n), n > 0);
    cp[id] = 0.0;
}
);
static const char *K_PROLONG = WGSL(
@group(0) @binding(1) var<storage, read> links: array<u32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@group(0) @binding(3) var<storage, read> cp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = tid(gid, nw);
    if (id >= lcount()) { return; }
    if (links[id] == 0u) { return; }
    p[id] += cp[cidx(lcell(id) / 2)];
}
);
/* the coarsest grid: 40 sweeps by one workgroup, which can wait for itself */
static const char *K_COARSEST = WGSL(
@group(0) @binding(1) var<storage, read> links: array<u32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@group(0) @binding(3) var<storage, read> rhs: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    let n = lcount();
    for (var it = 0; it < 40; it++) {
        for (var color = 0u; color < 2u; color++) {
            for (var i = li; i < n; i += 256u) {
                let q = lcell(i);
                if (((u32(q.x + q.y + q.z) + color) & 1u) == 0u) {
                    let m = links[i];
                    if (m != 0u) { p[i] = (nsum(i, m) - L.h.x * rhs[i]) / ncount(m); }
                }
            }
            storageBarrier();
            workgroupBarrier();
        }
    }
}
);

// ---- the C side ---------------------------------------------------------------------------------

typedef struct {
    uint32_t n[4];
    float lo[4], a[4], wind[4];
    uint32_t dn[4];
    float dlo[4], dhi[4];
    uint32_t m[4];
    float src[8][4];
    float rate[8];
} GFP;

typedef struct { uint32_t d[4], c[4]; float h[4]; } GLU;

enum { K_MARK_, K_BOUND_, K_FORCES_, K_VA_, K_VB_, K_VC_, K_VD_, K_ADV_, K_DIV_, K_GRAD_, K_D1_, K_D2_, K_SAMP_,
       K_LINKS_, K_CSOLID_, K_SMOOTH_, K_RESID_, K_RESTRICT_, K_PROLONG_, K_COARSEST_, NK };

// what each kernel is made of: its bindings, its body, and the pieces it needs
enum { NEED_VEL = 1, NEED_SOLID = 2, NEED_FIXED = 4, NEED_D0 = 8, NEED_D0MM = 16, NEED_D1 = 32, NEED_NSUM = 64, LEVEL = 128 };
static const struct { const char *name, *types; const char **body; int needs; } KERNELS[NK] = {
    {"air: obstacles",    "URWWR",    &K_MARK, 0},
    {"air: walls",        "UWWWRR",   &K_BOUND, NEED_SOLID | NEED_FIXED},
    {"air: forces",       "UWWWR",    &K_FORCES, 0},
    {"air: vorticity 1",  "URRRW",    &K_VA, 0},
    {"air: vorticity 2",  "URW",      &K_VB, 0},
    {"air: vorticity 3",  "URRW",     &K_VC, 0},
    {"air: vorticity 4",  "URWWW",    &K_VD, 0},
    {"air: advection",    "URRRWWW",  &K_ADV, NEED_VEL},
    {"air: divergence",   "URRRRW",   &K_DIV, 0},
    {"air: gradient",     "UWWWRRR",  &K_GRAD, NEED_SOLID | NEED_FIXED},
    {"cloud: advection",  "URRRRW",   &K_D1, NEED_VEL | NEED_D0},
    {"cloud: correction", "URRRRRWR", &K_D2, NEED_VEL | NEED_SOLID | NEED_D0MM | NEED_D1},
    {"air: around bodies","URRRRW",   &K_SAMP, NEED_VEL},
    {"pressure: links",   "URW",      &K_LINKS, LEVEL},
    {"pressure: coarse",  "URW",      &K_CSOLID, LEVEL},
    {"pressure: sweep",   "URWR",     &K_SMOOTH, LEVEL | NEED_NSUM},
    {"pressure: residual","URRRW",    &K_RESID, LEVEL | NEED_NSUM},
    {"pressure: restrict","URRWW",    &K_RESTRICT, LEVEL},
    {"pressure: prolong", "URWR",     &K_PROLONG, LEVEL},
    {"pressure: coarsest","URWR",     &K_COARSEST, LEVEL | NEED_NSUM},
};

typedef struct {
    int nx, ny, nz;
    WGPUBuffer links, res, solid, p, rhs;   // level 0: solid, p and rhs are the fluid's
    WGPUBuffer lu[2];                       // parameters, for the two colors of the sweeps
    WGPUBindGroup bg_links, bg_csolid, bg_smooth[2], bg_resid, bg_restrict, bg_prolong, bg_coarsest;
} GLevel;

struct GpuFluid {
    Gpu *g;
    PhysWorld *pw;
    Fluid *f;
    Volume *cloud;                // f->density, or NULL
    WGPUBindGroupLayout bgl[NK];
    WGPUComputePipeline pipe[NK];
    WGPUBuffer fp;                // GFP
    WGPUBuffer u, v, w, tu, tv, tw, p, div, solid, svel, cvel, omg, frc;
    WGPUBuffer dens, d0, d1;
    WGPUBuffer bodies, around;    // per body: 6 vec4 in, 1 vec4 out
    size_t bodies_cap, around_cap;
    WGPUBuffer occ;               // the mesh bodies' shapes: which cells they fill
    size_t occ_cap;
    int nocc;                     // shapes uploaded
    int nb_bodies;                // bodies the bind groups below were made for
    WGPUBindGroup bg[NK];         // the fluid kernels' bind groups (MARK and SAMP follow the bodies)
    GLevel lv[POISSON_MAX_LEVELS];
    int nlevels;
    float *body_data;
    // readbacks: the air around the bodies (every step), the cloud (when asked)
    WGPUBuffer around_rb, dens_rb;
    size_t around_rb_cap;
    int around_pending, around_n;
    int dens_pending, dens_wanted;
    uint64_t steps, dens_step;
    Vec3 *around_cpu;
    int around_cpu_cap;
    // back-pressure: steps submitted and not finished yet. A slow GPU must not fall behind
    // without limit (the frames would queue up behind the air): beyond MAX_INFLIGHT the step
    // waits, and its time is added to the next one (semi-Lagrangian steps take any dt).
    int inflight;
    real dt_owed;
    int wait;                     // offline: wait for the GPU instead of skipping
};

#define MAX_INFLIGHT 2
#define MAX_OWED 0.1f

// the GPU fluids alive: a "work done" callback can arrive after gpu_fluid_destroy
static GpuFluid *live[4];
static int is_live(const GpuFluid *gf){
    for (int i = 0; i < 4; i++) if (gf && live[i] == gf) return 1;
    return 0;
}

static void on_done(WGPUQueueWorkDoneStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)status; (void)msg; (void)u2;
    GpuFluid *gf = u1;
    if (is_live(gf) && gf->inflight > 0) gf->inflight--;
}

static size_t nu(const Fluid *f){ return (size_t)(f->nx + 1) * f->ny * f->nz; }
static size_t nv(const Fluid *f){ return (size_t)f->nx * (f->ny + 1) * f->nz; }
static size_t nw(const Fluid *f){ return (size_t)f->nx * f->ny * (f->nz + 1); }
static size_t ncells(const Fluid *f){ return (size_t)f->nx * f->ny * f->nz; }
static size_t npoints(const Fluid *f){ return (size_t)(f->nx + 1) * (f->ny + 1) * (f->nz + 1); }

// the WGSL of kernel k: the header, the samplers and helpers it needs, its body
static char *kernel_source(int k){
    const int nd = KERNELS[k].needs;
    size_t cap = 65536;
    char *s = malloc(cap);
    if (!s) return NULL;
    s[0] = 0;
#define ADD(t) do { strcat(s, t); strcat(s, "\n"); } while (0)
    ADD(nd & LEVEL ? LHEAD : FHEAD);
    ADD(*KERNELS[k].body);
    if (nd & NEED_NSUM) ADD(NSUM);
    if (nd & NEED_SOLID) ADD(SOLID);
    if (nd & NEED_FIXED) ADD(FIXED);
    // samplers: copies of SAMPLE with QN / QA replaced
    const char *names[6] = {"u", "v", "w", "d0", "d0", "d1"};
    const char *tmpl[6] = {SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE_MM, SAMPLE};
    int want[6] = {nd & NEED_VEL, nd & NEED_VEL, nd & NEED_VEL, (nd & (NEED_D0 | NEED_D0MM)) != 0, nd & NEED_D0MM, nd & NEED_D1};
    for (int i = 0; i < 6; i++){
        if (!want[i]) continue;
        const char *t = tmpl[i];
        char *o = s + strlen(s);
        for (; *t; t++){
            if (t[0] == 'Q' && (t[1] == 'N' || t[1] == 'A')){ strcpy(o, names[i]); o += strlen(names[i]); t++; }
            else *o++ = *t;
        }
        *o++ = '\n';
        *o = 0;
    }
    if (nd & NEED_VEL) ADD(VEL);
#undef ADD
    return s;
}

static WGPUBuffer storage(Gpu *g, const char *label, size_t bytes){
    return gpu_buffer(g, label, bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
}

static WGPUBuffer uniform(Gpu *g, const char *label, const void *data, size_t bytes){
    WGPUBuffer b = gpu_buffer(g, label, bytes, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    if (data) wgpuQueueWriteBuffer(g->queue, b, 0, data, bytes);
    return b;
}

#define BG(k, ...) do { WGPUBuffer b_[] = {__VA_ARGS__}; gf->bg[k] = gpu_bind(g, gf->bgl[k], b_, (int)(sizeof(b_) / sizeof(b_[0]))); } while (0)
#define LBG(dst, k, ...) do { WGPUBuffer b_[] = {__VA_ARGS__}; dst = gpu_bind(g, gf->bgl[k], b_, (int)(sizeof(b_) / sizeof(b_[0]))); } while (0)

// the bind groups that do not depend on the bodies
static void make_bind_groups(GpuFluid *gf){
    Gpu *g = gf->g;
    BG(K_BOUND_, gf->fp, gf->u, gf->v, gf->w, gf->solid, gf->svel);
    BG(K_FORCES_, gf->fp, gf->u, gf->v, gf->w, gf->dens);
    BG(K_VA_, gf->fp, gf->u, gf->v, gf->w, gf->cvel);
    BG(K_VB_, gf->fp, gf->cvel, gf->omg);
    BG(K_VC_, gf->fp, gf->omg, gf->solid, gf->frc);
    BG(K_VD_, gf->fp, gf->frc, gf->u, gf->v, gf->w);
    BG(K_ADV_, gf->fp, gf->u, gf->v, gf->w, gf->tu, gf->tv, gf->tw);
    BG(K_DIV_, gf->fp, gf->u, gf->v, gf->w, gf->solid, gf->div);
    BG(K_GRAD_, gf->fp, gf->u, gf->v, gf->w, gf->p, gf->solid, gf->svel);
    BG(K_D1_, gf->fp, gf->u, gf->v, gf->w, gf->d0, gf->d1);
    BG(K_D2_, gf->fp, gf->u, gf->v, gf->w, gf->d0, gf->d1, gf->dens, gf->solid);
    for (int k = 0; k < gf->nlevels; k++){
        GLevel *l = &gf->lv[k];
        LBG(l->bg_links, K_LINKS_, l->lu[0], l->solid, l->links);
        if (k > 0) LBG(l->bg_csolid, K_CSOLID_, gf->lv[k - 1].lu[0], gf->lv[k - 1].solid, l->solid);
        for (int c = 0; c < 2; c++) LBG(l->bg_smooth[c], K_SMOOTH_, l->lu[c], l->links, l->p, l->rhs);
        LBG(l->bg_resid, K_RESID_, l->lu[0], l->links, l->p, l->rhs, l->res);
        if (k + 1 < gf->nlevels){
            GLevel *c = &gf->lv[k + 1];
            LBG(l->bg_restrict, K_RESTRICT_, l->lu[0], l->links, l->res, c->rhs, c->p);
            LBG(l->bg_prolong, K_PROLONG_, l->lu[0], l->links, l->p, c->p);
        }
        LBG(l->bg_coarsest, K_COARSEST_, l->lu[0], l->links, l->p, l->rhs);
    }
}

// the bodies' buffers can grow: their bind groups follow
_Static_assert(OCC_RES == 48 && OCC_WORDS % 32 == 0, "K_MARK's OCC, and its offsets in units of 32 words");

static void reserve_bodies(GpuFluid *gf, const RigidWorld *rw){
    Gpu *g = gf->g;
    int n = rw->nbodies;
    int grown = gpu_buffer_reserve(g, &gf->bodies, &gf->bodies_cap, (size_t)(n > 0 ? n : 1) * 96,
                                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "bodies");
    grown |= gpu_buffer_reserve(g, &gf->around, &gf->around_cap, (size_t)(n > 0 ? n : 1) * 16,
                                WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc, "air around bodies");
    grown |= gpu_buffer_reserve(g, &gf->occ, &gf->occ_cap, (size_t)(rw->ngeoms > 0 ? rw->ngeoms : 1) * OCC_WORDS * 4,
                                WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "mesh bodies' cells");
    if (grown) gf->nocc = 0;
    for (; gf->nocc < rw->ngeoms; gf->nocc++)          // shapes never change once made
        wgpuQueueWriteBuffer(g->queue, gf->occ, (uint64_t)gf->nocc * OCC_WORDS * 4, rw->geoms[gf->nocc]->occ, OCC_WORDS * 4);
    if (!grown && gf->bg[K_MARK_]) return;
    if (gf->bg[K_MARK_]) wgpuBindGroupRelease(gf->bg[K_MARK_]);
    if (gf->bg[K_SAMP_]) wgpuBindGroupRelease(gf->bg[K_SAMP_]);
    BG(K_MARK_, gf->fp, gf->bodies, gf->solid, gf->svel, gf->occ);
    BG(K_SAMP_, gf->fp, gf->u, gf->v, gf->w, gf->bodies, gf->around);
}

static void level_params(GpuFluid *gf, int k){
    const Fluid *f = gf->f;
    const PoissonLevel *pl = &f->poisson.lv[k];
    GLevel *l = &gf->lv[k];
    for (int c = 0; c < 2; c++){
        GLU u;
        memset(&u, 0, sizeof(u));
        u.d[0] = (uint32_t)pl->nx; u.d[1] = (uint32_t)pl->ny; u.d[2] = (uint32_t)pl->nz; u.d[3] = f->open;
        if (k + 1 < f->poisson.nlevels){
            const PoissonLevel *cl = &f->poisson.lv[k + 1];
            u.c[0] = (uint32_t)cl->nx; u.c[1] = (uint32_t)cl->ny; u.c[2] = (uint32_t)cl->nz;
        }
        u.c[3] = (uint32_t)c;
        u.h[0] = (float)(pl->h * pl->h);
        if (!l->lu[c]) l->lu[c] = uniform(gf->g, "pressure level", &u, sizeof(u));
        else wgpuQueueWriteBuffer(gf->g->queue, l->lu[c], 0, &u, sizeof(u));
    }
}

static void free_all(GpuFluid *gf){
    WGPUBuffer bufs[] = {gf->fp, gf->u, gf->v, gf->w, gf->tu, gf->tv, gf->tw, gf->p, gf->div, gf->solid, gf->svel, gf->cvel,
                         gf->omg, gf->frc, gf->dens, gf->d0, gf->d1, gf->bodies, gf->around, gf->around_rb, gf->dens_rb, gf->occ};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) if (bufs[i]){ wgpuBufferDestroy(bufs[i]); wgpuBufferRelease(bufs[i]); }
    for (int k = 0; k < gf->nlevels; k++){
        GLevel *l = &gf->lv[k];
        WGPUBuffer lb[] = {k ? l->solid : NULL, l->links, l->res, k ? l->p : NULL, k ? l->rhs : NULL, l->lu[0], l->lu[1]};
        for (size_t i = 0; i < sizeof(lb) / sizeof(lb[0]); i++) if (lb[i]){ wgpuBufferDestroy(lb[i]); wgpuBufferRelease(lb[i]); }
        WGPUBindGroup bgs[] = {l->bg_links, l->bg_csolid, l->bg_smooth[0], l->bg_smooth[1], l->bg_resid, l->bg_restrict,
                               l->bg_prolong, l->bg_coarsest};
        for (size_t i = 0; i < sizeof(bgs) / sizeof(bgs[0]); i++) if (bgs[i]) wgpuBindGroupRelease(bgs[i]);
    }
    for (int k = 0; k < NK; k++){
        if (gf->bg[k]) wgpuBindGroupRelease(gf->bg[k]);
        if (gf->pipe[k]) wgpuComputePipelineRelease(gf->pipe[k]);
        if (gf->bgl[k]) wgpuBindGroupLayoutRelease(gf->bgl[k]);
    }
    free(gf->body_data);
    free(gf->around_cpu);
    free(gf);
}

static void air_step(void *user, PhysWorld *pw, real dt);

GpuFluid *gpu_fluid_create(Gpu *g, PhysWorld *pw){
    if (!g || gpu_state(g) != GPU_READY || !pw || !pw->fluid) return NULL;
    Fluid *f = pw->fluid;
    GpuFluid *gf = calloc(1, sizeof(*gf));
    if (!gf) return NULL;
    gf->g = g;
    gf->pw = pw;
    gf->f = f;
    gf->cloud = f->density;
    for (int k = 0; k < NK; k++){
        char *src = kernel_source(k);
        if (!src){ free_all(gf); return NULL; }
        gf->bgl[k] = gpu_layout(g, WGPUShaderStage_Compute, KERNELS[k].types);
        gf->pipe[k] = gpu_compute(g, KERNELS[k].name, src, "main", gf->bgl[k]);
        free(src);
    }
    const size_t n = ncells(f);
    gf->fp = uniform(g, "air parameters", NULL, sizeof(GFP));
    gf->u = storage(g, "air u", nu(f) * 4); gf->v = storage(g, "air v", nv(f) * 4); gf->w = storage(g, "air w", nw(f) * 4);
    gf->tu = storage(g, "air tu", nu(f) * 4); gf->tv = storage(g, "air tv", nv(f) * 4); gf->tw = storage(g, "air tw", nw(f) * 4);
    gf->p = storage(g, "pressure", n * 4);
    gf->div = storage(g, "divergence", n * 4);
    gf->solid = storage(g, "obstacles", n * 4);
    gf->svel = storage(g, "obstacle velocity", n * 16);
    gf->cvel = storage(g, "cell velocity", n * 16);
    gf->omg = storage(g, "vorticity", n * 16);
    gf->frc = storage(g, "vorticity force", n * 16);
    size_t nd = gf->cloud ? (size_t)gf->cloud->nx * gf->cloud->ny * gf->cloud->nz : 1;
    gf->dens = storage(g, "cloud", nd * 4);
    gf->d0 = storage(g, "cloud d0", nd * 4);
    gf->d1 = storage(g, "cloud d1", nd * 4);
    wgpuQueueWriteBuffer(g->queue, gf->u, 0, f->u, nu(f) * 4);
    wgpuQueueWriteBuffer(g->queue, gf->v, 0, f->v, nv(f) * 4);
    wgpuQueueWriteBuffer(g->queue, gf->w, 0, f->w, nw(f) * 4);
    wgpuQueueWriteBuffer(g->queue, gf->p, 0, f->p, n * 4);
    if (gf->cloud) wgpuQueueWriteBuffer(g->queue, gf->dens, 0, gf->cloud->density, nd * 4);
    // the pressure levels: the same sizes as the CPU's multigrid
    gf->nlevels = f->poisson.nlevels;
    for (int k = 0; k < gf->nlevels; k++){
        const PoissonLevel *pl = &f->poisson.lv[k];
        GLevel *l = &gf->lv[k];
        size_t ln = (size_t)pl->nx * pl->ny * pl->nz;
        l->nx = pl->nx; l->ny = pl->ny; l->nz = pl->nz;
        l->links = storage(g, "pressure links", ln * 4);
        l->res = storage(g, "pressure residual", ln * 4);
        l->solid = k ? storage(g, "pressure obstacles", ln * 4) : gf->solid;
        l->p = k ? storage(g, "pressure correction", ln * 4) : gf->p;
        l->rhs = k ? storage(g, "pressure rhs", ln * 4) : gf->div;
        level_params(gf, k);
    }
    make_bind_groups(gf);
    reserve_bodies(gf, &pw->rigid);
    for (int i = 0; i < 4; i++) if (!live[i]){ live[i] = gf; break; }
    pw->air_step = air_step;
    pw->air_user = gf;
    pw->air_around = NULL;
    pw->nair_around = 0;
    return gf;
}

void gpu_fluid_attach_view(GpuFluid *gf, GpuView *v){
    if (gf && gf->cloud) gpu_view_volume_on_gpu(v, gf->cloud, gf->dens);
}

// ---- readbacks ----------------------------------------------------------------------------------

static void on_around(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)msg; (void)u2;
    GpuFluid *gf = u1;
    if (status != WGPUMapAsyncStatus_Success || !is_live(gf)) return;   // cancelled: gf may be gone
    gf->around_pending = 0;
    const float *a = wgpuBufferGetConstMappedRange(gf->around_rb, 0, (size_t)gf->around_n * 16);
    if (a){
        if (gf->around_cpu_cap < gf->around_n){
            Vec3 *n = realloc(gf->around_cpu, (size_t)gf->around_n * sizeof(Vec3));
            if (n){ gf->around_cpu = n; gf->around_cpu_cap = gf->around_n; }
        }
        if (gf->around_cpu_cap >= gf->around_n){
            for (int i = 0; i < gf->around_n; i++) gf->around_cpu[i] = vec3(a[4 * i], a[4 * i + 1], a[4 * i + 2]);
            gf->pw->air_around = gf->around_cpu;
            gf->pw->nair_around = gf->around_n;
        }
    }
    wgpuBufferUnmap(gf->around_rb);
}

static void on_cloud(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2){
    (void)msg; (void)u2;
    GpuFluid *gf = u1;
    if (status != WGPUMapAsyncStatus_Success || !is_live(gf)) return;
    gf->dens_pending = 0;
    Volume *v = gf->cloud;
    size_t n = (size_t)v->nx * v->ny * v->nz;
    const float *d = wgpuBufferGetConstMappedRange(gf->dens_rb, 0, n * 4);
    if (d){
        memcpy(v->density, d, n * 4);
        volume_update(v);
    }
    wgpuBufferUnmap(gf->dens_rb);
}

static void map_read(WGPUBuffer b, size_t bytes, WGPUBufferMapCallback cb, void *user){
    WGPUBufferMapCallbackInfo mi;
    memset(&mi, 0, sizeof(mi));
    mi.mode = WGPUCallbackMode_AllowSpontaneous;
    mi.callback = cb;
    mi.userdata1 = user;
    wgpuBufferMapAsync(b, WGPUMapMode_Read, 0, bytes, mi);
}

void gpu_fluid_sync_cloud(GpuFluid *gf){
    if (!gf || !gf->cloud || gf->dens_pending || gf->dens_step == gf->steps) return;
    Gpu *g = gf->g;
    size_t bytes = (size_t)gf->cloud->nx * gf->cloud->ny * gf->cloud->nz * 4;
    if (!gf->dens_rb) gf->dens_rb = gpu_buffer(g, "cloud readback", bytes, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(enc, gf->dens, 0, gf->dens_rb, 0, bytes);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    gf->dens_pending = 1;
    gf->dens_step = gf->steps;
    map_read(gf->dens_rb, bytes, on_cloud, gf);
}

// ---- the step -----------------------------------------------------------------------------------

static void run(WGPUComputePassEncoder pass, GpuFluid *gf, int k, WGPUBindGroup bg, size_t threads){
    wgpuComputePassEncoderSetPipeline(pass, gf->pipe[k]);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    if (k == K_COARSEST_) wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    else gpu_dispatch(pass, threads);
}

static size_t lcells(const GLevel *l){ return (size_t)l->nx * l->ny * l->nz; }

static void vcycle(WGPUComputePassEncoder pass, GpuFluid *gf, int k){
    GLevel *l = &gf->lv[k];
    const Poisson *ps = &gf->f->poisson;
    if (k == gf->nlevels - 1){
        run(pass, gf, K_COARSEST_, l->bg_coarsest, 0);
        return;
    }
    for (int s = 0; s < ps->pre; s++)
        for (int c = 0; c < 2; c++) run(pass, gf, K_SMOOTH_, l->bg_smooth[c], lcells(l));
    run(pass, gf, K_RESID_, l->bg_resid, lcells(l));
    run(pass, gf, K_RESTRICT_, l->bg_restrict, lcells(&gf->lv[k + 1]));
    vcycle(pass, gf, k + 1);
    run(pass, gf, K_PROLONG_, l->bg_prolong, lcells(l));
    for (int s = 0; s < ps->post; s++)
        for (int c = 0; c < 2; c++) run(pass, gf, K_SMOOTH_, l->bg_smooth[c], lcells(l));
}

static void v4(float *d, Vec3 a, float w){ d[0] = (float)a.x; d[1] = (float)a.y; d[2] = (float)a.z; d[3] = w; }

static void air_step(void *user, PhysWorld *pw, real dt){
    GpuFluid *gf = user;
    Fluid *f = pw->fluid;
    Gpu *g = gf->g;
    if (dt <= 0 || gpu_state(g) != GPU_READY) return;
    if (gf->inflight >= MAX_INFLIGHT && gf->wait) gpu_wait(g);
    if (gf->inflight >= MAX_INFLIGHT){
        gf->dt_owed = gf->dt_owed + dt > MAX_OWED ? MAX_OWED : gf->dt_owed + dt;
        return;
    }
    dt += gf->dt_owed;
    gf->dt_owed = 0;
    const int nb = pw->rigid.nbodies;

    // the bodies
    reserve_bodies(gf, &pw->rigid);
    if (nb > 0){
        float *bd = realloc(gf->body_data, (size_t)nb * 96);
        if (!bd) return;
        gf->body_data = bd;
        for (int i = 0; i < nb; i++){
            const Body *b = &pw->rigid.bodies[i];
            float *o = bd + 24 * i;
            Mat3 r = quat_to_mat3(b->rot);
            int sphere = b->shape == BODY_SPHERE;
            real bound = sphere ? b->half.x : v_len(b->half);
            real shell = (sphere ? b->half.x : (b->half.x + b->half.y + b->half.z) / 3) * 1.15f + f->h;
            float kind = sphere ? 0.0f : 1.0f;
            for (int k = 0; b->g && k < pw->rigid.ngeoms; k++)
                if (pw->rigid.geoms[k] == b->g) kind = 2.0f + (float)k * (OCC_WORDS / 32);   // in 32-word units
            v4(o, b->pos, kind);
            v4(o + 4, sphere ? vec3(1, 0, 0) : mat3_col(r, 0), (float)b->half.x);
            v4(o + 8, sphere ? vec3(0, 1, 0) : mat3_col(r, 1), (float)b->half.y);
            v4(o + 12, sphere ? vec3(0, 0, 1) : mat3_col(r, 2), (float)b->half.z);
            v4(o + 16, b->vel, (float)bound);
            v4(o + 20, b->angvel, (float)shell);
        }
        wgpuQueueWriteBuffer(g->queue, gf->bodies, 0, bd, (size_t)nb * 96);
    }

    // the parameters
    GFP P;
    memset(&P, 0, sizeof(P));
    P.n[0] = (uint32_t)f->nx; P.n[1] = (uint32_t)f->ny; P.n[2] = (uint32_t)f->nz; P.n[3] = f->open;
    v4(P.lo, f->bounds.lo, (float)f->h);
    P.a[0] = (float)f->inv_h;
    P.a[1] = (float)dt;
    P.a[2] = (float)(f->wind_rate * dt > 1 ? 1 : f->wind_rate * dt);
    P.a[3] = (float)(gf->cloud ? f->buoyancy * dt : 0);
    v4(P.wind, f->wind, (float)(f->vorticity * f->h));
    const Volume *cv = gf->cloud;
    if (cv){
        P.dn[0] = (uint32_t)cv->nx; P.dn[1] = (uint32_t)cv->ny; P.dn[2] = (uint32_t)cv->nz; P.dn[3] = 1;
        v4(P.dlo, cv->bounds.lo, (float)(f->dissipation * dt >= 1 ? 0 : 1 - f->dissipation * dt));
        v4(P.dhi, cv->bounds.hi, (float)f->nsources);
    } else {
        P.dn[0] = P.dn[1] = P.dn[2] = 1;
        v4(P.dlo, vec3(0, 0, 0), 1);
        v4(P.dhi, vec3(1, 1, 1), (float)f->nsources);
    }
    P.m[0] = (uint32_t)nb;
    for (int s = 0; s < f->nsources && s < 8; s++){
        v4(P.src[s], f->sources[s].center, (float)f->sources[s].radius);
        P.rate[s] = (float)f->sources[s].rate;
    }
    wgpuQueueWriteBuffer(g->queue, gf->fp, 0, &P, sizeof(P));
    // the open faces may have changed (the levels' links use them)
    for (int k = 0; k < gf->nlevels; k++) level_params(gf, k);

    const size_t nc = ncells(f), np = npoints(f);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g->device, NULL);
    // 1. obstacles, forces, vorticity, advection (into tu, tv, tw)
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    run(pass, gf, K_MARK_, gf->bg[K_MARK_], nc);
    run(pass, gf, K_BOUND_, gf->bg[K_BOUND_], np);
    run(pass, gf, K_FORCES_, gf->bg[K_FORCES_], np);
    if (f->vorticity > 0){
        run(pass, gf, K_VA_, gf->bg[K_VA_], nc);
        run(pass, gf, K_VB_, gf->bg[K_VB_], nc);
        run(pass, gf, K_VC_, gf->bg[K_VC_], nc);
        run(pass, gf, K_VD_, gf->bg[K_VD_], nc);
    }
    run(pass, gf, K_BOUND_, gf->bg[K_BOUND_], np);
    run(pass, gf, K_ADV_, gf->bg[K_ADV_], np);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderCopyBufferToBuffer(enc, gf->tu, 0, gf->u, 0, nu(f) * 4);
    wgpuCommandEncoderCopyBufferToBuffer(enc, gf->tv, 0, gf->v, 0, nv(f) * 4);
    wgpuCommandEncoderCopyBufferToBuffer(enc, gf->tw, 0, gf->w, 0, nw(f) * 4);
    // 2. projection: divergence, multigrid (the levels' obstacles and links first), gradient
    pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    run(pass, gf, K_BOUND_, gf->bg[K_BOUND_], np);
    run(pass, gf, K_DIV_, gf->bg[K_DIV_], nc);
    for (int k = 0; k < gf->nlevels; k++){
        if (k > 0) run(pass, gf, K_CSOLID_, gf->lv[k].bg_csolid, lcells(&gf->lv[k]));
        run(pass, gf, K_LINKS_, gf->lv[k].bg_links, lcells(&gf->lv[k]));
    }
    for (int c = 0; c < f->pressure_cycles; c++) vcycle(pass, gf, 0);
    run(pass, gf, K_GRAD_, gf->bg[K_GRAD_], np);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    // 3. the cloud
    if (cv){
        size_t nd = (size_t)cv->nx * cv->ny * cv->nz;
        wgpuCommandEncoderCopyBufferToBuffer(enc, gf->dens, 0, gf->d0, 0, nd * 4);
        pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
        run(pass, gf, K_D1_, gf->bg[K_D1_], nd);
        run(pass, gf, K_D2_, gf->bg[K_D2_], nd);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }
    // 4. the air around the bodies, for the drag of the next step
    int sample = nb > 0 && !gf->around_pending;
    if (sample){
        pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
        run(pass, gf, K_SAMP_, gf->bg[K_SAMP_], (size_t)nb);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        gpu_buffer_reserve(g, &gf->around_rb, &gf->around_rb_cap, (size_t)nb * 16,
                           WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, "air around readback");
        wgpuCommandEncoderCopyBufferToBuffer(enc, gf->around, 0, gf->around_rb, 0, (size_t)nb * 16);
    }
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    WGPUQueueWorkDoneCallbackInfo wd;
    memset(&wd, 0, sizeof(wd));
    wd.mode = WGPUCallbackMode_AllowSpontaneous;
    wd.callback = on_done;
    wd.userdata1 = gf;
    gf->inflight++;
    wgpuQueueOnSubmittedWorkDone(g->queue, wd);
    if (sample){
        gf->around_pending = 1;
        gf->around_n = nb;
        map_read(gf->around_rb, (size_t)nb * 16, on_around, gf);
    }

    // as fluid_step: the expansion sources run out
    int k = 0;
    for (int s = 0; s < f->nsources; s++){
        f->sources[s].time_left -= dt;
        if (f->sources[s].time_left > 1e-6f) f->sources[k++] = f->sources[s];
    }
    f->nsources = k;
    gf->steps++;
    if (gf->cloud) gf->cloud->version++;   // the GPU's cloud changed (the CPU's copy did not)
}

int gpu_fluid_pending(const GpuFluid *gf){ return gf ? gf->inflight : 0; }
void gpu_fluid_set_wait(GpuFluid *gf, int wait){
#ifndef __EMSCRIPTEN__
    if (gf) gf->wait = wait;
#else
    (void)gf; (void)wait;
#endif
}

void gpu_fluid_destroy(GpuFluid *gf){
    if (!gf) return;
    PhysWorld *pw = gf->pw;
    if (pw->air_user == gf){
        pw->air_step = NULL;
        pw->air_user = NULL;
        pw->air_around = NULL;
        pw->nair_around = 0;
    }
    for (int i = 0; i < 4; i++) if (live[i] == gf) live[i] = NULL;
    free_all(gf);
}

#endif   // LR_GPU
