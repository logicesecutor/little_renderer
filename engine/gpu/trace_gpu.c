// trace_gpu.c - the path tracer of rt/tracer.c (trace_path_ex), on the GPU.
//
// One compute kernel, one thread per pixel, one sample per thread: the whole path is traced
// in a loop inside the shader (a "megakernel"), and the sample is added to an accumulation
// buffer (sum of the samples, and their count, per pixel) that the view shows. The same light
// transport as the CPU: diffuse / mirror / glass / lamp, next event estimation with multiple
// importance sampling, Russian roulette, and the media (delta tracking for the collisions,
// ratio tracking for the shadow rays, both walking the majorant bricks with a 3D DDA).
//
// The scene is the CPU's, flattened into buffers:
//   nodes   the BVH (rt/bvh.h), 32 bytes per node: box, and children or a range of `aux`
//   aux     the BVH's shape ids (leaves are ranges of it), then the vertex normals
//   prims   spheres and quads, with the precomputed data of core/scene.h
//   tris    mesh triangles, 48 bytes: corner, two edges, 3 normal indices, material
// What differs from the CPU: 32-bit floats whatever the build, and another random number
// generator (PCG with a 32-bit state: WGSL has no 64-bit integers). So the two images are
// not equal sample by sample; they converge to the same picture.
#include "gpu/gpu_internal.h"

#ifdef LR_GPU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/sampler.h"
#include "core/volume.h"
#include "rt/accel.h"
#include "rt/bvh.h"

static const char *TRACE = WGSL(
struct Node { lo: vec3f, a: u32, hi: vec3f, b: u32 };       /* leaf: a = first, b = 0x80000000 | count */
struct Prim {
    c: vec4f,      /* sphere: center, radius */
    q: vec4f,      /* quad: corner; w = 0 sphere, 1 quad */
    u: vec4f,      /* edge; w = material */
    v: vec4f,      /* edge; w = area */
    n: vec4f,      /* unit normal; w = plane_d */
    w: vec4f,      /* (u x v) / |u x v|^2 */
};
struct Tri { q: vec3f, n0: u32, u: vec3f, n1: u32, v: vec3f, nm: u32 };   /* nm: normal 2, material in the top 4 bits */
struct TMat { albedo: vec4f, emission: vec4f };             /* albedo.w = type, emission.w = ior */
struct TP {
    origin: vec4f, upper_left: vec4f, horizontal: vec4f, vertical: vec4f,   /* the camera */
    dims: vec4u,   /* image width, height; first row and rows of this band */
    opt: vec4u,    /* max depth, mis, number of prims, the lamp (a prim) */
    cnt: vec4u,    /* BVH nodes, where the normals start in aux, seed, media */
    f: vec4f,      /* clamp, ray epsilon, shadow margin, 1 = the denoiser's planes follow the samples */
    ext: vec4u,    /* 1 = Sobol + blue noise (else plain random numbers), the image's scramble,
                      1 = leave out what light tracing adds (skip_caustics), light paths */
    lt: vec4f,     /* light tracing: 2^32 / light paths per pass, the screen area of a pixel */
    ins: vec4u,    /* instances: how many, the first record in prims, the top-level BVH's root;
                      the world BVH's root */
    mats: array<TMat, 16>,
};
@group(0) @binding(3) var<storage, read> VS: array<VStat, 2>;
@group(0) @binding(4) var<uniform> T: TP;
@group(0) @binding(5) var<storage, read> nodes: array<Node>;
@group(0) @binding(6) var<storage, read> aux: array<u32>;
@group(0) @binding(7) var<storage, read> prims: array<Prim>;
@group(0) @binding(8) var<storage, read> tris: array<Tri>;
/* plane 0 (pixels 0..N-1): the samples, rgb = their sum, a = how many. With T.f.w = 1, for
   the denoiser (gpu/denoise_gpu.c): plane 1 the sums of l and l^2 (l the luminance of the
   sample over the albedo), plane 2 the first diffuse surface's normal and distance, plane 3
   its albedo and what the pixel is (0 a surface, 1 a lamp or nothing, 2 seen through a medium) */
@group(0) @binding(9) var<storage, read_write> accum: array<vec4f>;
@group(0) @binding(10) var<uniform> BN: array<vec4u, 512>;   /* the blue-noise mask: 4096 ranks, 16 bits each */
/* the light tracer's caustics (ltrace; bound instead of accum): per pixel and channel, a 64-bit
   32.32 fixed-point sum as two u32 (low, high) */
@group(0) @binding(11) var<storage, read_write> caus: array<atomic<u32>>;

const NONE: u32 = 0xffffffffu;
const PI_: f32 = 3.14159265;

struct Ray { o: vec3f, d: vec3f };
struct Found { t: f32, id: u32, b: vec2f, inst: u32 };   /* inst: the instance, or NONE */
struct Hit { t: f32, p: vec3f, n: vec3f, ng: vec3f, front: bool };

/* random numbers: a 32-bit PCG (an LCG whose output is scrambled) */
fn rnd(s: ptr<function, u32>) -> f32 {
    let old = *s;
    *s = old * 747796405u + 2891336453u;
    let w = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return f32(((w >> 22u) ^ w) >> 8u) * (1.0 / 16777216.0);
}
fn hash32(x0: u32) -> u32 {
    var x = x0;
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
    return x;
}

/* ---- the sampler (core/sampler.c): Owen-scrambled Sobol, shifted per pixel by blue noise ---- */
fn bn_value(x: u32, y: u32) -> f32 {
    let i = (y & 63u) * 64u + (x & 63u);
    let w = BN[i >> 3u][(i >> 1u) & 3u];
    return (f32((w >> ((i & 1u) * 16u)) & 0xffffu) + 0.5) / 4096.0;
}
fn owen(x0: u32, seed: u32) -> u32 {
    var x = reverseBits(x0);
    x ^= x * 0x3d20adeau;
    x += seed;
    x *= (seed >> 16u) | 1u;
    x ^= x * 0x05526c56u;
    x ^= x * 0x53a22864u;
    return reverseBits(x);
}
fn sobol1(i0: u32) -> u32 {
    var i = i0;
    var r = 0u;
    var v = 0x80000000u;
    loop {
        if (i == 0u) { break; }
        if ((i & 1u) != 0u) { r ^= v; }
        i >>= 1u;
        v ^= v >> 1u;
    }
    return r;
}
struct Smp { x: u32, y: u32, idx: u32, on: u32 };
const SMP_PAIRS = 13u;   /* SAMPLER_PAIRS of core/sampler.h */
fn s2(s: Smp, dim: u32, rng: ptr<function, u32>) -> vec2f {
    if (s.on == 0u || dim >= SMP_PAIRS) {
        let a = rnd(rng);
        return vec2f(a, rnd(rng));
    }
    let tile = hash32(((s.x >> 6u) * 73856093u) ^ ((s.y >> 6u) * 19349663u));
    let seed = hash32(T.ext.y ^ hash32((dim * 0x9e3779b9u) ^ tile));
    let i = owen(s.idx, seed);
    let a = owen(reverseBits(i), hash32(seed ^ 0xa511e9b3u));
    let b = owen(sobol1(i), hash32(seed ^ 0x63d83595u));
    let ox = s.x + 23u * dim;
    let oy = s.y + 41u * dim;
    let u = vec2f(f32(a >> 8u), f32(b >> 8u)) * (1.0 / 16777216.0) + vec2f(bn_value(ox, oy), bn_value(ox + 32u, oy + 17u));
    return min(select(u, u - 1.0, u >= vec2f(1.0)), vec2f(0.99999994));
}
fn s1(s: Smp, dim: u32, comp: u32, rng: ptr<function, u32>) -> f32 {
    if (s.on == 0u || dim >= SMP_PAIRS) { return rnd(rng); }
    return s2(s, dim, rng)[comp];
}

/* ---- intersections (rt/intersect.c); a miss is t = -1 ---- */

fn hit_sphere(pr: Prim, r: Ray, tmin: f32, tmax: f32) -> f32 {
    let oc = pr.c.xyz - r.o;
    let a = dot(r.d, r.d);
    let hh = dot(r.d, oc);
    let c = dot(oc, oc) - pr.c.w * pr.c.w;
    let disc = hh * hh - a * c;
    if (disc < 0.0) { return -1.0; }
    let sq = sqrt(disc);
    var t = (hh - sq) / a;
    if (t <= tmin || t >= tmax) {
        t = (hh + sq) / a;
        if (t <= tmin || t >= tmax) { return -1.0; }
    }
    return t;
}

fn hit_quad(pr: Prim, r: Ray, tmin: f32, tmax: f32) -> f32 {
    let denom = dot(pr.n.xyz, r.d);
    if (abs(denom) < 1e-12) { return -1.0; }
    let t = (pr.n.w - dot(pr.n.xyz, r.o)) / denom;
    if (t <= tmin || t >= tmax) { return -1.0; }
    let rel = r.o + r.d * t - pr.q.xyz;
    let a = dot(pr.w.xyz, cross(rel, pr.v.xyz));
    let b = dot(pr.w.xyz, cross(pr.u.xyz, rel));
    if (a < 0.0 || a > 1.0 || b < 0.0 || b > 1.0) { return -1.0; }
    return t;
}

/* Moller-Trumbore: (t, b1, b2) */
fn hit_tri(tr: Tri, r: Ray, tmin: f32, tmax: f32) -> vec3f {
    let pvec = cross(r.d, tr.v);
    let det = dot(tr.u, pvec);
    if (abs(det) < 1e-12) { return vec3f(-1.0); }
    let inv = 1.0 / det;
    let tvec = r.o - tr.q;
    let b1 = dot(tvec, pvec) * inv;
    if (b1 < 0.0 || b1 > 1.0) { return vec3f(-1.0); }
    let qvec = cross(tvec, tr.u);
    let b2 = dot(r.d, qvec) * inv;
    if (b2 < 0.0 || b1 + b2 > 1.0) { return vec3f(-1.0); }
    let t = dot(tr.v, qvec) * inv;
    if (t <= tmin || t >= tmax) { return vec3f(-1.0); }
    return vec3f(t, b1, b2);
}

/* slab test: the distance at which the ray enters the box, -1 if it misses (tmin > 0) */
fn box_t(lo: vec3f, hi: vec3f, o: vec3f, inv: vec3f, tmin: f32, tmax: f32) -> f32 {
    let t0 = (lo - o) * inv;
    let t1 = (hi - o) * inv;
    let a = min(t0, t1);
    let b = max(t0, t1);
    let enter = max(max(a.x, a.y), max(a.z, tmin));
    let leave = min(min(b.x, b.y), min(b.z, tmax));
    return select(enter, -1.0, leave < enter);
}

/* shape instances (the physics' balls and boxes): the unit sphere, the unit box [-1, 1]^3;
   Found.id is then one of these */
const UNIT_SPHERE: u32 = 0xfffffff0u;
const WIDE: u32 = 0x80000000u;   /* a stack entry that is a 4-wide node (its first slot) */
const UNIT_BOX: u32 = 0xfffffff1u;
fn hit_unit_sphere(r: Ray, tmin: f32, tmax: f32) -> f32 {
    let a = dot(r.d, r.d);
    let hh = -dot(r.d, r.o);
    let c = dot(r.o, r.o) - 1.0;
    let disc = hh * hh - a * c;
    if (disc < 0.0) { return -1.0; }
    let sq = sqrt(disc);
    var t = (hh - sq) / a;
    if (t <= tmin || t >= tmax) {
        t = (hh + sq) / a;
        if (t <= tmin || t >= tmax) { return -1.0; }
    }
    return t;
}
fn hit_unit_box(r: Ray, tmin: f32, tmax: f32) -> f32 {
    let inv = safe_inv(r.d);
    let t0 = (vec3f(-1.0) - r.o) * inv;
    let t1 = (vec3f(1.0) - r.o) * inv;
    let a = min(t0, t1);
    let b = max(t0, t1);
    let enter = max(max(a.x, a.y), a.z);
    let leave = min(min(b.x, b.y), b.z);
    if (leave < enter) { return -1.0; }
    if (enter > tmin && enter < tmax) { return enter; }
    if (leave > tmin && leave < tmax) { return leave; }
    return -1.0;
}

/* an instance's transforms: rows of world -> object (inv) and object -> world (fwd) */
fn inst_inv(i: u32, row: u32) -> vec4f {
    let r = prims[T.ins.y + 2u * i];
    if (row == 0u) { return r.c; }
    if (row == 1u) { return r.q; }
    return r.u;
}
fn to_object(i: u32, r: Ray) -> Ray {
    let a = inst_inv(i, 0u);
    let b = inst_inv(i, 1u);
    let c = inst_inv(i, 2u);
    return Ray(vec3f(dot(a.xyz, r.o) + a.w, dot(b.xyz, r.o) + b.w, dot(c.xyz, r.o) + c.w),
               vec3f(dot(a.xyz, r.d), dot(b.xyz, r.d), dot(c.xyz, r.d)));
}
/* an object-space normal in the world: by the inverse transpose */
fn normal_to_world(i: u32, n: vec3f) -> vec3f {
    return normalize(inst_inv(i, 0u).xyz * n.x + inst_inv(i, 1u).xyz * n.y + inst_inv(i, 2u).xyz * n.z);
}

/* The closest hit in (tmin, tmax) (or with stop_at_first any hit): the world's BVH
   (rt/bvh.c), then the top-level BVH over the instances (built on the GPU every frame:
   BUILD below), whose leaves lead into their meshes' BVHs, in the mesh's own space (the
   same t). One loop for all of them, so the hit tests exist once in the compiled code.

   The stack is one small array of u32 (its size is most of what a thread keeps in memory,
   and that decides how many threads the GPU runs at once: 64 entries with the instance and
   the distance of each made the whole path tracer 2-3x slower). An entry is
     a node (binary layout), WIDE | the first slot of a 4-wide node,
     INST | an instance (enter it: EXIT then its mesh's root are pushed), or EXIT (back to
     the world's space: instances are never inside instances).
   Entries are not culled by distance when they come off the stack (no distance is kept):
   their children's boxes are tested against the closest hit so far anyway. Measured on an
   RTX 3070 (docs/PERFORMANCE.md): 24 to 64 entries cost about the same, 40 leaves room for
   a mesh's tree under the top level's; a full stack drops what does not fit. */
const STACK = 40;
const INST: u32 = 0x40000000u;
const EXIT: u32 = 0xfffffffeu;
fn traverse(r: Ray, tmin: f32, tmax: f32, stop_at_first: bool) -> Found {
    var best = Found(tmax, NONE, vec2f(0.0), NONE);
    var stack: array<u32, STACK>;
    var sp = 0;
    if (T.ins.x > 0u) { stack[sp] = T.ins.z; sp++; }
    if ((T.cnt.x & 0x7fffffffu) > 0u) { stack[sp] = T.ins.w | (T.cnt.x & WIDE); sp++; }
    var cur = NONE;
    var ray = r;
    var inv = safe_inv(r.d);
    while (sp > 0) {
        sp--;
        let ni = stack[sp];
        if (ni == EXIT) {
            cur = NONE;
            ray = r;
            inv = safe_inv(r.d);
            continue;
        }
        /* a 4-wide node (rt/bvh.h): 4 slots, each a child's box; a leaf child is pushed as its
           slot (a leaf node of the binary layout), an inner one as WIDE | its slots; the hit
           children go on the stack farthest first */
        if ((ni & WIDE) != 0u) {
            let g = ni & 0x7fffffffu;
            var ce: array<u32, 4>;
            var ct: array<f32, 4>;
            var m = 0;
            for (var k = 0u; k < 4u; k++) {
                let c = nodes[g + k];
                if (c.b == NONE) { continue; }
                let tt = box_t(c.lo, c.hi, ray.o, inv, tmin, best.t);
                if (tt < 0.0) { continue; }
                let e = select(WIDE | c.a, g + k, (c.b & 0x80000000u) != 0u);
                var j = m;
                m++;
                while (j > 0 && ct[j - 1] < tt) { ce[j] = ce[j - 1]; ct[j] = ct[j - 1]; j--; }
                ce[j] = e;
                ct[j] = tt;
            }
            for (var q = 0; q < m; q++) {
                if (sp >= STACK) { break; }
                stack[sp] = ce[q];
                sp++;
            }
            continue;
        }
        if ((ni & INST) != 0u) {
            let i = ni & 0x3fffffffu;
            if (sp + 2 <= STACK) {
                cur = i;
                ray = to_object(i, r);
                inv = safe_inv(ray.d);
                stack[sp] = EXIT;
                stack[sp + 1] = bitcast<u32>(prims[T.ins.y + 2u * i + 1u].c.x);
                sp += 2;
            }
            continue;
        }
        let nd = nodes[ni];
        if ((nd.b & 0x80000000u) != 0u) {
            let count = nd.b & 0x7fffffffu;
            /* a leaf of the top level (nodes below the world's): instances, into their meshes;
               a ball or a box is hit right here, in its own space */
            if (cur == NONE && ni < T.ins.w) {
                for (var k = 0u; k < count; k++) {
                    let i = aux[nd.a + k];
                    let kind = bitcast<u32>(prims[T.ins.y + 2u * i + 1u].c.z);
                    if (kind >= UNIT_SPHERE) {
                        let o = to_object(i, r);
                        var t = -1.0;
                        if (kind == UNIT_SPHERE) { t = hit_unit_sphere(o, tmin, best.t); } else { t = hit_unit_box(o, tmin, best.t); }
                        if (t > 0.0) {
                            best = Found(t, kind, vec2f(0.0), i);
                            if (stop_at_first) { return best; }
                        }
                        continue;
                    }
                    if (sp < STACK) {
                        stack[sp] = INST | i;
                        sp++;
                    }
                }
                continue;
            }
            /* a mesh's leaves hold its triangles' indices: + the primitives = shape ids */
            let sid0 = select(0u, T.opt.z, cur != NONE);
            for (var k = 0u; k < count; k++) {
                let sid = aux[nd.a + k] + sid0;
                if (sid < T.opt.z) {
                    let pr = prims[sid];
                    var t = -1.0;
                    if (pr.q.w < 0.5) { t = hit_sphere(pr, ray, tmin, best.t); } else { t = hit_quad(pr, ray, tmin, best.t); }
                    if (t > 0.0) {
                        best = Found(t, sid, vec2f(0.0), cur);
                        if (stop_at_first) { return best; }
                    }
                } else {
                    let h = hit_tri(tris[sid - T.opt.z], ray, tmin, best.t);
                    if (h.x > 0.0) {
                        best = Found(h.x, sid, h.yz, cur);
                        if (stop_at_first) { return best; }
                    }
                }
            }
            continue;
        }
        /* inner node: the farther child is pushed first, so the nearer one is visited first */
        let tl = box_t(nodes[nd.a].lo, nodes[nd.a].hi, ray.o, inv, tmin, best.t);
        let tr = box_t(nodes[nd.b].lo, nodes[nd.b].hi, ray.o, inv, tmin, best.t);
        if (tl >= 0.0 && tr >= 0.0 && sp + 2 <= STACK) {
            let left_first = tl <= tr;
            stack[sp] = select(nd.a, nd.b, left_first);
            stack[sp + 1] = select(nd.b, nd.a, left_first);
            sp += 2;
        } else if (tl >= 0.0 && sp < STACK) {
            stack[sp] = nd.a;
            sp++;
        } else if (tr >= 0.0 && sp < STACK) {
            stack[sp] = nd.b;
            sp++;
        }
    }
    return best;
}

fn normal_at(i: u32) -> vec3f {
    let o = T.cnt.y + 3u * i;
    return vec3f(bitcast<f32>(aux[o]), bitcast<f32>(aux[o + 1u]), bitcast<f32>(aux[o + 2u]));
}

fn shape_mat(id: u32) -> u32 {
    if (id < T.opt.z) { return u32(prims[id].u.w); }
    return tris[id - T.opt.z].nm >> 28u;
}
fn hit_mat(f: Found) -> u32 {
    if (f.inst != NONE) {
        let ov = bitcast<u32>(prims[T.ins.y + 2u * f.inst + 1u].c.y);
        if (ov != NONE) { return ov; }
    }
    return shape_mat(f.id);
}

/* the hit record: point, normals facing the ray (shading and geometric), which side */
fn make_hit(r: Ray, f: Found) -> Hit {
    var h: Hit;
    h.t = f.t;
    h.p = r.o + r.d * f.t;
    var outward = vec3f(0.0, 1.0, 0.0);
    var ns = vec3f(0.0);
    var smooth_n = false;
    if (f.id >= UNIT_SPHERE) {                /* a ball or a box: the normal in its space */
        let o = to_object(f.inst, r);
        let q = o.o + o.d * f.t;
        var no = q;
        if (f.id == UNIT_BOX) {
            let aq = abs(q);
            if (aq.x >= aq.y && aq.x >= aq.z) { no = vec3f(sign(q.x), 0.0, 0.0); }
            else if (aq.y >= aq.z) { no = vec3f(0.0, sign(q.y), 0.0); }
            else { no = vec3f(0.0, 0.0, sign(q.z)); }
        }
        outward = normal_to_world(f.inst, no);
    } else if (f.id < T.opt.z) {
        let pr = prims[f.id];
        if (pr.q.w < 0.5) { outward = (h.p - pr.c.xyz) / pr.c.w; } else { outward = pr.n.xyz; }
    } else {
        let tr = tris[f.id - T.opt.z];
        outward = normalize(cross(tr.u, tr.v));
        if (tr.n0 != NONE) {
            ns = normalize(normal_at(tr.n0) * (1.0 - f.b.x - f.b.y) + normal_at(tr.n1) * f.b.x +
                           normal_at(tr.nm & 0x0fffffffu) * f.b.y);
            smooth_n = true;
        }
        if (f.inst != NONE) {                 /* an instance: its normals into the world */
            outward = normal_to_world(f.inst, outward);
            if (smooth_n) { ns = normal_to_world(f.inst, ns); }
        }
    }
    h.front = dot(r.d, outward) < 0.0;
    h.n = select(-outward, outward, h.front);
    h.ng = h.n;
    if (smooth_n) {
        if (dot(ns, h.n) < 0.0) { ns = -ns; }
        if (dot(ns, ns) > 0.5) { h.n = ns; }
    }
    return h;
}

/* ---- media (rt/medium.c) ---- */

/* the part of the ray inside a box: (t0, t1), t1 <= t0 when there is none */
fn clip_box(lo: vec3f, hi: vec3f, r: Ray, tmin: f32, tmax: f32) -> vec2f {
    var a = tmin;
    var b = tmax;
    for (var i = 0; i < 3; i++) {
        let d = r.d[i];
        let o = r.o[i];
        if (d == 0.0) {
            if (o < lo[i] || o > hi[i]) { return vec2f(1.0, 0.0); }
            continue;
        }
        let inv = 1.0 / d;
        let ta = min((lo[i] - o) * inv, (hi[i] - o) * inv);
        let tb = max((lo[i] - o) * inv, (hi[i] - o) * inv);
        a = max(a, ta);
        b = min(b, tb);
        if (a >= b) { return vec2f(1.0, 0.0); }
    }
    return vec2f(a, b);
}

/* Tracking through medium k along the ray, brick by brick (Amanatides and Woo DDA over the
   majorant grid), each brick with its own majorant.
   ratio = false: delta tracking, returns the distance of the first real collision, or -1;
   ratio = true: ratio tracking, returns the transmittance. */
fn track(k: u32, r: Ray, tmin: f32, tmax: f32, rng: ptr<function, u32>, ratio: bool) -> f32 {
    let nothing = select(-1.0, 1.0, ratio);
    let v = F.vols[k];
    let sig = v.albedo.w;
    let mx = VS[k].mx.x;
    if (!(mx > 0.0) || sig <= 0.0) { return nothing; }
    let seg = clip_box(VS[k].alo.xyz, VS[k].ahi.xyz, r, tmin, tmax);
    if (seg.y <= seg.x) { return nothing; }
    var t = seg.x;
    let t1 = seg.y;
    var tr = 1.0;
    /* thin: one segment with the global majorant is cheaper than walking the bricks */
    let whole = sig * mx * (t1 - t) < 4.0;
    var bi = vec3i(0);
    var sgn = vec3i(0);
    var nxt = vec3f(1e30);
    var dlt = vec3f(1e30);
    let nb = vec3i(v.bdims.xyz);
    if (!whole) {
        let bs = (v.hi.xyz - v.lo.xyz) / vec3f(v.dims.xyz) * 8.0;
        for (var a = 0; a < 3; a++) {
            let p = r.o[a] + r.d[a] * t;
            let i = clamp(i32(floor((p - v.lo[a]) / bs[a])), 0, nb[a] - 1);
            bi[a] = i;
            if (r.d[a] > 0.0) {
                sgn[a] = 1;
                nxt[a] = (v.lo[a] + f32(i + 1) * bs[a] - r.o[a]) / r.d[a];
                dlt[a] = bs[a] / r.d[a];
            } else if (r.d[a] < 0.0) {
                sgn[a] = -1;
                nxt[a] = (v.lo[a] + f32(i) * bs[a] - r.o[a]) / r.d[a];
                dlt[a] = -bs[a] / r.d[a];
            }
        }
    }
    for (var guard = 0; guard < 4096; guard++) {
        if (t >= t1) { break; }
        let ta = t;
        var tb = t1;
        var smax = sig * mx;
        if (!whole) {
            if (any(bi < vec3i(0)) || any(bi >= nb)) { break; }
            var ax = 2;
            if (nxt.x < nxt.y) { if (nxt.x < nxt.z) { ax = 0; } } else if (nxt.y < nxt.z) { ax = 1; }
            tb = max(min(nxt[ax], t1), t);
            smax = sig * bricks[v.bdims.w + u32((bi.z * nb.y + bi.y) * nb.x + bi.x)];
            bi[ax] += sgn[ax];
            nxt[ax] += dlt[ax];
        }
        t = tb;
        if (smax <= 0.0) { continue; }
        var tt = ta;
        for (var n = 0; n < 65536; n++) {
            tt -= log(1.0 - rnd(rng)) / smax;
            if (tt >= tb) { break; }
            let s = sig * density(k, r.o + r.d * tt);
            if (ratio) {
                tr *= 1.0 - min(s, smax) / smax;
                if (tr < 0.1) {
                    if (rnd(rng) < 0.5) { return 0.0; }
                    tr *= 2.0;
                }
            } else if (rnd(rng) * smax < s) {
                return tt;
            }
        }
    }
    return select(-1.0, tr, ratio);
}

fn transmittance(r: Ray, tmin: f32, tmax: f32, rng: ptr<function, u32>) -> f32 {
    var tr = 1.0;
    for (var k = 0u; k < T.cnt.w; k++) {
        if (tr <= 0.0) { break; }
        tr *= track(k, r, tmin, tmax, rng, true);
    }
    return tr;
}

fn hg(c: f32, g: f32) -> f32 {
    let d = 1.0 + g * g - 2.0 * g * c;
    return (1.0 - g * g) / (4.0 * PI_ * d * sqrt(d));
}

/* two tangents of n (Duff et al. 2017) */
fn onb(n: vec3f) -> mat2x3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let c = n.x * n.y * a;
    return mat2x3f(vec3f(1.0 + s * n.x * n.x * a, s * c, -s * n.x), vec3f(c, s + n.y * n.y * a, -n.y));
}

fn cosine_hemisphere(n: vec3f, u: vec2f) -> vec3f {
    let u1 = u.x;
    let u2 = u.y;
    let phi = 2.0 * PI_ * u1;
    let rad = sqrt(u2);
    let tb = onb(n);
    return tb[0] * (rad * cos(phi)) + tb[1] * (rad * sin(phi)) + n * sqrt(max(0.0, 1.0 - u2));
}

/* a direction drawn from the Henyey-Greenstein phase function around d */
fn sample_phase(d: vec3f, g: f32, u: vec2f) -> vec3f {
    let xi = u.x;
    var cos_t = 1.0 - 2.0 * xi;
    if (abs(g) >= 1e-3) {
        let sq = (1.0 - g * g) / (1.0 + g - 2.0 * g * xi);
        cos_t = (1.0 + g * g - sq * sq) / (2.0 * g);
    }
    cos_t = clamp(cos_t, -1.0, 1.0);
    let sin_t = sqrt(1.0 - cos_t * cos_t);
    let phi = 2.0 * PI_ * u.y;
    let tb = onb(d);
    return tb[0] * (sin_t * cos(phi)) + tb[1] * (sin_t * sin(phi)) + d * cos_t;
}

/* ---- light transport (rt/tracer.c) ---- */

fn power_heuristic(pa: f32, pb: f32) -> f32 {
    let a = pa * pa;
    let b = pb * pb;
    return select(0.0, a / (a + b), a + b > 0.0);
}
fn light_pdf(area: f32, dist2: f32, cos_light: f32) -> f32 { return dist2 / (cos_light * area); }

/* next event estimation for a diffuse surface: one point of the lamp */
/* (u: where on the lamp) */
fn direct_surface(h: Hit, albedo: vec3f, u: vec2f, rng: ptr<function, u32>) -> vec3f {
    if (T.opt.w == NONE) { return vec3f(0.0); }
    let lt = prims[T.opt.w];
    let le = T.mats[u32(lt.u.w)].emission.rgb;
    let r1 = u.x;
    let r2 = u.y;
    let to_light = lt.q.xyz + lt.u.xyz * r1 + lt.v.xyz * r2 - h.p;
    let dist2 = dot(to_light, to_light);
    let dist = sqrt(dist2);
    let wi = to_light / dist;
    let cos_surf = dot(h.n, wi);
    let cos_light = -dot(lt.n.xyz, wi);
    if (cos_surf <= 0.0 || cos_light <= 0.0) { return vec3f(0.0); }
    let shadow = Ray(h.p, wi);
    if (traverse(shadow, T.f.y, dist - T.f.z, true).id != NONE) { return vec3f(0.0); }
    var g = cos_surf * cos_light * lt.v.w / dist2;
    if (T.cnt.w > 0u) { g *= transmittance(shadow, 0.0, dist, rng); }
    if (T.opt.y != 0u) { g *= power_heuristic(light_pdf(lt.v.w, dist2, cos_light), cos_surf / PI_); }
    return albedo * le * (g / PI_);
}

/* the same for a point inside a medium, reached along d: the phase function replaces the BRDF */
fn direct_medium(p: vec3f, d: vec3f, g: f32, u: vec2f, rng: ptr<function, u32>) -> vec3f {
    if (T.opt.w == NONE) { return vec3f(0.0); }
    let lt = prims[T.opt.w];
    let le = T.mats[u32(lt.u.w)].emission.rgb;
    let r1 = u.x;
    let r2 = u.y;
    let to_light = lt.q.xyz + lt.u.xyz * r1 + lt.v.xyz * r2 - p;
    let dist2 = dot(to_light, to_light);
    let dist = sqrt(dist2);
    let wi = to_light / dist;
    let cos_light = -dot(lt.n.xyz, wi);
    if (cos_light <= 0.0) { return vec3f(0.0); }
    let shadow = Ray(p, wi);
    if (traverse(shadow, T.f.y, dist - T.f.z, true).id != NONE) { return vec3f(0.0); }
    let tr = transmittance(shadow, 0.0, dist, rng);
    if (tr <= 0.0) { return vec3f(0.0); }
    let ph = hg(dot(d, wi), g);
    var w = ph * tr * cos_light * lt.v.w / dist2;
    if (T.opt.y != 0u) { w *= power_heuristic(light_pdf(lt.v.w, dist2, cos_light), ph); }
    return le * w;
}

fn schlick(cos_i: f32, eta: f32) -> f32 {
    let r0 = (1.0 - eta) / (1.0 + eta);
    return r0 * r0 + (1.0 - r0 * r0) * pow(1.0 - cos_i, 5.0);
}

fn refract_dir(d: vec3f, n: vec3f, eta: f32) -> vec3f {
    let cos_i = min(dot(-d, n), 1.0);
    let perp = (d + n * cos_i) * eta;
    return perp - n * sqrt(abs(1.0 - dot(perp, perp)));
}

/* Russian roulette after 3 bounces: false = the path stops */
fn roulette(thr: ptr<function, vec3f>, depth: i32, sm: Smp, rng: ptr<function, u32>) -> bool {
    if (depth < 3) { return true; }
    let c = *thr;
    let p = min(max(c.x, max(c.y, c.z)), 0.95);
    if (s1(sm, 3u + 3u * u32(depth), 0u, rng) >= p) { return false; }
    *thr = c / p;
    return true;
}

/* The denoiser's guide for one pixel, along the ray through its centre: the first diffuse
   surface, through mirrors and glass (refracting, unless the glass reflects all), with the
   albedo of the whole way to it. A medium that hides it (transmittance under 1/2, the mean
   of 4 ratio tracking estimates) makes the pixel a medium's. */
struct Guide { n: vec3f, z: f32, a: vec3f, kind: f32, pos: vec3f, pt: f32 };
fn guide(ray0: Ray, rng: ptr<function, u32>) -> Guide {
    var g: Guide;
    g.n = vec3f(0.0); g.z = 1e30; g.a = vec3f(1.0); g.kind = 1.0;
    g.pos = vec3f(0.0); g.pt = 0.0;      /* the camera's first hit, if diffuse: where to look up caustics */
    var ray = ray0;
    var a = vec3f(1.0);
    var dist = 0.0;
    var tr = 1.0;
    var zmed = -1.0;
    for (var i = 0; i < 6; i++) {
        let f = traverse(ray, T.f.y, 1e30, false);
        let found = f.id != NONE;
        if (T.cnt.w > 0u && tr >= 0.5) {
            /* how much of what is behind shows through, and where the light scatters (the mean
               of 4 collision distances: a medium's pixel is reprojected from there) */
            let tmax = select(1e30, f.t, found);
            var s = 0.0;
            var zc = 0.0;
            var nc = 0.0;
            for (var j = 0; j < 4; j++) {
                s += transmittance(ray, 0.0, tmax, rng);
                var tc = tmax;
                for (var mv = 0u; mv < T.cnt.w; mv++) {
                    let ti = track(mv, ray, 0.0, tc, rng, false);
                    if (ti >= 0.0) { tc = ti; }
                }
                if (tc < tmax) { zc += tc; nc += 1.0; }
            }
            tr *= s / 4.0;
            if (tr < 0.5 && nc > 0.0) { zmed = dist + zc / nc; }
        }
        if (!found) { break; }
        let h = make_hit(ray, f);
        let m = T.mats[hit_mat(f)];
        let kind = u32(m.albedo.w);
        if (kind == 3u) { g.n = h.n; g.z = dist + h.t; break; }    /* a lamp: kept for reprojection */
        if (kind == 0u) {
            g.n = h.n; g.z = dist + h.t; g.a = a * m.albedo.rgb; g.kind = 0.0;
            if (i == 0) { g.pos = h.p; g.pt = max(tr, 1e-6); }
            break;
        }
        a *= m.albedo.rgb;
        dist += h.t;
        var nxt = reflect(ray.d, h.n);
        if (kind == 2u) {
            let eta = select(m.emission.w, 1.0 / m.emission.w, h.front);
            let cos_i = min(dot(-ray.d, h.n), 1.0);
            if (eta * eta * (1.0 - cos_i * cos_i) <= 1.0) { nxt = refract_dir(ray.d, h.n, eta); }
        }
        ray = Ray(h.p, normalize(nxt));
    }
    if (tr < 0.5) {
        g.kind = 2.0;               /* (the albedo is kept: only the caustics' lookup uses it then) */
        g.n = vec3f(0.0);
        if (zmed > 0.0) { g.z = zmed; }
    }
    return g;
}

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

/* the radiance arriving along one camera ray */
fn trace(ray0: Ray, sm: Smp, rng: ptr<function, u32>) -> vec3f {
    var ray = ray0;
    var radiance = vec3f(0.0);
    var thr = vec3f(1.0);
    var emit = 0;             /* a lamp hit counts: 0 fully, 1 not (NEE had it), 2 weighted against NEE */
    var bounce_pdf = 0.0;
    let max_depth = i32(T.opt.x);
    let mis = T.opt.y != 0u;
    let after_bounce = select(1, 2, mis);
    /* skip_caustics (rt/tracer.c): the camera saw a diffuse surface first (chain) and the path
       reaches the lamp right after a mirror / glass (spec): light tracing adds that one */
    var chain = false;
    var spec = false;
    for (var depth = 0; depth < 4096; depth++) {
        let f = traverse(ray, T.f.y, 1e30, false);
        let found = f.id != NONE;
        if (T.cnt.w > 0u) {
            /* the ray may scatter in a medium before the surface: the earliest collision wins */
            var ts = select(1e30, f.t, found);
            var which = NONE;
            for (var k = 0u; k < T.cnt.w; k++) {
                let ti = track(k, ray, 0.0, ts, rng, false);
                if (ti >= 0.0) { ts = ti; which = k; }
            }
            if (which != NONE) {
                if (depth >= max_depth) { break; }
                let vol = F.vols[which];
                let p = ray.o + ray.d * ts;
                thr *= vol.albedo.xyz;
                let dd = 3u * u32(depth);
                radiance += thr * direct_medium(p, ray.d, vol.p.x, s2(sm, 1u + dd, rng), rng);
                let nd = sample_phase(ray.d, vol.p.x, s2(sm, 2u + dd, rng));
                bounce_pdf = hg(dot(ray.d, nd), vol.p.x);
                ray = Ray(p, nd);
                emit = after_bounce;
                chain = false;
                if (!roulette(&thr, depth, sm, rng)) { break; }
                continue;
            }
        }
        if (!found) { break; }
        let h = make_hit(ray, f);
        let m = T.mats[hit_mat(f)];
        let kind = u32(m.albedo.w);
        if (kind == 3u) {
            if (h.front && !(chain && spec)) {
                if (emit == 0) {
                    radiance += thr * m.emission.rgb;
                } else if (emit == 2 && f.id == T.opt.w) {
                    let lt = prims[T.opt.w];
                    let cos_light = -dot(lt.n.xyz, ray.d);
                    if (cos_light > 0.0) {
                        radiance += thr * m.emission.rgb * power_heuristic(bounce_pdf, light_pdf(lt.v.w, h.t * h.t, cos_light));
                    }
                }
            }
            break;
        }
        if (depth >= max_depth) { break; }
        let d = ray.d;
        var nxt = d;
        if (kind == 0u) {
            let dd = 3u * u32(depth);
            radiance += thr * direct_surface(h, m.albedo.rgb, s2(sm, 1u + dd, rng), rng);
            nxt = cosine_hemisphere(h.n, s2(sm, 2u + dd, rng));
            bounce_pdf = dot(h.n, nxt) / PI_;
            thr *= m.albedo.rgb;
            emit = after_bounce;
            if (depth == 0) { chain = T.ext.z != 0u; }
            spec = false;
        } else if (kind == 1u) {
            nxt = reflect(d, h.n);
            thr *= m.albedo.rgb;
            emit = 0;
            spec = true;
        } else {
            let ior = m.emission.w;
            let eta = select(ior, 1.0 / ior, h.front);
            let cos_i = min(dot(-d, h.n), 1.0);
            let sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
            if (sin_t2 > 1.0 || s1(sm, 3u + 3u * u32(depth), 1u, rng) < schlick(cos_i, eta)) { nxt = reflect(d, h.n); } else { nxt = refract_dir(d, h.n, eta); }
            thr *= m.albedo.rgb;
            emit = 0;
            spec = true;
        }
        if (!roulette(&thr, depth, sm, rng)) { break; }
        ray = Ray(h.p, normalize(nxt));
    }
    return radiance;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = gid.x + gid.y * nw.x * 64u;
    let W = T.dims.x;
    if (id >= W * T.dims.w) { return; }
    let x = id % W;
    let y = T.dims.z + id / W;
    if (y >= T.dims.y) { return; }
    let k = y * W + x;
    let acc = accum[k];
    /* seeded from (pixel, sample number): every sample of every pixel is different */
    /* the plain random numbers: from (pixel, sample number), every sample of every pixel differs;
       the sequence's: sample T.cnt.z of the image (all pixels alike, shifted by blue noise) */
    var rng = hash32(k ^ hash32(T.cnt.z * 0x9e3779b9u + u32(acc.w)));
    let sm = Smp(x, y, T.cnt.z, T.ext.x);
    let j = s2(sm, 0u, &rng);
    let u = (f32(x) + j.x) / f32(W);
    let v = (f32(y) + j.y) / f32(T.dims.y);
    let through = T.upper_left.xyz + T.horizontal.xyz * u - T.vertical.xyz * v;
    var c = trace(Ray(T.origin.xyz, normalize(through - T.origin.xyz)), sm, &rng);
    let mx = max(c.x, max(c.y, c.z));
    if (!(mx >= 0.0 && mx < 1e30)) {
        c = vec3f(0.0);
    } else if (T.f.x > 0.0 && mx > T.f.x) {
        c *= T.f.x / mx;
    }
    accum[k] = acc + vec4f(c, 1.0);
    if (T.f.w > 0.5) {
        let N = W * T.dims.y;
        let ga = accum[3u * N + k];           /* the guide, made by gbuf before the first sample */
        let l = luma(select(c, c / max(ga.xyz, vec3f(1e-3)), ga.w == 0.0));
        accum[N + k] += vec4f(l, l * l, 0.0, 0.0);
    }
}

/* A new image's start (with T.f.w = 1): no samples, and each pixel's guide. The whole image
   (dims z and w, the band, are not used). */
@compute @workgroup_size(64)
fn gbuf(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let k = gid.x + gid.y * nw.x * 64u;
    let W = T.dims.x;
    let N = W * T.dims.y;
    if (k >= N) { return; }
    let x = k % W;
    let y = k / W;
    var rng = hash32(k ^ hash32(0x2545f491u + T.cnt.z));
    let pc = T.upper_left.xyz + T.horizontal.xyz * ((f32(x) + 0.5) / f32(W)) -
             T.vertical.xyz * ((f32(y) + 0.5) / f32(T.dims.y));
    let g = guide(Ray(T.origin.xyz, normalize(pc - T.origin.xyz)), &rng);
    accum[k] = vec4f(0.0);
    accum[N + k] = vec4f(0.0);
    accum[2u * N + k] = vec4f(g.n, g.z);
    accum[3u * N + k] = vec4f(g.a, g.kind);
    accum[4u * N + k] = vec4f(g.pos, g.pt);
}

/* ---- light tracing for caustics (light_trace_caustics in rt/tracer.c) ----
   One thread per light path: from the lamp, through mirrors and glass, onto diffuse surfaces,
   each connected to the camera and added to the pixel it lands in. Only the paths that pass a
   mirror or glass first; with a medium only those that do not scatter in it (the path tracer
   leaves exactly these out: chain / spec in trace). */
fn splat(pix: u32, c: vec3f) {
    for (var ch = 0u; ch < 3u; ch++) {
        let v = c[ch] * T.lt.x;
        if (!(v > 0.0) || v > 1.8e19) { continue; }
        let hi = floor(v / 4294967296.0);
        let lo = u32(min(v - hi * 4294967296.0, 4294967295.0));
        let at = (pix * 3u + ch) * 2u;
        let old = atomicAdd(&caus[at], lo);
        let carry = u32(hi) + select(0u, 1u, old > 0xffffffffu - lo);
        if (carry != 0u) { atomicAdd(&caus[at + 1u], carry); }
    }
}

/* The photon cache (viewer, real time): light paths leave their power at every diffuse vertex
   they reach, in a world-space hash grid of cells (T.lt.z wide, one per main normal axis too),
   which the resolve pass reads; it fades from frame to frame while things move. */
const PC_CELLS = 1048576u;
const PC_SCALE = 256.0;
fn pc_axis(n: vec3f) -> u32 {
    let a = abs(n);
    if (a.x >= a.y && a.x >= a.z) { return select(1u, 0u, n.x > 0.0); }
    if (a.y >= a.z) { return select(3u, 2u, n.y > 0.0); }
    return select(5u, 4u, n.z > 0.0);
}
/* 3 sizes of cells (x1, x4, x16) in one table: the lookup takes the finest that holds enough
   photons (adaptive density estimation). A slot: 3 x (low, high) sums, the photon count, the
   cell's key (0 = free). A cell takes the first slot from its hash on that is free or its own
   (up to 8 on): two cells never share a slot (that would light one with the other's photons). */
fn pc_key(c: vec3i, axis: u32, level: u32) -> u32 {
    return hash32((u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u) ^
                  ((axis + 6u * level) * 2654435761u));
}
fn pc_claim(c: vec3i, axis: u32, level: u32) -> u32 {
    let h = pc_key(c, axis, level);
    let key = hash32(h ^ 0x68e31da4u) | 1u;
    var idx = h & (PC_CELLS - 1u);
    var probe = 0u;
    for (var tries = 0u; tries < 16u && probe < 8u; tries++) {
        let r = atomicCompareExchangeWeak(&caus[idx * 8u + 7u], 0u, key);
        if (r.exchanged || r.old_value == key) { return idx; }
        if (r.old_value != 0u) {                       /* taken by another cell: the next slot */
            idx = (idx + 1u) & (PC_CELLS - 1u);
            probe++;
        }                                              /* (else a spurious failure: again) */
    }
    return 0xffffffffu;
}
fn deposit(p: vec3f, n: vec3f, c: vec3f) {
    let axis = pc_axis(n);
    var size = T.lt.z;
    for (var level = 0u; level < 3u; level++) {
        let cell = pc_claim(vec3i(floor(p / size)), axis, level);
        size *= 4.0;
        if (cell == 0xffffffffu) { continue; }
        for (var ch = 0u; ch < 3u; ch++) {
            let v = c[ch] * PC_SCALE;
            if (!(v > 0.0) || v > 1.8e19) { continue; }
            let hi = floor(v / 4294967296.0);
            let lo = u32(min(v - hi * 4294967296.0, 4294967295.0));
            let at = cell * 8u + ch * 2u;
            let old = atomicAdd(&caus[at], lo);
            let carry = u32(hi) + select(0u, 1u, old > 0xffffffffu - lo);
            if (carry != 0u) { atomicAdd(&caus[at + 1u], carry); }
        }
        atomicAdd(&caus[cell * 8u + 6u], 256u);       /* photons, 8 bits of fraction (it fades) */
    }
}

fn rr_light(beta: ptr<function, vec3f>, thr: ptr<function, vec3f>, depth: i32, rng: ptr<function, u32>) -> bool {
    if (depth < 3) { return true; }
    let c = *thr;
    let p = min(max(c.x, max(c.y, c.z)), 0.95);
    if (rnd(rng) >= p) { return false; }
    *beta = *beta / p;
    *thr = c / p;
    return true;
}

@compute @workgroup_size(64)
fn ltrace(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = gid.x + gid.y * nw.x * 64u;
    if (id >= T.ext.w || T.opt.w == NONE) { return; }
    let W = T.dims.x;
    let H = T.dims.y;
    var rng = hash32(id ^ hash32(T.cnt.z * 0x9e3779b9u + 0x632be5abu));
    let lt = prims[T.opt.w];
    let le = T.mats[u32(lt.u.w)].emission.rgb;
    let a0 = rnd(&rng);
    let a1 = rnd(&rng);
    let b0 = rnd(&rng);
    let b1 = rnd(&rng);
    var ray = Ray(lt.q.xyz + lt.u.xyz * a0 + lt.v.xyz * a1, cosine_hemisphere(lt.n.xyz, vec2f(b0, b1)));
    var beta = le * (lt.v.w * PI_);          /* Le cos / (pdf_area pdf_dir) */
    var thr = vec3f(1.0);
    var nspec = 0;
    var ndiff = 0;
    let center = T.upper_left.xyz + 0.5 * T.horizontal.xyz - 0.5 * T.vertical.xyz;
    let fw = center - T.origin.xyz;           /* the screen is 1 unit away */
    for (var depth = 0; depth < i32(T.opt.x); depth++) {
        let f = traverse(ray, T.f.y, 1e30, false);
        let found = f.id != NONE;
        if (T.cnt.w > 0u) {
            var scattered = false;
            for (var k = 0u; k < T.cnt.w; k++) {
                if (track(k, ray, 0.0, select(1e30, f.t, found), &rng, false) >= 0.0) { scattered = true; }
            }
            if (scattered) { break; }
        }
        if (!found) { break; }
        let h = make_hit(ray, f);
        let m = T.mats[hit_mat(f)];
        let kind = u32(m.albedo.w);
        if (kind == 3u) { break; }
        if (kind == 0u) {
            if (nspec == 0 && ndiff == 0) { break; }       /* lamp -> diffuse: the path tracer's (NEE) */
            let wi = -ray.d;
            let cg = abs(dot(h.ng, wi));
            let cs = abs(dot(h.n, wi));
            let corr = select(0.0, cs / cg, cg > 0.0);    /* smooth normals: the adjoint BSDF */
            if (T.lt.w > 0.5) {
                deposit(h.p, h.n, beta * corr);            /* photon cache mode: no camera */
            } else {
            /* 1. to the camera */
            let to_cam = T.origin.xyz - h.p;
            let dist2 = dot(to_cam, to_cam);
            let dist = sqrt(dist2);
            let wo = to_cam / dist;
            let cos_o = dot(h.ng, wo);
            let q = -to_cam;
            let z = dot(q, fw);
            if (cos_o > 0.0 && z > 0.0) {
                let e = q / z - fw;
                let sx = 0.5 + dot(e, T.horizontal.xyz) / dot(T.horizontal.xyz, T.horizontal.xyz);
                let sy = 0.5 - dot(e, T.vertical.xyz) / dot(T.vertical.xyz, T.vertical.xyz);
                if (sx >= 0.0 && sy >= 0.0 && sx < 1.0 && sy < 1.0) {
                    let px = min(u32(sx * f32(W)), W - 1u);
                    let py = min(u32(sy * f32(H)), H - 1u);
                    let sh = Ray(h.p, wo);
                    if (traverse(sh, T.f.y, dist * (1.0 - 1e-4), true).id == NONE) {
                        var tr = 1.0;
                        if (T.cnt.w > 0u) { tr = transmittance(sh, 0.0, dist, &rng); }
                        let cos_c = z / dist;
                        let wgt = corr * cos_o * tr / (PI_ * dist2 * T.lt.y * cos_c * cos_c * cos_c);
                        splat(py * W + px, beta * m.albedo.rgb * wgt);
                    }
                }
            }
            }
            /* 2. bounce on (cosine around the shading normal): later diffuse vertices count too */
            let c0 = rnd(&rng);
            let nd = cosine_hemisphere(h.n, vec2f(c0, rnd(&rng)));
            let og = dot(h.ng, nd);
            let os = dot(h.n, nd);
            if (og <= 0.0 || os <= 0.0) { break; }
            let fb = m.albedo.rgb * (corr * og / os);
            beta *= fb;
            thr *= fb;
            if (!rr_light(&beta, &thr, depth, &rng)) { break; }
            ray = Ray(h.p, nd);
            ndiff++;
            continue;
        }
        /* mirror or glass, as in the path tracer */
        let d = ray.d;
        var nxt = reflect(d, h.n);
        if (kind == 2u) {
            let eta = select(m.emission.w, 1.0 / m.emission.w, h.front);
            let cos_i = min(dot(-d, h.n), 1.0);
            let sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
            if (!(sin_t2 > 1.0 || rnd(&rng) < schlick(cos_i, eta))) { nxt = refract_dir(d, h.n, eta); }
        }
        beta *= m.albedo.rgb;
        thr *= m.albedo.rgb;
        nspec++;
        if (!rr_light(&beta, &thr, depth, &rng)) { break; }
        ray = Ray(h.p, normalize(nxt));
    }
}
);

// Reprojection: the new image's pixels take the samples of the previous image where their
// surface was (both images' planes, the new one's guide already made by gbuf). The guide's
// point (along the pixel's ray, at the distance of its first diffuse surface: through a
// mirror, the point behind it, which moves as the reflection does) is projected into the old
// camera; of the 4 old pixels around it (bilinear weights), those that saw the same thing
// count: same kind, a normal within ~25 degrees, a distance from the old eye within 2% (25%
// in a medium: its distance is a noisy estimate, and its light smooth) plus how much it changes from that old pixel to
// the next (on a wall seen edgewise it changes a lot within a pixel; the smaller of the two
// sides, so that an edge next to the pixel does not count). A pixel that matches none (it was hidden, or
// off screen) starts with no samples. The count is capped at `history`: older samples weigh
// less, so that a change of light fades in, and the blur of blending again every frame (and
// the streak it drags a bright sample into, along the motion) stays short. (Taking the
// nearest old pixel instead does not blur, but duplicates pixels under subpixel motion: the
// noise lines up in streaks, worse.)
static const char *WAVE = WGSL(
/* ---- the wavefront path tracer (-wavefront on; docs/PERFORMANCE.md R1) ----
   The same light transport as trace() above, cut into small kernels that each do one step
   for every path of the band at once, the paths' state kept in WF between them:
     wf_gen      a camera ray per pixel of the band (the paths, in queue 0)
     wf_ext      the closest hit of each path in the queue (the BVH walk alone)
     wf_shade    what the hit does: media, lamps, materials, the next ray (into the other
                 queue) and the shadow ray toward the lamp (into the shadow queue)
     wf_args     the queues' sizes into the next dispatches' workgroup counts (indirect)
     wf_shadow   whether each shadow ray reaches the lamp (and through the media)
     wf_resolve  every path's radiance into the image, as main() does
   wf_ext, wf_shade, wf_args and wf_shadow run once per bounce. Each kernel holds only what
   its step needs, so the GPU runs more threads of it at once, and a path that ends early
   no longer keeps a thread waiting for the longest one of its group.
   WF (vec4u): [0] the control: x = which queue is current (0 / 1), y = its size, z = the
   shadow queue's size; then per path 4 (state), 1 (its hit), 3 (its shadow ray); then the
   queues (path indices, 4 per vec4): 0, 1 and the shadow queue. WC: the counters the shade
   step adds to. WA: the indirect dispatches' workgroup counts (x, y, z) for the paths and the
   shadow rays. */
@group(0) @binding(12) var<storage, read_write> WF: array<vec4u>;
@group(0) @binding(13) var<storage, read_write> WC: array<atomic<u32>, 4>;
@group(0) @binding(14) var<storage, read_write> WA: array<u32, 8>;

fn wf_n() -> u32 { return T.dims.x * T.dims.w; }
fn wf_st(p: u32) -> u32 { return 1u + 4u * p; }
fn wf_hit(p: u32) -> u32 { return 1u + 4u * wf_n() + p; }
fn wf_sh(p: u32) -> u32 { return 1u + 5u * wf_n() + 3u * p; }
fn wf_q(which: u32) -> u32 { return 1u + 8u * wf_n() + which * ((wf_n() + 3u) / 4u); }
fn q_get(which: u32, i: u32) -> u32 { return WF[wf_q(which) + i / 4u][i % 4u]; }
fn q_set(which: u32, i: u32, v: u32) { WF[wf_q(which) + i / 4u][i % 4u] = v; }

/* a path: its ray, the pdf of the bounce that made it, its pixel, throughput, random state,
   radiance so far, and flags: depth (bits 0-15), emit (16-17), chain (18), spec (19) */
struct PS { o: vec3f, pdf: f32, d: vec3f, pix: u32, thr: vec3f, rng: u32, rad: vec3f, fl: u32 };
fn ps_load(p: u32) -> PS {
    let s = wf_st(p);
    let a = WF[s]; let b = WF[s + 1u]; let c = WF[s + 2u]; let e = WF[s + 3u];
    return PS(bitcast<vec3f>(a.xyz), bitcast<f32>(a.w), bitcast<vec3f>(b.xyz), b.w,
              bitcast<vec3f>(c.xyz), c.w, bitcast<vec3f>(e.xyz), e.w);
}
fn ps_store(p: u32, q: PS) {
    let s = wf_st(p);
    WF[s] = vec4u(bitcast<vec3u>(q.o), bitcast<u32>(q.pdf));
    WF[s + 1u] = vec4u(bitcast<vec3u>(q.d), q.pix);
    WF[s + 2u] = vec4u(bitcast<vec3u>(q.thr), q.rng);
    WF[s + 3u] = vec4u(bitcast<vec3u>(q.rad), q.fl);
}
fn wf_id(gid: vec3u, nw: vec3u) -> u32 { return gid.x + gid.y * nw.x * 64u; }

@compute @workgroup_size(64)
fn wf_gen(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = wf_id(gid, nw);
    let n = wf_n();
    if (id >= n) { return; }
    if (id == 0u) { WF[0] = vec4u(0u, n, 0u, 0u); }
    let W = T.dims.x;
    let x = id % W;
    let y = T.dims.z + id / W;
    let k = y * W + x;
    let acc = accum[k];
    var rng = hash32(k ^ hash32(T.cnt.z * 0x9e3779b9u + u32(acc.w)));
    let sm = Smp(x, y, T.cnt.z, T.ext.x);
    let j = s2(sm, 0u, &rng);
    let u = (f32(x) + j.x) / f32(W);
    let v = (f32(y) + j.y) / f32(T.dims.y);
    let through = T.upper_left.xyz + T.horizontal.xyz * u - T.vertical.xyz * v;
    ps_store(id, PS(T.origin.xyz, 0.0, normalize(through - T.origin.xyz), k, vec3f(1.0), rng, vec3f(0.0), 0u));
    q_set(0u, id, id);
}

@compute @workgroup_size(64)
fn wf_ext(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = wf_id(gid, nw);
    let c = WF[0];
    if (id >= c.y) { return; }
    let p = q_get(c.x & 1u, id);
    let s = wf_st(p);
    let o = bitcast<vec3f>(WF[s].xyz);
    let d = bitcast<vec3f>(WF[s + 1u].xyz);
    let f = traverse(Ray(o, d), T.f.y, 1e30, false);
    WF[wf_hit(p)] = vec4u(bitcast<u32>(f.t), f.id, f.inst, pack2x16float(f.b));
}

/* the shade step's shadow ray: from p along wi to the lamp at dist; c is what it brings if
   nothing is in the way (and times the media's transmittance) */
fn wf_shadow_ray(p: u32, o: vec3f, wi: vec3f, dist: f32, c: vec3f, seed: u32) {
    let s = wf_sh(p);
    WF[s] = vec4u(bitcast<vec3u>(o), bitcast<u32>(dist));
    WF[s + 1u] = vec4u(bitcast<vec3u>(wi), seed);
    WF[s + 2u] = vec4u(bitcast<vec3u>(c), 0u);
    let i = atomicAdd(&WC[1], 1u);
    q_set(2u, i, p);
}

/* direct_surface / direct_medium without the shadow ray: the light sample, and what it brings
   if the lamp is in sight (w = 0: nothing to send) */
struct LS { wi: vec3f, dist: f32, c: vec3f };
fn wf_light_surface(h: Hit, albedo: vec3f, u: vec2f) -> LS {
    var r = LS(vec3f(0.0), 0.0, vec3f(0.0));
    if (T.opt.w == NONE) { return r; }
    let lt = prims[T.opt.w];
    let le = T.mats[u32(lt.u.w)].emission.rgb;
    let to_light = lt.q.xyz + lt.u.xyz * u.x + lt.v.xyz * u.y - h.p;
    let dist2 = dot(to_light, to_light);
    let dist = sqrt(dist2);
    let wi = to_light / dist;
    let cos_surf = dot(h.n, wi);
    let cos_light = -dot(lt.n.xyz, wi);
    if (cos_surf <= 0.0 || cos_light <= 0.0) { return r; }
    var g = cos_surf * cos_light * lt.v.w / dist2;
    if (T.opt.y != 0u) { g *= power_heuristic(light_pdf(lt.v.w, dist2, cos_light), cos_surf / PI_); }
    return LS(wi, dist, albedo * le * (g / PI_));
}
fn wf_light_medium(p: vec3f, d: vec3f, gg: f32, u: vec2f) -> LS {
    var r = LS(vec3f(0.0), 0.0, vec3f(0.0));
    if (T.opt.w == NONE) { return r; }
    let lt = prims[T.opt.w];
    let le = T.mats[u32(lt.u.w)].emission.rgb;
    let to_light = lt.q.xyz + lt.u.xyz * u.x + lt.v.xyz * u.y - p;
    let dist2 = dot(to_light, to_light);
    let dist = sqrt(dist2);
    let wi = to_light / dist;
    let cos_light = -dot(lt.n.xyz, wi);
    if (cos_light <= 0.0) { return r; }
    let ph = hg(dot(d, wi), gg);
    var w = ph * cos_light * lt.v.w / dist2;
    if (T.opt.y != 0u) { w *= power_heuristic(light_pdf(lt.v.w, dist2, cos_light), ph); }
    return LS(wi, dist, le * w);
}

/* one bounce of trace() for each path of the queue */
@compute @workgroup_size(64)
fn wf_shade(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = wf_id(gid, nw);
    let cq = WF[0];
    if (id >= cq.y) { return; }
    let par = cq.x & 1u;
    let p = q_get(par, id);
    var s = ps_load(p);
    let hv = WF[wf_hit(p)];
    let f = Found(bitcast<f32>(hv.x), hv.y, unpack2x16float(hv.w), hv.z);
    var rng = s.rng;
    let depth = i32(s.fl & 0xffffu);
    var emit = i32((s.fl >> 16u) & 3u);
    var chain = (s.fl & (1u << 18u)) != 0u;
    var spec = (s.fl & (1u << 19u)) != 0u;
    var thr = s.thr;
    var ray = Ray(s.o, s.d);
    let max_depth = i32(T.opt.x);
    let after_bounce = select(1, 2, T.opt.y != 0u);
    let W = T.dims.x;
    let sm = Smp(s.pix % W, s.pix / W, T.cnt.z, T.ext.x);
    let found = f.id != NONE;
    var go_on = false;
    var nxt_o = vec3f(0.0);
    var nxt_d = vec3f(0.0);
    var done = false;
    if (T.cnt.w > 0u) {
        /* the ray may scatter in a medium before the surface: the earliest collision wins */
        var ts = select(1e30, f.t, found);
        var which = NONE;
        for (var k = 0u; k < T.cnt.w; k++) {
            let ti = track(k, ray, 0.0, ts, &rng, false);
            if (ti >= 0.0) { ts = ti; which = k; }
        }
        if (which != NONE) {
            done = true;
            if (depth < max_depth) {
                let vol = F.vols[which];
                let pp = ray.o + ray.d * ts;
                thr *= vol.albedo.xyz;
                let dd = 3u * u32(depth);
                let ls = wf_light_medium(pp, ray.d, vol.p.x, s2(sm, 1u + dd, &rng));
                if (ls.dist > 0.0) { wf_shadow_ray(p, pp, ls.wi, ls.dist, thr * ls.c, hash32(rng ^ 0x68bc21ebu)); }
                let nd = sample_phase(ray.d, vol.p.x, s2(sm, 2u + dd, &rng));
                s.pdf = hg(dot(ray.d, nd), vol.p.x);
                emit = after_bounce;
                chain = false;
                if (roulette(&thr, depth, sm, &rng)) { go_on = true; nxt_o = pp; nxt_d = nd; }
            }
        }
    }
    if (!done && found) {
        let h = make_hit(ray, f);
        let m = T.mats[hit_mat(f)];
        let kind = u32(m.albedo.w);
        if (kind == 3u) {
            if (h.front && !(chain && spec)) {
                if (emit == 0) {
                    s.rad += thr * m.emission.rgb;
                } else if (emit == 2 && f.id == T.opt.w) {
                    let lt = prims[T.opt.w];
                    let cos_light = -dot(lt.n.xyz, ray.d);
                    if (cos_light > 0.0) {
                        s.rad += thr * m.emission.rgb * power_heuristic(s.pdf, light_pdf(lt.v.w, h.t * h.t, cos_light));
                    }
                }
            }
        } else if (depth < max_depth) {
            let d = ray.d;
            var nxt = d;
            if (kind == 0u) {
                let dd = 3u * u32(depth);
                let ls = wf_light_surface(h, m.albedo.rgb, s2(sm, 1u + dd, &rng));
                if (ls.dist > 0.0) { wf_shadow_ray(p, h.p, ls.wi, ls.dist, thr * ls.c, hash32(rng ^ 0x68bc21ebu)); }
                nxt = cosine_hemisphere(h.n, s2(sm, 2u + dd, &rng));
                s.pdf = dot(h.n, nxt) / PI_;
                thr *= m.albedo.rgb;
                emit = after_bounce;
                if (depth == 0) { chain = T.ext.z != 0u; }
                spec = false;
            } else if (kind == 1u) {
                nxt = reflect(d, h.n);
                thr *= m.albedo.rgb;
                emit = 0;
                spec = true;
            } else {
                let ior = m.emission.w;
                let eta = select(ior, 1.0 / ior, h.front);
                let cos_i = min(dot(-d, h.n), 1.0);
                let sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
                if (sin_t2 > 1.0 || s1(sm, 3u + 3u * u32(depth), 1u, &rng) < schlick(cos_i, eta)) { nxt = reflect(d, h.n); } else { nxt = refract_dir(d, h.n, eta); }
                thr *= m.albedo.rgb;
                emit = 0;
                spec = true;
            }
            if (roulette(&thr, depth, sm, &rng)) { go_on = true; nxt_o = h.p; nxt_d = normalize(nxt); }
        }
    }
    s.thr = thr;
    s.rng = rng;
    s.fl = u32(depth + 1) | (u32(emit) << 16u) | select(0u, 1u << 18u, chain) | select(0u, 1u << 19u, spec);
    if (go_on) {
        s.o = nxt_o;
        s.d = nxt_d;
        let i = atomicAdd(&WC[0], 1u);
        q_set(1u - par, i, p);
    }
    ps_store(p, s);
}

fn wf_groups(n: u32, at: u32) {
    let g = (n + 63u) / 64u;
    let x = min(g, 65535u);
    WA[at] = x;
    WA[at + 1u] = select(1u, (g + x - 1u) / x, x > 0u);
    WA[at + 2u] = 1u;
}

@compute @workgroup_size(1)
fn wf_args() {
    let nn = atomicLoad(&WC[0]);
    let ns = atomicLoad(&WC[1]);
    var c = WF[0];
    c.x = c.x ^ 1u;
    c.y = nn;
    c.z = ns;
    WF[0] = c;
    wf_groups(nn, 0u);
    wf_groups(ns, 4u);
    atomicStore(&WC[0], 0u);
    atomicStore(&WC[1], 0u);
}

@compute @workgroup_size(64)
fn wf_shadow(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = wf_id(gid, nw);
    if (id >= WF[0].z) { return; }
    let p = q_get(2u, id);
    let s = wf_sh(p);
    let a = WF[s];
    let b = WF[s + 1u];
    let dist = bitcast<f32>(a.w);
    let r = Ray(bitcast<vec3f>(a.xyz), bitcast<vec3f>(b.xyz));
    if (traverse(r, T.f.y, dist - T.f.z, true).id != NONE) { return; }
    var c = bitcast<vec3f>(WF[s + 2u].xyz);
    if (T.cnt.w > 0u) {
        var rng = b.w;
        c *= transmittance(r, 0.0, dist, &rng);
    }
    let st = wf_st(p) + 3u;
    let e = WF[st];
    WF[st] = vec4u(bitcast<vec3u>(bitcast<vec3f>(e.xyz) + c), e.w);
}

@compute @workgroup_size(64)
fn wf_resolve(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = wf_id(gid, nw);
    if (id >= wf_n()) { return; }
    let st = wf_st(id);
    let k = WF[st + 1u].w;
    var c = bitcast<vec3f>(WF[st + 3u].xyz);
    let mx = max(c.x, max(c.y, c.z));
    if (!(mx >= 0.0 && mx < 1e30)) {
        c = vec3f(0.0);
    } else if (T.f.x > 0.0 && mx > T.f.x) {
        c *= T.f.x / mx;
    }
    accum[k] = accum[k] + vec4f(c, 1.0);
    if (T.f.w > 0.5) {
        let N = T.dims.x * T.dims.y;
        let ga = accum[3u * N + k];
        let l = luma(select(c, c / max(ga.xyz, vec3f(1e-3)), ga.w == 0.0));
        accum[N + k] += vec4f(l, l * l, 0.0, 0.0);
    }
}
);

static const char *REPROJECT = WGSL(
struct RP {
    oo: vec4f, oul: vec4f, oh: vec4f, ov: vec4f,     /* the old camera */
    no: vec4f, nul: vec4f, nh: vec4f, nv: vec4f,     /* the new one */
    dims: vec4u,                                     /* width, height, history */
};
@group(0) @binding(0) var<uniform> R: RP;
@group(0) @binding(1) var<storage, read> old: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cur: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let W = R.dims.x;
    let H = R.dims.y;
    let N = W * H;
    let k = gid.x + gid.y * nw.x * 64u;
    if (k >= N) { return; }
    let g2 = cur[2u * N + k];
    let g3 = cur[3u * N + k];
    let x = f32(k % W);
    let y = f32(k / W);
    let d = normalize(R.nul.xyz + R.nh.xyz * ((x + 0.5) / f32(W)) - R.nv.xyz * ((y + 0.5) / f32(H)) - R.no.xyz);
    let far = g2.w > 1e29;
    let p = R.no.xyz + d * select(g2.w, 1e6, far);
    /* into the old camera: the screen is 1 unit in front of the eye, horizontal and vertical
       perpendicular to the view direction c */
    let f = R.oul.xyz - R.oo.xyz;
    let c = f + 0.5 * R.oh.xyz - 0.5 * R.ov.xyz;
    let q0 = p - R.oo.xyz;
    let den = dot(q0, c);
    var acc0 = vec4f(0.0);
    var acc1 = vec4f(0.0);
    var nsum = 0.0;
    var wsum = 0.0;
    if (den > 0.0) {
        let q = q0 * (dot(c, c) / den) - f;
        let fx = dot(q, R.oh.xyz) / dot(R.oh.xyz, R.oh.xyz) * f32(W) - 0.5;
        let fy = -dot(q, R.ov.xyz) / dot(R.ov.xyz, R.ov.xyz) * f32(H) - 0.5;
        let x0 = floor(fx);
        let y0 = floor(fy);
        let dist = length(q0);
        for (var j = 0; j < 4; j++) {
            let tx = x0 + f32(j & 1);
            let ty = y0 + f32(j >> 1);
            if (tx < 0.0 || ty < 0.0 || tx >= f32(W) || ty >= f32(H)) { continue; }
            let o = u32(ty) * W + u32(tx);
            let s = old[o];
            if (s.w == 0.0) { continue; }
            let o3 = old[3u * N + o];
            if (o3.w != g3.w) { continue; }
            let o2 = old[2u * N + o];
            if (!far) {
                let ux = u32(tx);
                let uy = u32(ty);
                let zl = old[2u * N + o - select(0u, 1u, ux > 0u)].w;
                let zr = old[2u * N + o + select(0u, 1u, ux + 1u < W)].w;
                let zu = old[2u * N + o - select(0u, W, uy > 0u)].w;
                let zd = old[2u * N + o + select(0u, W, uy + 1u < H)].w;
                let grad = min(abs(zl - o2.w), abs(zr - o2.w)) + min(abs(zu - o2.w), abs(zd - o2.w));
                if (abs(o2.w - dist) > select(0.02, 0.25, g3.w == 2.0) * dist + grad) { continue; }
            }
            if (g3.w == 0.0 && dot(o2.xyz, g2.xyz) < 0.9) { continue; }
            let w = (1.0 - abs(fx - tx)) * (1.0 - abs(fy - ty));
            acc0 += w * s / s.w;
            acc1 += w * old[N + o] / s.w;
            nsum += w * s.w;
            wsum += w;
        }
    }
    if (wsum < 0.05) {
        cur[k] = vec4f(0.0);
        cur[N + k] = vec4f(0.0);
        return;
    }
    let n = min(nsum / wsum, f32(R.dims.z));
    cur[k] = vec4f(acc0.rgb / wsum * n, n);
    cur[N + k] = acc1 / wsum * n;
}
);

// The image to show or read back: the path tracer's (its samples, or denoised) plus the
// caustics the path tracer left out: the light tracer's (exact, averaged over its passes),
// blended with the photon cache's while the light tracer has few passes (weight P / (P + 8)).
// The cache is read where the camera sees a diffuse surface directly (the 5th plane: the
// point and the transmittance up to it), bilinearly between the cells along the surface.
static const char *RESOLVE = WGSL(
struct RS {
    dims: vec4u,   /* width, height, 1 = use the photon cache */
    k: vec4f,      /* 1 / (2^32 passes), the light tracer's weight, 1 / (paths PC_SCALE), finest cell */
};
@group(0) @binding(0) var<uniform> R: RS;
@group(0) @binding(1) var<storage, read> base: array<vec4f>;
@group(0) @binding(2) var<storage, read> caus: array<u32>;
@group(0) @binding(3) var<storage, read_write> outp: array<vec4f>;
@group(0) @binding(4) var<storage, read> grid: array<u32>;
@group(0) @binding(5) var<storage, read> planes: array<vec4f>;
const PC_CELLS = 1048576u;
fn hash32(x0: u32) -> u32 {
    var x = x0;
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
    return x;
}
fn pc_axis(n: vec3f) -> u32 {
    let a = abs(n);
    if (a.x >= a.y && a.x >= a.z) { return select(1u, 0u, n.x > 0.0); }
    if (a.y >= a.z) { return select(3u, 2u, n.y > 0.0); }
    return select(5u, 4u, n.z > 0.0);
}
fn pc_key(c: vec3i, axis: u32, level: u32) -> u32 {
    return hash32((u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u) ^
                  ((axis + 6u * level) * 2654435761u));
}
/* rgb sums and photons of a cell (0 if it has none): its slot, as pc_claim found it */
fn cell_sum(c: vec3i, axis: u32, level: u32) -> vec4f {
    let h = pc_key(c, axis, level);
    let key = hash32(h ^ 0x68e31da4u) | 1u;
    var idx = h & (PC_CELLS - 1u);
    var found = false;
    for (var probe = 0u; probe < 8u; probe++) {
        let k = grid[idx * 8u + 7u];
        if (k == key) { found = true; break; }
        if (k == 0u) { break; }
        idx = (idx + 1u) & (PC_CELLS - 1u);
    }
    if (!found) { return vec4f(0.0); }
    let at = idx * 8u;
    return vec4f(f32(grid[at + 1u]) * 4294967296.0 + f32(grid[at]),
                 f32(grid[at + 3u]) * 4294967296.0 + f32(grid[at + 2u]),
                 f32(grid[at + 5u]) * 4294967296.0 + f32(grid[at + 4u]),
                 f32(grid[at + 6u]) / 256.0);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let N = R.dims.x * R.dims.y;
    let k = gid.x + gid.y * nw.x * 64u;
    if (k >= N) { return; }
    let b = base[k];
    if (b.w <= 0.0) { outp[k] = vec4f(0.0); return; }   /* no sample yet: the raster shows */
    var add = vec3f(0.0);
    if (R.k.y > 0.0) {
        var c = vec3f(0.0);
        for (var ch = 0u; ch < 3u; ch++) {
            let at = (k * 3u + ch) * 2u;
            c[ch] = (f32(caus[at + 1u]) * 4294967296.0 + f32(caus[at])) * R.k.x;
        }
        add += c * R.k.y;
    }
    if (R.dims.z != 0u && R.k.y < 1.0) {
        let p4 = planes[4u * N + k];
        if (p4.w > 0.0) {
            let n = planes[2u * N + k].xyz;
            let alb = planes[3u * N + k].xyz;
            let axis = pc_axis(n);
            let ax = axis / 2u;
            let t1 = (ax + 1u) % 3u;
            let t2 = (ax + 2u) % 3u;
            /* the finest size of cell holding at least 16 photons around the point */
            var size = R.k.w;
            var e = vec4f(0.0);
            for (var level = 0u; level < 3u; level++) {
                let fp = p4.xyz / size;
                let c0 = vec3i(floor(fp));
                let u = fp[t1] - 0.5;
                let v = fp[t2] - 0.5;
                let u0 = floor(u);
                let v0 = floor(v);
                let fu = u - u0;
                let fv = v - v0;
                e = vec4f(0.0);
                for (var j = 0; j < 4; j++) {
                    var c = c0;
                    c[t1] = i32(u0) + (j & 1);
                    c[t2] = i32(v0) + (j >> 1);
                    let w = select(1.0 - fu, fu, (j & 1) != 0) * select(1.0 - fv, fv, (j >> 1) != 0);
                    e += w * cell_sum(c, axis, level);
                }
                if (e.w >= 16.0 || level == 2u) { break; }
                size *= 4.0;
            }
            add += (1.0 - R.k.y) * alb * (e.rgb * R.k.z / (size * size)) * (p4.w / 3.14159265);
        }
    }
    outp[k] = vec4f(b.rgb / b.w + add, 1.0);
}
);

// the photon cache fades: every sum times a factor
static const char *DECAY = WGSL(
struct DC { k: vec4f, n: vec4u };
@group(0) @binding(0) var<uniform> D: DC;
@group(0) @binding(1) var<storage, read_write> grid: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = gid.x + gid.y * nw.x * 64u;
    if (i >= D.n.x) { return; }
    if ((i & 3u) == 3u) {                 /* the photon count fades, the cell's key stays */
        grid[2u * i] = u32(f32(grid[2u * i]) * D.k.x);
        return;
    }
    let hi = grid[2u * i + 1u];
    let lo = grid[2u * i];
    if (hi == 0u && lo == 0u) { return; }
    let v = (f32(hi) * 4294967296.0 + f32(lo)) * D.k.x;
    let h = floor(v / 4294967296.0);
    grid[2u * i + 1u] = u32(h);
    grid[2u * i] = u32(min(v - h * 4294967296.0, 4294967295.0));
}
);

// ---- the BVH builder: a linear BVH (Karras 2012, "Maximizing Parallelism in the
// Construction of BVHs, Octrees, and k-d Trees"), entirely on the GPU ----
// morton:  each item's box centre, as a 30-bit Morton code (10 bits per axis inside the
//          bounds), with the item's index: (code, index), padded with the largest key
// bsort:   a bitonic sort of the keys, one dispatch per (k, j) step
// hier:    internal node i (of n - 1) finds the range of keys it covers and where it splits,
//          from the length of the codes' common prefix (equal codes: of their positions)
// leaves:  leaf j: item keys[j].y's box, its shape id in aux
// up:      the boxes from the leaves up, in passes: a node is done in pass p when both its
//          children were done before (flags: the pass a node was done in + 1; leaves 1);
//          one dispatch's threads do not see each other's writes, the next dispatch does
// The nodes are this path tracer's: internal i at node0 + i, leaf j at node0 + n - 1 + j,
// the root at node0.
static const char *BUILD = WGSL(
struct Node { lo: vec3f, a: u32, hi: vec3f, b: u32 };
struct BP {
    n: vec4u,      /* items, padded to a power of 2, node0, ref0 */
    m: vec4u,      /* bsort: j, k; up: the pass; a leaf's shape = item + m.z */
    lo: vec4f,     /* the bounds' low corner */
    ext: vec4f,    /* 1 / their size */
};
@group(0) @binding(0) var<uniform> U: BP;
@group(0) @binding(1) var<storage, read> boxes: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> keys: array<vec2u>;
@group(0) @binding(3) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(4) var<storage, read_write> aux: array<u32>;
@group(0) @binding(5) var<storage, read_write> flags: array<u32>;

fn tid(gid: vec3u, nw: vec3u) -> u32 { return gid.x + gid.y * nw.x * 64u; }
fn spread(v: u32) -> u32 {
    var x = v & 1023u;
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x << 8u)) & 0x0300f00fu;
    x = (x | (x << 4u)) & 0x030c30c3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}

@compute @workgroup_size(64)
fn morton(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i >= U.n.y) { return; }
    if (i >= U.n.x) { keys[i] = vec2u(0xffffffffu, 0xffffffffu); return; }
    let c = (boxes[2u * i].xyz + boxes[2u * i + 1u].xyz) * 0.5;
    let q = clamp((c - U.lo.xyz) * U.ext.xyz, vec3f(0.0), vec3f(0.999)) * 1024.0;
    keys[i] = vec2u((spread(u32(q.x)) << 2u) | (spread(u32(q.y)) << 1u) | spread(u32(q.z)), i);
}

fn key_gt(a: vec2u, b: vec2u) -> bool { return a.x > b.x || (a.x == b.x && a.y > b.y); }

@compute @workgroup_size(64)
fn bsort(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i >= U.n.y) { return; }
    let p = i ^ U.m.x;
    if (p <= i) { return; }
    let a = keys[i];
    let b = keys[p];
    let ascending = (i & U.m.y) == 0u;
    if (key_gt(a, b) == ascending) {
        keys[i] = b;
        keys[p] = a;
    }
}

fn delta(i: i32, j: i32) -> i32 {
    if (j < 0 || j >= i32(U.n.x)) { return -1; }
    let a = keys[i].x;
    let b = keys[j].x;
    if (a == b) { return 32 + i32(countLeadingZeros(u32(i) ^ u32(j))); }
    return i32(countLeadingZeros(a ^ b));
}

@compute @workgroup_size(64)
fn hier(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let ii = tid(gid, nw);
    let n = i32(U.n.x);
    if (i32(ii) >= n - 1) { return; }
    let i = i32(ii);
    let d = select(-1, 1, delta(i, i + 1) > delta(i, i - 1));
    let dmin = delta(i, i - d);
    var lmax = 2;
    for (var g = 0; g < 32; g++) {
        if (lmax >= 2 * n || delta(i, i + lmax * d) <= dmin) { break; }
        lmax *= 2;
    }
    var l = 0;
    var t = lmax / 2;
    for (var g = 0; g < 32; g++) {
        if (t < 1) { break; }
        if (delta(i, i + (l + t) * d) > dmin) { l += t; }
        t /= 2;
    }
    let j = i + l * d;
    let dnode = delta(i, j);
    var s = 0;
    var div = 2;
    for (var g = 0; g < 32; g++) {
        let st = (l + div - 1) / div;
        if (delta(i, i + (s + st) * d) > dnode) { s += st; }
        if (st <= 1) { break; }
        div *= 2;
    }
    let gam = i + s * d + min(d, 0);
    let left = select(gam, n - 1 + gam, min(i, j) == gam);
    let right = select(gam + 1, n + gam, max(i, j) == gam + 1);
    nodes[U.n.z + ii].a = U.n.z + u32(left);
    nodes[U.n.z + ii].b = U.n.z + u32(right);
    flags[ii] = 0u;
}

@compute @workgroup_size(64)
fn leaves(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let j = tid(gid, nw);
    let n = U.n.x;
    if (j >= n) { return; }
    let item = keys[j].y;
    nodes[U.n.z + n - 1u + j] = Node(boxes[2u * item].xyz, U.n.w + j, boxes[2u * item + 1u].xyz, 0x80000001u);
    aux[U.n.w + j] = item + U.m.z;
    flags[n - 1u + j] = 1u;
}

@compute @workgroup_size(64)
fn up(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i + 1u >= U.n.x || flags[i] != 0u) { return; }
    let nd = nodes[U.n.z + i];
    let fl = flags[nd.a - U.n.z];
    let fr = flags[nd.b - U.n.z];
    if (fl == 0u || fr == 0u || fl > U.m.y || fr > U.m.y) { return; }
    let a = nodes[nd.a];
    let b = nodes[nd.b];
    nodes[U.n.z + i].lo = min(a.lo, b.lo);
    nodes[U.n.z + i].hi = max(a.hi, b.hi);
    flags[i] = U.m.y + 1u;
}
);

#define RT_EPS_F 2e-3f           // rt/tracer.c, float build: minimum ray distance
#define RT_SHADOW_MARGIN_F 2e-2f // shadow rays stop this far before the lamp
#define CHUNK_TRIS 32768

typedef struct {
    float origin[4], upper_left[4], horizontal[4], vertical[4];
    uint32_t dims[4], opt[4], cnt[4];
    float f[4];
    uint32_t ext[4];
    float lt[4];
    uint32_t ins[4];
    float mats[MAX_MATS][8];
} GTP;
// one copy of the parameters per pass of a submission (each its own sample index), 256-byte
// aligned (WebGPU's minimum offset alignment for uniform bindings)
#define MAX_REPS 16
#define GTP_STRIDE ((sizeof(GTP) + 255) / 256 * 256)

// where an instanced mesh is on the GPU (and what it was made from)
typedef struct {
    const Triangle *tris; int ntris;
    const Vec3 *normals; int nnormals;
    int dynamic;
    unsigned version;
    uint32_t node0, ref0, tri0, nrm0;   // its BVH's root, its leaves' shapes, triangles, normals
    uint32_t root;                      // the root as the kernel's stack takes it (WIDE | node0 if 4-wide)
} GpuMeshCopy;

// a BVH for the GPU to build (gpu_trace_build): n boxes, into nodes[node0 ..], aux[ref0 ..]
typedef struct {
    int n;
    uint32_t node0, ref0, ref_add;      // a leaf's shape id: its item + ref_add
    size_t box_at, par_at;              // its boxes and parameters, in the arena
    int npar;
} GpuBuild;

struct GpuTrace {
    Gpu *g;
    int w, h;
    WGPUBindGroupLayout bgl, rp_bgl;
    WGPUComputePipeline pipe, gbuf_pipe, rp_pipe;
    WGPUBuffer params, rp_params, blue;   // parameters (MAX_REPS copies), reprojection's, blue noise
    WGPUBuffer acc[2];       // the image's samples, and (once reprojected) the previous image's
    int cur;                 // which one is the image's
    int planes;              // 5 with the guide's planes (denoiser, caustics), 1 if this GPU's buffers are too small
    Camera cam;              // the image's camera
    int has_image;
    // light-traced caustics: the 64-bit sums (ltrace), their passes, and the resolved image
    WGPUBindGroupLayout lt_bgl, rs_bgl;
    WGPUComputePipeline lt_pipe, rs_pipe;
    WGPUBuffer caus, rs_params, final, lt_params;   // (its own parameters: written in the same submit)
    int caus_passes, specular;
    // the photon cache (the viewer): grid, its fading, its light paths' parameters
    WGPUBindGroupLayout dc_bgl;
    WGPUComputePipeline dc_pipe;
    WGPUBuffer grid, dc_params, pc_params;
    double pc_paths;         // light paths in the grid, faded as the grid is
    float cell;              // its cells' size
    WGPUBuffer nodes, aux, prims, tris;
    size_t nodes_cap, aux_cap, prims_cap, tris_cap;
    uint32_t nnodes, normals_at;   // (nnodes: the world's, in slots)
    int world_wide;
    int wide_ok;             // LR_GPU_WIDE=1: the CPU's 4-wide trees as such (else binary: faster here)
    int tlas_mode;           // the top level: 1 the CPU's (SAH, refit) when it is current, else
                             // (or LR_GPU_TLAS=lbvh: 0) built here (LBVH)
    // the layout (see "the scene on the GPU"): instanced meshes, instances, the top level
    GpuMeshCopy *mc;
    int nmc, laid_out;
    int inst_cap, ninst;
    uint32_t tlas_node0, tlas_ref0, world_node0, world_ref0, layout_nodes, layout_aux;
    // the BVH builder (BUILD): pipelines, scratch (keys, flags), this frame's builds and their
    // boxes and parameters (an arena: written once, then the builds run in order)
    WGPUBindGroupLayout bd_bgl;
    WGPUComputePipeline bd_morton, bd_sort, bd_hier, bd_leaves, bd_up;
    WGPUBuffer bd_keys, bd_flags, bd_boxes, bd_par;
    size_t bd_keys_cap, bd_flags_cap, bd_boxes_cap, bd_par_cap;
    GpuBuild builds[8];
    int nbuild;
    unsigned char *arena_boxes, *arena_par;
    size_t arena_nboxes, arena_npar, arena_boxes_cap, arena_par_cap;
    // what is on the GPU now
    Primitive *prim_copy;
    int nprim_copy;
    const Triangle *mesh_tris;
    const Vec3 *mesh_normals;
    int mesh_ntris, mesh_nnormals;
    int valid;
    Bvh own;                 // built here when the scene has no BVH of its own
    int own_built;
    GpuTimer *timer;         // LR_GPU_TIMING=1 (natively): the GPU time of the passes, by stage
    // the wavefront tracer (WAVE; gpu_trace_set_wavefront): its kernels and their layouts,
    // the paths' state (WF), the counters (WC), the indirect dispatches' sizes (WA), and the
    // parameters of each wave (a band of more paths than WF holds goes in several waves)
    int wavefront, wf_made, wf_failed;
    WGPUBindGroupLayout wf_bgl[6];
    WGPUComputePipeline wf_pipe[6];
    WGPUBuffer wf, wfc, wfa, wf_params;
    int wf_rows;             // rows per wave
    const char *types_wgsl, *media_wgsl;
};
enum { TS_TRACE, TS_LIGHT, TS_BUILD, TS_WF0, NTS = TS_WF0 + 6 };
static const char *const TS_NAMES[NTS] = {"path tracer", "light tracer (caustics)", "BVH builds",
    "  wavefront: camera rays", "  wavefront: extend", "  wavefront: shade", "  wavefront: sizes", "  wavefront: shadow rays",
    "  wavefront: into the image"};

// a compute pass, timed under `stage` when timing is on; `units`: the work it does
static WGPUComputePassEncoder timed_pass(GpuTrace *t, WGPUCommandEncoder enc, int stage, double units){
    WGPUComputePassDescriptor pd;
    memset(&pd, 0, sizeof(pd));
    pd.timestampWrites = gpu_timer_next(t->timer, stage);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &pd);
    if (pd.timestampWrites) gpu_timer_units(t->timer, units);
    return pass;
}

void gpu_trace_before_submit(GpuTrace *t, WGPUCommandEncoder enc){ if (t) gpu_timer_finish(t->timer, enc); }
void gpu_trace_after_submit(GpuTrace *t){ if (t) gpu_timer_submitted(t->timer); }

GpuTrace *gpu_trace_create(Gpu *g, int width, int height, const char *types_wgsl, const char *media_wgsl){
    if (!g || g->state != GPU_READY) return NULL;
    GpuTrace *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->g = g;
    t->w = width;
    t->h = height;
    char *src = malloc(strlen(types_wgsl) + strlen(media_wgsl) + strlen(TRACE) + 8);
    if (!src){ free(t); return NULL; }
    strcpy(src, types_wgsl); strcat(src, "\n"); strcat(src, media_wgsl); strcat(src, "\n"); strcat(src, TRACE);
    // frame, densities, bricks, media boxes, parameters, nodes, aux, prims, tris, accumulation,
    // blue noise
    t->bgl = gpu_layout(g, WGPUShaderStage_Compute, "URRRURRRRWU");
    t->timer = gpu_timer_create(g, 1024, NTS);
    t->types_wgsl = types_wgsl;
    t->media_wgsl = media_wgsl;
    const char *wide_env = getenv("LR_GPU_WIDE");
    t->wide_ok = wide_env && atoi(wide_env) > 0;
    const char *tlas_env = getenv("LR_GPU_TLAS");
    t->tlas_mode = tlas_env && !strcmp(tlas_env, "lbvh") ? 0 : 1;
    t->pipe = gpu_compute(g, "path tracer", src, "main", t->bgl);
    t->planes = (uint64_t)width * height * 80 <= g->max_storage ? 5 : 1;
    if (t->planes == 5){
        t->gbuf_pipe = gpu_compute(g, "path tracer: guide", src, "gbuf", t->bgl);
        t->rp_bgl = gpu_layout(g, WGPUShaderStage_Compute, "URW");
        t->rp_pipe = gpu_compute(g, "path tracer: reprojection", REPROJECT, "main", t->rp_bgl);
        t->rp_params = gpu_buffer(g, "reprojection parameters", 9 * 16, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    }
    // the BVH builder
    t->bd_bgl = gpu_layout(g, WGPUShaderStage_Compute, "URWWWW");
    t->bd_morton = gpu_compute(g, "BVH builder: morton", BUILD, "morton", t->bd_bgl);
    t->bd_sort = gpu_compute(g, "BVH builder: sort", BUILD, "bsort", t->bd_bgl);
    t->bd_hier = gpu_compute(g, "BVH builder: hierarchy", BUILD, "hier", t->bd_bgl);
    t->bd_leaves = gpu_compute(g, "BVH builder: leaves", BUILD, "leaves", t->bd_bgl);
    t->bd_up = gpu_compute(g, "BVH builder: boxes", BUILD, "up", t->bd_bgl);
    // the light tracer: the same bindings, its sums (11) instead of the samples (9)
    t->lt_bgl = gpu_layout(g, WGPUShaderStage_Compute, "URRRURRRR-UW");
    t->lt_pipe = gpu_compute(g, "light tracer", src, "ltrace", t->lt_bgl);
    t->rs_bgl = gpu_layout(g, WGPUShaderStage_Compute, "URRWRR");
    t->dc_bgl = gpu_layout(g, WGPUShaderStage_Compute, "UW");
    t->dc_pipe = gpu_compute(g, "photon cache: fade", DECAY, "main", t->dc_bgl);
    t->dc_params = gpu_buffer(g, "photon cache: fade parameters", 32, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    t->pc_params = gpu_buffer(g, "photon cache: light paths' parameters", GTP_STRIDE * MAX_REPS,
                              WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    t->rs_pipe = gpu_compute(g, "caustics: resolve", RESOLVE, "main", t->rs_bgl);
    t->rs_params = gpu_buffer(g, "caustics: resolve parameters", 32, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    t->lt_params = gpu_buffer(g, "light tracer parameters", GTP_STRIDE * MAX_REPS, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    t->caus = gpu_buffer(g, "caustics", (size_t)width * height * 24, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    t->final = gpu_buffer(g, "path traced image with caustics", (size_t)width * height * 16,
                          WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    free(src);
    t->params = gpu_buffer(g, "path tracer parameters", GTP_STRIDE * MAX_REPS, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    {
        // the blue-noise mask (core/sampler.c), 2 ranks per 32 bits
        const uint16_t *m = blue_noise_mask();
        uint32_t packed[BLUE_NOISE_SIZE * BLUE_NOISE_SIZE / 2];
        for (int i = 0; i < BLUE_NOISE_SIZE * BLUE_NOISE_SIZE / 2; i++) packed[i] = (uint32_t)m[2 * i] | ((uint32_t)m[2 * i + 1] << 16);
        t->blue = gpu_buffer(g, "blue noise", sizeof(packed), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
        wgpuQueueWriteBuffer(g->queue, t->blue, 0, packed, sizeof(packed));
    }
    t->acc[0] = gpu_buffer(g, "path tracer samples", (size_t)width * height * 16 * t->planes,
                           WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
    return t;
}

WGPUBuffer gpu_trace_accum(GpuTrace *t){ return t->acc[t->cur]; }
int gpu_trace_guided(const GpuTrace *t){ return t->planes == 5; }

static void put(float *d, Vec3 a, float w){ d[0] = (float)a.x; d[1] = (float)a.y; d[2] = (float)a.z; d[3] = w; }

// ---- the BVH builder's C side --------------------------------------------------------------
typedef struct { uint32_t n[4], m[4]; float lo[4], ext[4]; } BuildParams;
#define BP_STRIDE 256

static uint32_t pow2_at_least(uint32_t n){ uint32_t p = 2; while (p < n) p <<= 1; return p; }
static int up_passes(int n){ int k = 0; while ((1 << k) < n) k++; return n > 1 ? 34 + k : 0; }
static int sort_passes(uint32_t p){ int c = 0; for (uint32_t k = 2; k <= p; k <<= 1) for (uint32_t j = k >> 1; j > 0; j >>= 1) c++; return c; }

static void *arena_put(unsigned char **buf, size_t *len, size_t *cap, const void *data, size_t bytes){
    size_t at = (*len + 255) / 256 * 256;
    if (at + bytes > *cap){
        size_t want = (at + bytes) * 2;
        unsigned char *nb = realloc(*buf, want);
        if (!nb) return NULL;
        *buf = nb;
        *cap = want;
    }
    if (data) memcpy(*buf + at, data, bytes);
    *len = at + bytes;
    return *buf + at;
}

// a BVH over n boxes (2 x vec4 each: lo, hi) to build on the GPU at the next gpu_trace_build
static void build_queue(GpuTrace *t, const float *boxes, int n, uint32_t node0, uint32_t ref0, uint32_t ref_add, Aabb bounds){
    if (n <= 0 || t->nbuild >= (int)(sizeof(t->builds) / sizeof(t->builds[0]))) return;
    if (t->nbuild == 0) t->arena_nboxes = t->arena_npar = 0;
    GpuBuild *b = &t->builds[t->nbuild];
    b->n = n; b->node0 = node0; b->ref0 = ref0; b->ref_add = ref_add;
    unsigned char *at = arena_put(&t->arena_boxes, &t->arena_nboxes, &t->arena_boxes_cap, boxes, (size_t)n * 32);
    if (!at) return;
    b->box_at = (size_t)(at - t->arena_boxes);
    uint32_t p = pow2_at_least((uint32_t)n);
    BuildParams base;
    memset(&base, 0, sizeof(base));
    base.n[0] = (uint32_t)n; base.n[1] = p; base.n[2] = node0; base.n[3] = ref0;
    base.m[2] = ref_add;
    Vec3 ext = v_sub(bounds.hi, bounds.lo);
    put(base.lo, bounds.lo, 0);
    base.ext[0] = ext.x > 0 ? (float)(1.0 / ext.x) : 0.0f;
    base.ext[1] = ext.y > 0 ? (float)(1.0 / ext.y) : 0.0f;
    base.ext[2] = ext.z > 0 ? (float)(1.0 / ext.z) : 0.0f;
    // parameters: the base one, one per sort step, one per pass up
    int npar = 1 + sort_passes(p) + up_passes(n);
    size_t first = (t->arena_npar + 255) / 256 * 256;
    unsigned char *par = arena_put(&t->arena_par, &t->arena_npar, &t->arena_par_cap, NULL, (size_t)npar * BP_STRIDE);
    if (!par) return;
    memset(par, 0, (size_t)npar * BP_STRIDE);
    int k = 0;
    memcpy(par + (size_t)k++ * BP_STRIDE, &base, sizeof(base));
    for (uint32_t kk = 2; kk <= p; kk <<= 1)
        for (uint32_t j = kk >> 1; j > 0; j >>= 1){
            BuildParams q = base;
            q.m[0] = j; q.m[1] = kk;
            memcpy(par + (size_t)k++ * BP_STRIDE, &q, sizeof(q));
        }
    for (int u = 1; u <= up_passes(n); u++){
        BuildParams q = base;
        q.m[1] = (uint32_t)u;
        memcpy(par + (size_t)k++ * BP_STRIDE, &q, sizeof(q));
    }
    b->par_at = first;
    b->npar = npar;
    t->nbuild++;
}

static void build_dispatch(GpuTrace *t, WGPUComputePassEncoder pass, WGPUComputePipeline pipe, const GpuBuild *b,
                           int par, size_t threads, size_t keys_bytes, size_t flags_bytes){
    WGPUBindGroupEntry e[6];
    memset(e, 0, sizeof(e));
    WGPUBuffer bufs[6] = {t->bd_par, t->bd_boxes, t->bd_keys, t->nodes, t->aux, t->bd_flags};
    for (int i = 0; i < 6; i++){ e[i].binding = (uint32_t)i; e[i].buffer = bufs[i]; e[i].size = WGPU_WHOLE_SIZE; }
    e[0].offset = b->par_at + (size_t)par * BP_STRIDE; e[0].size = sizeof(BuildParams);
    e[1].offset = b->box_at; e[1].size = (size_t)b->n * 32;
    e[2].size = keys_bytes;
    e[5].size = flags_bytes;
    WGPUBindGroupDescriptor d;
    memset(&d, 0, sizeof(d));
    d.layout = t->bd_bgl;
    d.entryCount = 6;
    d.entries = e;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(t->g->device, &d);
    wgpuComputePassEncoderSetPipeline(pass, pipe);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    gpu_dispatch(pass, threads);
    wgpuBindGroupRelease(bg);
}

// the BVHs queued since the last call (the top level over the instances, dynamic meshes')
static void gpu_trace_build(GpuTrace *t, WGPUCommandEncoder enc){
    if (t->nbuild == 0) return;
    Gpu *g = t->g;
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    size_t keys_bytes = 8, flags_bytes = 8;
    for (int i = 0; i < t->nbuild; i++){
        size_t kb = (size_t)pow2_at_least((uint32_t)t->builds[i].n) * 8, fb = (size_t)t->builds[i].n * 8;
        if (kb > keys_bytes) keys_bytes = kb;
        if (fb > flags_bytes) flags_bytes = fb;
    }
    gpu_buffer_reserve(g, &t->bd_keys, &t->bd_keys_cap, keys_bytes, st, "BVH builder: keys");
    gpu_buffer_reserve(g, &t->bd_flags, &t->bd_flags_cap, flags_bytes, st, "BVH builder: flags");
    gpu_buffer_reserve(g, &t->bd_boxes, &t->bd_boxes_cap, t->arena_nboxes, st, "BVH builder: boxes");
    gpu_buffer_reserve(g, &t->bd_par, &t->bd_par_cap, t->arena_npar, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                       "BVH builder: parameters");
    wgpuQueueWriteBuffer(g->queue, t->bd_boxes, 0, t->arena_boxes, (t->arena_nboxes + 3) / 4 * 4);
    wgpuQueueWriteBuffer(g->queue, t->bd_par, 0, t->arena_par, (t->arena_npar + 3) / 4 * 4);
    WGPUComputePassEncoder pass = timed_pass(t, enc, TS_BUILD, 1);
    for (int i = 0; i < t->nbuild; i++){
        const GpuBuild *b = &t->builds[i];
        uint32_t p = pow2_at_least((uint32_t)b->n);
        int par = 0;
        build_dispatch(t, pass, t->bd_morton, b, par++, p, keys_bytes, flags_bytes);
        for (uint32_t kk = 2; kk <= p; kk <<= 1)
            for (uint32_t j = kk >> 1; j > 0; j >>= 1) build_dispatch(t, pass, t->bd_sort, b, par++, p, keys_bytes, flags_bytes);
        if (b->n > 1) build_dispatch(t, pass, t->bd_hier, b, 0, (size_t)b->n - 1, keys_bytes, flags_bytes);
        build_dispatch(t, pass, t->bd_leaves, b, 0, (size_t)b->n, keys_bytes, flags_bytes);
        for (int u = 0; u < up_passes(b->n); u++) build_dispatch(t, pass, t->bd_up, b, par++, (size_t)b->n - 1, keys_bytes, flags_bytes);
    }
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    t->nbuild = 0;
}

// ---- the scene on the GPU ----------------------------------------------------------------
// nodes: each instanced mesh's BVH | the top-level BVH over the instances | the world's BVH
// aux:   the meshes' leaves' shapes | the top level's instances | the vertex normals (the
//        world's, then each mesh's) | the world's leaves' shapes
// tris:  the world's triangles | each mesh's (its normal indices moved past the ones before)
// prims: the world's primitives | 2 records per instance (transforms; mesh BVH root, material,
//        world box)
// The world's BVH comes last, so that a new one (the physics moved a body) is uploaded alone.
// Moving instances only rewrites their records, and the top level is rebuilt on the GPU
// (gpu_trace_build, BUILD). A dynamic mesh (its triangles change) gets its triangles
// rewritten and its BVH rebuilt on the GPU too, in place.

static int meshes_same(const GpuTrace *t, const Scene *s){
    if (t->nmc != s->nmeshes) return 0;
    for (int m = 0; m < s->nmeshes; m++){
        const SceneMesh *sm = &s->meshes[m];
        const GpuMeshCopy *c = &t->mc[m];
        if (c->tris != sm->tris || c->ntris != sm->ntris || c->normals != sm->normals ||
            c->nnormals != sm->nnormals || c->dynamic != sm->dynamic) return 0;
        if (!sm->dynamic && c->version != sm->version) return 0;
    }
    return 1;
}

// all but the world's BVH and primitives stay where they are
static int layout_same(const GpuTrace *t, const Scene *s){
    return t->laid_out && t->mesh_tris == s->tris && t->mesh_ntris == s->ntris && t->mesh_normals == s->normals &&
           t->mesh_nnormals == s->nnormals && meshes_same(t, s) && s->ninsts <= t->inst_cap;
}

static int geometry_same(const GpuTrace *t, const Scene *s){
    return t->valid && layout_same(t, s) && t->nprim_copy == s->nprims &&
           (s->nprims == 0 || memcmp(t->prim_copy, s->prims, (size_t)s->nprims * sizeof(Primitive)) == 0);
}

// a BVH's nodes as the GPU's, children and leaves moved by node0 / ref0
#define GPU_WIDE 0x80000000u
// how many 32-byte slots a BVH takes on the GPU: its 4-wide nodes (4 slots each), or its nodes
static uint32_t gpu_slots(const Bvh *b, int wide){ return wide ? 4u * (uint32_t)b->nwide : (uint32_t)b->nnodes; }

// a 4-wide BVH: each node 4 slots, one per child: its box; inner: a = its first slot, b = 0;
// leaf: a = its first shape, b = 0x80000000 | count (a leaf node of the binary layout); empty:
// b = 0xffffffff (and a box that no ray meets)
static void put_wide(uint32_t *nd, const Bvh *b, uint32_t node0, uint32_t ref0){
    for (int i = 0; i < b->nwide; i++){
        const BvhWideNode *n = &b->wide[i];
        for (int k = 0; k < 4; k++){
            uint32_t *o = nd + 8 * (4 * (size_t)i + k);
            float lo[3] = {(float)n->b[0][k], (float)n->b[1][k], (float)n->b[2][k]};
            float hi[3] = {(float)n->b[3][k], (float)n->b[4][k], (float)n->b[5][k]};
            if (n->count[k] < 0){ lo[0] = lo[1] = lo[2] = hi[0] = hi[1] = hi[2] = 1e30f; }
            memcpy(o, lo, 12);
            memcpy(o + 4, hi, 12);
            o[3] = n->count[k] > 0 ? ref0 + (uint32_t)n->child[k] : n->count[k] == 0 ? node0 + 4u * (uint32_t)n->child[k] : 0;
            o[7] = n->count[k] > 0 ? (0x80000000u | (uint32_t)n->count[k]) : n->count[k] == 0 ? 0 : 0xffffffffu;
        }
    }
}

static void put_nodes(uint32_t *nd, const Bvh *b, uint32_t node0, uint32_t ref0){
    for (int i = 0; i < b->nnodes; i++){
        const BvhNode *n = &b->nodes[i];
        float lo[4], hi[4];
        put(lo, n->box.lo, 0);
        put(hi, n->box.hi, 0);
        uint32_t *o = nd + 8 * (size_t)i;
        memcpy(o, lo, 12);
        memcpy(o + 4, hi, 12);
        o[3] = n->count > 0 ? ref0 + (uint32_t)n->start : node0 + (uint32_t)n->left;
        o[7] = n->count > 0 ? (0x80000000u | (uint32_t)n->count) : node0 + (uint32_t)n->right;
    }
}

// triangles as the GPU's (48 bytes), normal indices moved by nrm0
static void put_tris(uint32_t *tr, const Triangle *src, int n, uint32_t nrm0){
    for (int i = 0; i < n; i++){
        const Triangle *q = &src[i];
        uint32_t *o = tr + 12 * (size_t)i;
        float a[4];
        put(a, q->q, 0); memcpy(o, a, 12);
        put(a, q->u, 0); memcpy(o + 4, a, 12);
        put(a, q->v, 0); memcpy(o + 8, a, 12);
        int smooth_n = q->n[0] >= 0;
        o[3] = smooth_n ? nrm0 + (uint32_t)q->n[0] : 0xffffffffu;
        o[7] = smooth_n ? nrm0 + (uint32_t)q->n[1] : 0;
        o[11] = ((uint32_t)(q->material & 15) << 28) | (smooth_n ? ((nrm0 + (uint32_t)q->n[2]) & 0x0fffffffu) : 0);
    }
}

static int upload_tris(GpuTrace *t, const Triangle *src, int n, uint32_t first, uint32_t nrm0){
    uint32_t *tr = malloc((size_t)CHUNK_TRIS * 48);
    if (!tr) return 0;
    for (int f = 0; f < n; f += CHUNK_TRIS){
        int k = n - f < CHUNK_TRIS ? n - f : CHUNK_TRIS;
        put_tris(tr, src + f, k, nrm0);
        wgpuQueueWriteBuffer(t->g->queue, t->tris, (uint64_t)(first + (uint32_t)f) * 48, tr, (size_t)k * 48);
    }
    free(tr);
    return 1;
}

static int too_big(Gpu *g, size_t bytes){
    if (bytes <= g->max_storage) return 0;
    fprintf(stderr, "GPU: the scene needs a %.0f MB buffer, this GPU allows %.0f MB: the CPU path traces\n",
            (double)bytes / 1048576, (double)g->max_storage / 1048576);
    return 1;
}

// everything but the world's BVH and primitives: the meshes, the room for the instances and
// the top level, the normals, the world's triangles
static int upload_layout(GpuTrace *t, const Scene *s){
    Gpu *g = t->g;
    t->laid_out = 0;
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    GpuMeshCopy *mc = realloc(t->mc, (size_t)(s->nmeshes ? s->nmeshes : 1) * sizeof(GpuMeshCopy));
    if (!mc) return 0;
    t->mc = mc;
    uint32_t nodes = 0, refs = 0, tris = (uint32_t)s->ntris, nrm = (uint32_t)s->nnormals;
    size_t most_nodes = 1, most_refs = 1;
    for (int m = 0; m < s->nmeshes; m++){
        const SceneMesh *sm = &s->meshes[m];
        const Bvh *mb = scene_mesh_bvh(s, m);
        if (!mb){
            fprintf(stderr, "GPU path tracer: mesh %d has no BVH (scene_set_accel builds them): the CPU path traces\n", m);
            return 0;
        }
        GpuMeshCopy *c = &mc[m];
        c->tris = sm->tris; c->ntris = sm->ntris; c->normals = sm->normals; c->nnormals = sm->nnormals;
        c->dynamic = sm->dynamic; c->version = sm->version;
        c->node0 = nodes; c->ref0 = refs; c->tri0 = tris; c->nrm0 = nrm;
        // a dynamic mesh: room for the GPU's BVH of it too (2n - 1 nodes, n leaves of one
        // triangle), binary as the GPU builds it; the others 4-wide when the CPU made them so
        int wide = t->wide_ok && mb->wide && !sm->dynamic;
        uint32_t nn = gpu_slots(mb, wide), nr = (uint32_t)mb->nrefs;
        c->root = wide ? (GPU_WIDE | c->node0) : c->node0;
        if (sm->dynamic && (uint32_t)(2 * sm->ntris - 1) > nn) nn = (uint32_t)(2 * sm->ntris - 1);
        if (sm->dynamic && (uint32_t)sm->ntris > nr) nr = (uint32_t)sm->ntris;
        nodes += nn; refs += nr; tris += (uint32_t)sm->ntris; nrm += (uint32_t)sm->nnormals;
        if ((size_t)gpu_slots(mb, wide) > most_nodes) most_nodes = (size_t)gpu_slots(mb, wide);
        if ((size_t)mb->nrefs > most_refs) most_refs = (size_t)mb->nrefs;
    }
    t->nmc = s->nmeshes;
    // room for more (bodies keep appearing): doubling, so the layout is rarely redone
    t->inst_cap = 0;
    if (s->ninsts > 0){ t->inst_cap = 64; while (t->inst_cap < s->ninsts) t->inst_cap *= 2; }
    t->tlas_node0 = nodes;
    t->tlas_ref0 = refs;
    if (t->inst_cap){ nodes += (uint32_t)(2 * t->inst_cap - 1); refs += (uint32_t)t->inst_cap; }
    t->normals_at = refs;
    t->world_node0 = nodes;
    t->world_ref0 = refs + 3 * nrm;
    size_t bytes_tris = (size_t)(tris ? tris : 1) * 48;
    if (too_big(g, bytes_tris)) return 0;
    gpu_buffer_reserve(g, &t->tris, &t->tris_cap, bytes_tris, st, "triangles");
    // (the node and shape buffers: sized with the world's part, by upload_world)
    t->layout_nodes = nodes;
    t->layout_aux = t->world_ref0;
    gpu_buffer_reserve(g, &t->nodes, &t->nodes_cap, ((size_t)nodes + t->nnodes + 1) * 32, st, "BVH nodes");
    gpu_buffer_reserve(g, &t->aux, &t->aux_cap, ((size_t)t->world_ref0 + 2 * (size_t)t->nnodes + 1) * 4, st, "BVH shapes and normals");

    uint32_t *nd = malloc(most_nodes * 32);
    uint32_t *ax = malloc(most_refs * 4);
    if (!nd || !ax){ free(nd); free(ax); return 0; }
    for (int m = 0; m < s->nmeshes; m++){
        const Bvh *mb = scene_mesh_bvh(s, m);
        const GpuMeshCopy *c = &mc[m];
        int wide = (c->root & GPU_WIDE) != 0;
        if (wide) put_wide(nd, mb, c->node0, c->ref0); else put_nodes(nd, mb, c->node0, c->ref0);
        wgpuQueueWriteBuffer(g->queue, t->nodes, (uint64_t)c->node0 * 32, nd, (size_t)gpu_slots(mb, wide) * 32);
        for (int i = 0; i < mb->nrefs; i++) ax[i] = c->tri0 + (uint32_t)mb->refs[i];   // + the primitives, when walked
        wgpuQueueWriteBuffer(g->queue, t->aux, (uint64_t)c->ref0 * 4, ax, (size_t)mb->nrefs * 4);
    }
    free(nd);
    free(ax);
    // the normals, the world's then each mesh's
    size_t most_n = (size_t)s->nnormals + 1;
    for (int m = 0; m < s->nmeshes; m++) if ((size_t)s->meshes[m].nnormals > most_n) most_n = (size_t)s->meshes[m].nnormals;
    float *nf = malloc(most_n * 12);
    if (!nf) return 0;
    for (int i = 0; i < s->nnormals; i++){
        nf[3 * i] = (float)s->normals[i].x; nf[3 * i + 1] = (float)s->normals[i].y; nf[3 * i + 2] = (float)s->normals[i].z;
    }
    if (s->nnormals) wgpuQueueWriteBuffer(g->queue, t->aux, (uint64_t)t->normals_at * 4, nf, (size_t)s->nnormals * 12);
    for (int m = 0; m < s->nmeshes; m++){
        const SceneMesh *sm = &s->meshes[m];
        for (int i = 0; i < sm->nnormals; i++){
            nf[3 * i] = (float)sm->normals[i].x; nf[3 * i + 1] = (float)sm->normals[i].y; nf[3 * i + 2] = (float)sm->normals[i].z;
        }
        if (sm->nnormals)
            wgpuQueueWriteBuffer(g->queue, t->aux, ((uint64_t)t->normals_at + 3 * (uint64_t)mc[m].nrm0) * 4, nf, (size_t)sm->nnormals * 12);
    }
    free(nf);
    // the triangles: the world's (they are many: only now), then each mesh's
    if (!upload_tris(t, s->tris, s->ntris, 0, 0)) return 0;
    for (int m = 0; m < s->nmeshes; m++)
        if (!upload_tris(t, s->meshes[m].tris, s->meshes[m].ntris, mc[m].tri0, mc[m].nrm0)) return 0;
    t->mesh_tris = s->tris;
    t->mesh_normals = s->normals;
    t->mesh_ntris = s->ntris;
    t->mesh_nnormals = s->nnormals;
    t->laid_out = 1;
    return 1;
}

// the world's BVH and primitives (after the layout; a new layout if the buffers had to grow)
static int upload_world(GpuTrace *t, const Scene *s){
    Gpu *g = t->g;
    t->valid = 0;
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    // the BVH: the scene's own if it has one (it must be up to date: scene_set_accel after
    // the geometry changed), otherwise one is built here
    const Bvh *b = scene_bvh(s);
    if (!b){
        if (t->own_built) bvh_free(&t->own);
        AccelParams ap = accel_default_params();
        bvh_build(&t->own, s, ap.bvh_leaf, ap.bvh_bins, ap.bvh_split);
        t->own_built = 1;
        b = &t->own;
    }
    if (b->depth > 36) fprintf(stderr, "GPU path tracer: the BVH is %d levels deep, its stack holds 40: some hits may be missed\n", b->depth);
    const int wide = t->wide_ok && b->wide != NULL;
    const uint32_t slots = gpu_slots(b, wide);
    size_t bytes_nodes = ((size_t)t->layout_nodes + (size_t)slots + 1) * 32;
    size_t bytes_aux = ((size_t)t->layout_aux + (size_t)b->nrefs + 1) * 4;
    size_t bytes_prims = ((size_t)s->nprims + 2 * (size_t)t->inst_cap + 1) * 96;
    if (too_big(g, bytes_nodes) || too_big(g, bytes_aux)) return 0;
    WGPUBuffer old_nodes = t->nodes, old_aux = t->aux;
    gpu_buffer_reserve(g, &t->nodes, &t->nodes_cap, bytes_nodes, st, "BVH nodes");
    gpu_buffer_reserve(g, &t->aux, &t->aux_cap, bytes_aux, st, "BVH shapes and normals");
    gpu_buffer_reserve(g, &t->prims, &t->prims_cap, bytes_prims, st, "primitives and instances");
    if (t->nodes != old_nodes || t->aux != old_aux){
        // they grew (a new buffer, empty): everything again
        if (!upload_layout(t, s)) return 0;
    }
    uint32_t *nd = malloc((size_t)(slots ? slots : 1) * 32);
    uint32_t *ax = malloc((size_t)(b->nrefs ? b->nrefs : 1) * 4);
    if (!nd || !ax){ free(nd); free(ax); return 0; }
    if (wide) put_wide(nd, b, t->world_node0, t->world_ref0); else put_nodes(nd, b, t->world_node0, t->world_ref0);
    if (slots) wgpuQueueWriteBuffer(g->queue, t->nodes, (uint64_t)t->world_node0 * 32, nd, (size_t)slots * 32);
    for (int i = 0; i < b->nrefs; i++) ax[i] = (uint32_t)b->refs[i];
    if (b->nrefs) wgpuQueueWriteBuffer(g->queue, t->aux, (uint64_t)t->world_ref0 * 4, ax, (size_t)b->nrefs * 4);
    free(nd);
    free(ax);
    t->nnodes = slots;
    t->world_wide = wide;
    // spheres and quads
    float *pr = malloc((size_t)(s->nprims ? s->nprims : 1) * 96);
    if (!pr) return 0;
    for (int i = 0; i < s->nprims; i++){
        const Primitive *p = &s->prims[i];
        float *o = pr + 24 * (size_t)i;
        put(o, p->center, (float)p->radius);
        put(o + 4, p->q, p->type == PRIM_QUAD ? 1.0f : 0.0f);
        put(o + 8, p->u, (float)p->material);
        put(o + 12, p->v, (float)p->area);
        put(o + 16, p->normal, (float)p->plane_d);
        put(o + 20, p->w, 0);
    }
    if (s->nprims) wgpuQueueWriteBuffer(g->queue, t->prims, 0, pr, (size_t)s->nprims * 96);
    free(pr);
    Primitive *c = realloc(t->prim_copy, (size_t)(s->nprims ? s->nprims : 1) * sizeof(Primitive));
    if (!c) return 0;
    t->prim_copy = c;
    if (s->nprims) memcpy(c, s->prims, (size_t)s->nprims * sizeof(Primitive));
    t->nprim_copy = s->nprims;
    // the photon cache's cells: from the size of everything
    Aabb all = {vec3(1e30f, 1e30f, 1e30f), vec3(-1e30f, -1e30f, -1e30f)};
    if (b->nnodes > 0) all = b->nodes[0].box;
    for (int i = 0; i < s->ninsts; i++){
        all.lo = vec3(r_fmin(all.lo.x, s->insts[i].box.lo.x), r_fmin(all.lo.y, s->insts[i].box.lo.y), r_fmin(all.lo.z, s->insts[i].box.lo.z));
        all.hi = vec3(r_fmax(all.hi.x, s->insts[i].box.hi.x), r_fmax(all.hi.y, s->insts[i].box.hi.y), r_fmax(all.hi.z, s->insts[i].box.hi.z));
    }
    Vec3 ext = v_sub(all.hi, all.lo);
    real e = ext.x > ext.y ? (ext.x > ext.z ? ext.x : ext.z) : (ext.y > ext.z ? ext.y : ext.z);
    t->cell = (float)(e / 256.0);
    if (!(t->cell > 0) || t->cell > 1e20f) t->cell = 1.0f;
    t->specular = 0;
    for (int m = 0; m < s->nmats; m++) if (s->mats[m].type == MAT_MIRROR || s->mats[m].type == MAT_GLASS) t->specular = 1;
    t->valid = 1;
    return 1;
}

// the instances' records, the dynamic meshes' triangles; their BVHs to build (gpu_trace_build)
static int upload_instances(GpuTrace *t, const Scene *s){
    Gpu *g = t->g;
    t->ninst = s->ninsts;
    t->nbuild = 0;
    if (s->ninsts > 0){
        float *rec = malloc((size_t)s->ninsts * 192);
        float *box = malloc((size_t)s->ninsts * 32);
        if (!rec || !box){ free(rec); free(box); return 0; }
        Aabb all = {vec3(1e30f, 1e30f, 1e30f), vec3(-1e30f, -1e30f, -1e30f)};
        for (int i = 0; i < s->ninsts; i++){
            const Instance *in = &s->insts[i];
            float *o = rec + 48 * (size_t)i;
            for (int r = 0; r < 3; r++)
                for (int k = 0; k < 4; k++){ o[4 * r + k] = (float)in->inv.m[r][k]; o[12 + 4 * r + k] = (float)in->xf.m[r][k]; }
            // a mesh: its BVH's root, its index; a shape (scene.h INST_*): the kind the kernel tests
            uint32_t kind = in->mesh >= 0 ? (uint32_t)in->mesh : (in->mesh == INST_SPHERE ? 0xfffffff0u : 0xfffffff1u);
            uint32_t meta[4] = {in->mesh >= 0 ? t->mc[in->mesh].root : 0, in->material >= 0 ? (uint32_t)in->material : 0xffffffffu, kind, 0};
            memcpy(o + 24, meta, 16);
            put(o + 28, in->box.lo, 0);
            put(o + 32, in->box.hi, 0);
            memset(o + 36, 0, 48);
            put(box + 8 * (size_t)i, in->box.lo, 0);
            put(box + 8 * (size_t)i + 4, in->box.hi, 0);
            all.lo = vec3(r_fmin(all.lo.x, in->box.lo.x), r_fmin(all.lo.y, in->box.lo.y), r_fmin(all.lo.z, in->box.lo.z));
            all.hi = vec3(r_fmax(all.hi.x, in->box.hi.x), r_fmax(all.hi.y, in->box.hi.y), r_fmax(all.hi.z, in->box.hi.z));
        }
        wgpuQueueWriteBuffer(g->queue, t->prims, (uint64_t)s->nprims * 96, rec, (size_t)s->ninsts * 192);
        free(rec);
        const Bvh *tl = t->tlas_mode == 1 ? scene_tlas(s) : NULL;
        if (tl && (uint32_t)tl->nnodes <= (uint32_t)(2 * t->inst_cap - 1) && tl->nrefs <= t->inst_cap){
            // the CPU's (SAH) top level, as it is
            uint32_t *nd = malloc((size_t)tl->nnodes * 32), *ax = malloc((size_t)tl->nrefs * 4);
            if (nd && ax){
                put_nodes(nd, tl, t->tlas_node0, t->tlas_ref0);
                for (int i = 0; i < tl->nrefs; i++) ax[i] = (uint32_t)tl->refs[i];
                wgpuQueueWriteBuffer(g->queue, t->nodes, (uint64_t)t->tlas_node0 * 32, nd, (size_t)tl->nnodes * 32);
                wgpuQueueWriteBuffer(g->queue, t->aux, (uint64_t)t->tlas_ref0 * 4, ax, (size_t)tl->nrefs * 4);
            } else tl = NULL;
            free(nd);
            free(ax);
        } else tl = NULL;
        if (!tl) build_queue(t, box, s->ninsts, t->tlas_node0, t->tlas_ref0, 0, all);   // the top level: instance ids
        free(box);
    }
    // dynamic meshes: their new triangles, and a BVH of them built on the GPU in their place
    for (int m = 0; m < s->nmeshes; m++){
        const SceneMesh *sm = &s->meshes[m];
        GpuMeshCopy *c = &t->mc[m];
        if (!sm->dynamic || c->version == sm->version) continue;
        if (!upload_tris(t, sm->tris, sm->ntris, c->tri0, c->nrm0)) return 0;
        float *box = malloc((size_t)(sm->ntris ? sm->ntris : 1) * 32);
        if (!box) return 0;
        for (int i = 0; i < sm->ntris; i++){
            Aabb bb = triangle_bounds(&sm->tris[i]);
            put(box + 8 * (size_t)i, bb.lo, 0);
            put(box + 8 * (size_t)i + 4, bb.hi, 0);
        }
        build_queue(t, box, sm->ntris, c->node0, c->ref0, c->tri0, sm->box);
        free(box);
        c->version = sm->version;
    }
    return 1;
}

int gpu_trace_sync(GpuTrace *t, const Scene *s){
    if (!geometry_same(t, s)){
        if (!layout_same(t, s) && !upload_layout(t, s)) return 0;
        if (!upload_world(t, s)) return 0;
    }
    return upload_instances(t, s);
}

void gpu_trace_clear(GpuTrace *t, WGPUCommandEncoder enc){
    wgpuCommandEncoderClearBuffer(enc, t->acc[t->cur], 0, (uint64_t)t->w * t->h * 16 * t->planes);
}

int gpu_trace_caustics_on(const GpuTrace *t, const GpuTraceParams *tp){ return tp->photons > 0 && t->specular; }

// light paths per pass: photons per pixel
static uint32_t light_paths(const GpuTrace *t, const GpuTraceParams *tp){
    if (!gpu_trace_caustics_on(t, tp)) return 0;
    double n = tp->photons * t->w * t->h;
    return n < 1 ? 1u : (n > 1.6e7 ? 16000000u : (uint32_t)n);
}

// the parameters of `reps` passes, sample indices index, index + 1, ...
static void fill_params(GpuTrace *t, GTP *Pp, const Scene *s, const GpuTraceParams *tp, int y0, int rows,
                        int nvolumes, float lt_mode){
    GTP P;
    memset(&P, 0, sizeof(P));
    put(P.origin, tp->cam.origin, 0);
    put(P.upper_left, tp->cam.upper_left, 0);
    put(P.horizontal, tp->cam.horizontal, 0);
    put(P.vertical, tp->cam.vertical, 0);
    P.dims[0] = (uint32_t)t->w; P.dims[1] = (uint32_t)t->h; P.dims[2] = (uint32_t)y0; P.dims[3] = (uint32_t)rows;
    P.opt[0] = (uint32_t)(tp->max_depth > 0 ? tp->max_depth : 0);
    P.opt[1] = tp->mis ? 1u : 0u;
    P.opt[2] = (uint32_t)s->nprims;
    P.opt[3] = s->light >= 0 && s->light < s->nprims ? (uint32_t)s->light : 0xffffffffu;
    P.cnt[0] = t->nnodes | (t->world_wide && t->nnodes ? GPU_WIDE : 0u);
    P.cnt[1] = t->normals_at;
    P.cnt[2] = 0;
    P.cnt[3] = (uint32_t)nvolumes;
    P.f[0] = (float)tp->clamp;
    P.f[1] = RT_EPS_F;
    P.f[2] = RT_SHADOW_MARGIN_F;
    P.f[3] = t->planes == 5 ? 1.0f : 0.0f;
    P.ext[0] = tp->random ? 0u : 1u;
    P.ext[1] = 0x2545f491u;
    P.ext[2] = gpu_trace_caustics_on(t, tp) ? 1u : 0u;
    P.ext[3] = light_paths(t, tp);
    P.lt[0] = P.ext[3] ? (float)(4294967296.0 / P.ext[3]) : 0.0f;
    P.lt[1] = (float)(camera_screen_area(&tp->cam) / ((double)t->w * t->h));
    P.lt[2] = t->cell;
    P.lt[3] = lt_mode;              // 1 = the light paths fill the photon cache
    P.ins[0] = (uint32_t)t->ninst;
    P.ins[1] = (uint32_t)s->nprims;
    P.ins[2] = t->tlas_node0;
    P.ins[3] = t->world_node0;
    for (int m = 0; m < s->nmats && m < MAX_MATS; m++){
        put(P.mats[m], s->mats[m].albedo, (float)s->mats[m].type);
        put(P.mats[m] + 4, s->mats[m].emission, (float)s->mats[m].ior);
    }
    *Pp = P;
}

static void write_params(GpuTrace *t, WGPUBuffer dst, const Scene *s, const GpuTraceParams *tp, int y0, int rows,
                         int nvolumes, uint32_t index, int reps, float lt_mode){
    Gpu *g = t->g;
    GTP P;
    fill_params(t, &P, s, tp, y0, rows, nvolumes, lt_mode);
    if (reps < 1) reps = 1;
    if (reps > MAX_REPS) reps = MAX_REPS;
    unsigned char buf[GTP_STRIDE * MAX_REPS];
    for (int r = 0; r < reps; r++){
        P.cnt[2] = index + (uint32_t)r;
        memcpy(buf + (size_t)r * GTP_STRIDE, &P, sizeof(P));
    }
    wgpuQueueWriteBuffer(g->queue, dst, 0, buf, (size_t)(reps - 1) * GTP_STRIDE + sizeof(P));
}

// `repeats` dispatches, the r-th with the r-th copy of the parameters (light = 1: the light
// tracer, whose binding 11 takes the place of 9)
static void dispatch_ex(GpuTrace *t, WGPUCommandEncoder enc, WGPUComputePipeline pipe, size_t threads, int repeats,
                        WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat, int light){
    WGPUBuffer b[11] = {frame_u, dens, bricks, vstat, light == 2 ? t->pc_params : light ? t->lt_params : t->params,
                        t->nodes, t->aux, t->prims, t->tris,
                        light == 2 ? t->grid : light ? t->caus : t->acc[t->cur], t->blue};
    WGPUBindGroupEntry e[11];
    memset(e, 0, sizeof(e));
    for (int i = 0; i < 11; i++){
        e[i].binding = (uint32_t)(light && i == 9 ? 11 : i);
        e[i].buffer = b[i];
        e[i].size = i == 4 ? sizeof(GTP) : WGPU_WHOLE_SIZE;
    }
    if (repeats < 1) repeats = 1;
    if (repeats > MAX_REPS) repeats = MAX_REPS;
    // timed: the tracer per full-image pass (1 sample per pixel), the light tracer per pass
    WGPUComputePassEncoder pass = light == 2 ? wgpuCommandEncoderBeginComputePass(enc, NULL)
                                : timed_pass(t, enc, light ? TS_LIGHT : TS_TRACE,
                                             light ? repeats : (double)threads * repeats / ((double)t->w * t->h));
    wgpuComputePassEncoderSetPipeline(pass, pipe);
    for (int r = 0; r < repeats; r++){
        e[4].offset = (uint64_t)r * GTP_STRIDE;
        WGPUBindGroupDescriptor d;
        memset(&d, 0, sizeof(d));
        d.layout = light ? t->lt_bgl : t->bgl;
        d.entryCount = 11;
        d.entries = e;
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(t->g->device, &d);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
        gpu_dispatch(pass, threads);
        wgpuBindGroupRelease(bg);
    }
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

static void dispatch(GpuTrace *t, WGPUCommandEncoder enc, WGPUComputePipeline pipe, size_t threads, int repeats,
                     WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat){
    dispatch_ex(t, enc, pipe, threads, repeats, frame_u, dens, bricks, vstat, 0);
}

// ---- the wavefront tracer --------------------------------------------------------------

enum { WF_GEN, WF_EXT, WF_SHADE, WF_ARGS, WF_SHADOW, WF_RESOLVE, NWF };
static const char *const WF_ENTRY[NWF] = {"wf_gen", "wf_ext", "wf_shade", "wf_args", "wf_shadow", "wf_resolve"};
// the bindings each kernel uses (index = binding; as gpu_layout): within 8 storage buffers each
static const char *const WF_LAYOUT[NWF] = {
    "----U----WU-W",      // gen: parameters, samples (their count seeds the path), blue noise, WF
    "----URRRR---W",      // ext: the BVH and shapes, WF
    "URRRU-RRR-U-WW",     // shade: media, shapes and normals, blue noise, WF, WC
    "------------WWW",    // args: WF, WC, WA
    "URRRURRRR---W",      // shadow: media, the BVH and shapes, WF
    "----U----W--W"};     // resolve: samples, WF
#define WF_PATH_BYTES 140          // per path: 8 vec4 of state, hit and shadow ray, 3 queue entries
#define WF_BYTES_MAX (96u << 20)   // WF at most 96 MB (browsers' default limit per buffer: 128 MB)
#define MAX_WAVES 64

void gpu_trace_set_wavefront(GpuTrace *t, int on){ if (t) t->wavefront = on; }

static WGPUComputePipeline pipeline_of(Gpu *g, const char *label, WGPUShaderModule m, const char *entry, WGPUBindGroupLayout bgl){
    WGPUPipelineLayoutDescriptor ld;
    memset(&ld, 0, sizeof(ld));
    ld.bindGroupLayoutCount = 1;
    ld.bindGroupLayouts = &bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(g->device, &ld);
    WGPUComputePipelineDescriptor cd;
    memset(&cd, 0, sizeof(cd));
    cd.label = gpu_str(label);
    cd.layout = pl;
    cd.compute.module = m;
    cd.compute.entryPoint = gpu_str(entry);
    WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(g->device, &cd);
    wgpuPipelineLayoutRelease(pl);
    return p;
}

// made the first time it is used: one shader module, six pipelines, the buffers
static int wf_ready(GpuTrace *t){
    if (t->wf_made) return 1;
    if (t->wf_failed) return 0;
    Gpu *g = t->g;
    size_t bytes = WF_BYTES_MAX < g->max_storage ? WF_BYTES_MAX : (size_t)g->max_storage;
    size_t paths = bytes / WF_PATH_BYTES;
    t->wf_rows = (int)(paths / (size_t)t->w);
    if (t->wf_rows < 1){ t->wf_failed = 1; return 0; }
    if (t->wf_rows > t->h) t->wf_rows = t->h;
    if ((t->h + t->wf_rows - 1) / t->wf_rows > MAX_WAVES) t->wf_rows = (t->h + MAX_WAVES - 1) / MAX_WAVES;
    size_t n = (size_t)t->wf_rows * t->w;
    size_t wf_bytes = (1 + 8 * n + 3 * ((n + 3) / 4)) * 16;
    const char *parts[] = {t->types_wgsl, "\n", t->media_wgsl, "\n", TRACE, "\n", WAVE};
    size_t len = 1;
    for (int i = 0; i < 7; i++) len += strlen(parts[i]);
    char *src = malloc(len);
    if (!src){ t->wf_failed = 1; return 0; }
    src[0] = 0;
    for (int i = 0; i < 7; i++) strcat(src, parts[i]);
    WGPUShaderModule m = gpu_shader(g, "path tracer: wavefront", src);
    free(src);
    for (int k = 0; k < NWF; k++){
        char label[64];
        snprintf(label, sizeof(label), "path tracer: %s", WF_ENTRY[k]);
        t->wf_bgl[k] = gpu_layout(g, WGPUShaderStage_Compute, WF_LAYOUT[k]);
        t->wf_pipe[k] = pipeline_of(g, label, m, WF_ENTRY[k], t->wf_bgl[k]);
    }
    wgpuShaderModuleRelease(m);
    t->wf = gpu_buffer(g, "wavefront: paths", wf_bytes, WGPUBufferUsage_Storage);
    t->wfc = gpu_buffer(g, "wavefront: counters", 16, WGPUBufferUsage_Storage);
    t->wfa = gpu_buffer(g, "wavefront: dispatch sizes", 32, WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect);
    t->wf_params = gpu_buffer(g, "wavefront: parameters", GTP_STRIDE * MAX_REPS * MAX_WAVES,
                              WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    t->wf_made = t->wf && t->wfc && t->wfa && t->wf_params;
    if (!t->wf_made) t->wf_failed = 1;
    return t->wf_made;
}

static WGPUBindGroup wf_bind(GpuTrace *t, int k, uint64_t toff, WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks,
                             WGPUBuffer vstat){
    WGPUBuffer b[15] = {frame_u, dens, bricks, vstat, t->wf_params, t->nodes, t->aux, t->prims, t->tris,
                        t->acc[t->cur], t->blue, NULL, t->wf, t->wfc, t->wfa};
    WGPUBindGroupEntry e[15];
    memset(e, 0, sizeof(e));
    int n = 0;
    for (int i = 0; WF_LAYOUT[k][i]; i++){
        if (WF_LAYOUT[k][i] == '-') continue;
        e[n].binding = (uint32_t)i;
        e[n].buffer = b[i];
        e[n].offset = i == 4 ? toff : 0;
        e[n].size = i == 4 ? sizeof(GTP) : WGPU_WHOLE_SIZE;
        n++;
    }
    WGPUBindGroupDescriptor d;
    memset(&d, 0, sizeof(d));
    d.layout = t->wf_bgl[k];
    d.entryCount = (size_t)n;
    d.entries = e;
    return wgpuDeviceCreateBindGroup(t->g->device, &d);
}

static void write_params(GpuTrace *t, WGPUBuffer dst, const Scene *s, const GpuTraceParams *tp, int y0, int rows,
                         int nvolumes, uint32_t index, int reps, float lt_mode);
static void fill_params(GpuTrace *t, GTP *P, const Scene *s, const GpuTraceParams *tp, int y0, int rows,
                        int nvolumes, float lt_mode);

// a band: per pass and wave, the camera rays, then per bounce extend, shade, sizes, shadow
// rays, then the image; all in one compute pass
static void wf_encode(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp,
                      int y0, int rows, int repeats, int nvolumes, uint32_t index,
                      WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat){
    const int waves = (rows + t->wf_rows - 1) / t->wf_rows;
    if (repeats < 1) repeats = 1;
    if (repeats > MAX_REPS) repeats = MAX_REPS;
    GTP P;
    fill_params(t, &P, s, tp, y0, rows, nvolumes, 0);
    size_t used = (size_t)(repeats * waves - 1) * GTP_STRIDE + sizeof(GTP);
    unsigned char *buf = calloc(1, (size_t)repeats * waves * GTP_STRIDE);
    if (!buf) return;
    for (int r = 0; r < repeats; r++)
        for (int w = 0; w < waves; w++){
            P.dims[2] = (uint32_t)(y0 + w * t->wf_rows);
            P.dims[3] = (uint32_t)(rows - w * t->wf_rows < t->wf_rows ? rows - w * t->wf_rows : t->wf_rows);
            P.cnt[2] = index + (uint32_t)r;
            memcpy(buf + ((size_t)r * waves + w) * GTP_STRIDE, &P, sizeof(P));
        }
    wgpuQueueWriteBuffer(t->g->queue, t->wf_params, 0, buf, used);
    free(buf);
    const int iters = (tp->max_depth > 0 ? (tp->max_depth < 4096 ? tp->max_depth : 4096) : 0) + 1;
    static int prof = -1;
    if (prof < 0){ const char *e = getenv("LR_WF_PROFILE"); prof = e && atoi(e) > 0; }
    const double units = (double)t->w * rows * repeats / ((double)t->w * t->h);
    WGPUComputePassEncoder pass = prof ? NULL : timed_pass(t, enc, TS_TRACE, units);
    // (LR_WF_PROFILE=1: every dispatch in a pass of its own, timed by kernel)
#define WF_STEP(k, call) do { \
        if (prof) pass = timed_pass(t, enc, TS_WF0 + (k), (k) == WF_GEN || (k) == WF_RESOLVE ? units / waves : units / waves / iters); \
        wgpuComputePassEncoderSetPipeline(pass, t->wf_pipe[k]); \
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg[k], 0, NULL); \
        call; \
        if (prof){ wgpuComputePassEncoderEnd(pass); wgpuComputePassEncoderRelease(pass); } \
    } while (0)
    for (int r = 0; r < repeats; r++)
        for (int w = 0; w < waves; w++){
            uint64_t toff = ((uint64_t)r * waves + w) * GTP_STRIDE;
            size_t n = (size_t)t->w * (rows - w * t->wf_rows < t->wf_rows ? rows - w * t->wf_rows : t->wf_rows);
            WGPUBindGroup bg[NWF];
            for (int k = 0; k < NWF; k++) bg[k] = wf_bind(t, k, toff, frame_u, dens, bricks, vstat);
            WF_STEP(WF_GEN, gpu_dispatch(pass, n));
            for (int it = 0; it < iters; it++){
                WF_STEP(WF_EXT, if (it == 0) gpu_dispatch(pass, n); else wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, t->wfa, 0));
                WF_STEP(WF_SHADE, if (it == 0) gpu_dispatch(pass, n); else wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, t->wfa, 0));
                WF_STEP(WF_ARGS, wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1));
                WF_STEP(WF_SHADOW, wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, t->wfa, 16));
            }
            WF_STEP(WF_RESOLVE, gpu_dispatch(pass, n));
            for (int k = 0; k < NWF; k++) wgpuBindGroupRelease(bg[k]);
        }
#undef WF_STEP
    if (!prof){
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }
}

void gpu_trace_encode(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp,
                      int y0, int rows, int repeats, int nvolumes, uint32_t index,
                      WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat){
    gpu_trace_build(t, enc);
    if (t->wavefront && wf_ready(t)){
        wf_encode(t, enc, s, tp, y0, rows, repeats, nvolumes, index, frame_u, dens, bricks, vstat);
        return;
    }
    write_params(t, t->params, s, tp, y0, rows, nvolumes, index, repeats, 0);
    dispatch(t, enc, t->pipe, (size_t)t->w * rows, repeats, frame_u, dens, bricks, vstat);
}

void gpu_trace_light(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp,
                     int passes, int nvolumes, uint32_t index,
                     WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat){
    if (!gpu_trace_caustics_on(t, tp) || passes < 1) return;
    write_params(t, t->lt_params, s, tp, 0, t->h, nvolumes, index, passes, 0);
    dispatch_ex(t, enc, t->lt_pipe, light_paths(t, tp), passes, frame_u, dens, bricks, vstat, 1);
    t->caus_passes += passes;
}

void gpu_trace_caustics_clear(GpuTrace *t, WGPUCommandEncoder enc){
    wgpuCommandEncoderClearBuffer(enc, t->caus, 0, (uint64_t)t->w * t->h * 24);
    t->caus_passes = 0;
}

int gpu_trace_caustic_passes(const GpuTrace *t){ return t->caus_passes; }

#define PC_CELLS (1u << 20)
void gpu_trace_photons(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp, int nvolumes,
                       uint32_t index, double fade, WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks,
                       WGPUBuffer vstat){
    if (!gpu_trace_caustics_on(t, tp) || t->planes != 5) return;
    Gpu *g = t->g;
    if (!t->grid){
        t->grid = gpu_buffer(g, "photon cache", (size_t)PC_CELLS * 32, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
        wgpuCommandEncoderClearBuffer(enc, t->grid, 0, (uint64_t)PC_CELLS * 32);
        t->pc_paths = 0;
    }
    if (fade < 1 && t->pc_paths > 0){
        struct { float k[4]; uint32_t n[4]; } D = {{(float)fade, 0, 0, 0}, {PC_CELLS * 4, 0, 0, 0}};
        wgpuQueueWriteBuffer(g->queue, t->dc_params, 0, &D, sizeof(D));
        WGPUBuffer b[2] = {t->dc_params, t->grid};
        WGPUBindGroup bg = gpu_bind(g, t->dc_bgl, b, 2);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
        wgpuComputePassEncoderSetPipeline(pass, t->dc_pipe);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
        gpu_dispatch(pass, (size_t)PC_CELLS * 4);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuBindGroupRelease(bg);
        t->pc_paths *= fade;
    }
    write_params(t, t->pc_params, s, tp, 0, t->h, nvolumes, index ^ 0x5bd1e995u, 1, 1.0f);
    dispatch_ex(t, enc, t->lt_pipe, light_paths(t, tp), 1, frame_u, dens, bricks, vstat, 2);
    t->pc_paths += light_paths(t, tp);
}

void gpu_trace_photons_clear(GpuTrace *t, WGPUCommandEncoder enc){
    if (t->grid) wgpuCommandEncoderClearBuffer(enc, t->grid, 0, (uint64_t)PC_CELLS * 32);
    t->pc_paths = 0;
}

double gpu_trace_photon_paths(const GpuTrace *t){ return t->pc_paths; }

WGPUBuffer gpu_trace_resolve(GpuTrace *t, WGPUCommandEncoder enc, WGPUBuffer base, int cache){
    int use_cache = cache && t->grid && t->pc_paths > 0;
    double we = t->caus_passes ? (use_cache ? t->caus_passes / (t->caus_passes + 8.0) : 1.0) : 0.0;
    struct { uint32_t dims[4]; float k[4]; } R = {{(uint32_t)t->w, (uint32_t)t->h, use_cache ? 1u : 0u, 0},
        {t->caus_passes ? (float)(1.0 / (4294967296.0 * t->caus_passes)) : 0.0f, (float)we,
         use_cache ? (float)(1.0 / (t->pc_paths * 256.0)) : 0.0f, t->cell}};
    wgpuQueueWriteBuffer(t->g->queue, t->rs_params, 0, &R, sizeof(R));
    WGPUBuffer b[6] = {t->rs_params, base, t->caus, t->final, t->grid ? t->grid : t->caus, t->acc[t->cur]};
    WGPUBindGroup bg = gpu_bind(t->g, t->rs_bgl, b, 6);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    wgpuComputePassEncoderSetPipeline(pass, t->rs_pipe);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    gpu_dispatch(pass, (size_t)t->w * t->h);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    wgpuBindGroupRelease(bg);
    return t->final;
}

static void put_cam(float *d, const Camera *c){
    put(d, c->origin, 0); put(d + 4, c->upper_left, 0); put(d + 8, c->horizontal, 0); put(d + 12, c->vertical, 0);
}

void gpu_trace_begin_image(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp,
                           int history, int nvolumes,
                           WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat){
    Gpu *g = t->g;
    gpu_trace_build(t, enc);         // the top level over the instances (they may have moved)
    if (t->planes != 5){             // no guide on this GPU: no reprojection either
        gpu_trace_clear(t, enc);
        t->cam = tp->cam;
        t->has_image = 1;
        return;
    }
    int reproject = history > 0 && t->has_image;
    if (reproject && !t->acc[1 - t->cur])
        t->acc[1 - t->cur] = gpu_buffer(g, "path tracer samples (previous image)", (size_t)t->w * t->h * 16 * t->planes,
                                        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
    if (reproject) t->cur = 1 - t->cur;
    write_params(t, t->params, s, tp, 0, t->h, nvolumes, 0, 1, 0);
    dispatch(t, enc, t->gbuf_pipe, (size_t)t->w * t->h, 1, frame_u, dens, bricks, vstat);
    if (reproject){
        struct { float cams[8][4]; uint32_t dims[4]; } R;
        memset(&R, 0, sizeof(R));
        put_cam(R.cams[0], &t->cam);
        put_cam(R.cams[4], &tp->cam);
        R.dims[0] = (uint32_t)t->w; R.dims[1] = (uint32_t)t->h; R.dims[2] = (uint32_t)history;
        wgpuQueueWriteBuffer(g->queue, t->rp_params, 0, &R, sizeof(R));
        WGPUBuffer b[3] = {t->rp_params, t->acc[1 - t->cur], t->acc[t->cur]};
        WGPUBindGroup bg = gpu_bind(g, t->rp_bgl, b, 3);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
        wgpuComputePassEncoderSetPipeline(pass, t->rp_pipe);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
        gpu_dispatch(pass, (size_t)t->w * t->h);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuBindGroupRelease(bg);
    }
    t->cam = tp->cam;
    t->has_image = 1;
}

void gpu_trace_destroy(GpuTrace *t){
    if (!t) return;
    if (t->timer){
        for (int k = 0; k < NTS; k++){
            double ms = gpu_timer_ms_per_unit(t->timer, k);
            if (ms > 0) printf("GPU time, %s: %.3f ms per %s\n", TS_NAMES[k], ms,
                               k == TS_TRACE || k >= TS_WF0 ? "pass (1 sample per pixel, the whole image)" : k == TS_LIGHT ? "pass" : "frame with builds");
        }
        gpu_timer_destroy(t->timer);
    }
    for (int k = 0; k < NWF; k++){
        if (t->wf_pipe[k]) wgpuComputePipelineRelease(t->wf_pipe[k]);
        if (t->wf_bgl[k]) wgpuBindGroupLayoutRelease(t->wf_bgl[k]);
    }
    WGPUBuffer wfb[] = {t->wf, t->wfc, t->wfa, t->wf_params};
    for (int i = 0; i < 4; i++) if (wfb[i]){ wgpuBufferDestroy(wfb[i]); wgpuBufferRelease(wfb[i]); }
    WGPUBuffer bufs[] = {t->params, t->rp_params, t->blue, t->acc[0], t->acc[1], t->nodes, t->aux, t->prims, t->tris,
                         t->caus, t->rs_params, t->final, t->lt_params, t->grid, t->dc_params, t->pc_params,
                         t->bd_keys, t->bd_flags, t->bd_boxes, t->bd_par};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) if (bufs[i]){ wgpuBufferDestroy(bufs[i]); wgpuBufferRelease(bufs[i]); }
    WGPUComputePipeline pipes[] = {t->pipe, t->gbuf_pipe, t->rp_pipe, t->lt_pipe, t->rs_pipe, t->dc_pipe,
                                   t->bd_morton, t->bd_sort, t->bd_hier, t->bd_leaves, t->bd_up};
    for (size_t i = 0; i < sizeof(pipes) / sizeof(pipes[0]); i++) if (pipes[i]) wgpuComputePipelineRelease(pipes[i]);
    WGPUBindGroupLayout bgls[] = {t->bgl, t->rp_bgl, t->lt_bgl, t->rs_bgl, t->dc_bgl, t->bd_bgl};
    for (size_t i = 0; i < sizeof(bgls) / sizeof(bgls[0]); i++) if (bgls[i]) wgpuBindGroupLayoutRelease(bgls[i]);
    free(t->mc);
    free(t->arena_boxes);
    free(t->arena_par);
    if (t->own_built) bvh_free(&t->own);
    free(t->prim_copy);
    free(t);
}

#else
typedef int trace_gpu_unused;
#endif
