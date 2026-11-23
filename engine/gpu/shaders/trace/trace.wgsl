// trace.wgsl - used by engine/gpu/trace_gpu.c (TRACE)
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
/* albedo: the base colour, w = the type (3 a lamp); emission, w = ior; pbr: metallic, roughness,
   transmission (material_resolved) */
struct TMat { albedo: vec4f, emission: vec4f, pbr: vec4f, tex: vec4u };   /* tex: as RMat's (raster_gpu.c) */
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
    sky: vec4f,    /* the sky (SKY, SKYC): its scale, cos and sin of its turn, 1 = there is one */
    txl: vec4u,    /* the triangles' texture coordinates in aux: where their indices start (3 per
                      triangle), where the coordinates start (6 floats each: u, v, tangent), 1 = there are */
    mats: array<TMat, 16>,
    em: vec4u,     /* the glowing shapes sampled as lights (Scene.emitters): how many */
    emi: array<vec4u, 4>,   /* ... their instances, 4 to an entry */
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
/* the blue-noise mask (4096 ranks, 16 bits each), then the BSDF's tables (gpu_bsdf_tables) */
struct TBN { bn: array<vec4u, 512>, tab: array<vec4f, 528> };
@group(0) @binding(10) var<uniform> TB: TBN;
fn bsdf_tab(k: u32) -> f32 { return TB.tab[k >> 2u][k & 3u]; }
/* the light tracer's caustics (ltrace; bound instead of accum): per pixel and channel, a 64-bit
   32.32 fixed-point sum as two u32 (low, high) */
@group(0) @binding(11) var<storage, read_write> caus: array<atomic<u32>>;
/* the sky (scene/env.h): SKY rgb = the radiance (times T.sky.x), a = the pixel's probability;
   SKYC the cumulative distributions, rows 0..h-1 each row's (entry x: up to pixel x, with it),
   row h the rows' (entry y: up to row y) */
@group(0) @binding(16) var SKY: texture_2d<f32>;
@group(0) @binding(17) var SKYC: texture_2d<f32>;
/* the materials' textures (gpu/tex_gpu.c) */
@group(0) @binding(18) var TEXA: texture_2d<f32>;
struct TT { e: array<vec4u, 512> };
@group(0) @binding(19) var<uniform> TXT: TT;
fn tex_entry(k: u32) -> vec4u { return TXT.e[k]; }

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

/* ---- the sampler (base/sampler.c): Owen-scrambled Sobol, shifted per pixel by blue noise ---- */
fn bn_value(x: u32, y: u32) -> f32 {
    let i = (y & 63u) * 64u + (x & 63u);
    let w = TB.bn[i >> 3u][(i >> 1u) & 3u];
    return (f32((w >> ((i & 1u) * 16u)) & 0xffffu) + 0.5) / 4096.0;
}
/* owen_r: x's bits already reversed (base/sampler.c, owen_reversed) */
fn owen(x0: u32, seed: u32) -> u32 { return owen_r(reverseBits(x0), seed); }
fn owen_r(x0: u32, seed: u32) -> u32 {
    var x = x0;
    x ^= x * 0x3d20adeau;
    x += seed;
    x *= (seed >> 16u) | 1u;
    x ^= x * 0x05526c56u;
    x ^= x * 0x53a22864u;
    return reverseBits(x);
}
fn sobol1_r(i0: u32) -> u32 {   /* the second Sobol dimension, bits reversed (base/sampler.c) */
    var i = i0;
    i ^= (i >> 1u) & 0x55555555u;
    i ^= (i >> 2u) & 0x33333333u;
    i ^= (i >> 4u) & 0x0f0f0f0fu;
    i ^= (i >> 8u) & 0x00ff00ffu;
    return i ^ (i >> 16u);
}
struct Smp { x: u32, y: u32, idx: u32, on: u32 };
const SMP_PAIRS = 13u;   /* SAMPLER_PAIRS of base/sampler.h */
fn s2(s: Smp, dim: u32, rng: ptr<function, u32>) -> vec2f {
    if (s.on == 0u || dim >= SMP_PAIRS) {
        let a = rnd(rng);
        return vec2f(a, rnd(rng));
    }
    let tile = hash32(((s.x >> 6u) * 73856093u) ^ ((s.y >> 6u) * 19349663u));
    let seed = hash32(T.ext.y ^ hash32((dim * 0x9e3779b9u) ^ tile));
    let i = owen(s.idx, seed);
    let a = owen_r(i, hash32(seed ^ 0xa511e9b3u));
    let b = owen_r(sobol1_r(i), hash32(seed ^ 0x63d83595u));
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
   RTX 3070 (docs/STATUS.md): 24 to 64 entries cost about the same, 40 leaves room for
   a mesh's tree under the top level's; a full stack drops what does not fit. */
const STACK = 40;
const INST: u32 = 0x40000000u;
const EXIT: u32 = 0xfffffffeu;
fn traverse(r: Ray, tmin: f32, tmax: f32, stop_at_first: bool) -> Found {
    var best = Found(tmax, NONE, vec2f(0.0), NONE);
    var stack: array<u32, STACK>;
    var sp = 0;
    if (T.ins.x > 0u) { stack[SI(sp)] = T.ins.z; sp++; }
    if ((T.cnt.x & 0x7fffffffu) > 0u) { stack[SI(sp)] = T.ins.w | (T.cnt.x & WIDE); sp++; }
    var cur = NONE;
    var ray = r;
    var inv = safe_inv(r.d);
    while (sp > 0) {
        sp--;
        let ni = stack[SI(sp)];
        if (ni == EXIT) {
            cur = NONE;
            ray = r;
            inv = safe_inv(r.d);
            continue;
        }
        /* a 4-wide node (rt/bvh.h): 4 slots, each a child's box; a leaf child is pushed as its
           slot (a leaf node of the binary layout), an inner one as WIDE | its slots; the hit
           children go on the stack farthest first */
        if (WIDE_ON && (ni & WIDE) != 0u) {
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
                stack[SI(sp)] = ce[q];
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
                stack[SI(sp)] = EXIT;
                stack[SI(sp + 1)] = bitcast<u32>(prims[T.ins.y + 2u * i + 1u].c.x);
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
                        stack[SI(sp)] = INST | i;
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
            stack[SI(sp)] = select(nd.a, nd.b, left_first);
            stack[SI(sp + 1)] = select(nd.b, nd.a, left_first);
            sp += 2;
        } else if (tl >= 0.0 && sp < STACK) {
            stack[SI(sp)] = nd.a;
            sp++;
        } else if (tr >= 0.0 && sp < STACK) {
            stack[SI(sp)] = nd.b;
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

/* ---- textures (PBR5): where a hit is in its shape's texture (rt/intersect.c hit_texcoords) ---- */
struct TexCo { uv: vec2f, t: vec4f };
fn sphere_uv(n: vec3f) -> vec2f {
    let u = 0.5 + atan2(n.x, n.z) * (0.5 / PI_);
    return vec2f(u - floor(u), acos(clamp(n.y, -1.0, 1.0)) / PI_);
}
fn sphere_tan(uv: vec2f) -> vec4f {
    let ph = (uv.x - 0.5) * 2.0 * PI_;
    return vec4f(cos(ph), 0.0, -sin(ph), 1.0);
}
fn auxf(i: u32) -> f32 { return bitcast<f32>(aux[i]); }
fn hit_texco(r: Ray, f: Found, h: Hit) -> TexCo {
    var c = TexCo(vec2f(0.0), vec4f(1.0, 0.0, 0.0, 1.0));
    if (f.id >= UNIT_SPHERE) {                /* a ball or a box, in its own space */
        let o = to_object(f.inst, r);
        let q = o.o + o.d * f.t;
        if (f.id == UNIT_SPHERE) {
            c.uv = sphere_uv(normalize(q));
            c.t = sphere_tan(c.uv);
        } else {
            /* scene.c unit_box_face(2 k + (side > 0)): u, v the next two axes (twice), swapped
               on the negative side; (a, b) of the point, tangent along u */
            let aq = abs(q);
            var k = 2u;
            if (aq.x >= aq.y && aq.x >= aq.z) { k = 0u; } else if (aq.y >= aq.z) { k = 1u; }
            let pos = q[k] > 0.0;
            var eu = vec3f(0.0); eu[(k + 1u) % 3u] = 2.0;
            var ev = vec3f(0.0); ev[(k + 2u) % 3u] = 2.0;
            var ek = vec3f(0.0); ek[k] = select(-1.0, 1.0, pos);
            let q0 = ek - (eu + ev) * 0.5;
            var u = eu;
            var v = ev;
            if (!pos) { u = ev; v = eu; }
            let rel = q - q0;
            c.uv = vec2f(dot(rel, u) / dot(u, u), dot(rel, v) / dot(v, v));
            c.t = vec4f(normalize(u), -1.0);
        }
        c.t = vec4f(normal_to_world(f.inst, c.t.xyz), c.t.w);   /* (a rotation and a uniform scale) */
        return c;
    }
    if (f.id < T.opt.z) {
        let pr = prims[f.id];
        if (pr.q.w < 0.5) {
            c.uv = sphere_uv((h.p - pr.c.xyz) / pr.c.w);
            c.t = sphere_tan(c.uv);
        } else {
            let rel = h.p - pr.q.xyz;
            c.uv = vec2f(dot(pr.w.xyz, cross(rel, pr.v.xyz)), dot(pr.w.xyz, cross(pr.u.xyz, rel)));
            c.t = vec4f(normalize(pr.u.xyz), -1.0);
        }
        return c;
    }
    let gi = f.id - T.opt.z;
    let tr = tris[gi];
    c.uv = f.b;
    c.t = vec4f(normalize(tr.u), 1.0);
    if (T.txl.z != 0u) {
        let o = T.txl.x + 3u * gi;
        let i0 = aux[o];
        if (i0 != NONE) {
            let w = vec3f(1.0 - f.b.x - f.b.y, f.b.x, f.b.y);
            var uv = vec2f(0.0);
            var t = vec3f(0.0);
            var hand = 1.0;
            for (var k = 0u; k < 3u; k++) {
                let e = T.txl.y + 6u * aux[o + k];
                uv += w[k] * vec2f(auxf(e), auxf(e + 1u));
                t += w[k] * vec3f(auxf(e + 2u), auxf(e + 3u), auxf(e + 4u));
                if (k == 0u) { hand = auxf(e + 5u); }
            }
            c.uv = uv;
            c.t = vec4f(t, hand);
        }
    }
    if (f.inst != NONE) { c.t = vec4f(normal_to_world(f.inst, c.t.xyz), c.t.w); }
    return c;
}
/* the material at a hit with its textures (scene.c material_at, level 0), the shading normal
   bent by its normal map and tilted back towards where the ray came from (rt/tracer.c
   hit_shading) */
fn shade_tex(m0: TMat, r: Ray, f: Found, h: ptr<function, Hit>) -> TMat {
    if (all(m0.tex.xyz == vec3u(0u)) || u32(m0.albedo.w) == 3u) { return m0; }
    let tc = hit_texco(r, f, *h);
    var m = m0;
    let tm = tex_material(m0.tex, m0.albedo.rgb, m0.pbr.x, m0.pbr.y, m0.emission.rgb, tc.uv, -1000.0);
    m.albedo = vec4f(tm.base, m0.albedo.w);
    m.pbr.x = tm.metallic;
    m.pbr.y = tm.rough;
    m.emission = vec4f(tm.emission, m0.emission.w);
    if ((m0.tex.z & 255u) != 0u) {
        var n = tex_normal(m0.tex, tc.uv, -1000.0, (*h).n, tc.t);
        let v = -r.d;
        let c = dot(n, v);
        if (c < 1e-3) { n = normalize(n + v * (1e-3 - c)); }
        (*h).n = n;
    }
    return m;
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

/* ---- the sky (scene/env.c): lookups, its pdf, directions drawn by its luminance ---- */
/* SKY_ON: a constant the module is compiled with (two variants: the sky's code costs
   registers, so the kernels of scenes without one leave it out) */
fn sky_on() -> bool { return SKY_ON && T.sky.w > 0.5; }
fn sky_uv(d: vec3f) -> vec2f {
    let x = d.x * T.sky.y - d.z * T.sky.z;           /* turned back by the sky's turn */
    let z = d.x * T.sky.z + d.z * T.sky.y;
    let u = 0.5 + atan2(x, z) * (0.5 / PI_);
    return vec2f(u - floor(u), acos(clamp(d.y, -1.0, 1.0)) / PI_);
}
fn sky_dir(u: f32, v: f32) -> vec3f {
    let ph = (u - 0.5) * 2.0 * PI_;
    let th = v * PI_;
    let st = sin(th);
    let x = st * sin(ph);
    let z = st * cos(ph);
    return vec3f(x * T.sky.y + z * T.sky.z, cos(th), -x * T.sky.z + z * T.sky.y);
}
fn sky_eval(d: vec3f) -> vec3f {
    let dims = vec2i(textureDimensions(SKY));
    let w = dims.x;
    let h = dims.y;
    let uv = sky_uv(d);
    let x = uv.x * f32(w) - 0.5;
    let y = uv.y * f32(h) - 0.5;
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let ya = clamp(y0, 0, h - 1);
    let yb = clamp(y0 + 1, 0, h - 1);
    let xa = ((x0 % w) + w) % w;
    let xb = (xa + 1) % w;
    let a = textureLoad(SKY, vec2i(xa, ya), 0).rgb;
    let b = textureLoad(SKY, vec2i(xb, ya), 0).rgb;
    let c = textureLoad(SKY, vec2i(xa, yb), 0).rgb;
    let e = textureLoad(SKY, vec2i(xb, yb), 0).rgb;
    return ((1.0 - fy) * ((1.0 - fx) * a + fx * b) + fy * ((1.0 - fx) * c + fx * e)) * T.sky.x;
}
fn sky_pdf(d: vec3f) -> f32 {
    let dims = vec2i(textureDimensions(SKY));
    let uv = sky_uv(d);
    let st = sin(uv.y * PI_);
    if (st < 1e-6) { return 0.0; }
    let x = min(i32(uv.x * f32(dims.x)), dims.x - 1);
    let y = min(i32(uv.y * f32(dims.y)), dims.y - 1);
    return textureLoad(SKY, vec2i(x, y), 0).a * f32(dims.x) * f32(dims.y) / (2.0 * PI_ * PI_ * st);
}
/* the first of n entries of row `row` of SKYC above x (n - 1 if none) */
fn sky_search(row: i32, n: i32, x: f32) -> i32 {
    var lo = 0;
    var hi = n;
    while (lo < hi) {
        let mid = (lo + hi) / 2;
        if (textureLoad(SKYC, vec2i(mid, row), 0).r <= x) { lo = mid + 1; } else { hi = mid; }
    }
    return min(lo, n - 1);
}
/* a direction drawn by the sky's luminance: xyz, w = its pdf (0: none) */
fn sky_sample(u: vec2f) -> vec4f {
    let dims = vec2i(textureDimensions(SKY));
    let w = dims.x;
    let h = dims.y;
    let y = sky_search(h, h, u.y);
    let m0 = select(0.0, textureLoad(SKYC, vec2i(y - 1, h), 0).r, y > 0);
    let m1 = textureLoad(SKYC, vec2i(y, h), 0).r;
    let dv = select(0.5, (u.y - m0) / (m1 - m0), m1 > m0);
    let x = sky_search(y, w, u.x);
    let c0 = select(0.0, textureLoad(SKYC, vec2i(x - 1, y), 0).r, x > 0);
    let c1 = textureLoad(SKYC, vec2i(x, y), 0).r;
    let du = select(0.5, (u.x - c0) / (c1 - c0), c1 > c0);
    let v = (f32(y) + clamp(dv, 0.0, 0.9999)) / f32(h);
    let st = sin(v * PI_);
    if (st < 1e-6) { return vec4f(0.0); }
    let d = sky_dir((f32(x) + clamp(du, 0.0, 0.9999)) / f32(w), v);
    let p = textureLoad(SKY, vec2i(x, y), 0).a * f32(w) * f32(h) / (2.0 * PI_ * PI_ * st);
    return vec4f(d, p);
}
/* the numbers for the sky's sample: the lamp's pair without a lamp, else plain ones */
fn sky_u(sm: Smp, depth: i32, rng: ptr<function, u32>) -> vec2f {
    if (T.opt.w == NONE) { return s2(sm, 1u + 3u * u32(depth), rng); }
    let a = rnd(rng);
    return vec2f(a, rnd(rng));
}

/* ---- the glowing shapes as lights (rt/tracer.c: emit_sample, emit_pdf) ----
   One of them picked uniformly; a ball by the cone it fills, a box by area. With none
   (T.em.x = 0) nothing here runs and no number is drawn. */
fn em_inst(k: u32) -> u32 { return T.emi[k >> 2u][k & 3u]; }
fn em_slot(inst: u32) -> u32 {
    for (var k = 0u; k < T.em.x; k++) { if (em_inst(k) == inst) { return k; } }
    return NONE;
}
struct Em { ball: bool, c: vec3f, a0: vec3f, a1: vec3f, a2: vec3f, r: f32, area: f32, le: vec3f };
fn emitter(i: u32) -> Em {
    let x = prims[T.ins.y + 2u * i].v;      /* object -> world, by rows */
    let y = prims[T.ins.y + 2u * i].n;
    let z = prims[T.ins.y + 2u * i].w;
    let md = prims[T.ins.y + 2u * i + 1u].c;   /* (its mesh, material, kind) */
    var e: Em;
    e.ball = bitcast<u32>(md.z) == UNIT_SPHERE;
    e.c = vec3f(x.w, y.w, z.w);
    e.a0 = vec3f(x.x, y.x, z.x);
    e.a1 = vec3f(x.y, y.y, z.y);
    e.a2 = vec3f(x.z, y.z, z.z);
    e.le = T.mats[bitcast<u32>(md.y)].emission.rgb;
    e.r = length(e.a0);
    if (e.ball) { e.area = 4.0 * PI_ * e.r * e.r; }
    else { e.area = 8.0 * (length(cross(e.a1, e.a2)) + length(cross(e.a0, e.a2)) + length(cross(e.a0, e.a1))); }
    return e;
}
/* 1 - cos_max of a ball of radius r, its centre dist2 away (squared); 0 from inside */
fn ball_cone(r: f32, dist2: f32) -> f32 {
    let s2v = r * r / dist2;
    if (!(s2v < 1.0)) { return 0.0; }
    return s2v / (1.0 + sqrt(1.0 - s2v));
}
/* the density (per unit solid angle, the pick included) of reaching the shape's point t along
   unit d from p, its outward normal n there; 0: NEE could not have */
fn em_pdf(k: u32, p: vec3f, d: vec3f, t: f32, n: vec3f) -> f32 {
    let e = emitter(em_inst(k));
    let cnt = f32(T.em.x);
    if (e.ball) {
        let om = ball_cone(e.r, dot(e.c - p, e.c - p));
        return select(0.0, 1.0 / (2.0 * PI_ * om * cnt), om > 0.0);
    }
    let cl = -dot(n, d);
    return select(0.0, t * t / (cl * e.area * cnt), cl > 0.0);
}
struct ES { wi: vec3f, dist: f32, pdf: f32, le: vec3f };
/* a point of one of them seen from p (pdf 0: none) */
fn em_sample(p: vec3f, u0: f32, ua: f32, ub: f32) -> ES {
    var r = ES(vec3f(0.0), 0.0, 0.0, vec3f(0.0));
    let k = min(u32(u0 * f32(T.em.x)), T.em.x - 1u);
    let e = emitter(em_inst(k));
    let cnt = f32(T.em.x);
    r.le = e.le;
    if (e.ball) {
        let to = e.c - p;
        let d2 = dot(to, to);
        let om = ball_cone(e.r, d2);
        if (!(om > 0.0)) { return r; }
        let dc = sqrt(d2);
        let ct = 1.0 - ua * om;
        let st = sqrt(max(0.0, 1.0 - ct * ct));
        let ph = 2.0 * PI_ * ub;
        let w = to / dc;
        let tb = onb(w);
        r.wi = w * ct + tb[0] * (st * cos(ph)) + tb[1] * (st * sin(ph));
        r.dist = dc * ct - sqrt(max(0.0, e.r * e.r - d2 * st * st));
        if (!(r.dist > 0.0)) { return r; }
        r.pdf = 1.0 / (2.0 * PI_ * om * cnt);
        return r;
    }
    let ax = array<vec3f, 3>(e.a0, e.a1, e.a2);
    var fa = vec3f(4.0 * length(cross(e.a1, e.a2)), 4.0 * length(cross(e.a2, e.a0)), 4.0 * length(cross(e.a0, e.a1)));
    var x = ua * e.area;
    var f = 0u;
    while (f < 5u && x >= fa[f >> 1u]) { x -= fa[f >> 1u]; f++; }
    let i = f >> 1u;
    let sg = select(1.0, -1.0, (f & 1u) != 0u);
    let aj = ax[(i + 1u) % 3u];
    let ak = ax[(i + 2u) % 3u];
    let uu = min(x / fa[i], 1.0) * 2.0 - 1.0;
    let vv = ub * 2.0 - 1.0;
    let q = e.c + ax[i] * sg + aj * uu + ak * vv;
    var n = normalize(cross(aj, ak)) * sg;
    if (dot(n, ax[i]) * sg < 0.0) { n = -n; }
    let to = q - p;
    let d2 = dot(to, to);
    r.dist = sqrt(d2);
    r.wi = to / r.dist;
    let cl = -dot(n, r.wi);
    if (!(cl > 0.0)) { r.dist = 0.0; return r; }
    r.pdf = d2 / (cl * e.area * cnt);
    return r;
}
/* next event estimation of them at a surface (as direct_surface) */
fn direct_emit(h: Hit, b: ptr<function, Bsdf>, v: vec3f, rng: ptr<function, u32>) -> vec3f {
    let u0 = rnd(rng);
    let ua = rnd(rng);
    let ub = rnd(rng);
    let es = em_sample(h.p, u0, ua, ub);
    if (!(es.pdf > 0.0)) { return vec3f(0.0); }
    let fp = bsdf_eval(b, h.n, v, es.wi);
    if (!(max(fp.x, max(fp.y, fp.z)) > 0.0)) { return vec3f(0.0); }
    let shadow = Ray(h.p, es.wi);
    if (traverse(shadow, T.f.y, es.dist - T.f.z, true).id != NONE) { return vec3f(0.0); }
    var g = abs(dot(h.n, es.wi)) / es.pdf;
    if (T.cnt.w > 0u) { g *= transmittance(shadow, 0.0, es.dist, rng); }
    if (T.opt.y != 0u) { g *= power_heuristic(es.pdf, fp.w); }
    return fp.xyz * es.le * g;
}
/* ... and from a point in a medium (the phase function for the BSDF) */
fn direct_emit_medium(p: vec3f, d: vec3f, gg: f32, rng: ptr<function, u32>) -> vec3f {
    let u0 = rnd(rng);
    let ua = rnd(rng);
    let ub = rnd(rng);
    let es = em_sample(p, u0, ua, ub);
    if (!(es.pdf > 0.0)) { return vec3f(0.0); }
    let shadow = Ray(p, es.wi);
    if (traverse(shadow, T.f.y, es.dist - T.f.z, true).id != NONE) { return vec3f(0.0); }
    let ph = hg(dot(d, es.wi), gg);
    var w = ph * transmittance(shadow, 0.0, es.dist, rng) / es.pdf;
    if (T.opt.y != 0u) { w *= power_heuristic(es.pdf, ph); }
    return es.le * w;
}
/* the weight of a glowing surface's light found by a path (emit: 0 fully, 1 NEE had it, 2
   against NEE): 1 but for a glowing shape NEE could have reached from the last vertex o */
fn emit_weight(f: Found, h: Hit, o: vec3f, d: vec3f, emit: i32, bounce_pdf: f32, sel: f32) -> f32 {
    if (T.em.x == 0u || emit == 0 || f.inst == NONE || !h.front) { return 1.0; }
    let k = em_slot(f.inst);
    if (k == NONE) { return 1.0; }
    let pe = sel * em_pdf(k, o, d, h.t, h.ng);
    if (!(pe > 0.0)) { return 1.0; }
    return select(0.0, power_heuristic(bounce_pdf, pe), emit == 2);
}

/* a material's BSDF (GPU_WGSL_BSDF), seen from the side `outside` says */
fn mat_bsdf(m: TMat, outside: bool) -> Bsdf {
    return bsdf_make(m.albedo.rgb, m.pbr.x, m.pbr.y, m.emission.w, m.pbr.z, outside);
}

/* next event estimation at a surface: one point of the lamp, the BSDF's rough part towards it
   (v: towards the viewer; u: where on the lamp) */
fn direct_surface(h: Hit, b: ptr<function, Bsdf>, v: vec3f, u: vec2f, rng: ptr<function, u32>) -> vec3f {
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
    if (cos_light <= 0.0) { return vec3f(0.0); }
    let fp = bsdf_eval(b, h.n, v, wi);
    if (!(max(fp.x, max(fp.y, fp.z)) > 0.0)) { return vec3f(0.0); }
    let shadow = Ray(h.p, wi);
    if (traverse(shadow, T.f.y, dist - T.f.z, true).id != NONE) { return vec3f(0.0); }
    var g = abs(cos_surf) * cos_light * lt.v.w / dist2;
    if (T.cnt.w > 0u) { g *= transmittance(shadow, 0.0, dist, rng); }
    if (T.opt.y != 0u) { g *= power_heuristic(light_pdf(lt.v.w, dist2, cos_light), fp.w); }
    return fp.xyz * le * g;
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

/* next event estimation towards the sky (rt/tracer.c sample_direct_env) */
fn direct_sky(h: Hit, b: ptr<function, Bsdf>, v: vec3f, u: vec2f, rng: ptr<function, u32>) -> vec3f {
    let sp = sky_sample(u);
    if (!(sp.w > 0.0)) { return vec3f(0.0); }
    let wi = sp.xyz;
    let fp = bsdf_eval(b, h.n, v, wi);
    if (!(max(fp.x, max(fp.y, fp.z)) > 0.0)) { return vec3f(0.0); }
    let le = sky_eval(wi);
    let shadow = Ray(h.p, wi);
    if (traverse(shadow, T.f.y, 1e30, true).id != NONE) { return vec3f(0.0); }
    var g = abs(dot(h.n, wi)) / sp.w;
    if (T.cnt.w > 0u) { g *= transmittance(shadow, 0.0, 1e6, rng); }
    if (T.opt.y != 0u) { g *= power_heuristic(sp.w, fp.w); }
    return fp.xyz * le * g;
}
fn direct_sky_medium(p: vec3f, d: vec3f, g: f32, u: vec2f, rng: ptr<function, u32>) -> vec3f {
    let sp = sky_sample(u);
    if (!(sp.w > 0.0)) { return vec3f(0.0); }
    let wi = sp.xyz;
    let shadow = Ray(p, wi);
    if (traverse(shadow, T.f.y, 1e30, true).id != NONE) { return vec3f(0.0); }
    let tr = transmittance(shadow, 0.0, 1e6, rng);
    if (tr <= 0.0) { return vec3f(0.0); }
    let ph = hg(dot(d, wi), g);
    var w = ph * tr / sp.w;
    if (T.opt.y != 0u) { w *= power_heuristic(sp.w, ph); }
    return sky_eval(wi) * w;
}

fn refract_dir(d: vec3f, n: vec3f, eta: f32) -> vec3f {
    let cos_i = min(dot(-d, n), 1.0);
    let perp = (d + n * cos_i) * eta;
    return perp - n * sqrt(abs(1.0 - dot(perp, perp)));
}

/* Russian roulette after 3 bounces: false = the path stops (u: the number that decides) */
fn roulette_u(thr: ptr<function, vec3f>, depth: i32, u: f32) -> bool {
    if (depth < 3) { return true; }
    let c = *thr;
    let p = min(max(c.x, max(c.y, c.z)), 0.95);
    if (u >= p) { return false; }
    *thr = c / p;
    return true;
}
fn roulette(thr: ptr<function, vec3f>, depth: i32, sm: Smp, rng: ptr<function, u32>) -> bool {
    if (depth < 3) { return true; }
    return roulette_u(thr, depth, s1(sm, 3u + 3u * u32(depth), 0u, rng));
}

/* a surface's bounce (rt/tracer.c): the lobe and the direction from the BSDF, the numbers drawn
   only when used (SAMPLE_MISC's pair for Russian roulette and the lobe, SAMPLE_DIR's for a
   direction that is not fixed); u.x of the result: Russian roulette's number */
struct Bounce { s: BsdfSample, ur: f32 };
fn surface_bounce(b: ptr<function, Bsdf>, h: Hit, v: vec3f, depth: i32, sm: Smp, rng: ptr<function, u32>) -> Bounce {
    bsdf_view(b, dot(v, h.n));
    let dd = 3u * u32(depth);
    var mu = vec2f(0.0);
    if (depth >= 3 || bsdf_nlobes(b) > 1u) { mu = s2(sm, 3u + dd, rng); }
    let lobe = bsdf_pick(b, mu.y);
    var du = vec2f(0.0);
    if (lobe >= 0 && !bsdf_lobe_delta(b, lobe)) { du = s2(sm, 2u + dd, rng); }
    return Bounce(bsdf_sample_lobe(b, lobe, h.n, v, du), mu.x);
}

/* The denoiser's guide for one pixel, along the ray through its centre: the first diffuse
   surface, through mirrors and glass (refracting, unless the glass reflects all), with the
   albedo of the whole way to it. A medium that hides it (transmittance under 1/2, the mean
   of 4 ratio tracking estimates) makes the pixel a medium's. */
struct Guide { n: vec3f, z: f32, a: vec3f, kind: f32, pos: vec3f, pt: f32, inst: u32 };
/* The guide's 4th plane's w: its kind (0 a surface, 1 the sky or a lamp, 2 a medium) plus 4
   times the instance seen + 1 (0: none), so that reprojection keeps a pixel's samples only if it
   still sees the same object (a body moving over another at the same distance does not smear
   its light onto it). Exact in f32 up to 2^22 instances. gkind: the kind alone. */
fn gpack(kind: f32, inst: u32) -> f32 { return kind + 4.0 * f32(select(0u, (inst & 0x3fffffu) + 1u, inst != NONE)); }
fn gkind(w: f32) -> f32 { return w - 4.0 * floor(w * 0.25); }
fn guide(ray0: Ray, rng: ptr<function, u32>) -> Guide {
    var g: Guide;
    g.n = vec3f(0.0); g.z = 1e30; g.a = vec3f(1.0); g.kind = 1.0;
    g.pos = vec3f(0.0); g.pt = 0.0;      /* the camera's first hit, if diffuse: where to look up caustics */
    g.inst = NONE;
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
        var h = make_hit(ray, f);
        let m = shade_tex(T.mats[hit_mat(f)], ray, f, &h);
        let kind = u32(m.albedo.w);
        /* a lamp: kept for reprojection. So is a surface glowing at least as bright as white
           paper under a light of 1 (a lamp's bulb): the denoiser leaves it as it is, and does not
           mix it into its neighbours (thousands of times brighter, it left a halo of speckles
           round it); a dimmer glow (a glTF's emissive details) is denoised with its surface */
        if (kind == 3u || glow_bright(m.emission.rgb)) { g.n = h.n; g.z = dist + h.t; g.inst = f.inst; break; }
        var b = mat_bsdf(m, h.front);
        if (bsdf_rough_part(&b)) {     /* a surface the light scatters on (not only a mirror or glass) */
            g.n = h.n; g.z = dist + h.t; g.a = a * m.albedo.rgb; g.kind = 0.0; g.inst = f.inst;
            if (i == 0) { g.pos = h.p; g.pt = max(tr, 1e-6); }
            break;
        }
        a *= m.albedo.rgb;
        dist += h.t;
        var nxt = reflect(ray.d, h.n);
        if (b.transmission > 0.5 && b.metallic < 0.5) {
            let cos_i = min(dot(-ray.d, h.n), 1.0);
            if (b.eta * b.eta * (1.0 - cos_i * cos_i) <= 1.0) { nxt = refract_dir(ray.d, h.n, b.eta); }
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


/* the camera ray's first hit was a surface glowing at least as bright as a lamp (glow_bright):
   that sample is the pixel's coverage of it (its edge, anti-aliased), not noise of the surface
   the pixel's guide sees, so it is left out of the denoiser's variance (else the denoiser,
   spreading what looks like a firefly, left a halo of speckles round a bulb) */
var<private> cam_glow: bool;
fn glow_bright(e: vec3f) -> bool { return max(e.r, max(e.g, e.b)) >= 1.0; }

/* the radiance arriving along one camera ray */
fn trace(ray0: Ray, sm: Smp, rng: ptr<function, u32>) -> vec3f {
    cam_glow = false;
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
                if (sky_on()) { radiance += thr * direct_sky_medium(p, ray.d, vol.p.x, sky_u(sm, depth, rng), rng); }
                if (T.em.x > 0u) { radiance += thr * direct_emit_medium(p, ray.d, vol.p.x, rng); }
                let nd = sample_phase(ray.d, vol.p.x, s2(sm, 2u + dd, rng));
                bounce_pdf = hg(dot(ray.d, nd), vol.p.x);
                ray = Ray(p, nd);
                emit = after_bounce;
                chain = false;
                if (!roulette(&thr, depth, sm, rng)) { break; }
                continue;
            }
        }
        if (!found) {                     /* escaped: the sky (black without one) */
            if (sky_on() && emit != 1) {
                var le = sky_eval(ray.d);
                if (emit == 2) { le *= power_heuristic(bounce_pdf, sky_pdf(ray.d)); }
                radiance += thr * le;
            }
            break;
        }
        var h = make_hit(ray, f);
        let m = shade_tex(T.mats[hit_mat(f)], ray, f, &h);
        let kind = u32(m.albedo.w);
        /* a glowing surface (glTF's emissive): found by the paths that hit it; a glowing shape
           sampled as a light: weighted against that (emit_weight) */
        if (kind != 3u) {
            radiance += thr * m.emission.rgb * emit_weight(f, h, ray.o, ray.d, emit, bounce_pdf, 1.0);
            if (depth == 0 && glow_bright(m.emission.rgb)) { cam_glow = true; }
        }
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
        /* the surface's BSDF, seen from v (back along the ray): a direction drawn from it,
           weighted f cos / pdf, and light from the lamp (its rough part). (In this order the
           BSDF's state is done with before the shadow ray's walk: fewer registers held.) */
        let v = -ray.d;
        var b = mat_bsdf(m, h.front);
        let bo = surface_bounce(&b, h, v, depth, sm, rng);
        if (bsdf_rough_part(&b)) {
            radiance += thr * direct_surface(h, &b, v, s2(sm, 1u + 3u * u32(depth), rng), rng);
            if (sky_on()) { radiance += thr * direct_sky(h, &b, v, sky_u(sm, depth, rng), rng); }
            if (T.em.x > 0u) { radiance += thr * direct_emit(h, &b, v, rng); }
        }
        if (!bo.s.ok) { break; }
        thr *= bo.s.w;
        if (bo.s.delta) {                 /* a perfect mirror or refraction */
            emit = 0;
            spec = true;
        } else {
            bounce_pdf = bo.s.pdf;
            emit = after_bounce;
            if (depth == 0) { chain = T.ext.z != 0u; }
            spec = false;
        }
        if (!roulette_u(&thr, depth, bo.ur)) { break; }
        ray = Ray(h.p, normalize(bo.s.l));
    }
    return radiance;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_index) li: u32) {
    tlid = li;
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
        let l = luma(select(c, c / max(ga.xyz, vec3f(1e-3)), gkind(ga.w) == 0.0));
        if (!(cam_glow && gkind(ga.w) == 0.0)) { accum[N + k] += vec4f(l, l * l, 0.0, 0.0); }
    }
}

/* A new image's start (with T.f.w = 1): no samples, and each pixel's guide. The whole image
   (dims z and w, the band, are not used). */
@compute @workgroup_size(64)
fn gbuf(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_index) li: u32) {
    tlid = li;
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
    accum[3u * N + k] = vec4f(g.a, gpack(g.kind, g.inst));
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
/* 3 sizes of cells (x1, x4, x16) in one table: the lookup takes the finest that holds enough
   photons (adaptive density estimation). A slot: 3 x (low, high) sums, the photon count, the
   cell's key (0 = free). A cell takes the first slot from its hash on that is free or its own
   (up to 8 on): two cells never share a slot (that would light one with the other's photons). */
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
fn ltrace(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_index) li: u32) {
    tlid = li;
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
        var h = make_hit(ray, f);
        let m = shade_tex(T.mats[hit_mat(f)], ray, f, &h);
        let kind = u32(m.albedo.w);
        if (kind == 3u) { break; }
        var b = mat_bsdf(m, h.front);
        let wi = -ray.d;                                   /* towards where the light came from */
        let cg = abs(dot(h.ng, wi));
        let cs = abs(dot(h.n, wi));
        let corr = select(0.0, cs / cg, cg > 0.0);        /* smooth normals: the adjoint BSDF */
        /* 1. the BSDF's rough part to the camera (not at the first vertex: lamp -> rough surface
              is the path tracer's, by NEE) */
        if (depth > 0 && bsdf_rough_part(&b)) {
            if (T.lt.w > 0.5) {
                deposit(h.p, h.n, beta * corr);            /* photon cache mode: no camera */
            } else {
            let to_cam = T.origin.xyz - h.p;
            let dist2 = dot(to_cam, to_cam);
            let dist = sqrt(dist2);
            let wo = to_cam / dist;
            let cos_o = dot(h.ng, wo);                     /* > 0: the camera on the light's side */
            let q = -to_cam;
            let z = dot(q, fw);
            /* the BSDF seen from the camera (v = wo, l = wi): its side's normal and index */
            var bc = b;
            var nc = h.n;
            if (cos_o < 0.0) { bc = mat_bsdf(m, !h.front); nc = -h.n; }
            if (cos_o != 0.0 && dot(nc, wo) > 0.0 && z > 0.0) {
                let fp = bsdf_eval(&bc, nc, wo, wi);
                let e = q / z - fw;
                let sx = 0.5 + dot(e, T.horizontal.xyz) / dot(T.horizontal.xyz, T.horizontal.xyz);
                let sy = 0.5 - dot(e, T.vertical.xyz) / dot(T.vertical.xyz, T.vertical.xyz);
                if (max(fp.x, max(fp.y, fp.z)) > 0.0 && sx >= 0.0 && sy >= 0.0 && sx < 1.0 && sy < 1.0) {
                    let px = min(u32(sx * f32(W)), W - 1u);
                    let py = min(u32(sy * f32(H)), H - 1u);
                    let sh = Ray(h.p, wo);
                    if (traverse(sh, T.f.y, dist * (1.0 - 1e-4), true).id == NONE) {
                        var tr = 1.0;
                        if (T.cnt.w > 0u) { tr = transmittance(sh, 0.0, dist, &rng); }
                        let cos_c = z / dist;
                        let wgt = corr * abs(cos_o) * tr / (dist2 * T.lt.y * cos_c * cos_c * cos_c);
                        splat(py * W + px, beta * fp.xyz * wgt);
                    }
                }
            }
            }
        }
        /* 2. bounce on, in a direction drawn from the BSDF (seen from the light's side): later
              rough vertices count too. Rough lobes: * |wi.ns| / |wi.ng| * |wo.ng| / |wo.ns|;
              through the surface: * (eta_from / eta_to)^2 */
        let ul = rnd(&rng);
        let u1 = rnd(&rng);
        let bs = bsdf_sample(&b, h.n, wi, ul, vec2f(u1, rnd(&rng)));
        if (!bs.ok) { break; }
        if (depth == 0 && !bs.delta) { break; }            /* lamp -> rough: the path tracer's */
        let og = dot(h.ng, bs.l);
        let os = dot(h.n, bs.l);
        if (!bs.delta && select(og <= 0.0, og >= 0.0, bs.through)) { break; }
        var fb = bs.w;
        if (!bs.delta) { fb *= select(0.0, corr * abs(og / os), os != 0.0); }
        if (bs.through) { fb *= b.eta * b.eta; }
        beta *= fb;
        thr *= fb;
        let nxt = bs.l;
        if (!rr_light(&beta, &thr, depth, &rng)) { break; }
        ray = Ray(h.p, normalize(nxt));
    }
}
