// fluid_head.wgsl - used by engine/gpu/fluid_gpu.c (FHEAD)
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
