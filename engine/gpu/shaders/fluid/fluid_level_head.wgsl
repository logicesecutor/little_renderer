// fluid_level_head.wgsl - used by engine/gpu/fluid_gpu.c (LHEAD)
struct LU {
    d: vec4u,    /* this level: nx, ny, nz, open faces */
    c: vec4u,    /* the next coarser level: nx, ny, nz; w = the color of a red-black sweep */
    h: vec4f,    /* h^2 */
};
@group(0) @binding(0) var<uniform> L: LU;
fn lcount() -> u32 { return L.d.x * L.d.y * L.d.z; }
fn ccount() -> u32 { return L.c.x * L.c.y * L.c.z; }
fn lcell(id: u32) -> vec3i { let i = i32(id); let n = vec3i(L.d.xyz); return vec3i(i % n.x, (i / n.x) % n.y, i / (n.x * n.y)); }
fn ccell(id: u32) -> vec3i { let i = i32(id); let n = vec3i(L.c.xyz); return vec3i(i % n.x, (i / n.x) % n.y, i / (n.x * n.y)); }
fn lidx(q: vec3i) -> u32 { return u32((q.z * i32(L.d.y) + q.y) * i32(L.d.x) + q.x); }
fn cidx(q: vec3i) -> u32 { return u32((q.z * i32(L.c.y) + q.y) * i32(L.c.x) + q.x); }
