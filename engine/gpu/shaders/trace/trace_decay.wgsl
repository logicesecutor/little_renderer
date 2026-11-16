// trace_decay.wgsl - used by engine/gpu/trace_gpu.c (DECAY)
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
