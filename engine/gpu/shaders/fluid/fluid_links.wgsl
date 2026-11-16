// fluid_links.wgsl - used by engine/gpu/fluid_gpu.c (K_LINKS)
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
