// trace_resolve.wgsl - used by engine/gpu/trace_gpu.c (RESOLVE)
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
