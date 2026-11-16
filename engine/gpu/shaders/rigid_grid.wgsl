// rigid_grid.wgsl - used by engine/gpu/rigid_gpu.c (K_GRID)
/* The bodies by where they are, so that each finds its neighbours without looking at all the
   others (K_BROAD; it was one thread per pair of bodies: 8000 bodies, 64 million threads, 3.5
   ms a step for the pairs and the wake). Cells of P.q.x (three times the radius of the bodies
   smaller than P.q.y), hashed into H = P.q.z slots; a body bigger than that (a fixed mesh, a
   boulder among pebbles) goes on a list of its own that every body looks through.
   grid: [0, H] the slots' counts, then (GRID_SCAN) where each slot's bodies start, [H] all
   of them; [H + 1] the big ones' count, [H + 2, + n) their list; then the slots' bodies (n),
   then each body's slot and its place in it (2 n). Three passes (GRID_STEP): count, scan
   (one workgroup: each thread a run of H / 64 slots), place. */
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> grid: array<atomic<u32>>;
fn grid_h() -> u32 { return u32(P.q.z); }
fn grid_big(i: u32) -> bool { return body_radius(B[i]) > P.q.y; }
fn grid_cell(p: vec3f) -> vec3i { return vec3i(floor(p / P.q.x)); }
fn grid_hash(c: vec3i) -> u32 {   /* (K_BROAD's cell_hash) */
    return (u32(c.x) * 73856093u ^ u32(c.y) * 19349663u ^ u32(c.z) * 83492791u) & (grid_h() - 1u);
}
var<workgroup> sums: array<u32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lv: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let n = P.n.x;
    let H = grid_h();
    let lists = H + 2u + n;
    if (GRID_STEP == 1u) {                  /* the counts into starts */
        let t = lv.x;
        let run = H / 64u;
        var s = 0u;
        for (var k = 0u; k < run; k++) { s += atomicLoad(&grid[t * run + k]); }
        sums[t] = s;
        workgroupBarrier();
        for (var d = 1u; d < 64u; d = d << 1u) {
            var v = 0u;
            if (t >= d) { v = sums[t - d]; }
            workgroupBarrier();
            sums[t] += v;
            workgroupBarrier();
        }
        var at = sums[t] - s;
        for (var k = 0u; k < run; k++) {
            let c = atomicLoad(&grid[t * run + k]);
            atomicStore(&grid[t * run + k], at);
            at += c;
        }
        if (t == 63u) { atomicStore(&grid[H], sums[63]); }
        return;
    }
    let i = tid(gid, nw);
    if (i >= n) { return; }
    if (GRID_STEP == 0u) {
        if (grid_big(i)) { atomicStore(&grid[H + 2u + atomicAdd(&grid[H + 1u], 1u)], i); return; }
        let h = grid_hash(grid_cell(B[i].pos.xyz));
        atomicStore(&grid[lists + n + 2u * i], h);
        atomicStore(&grid[lists + n + 2u * i + 1u], atomicAdd(&grid[h], 1u));
        return;
    }
    if (grid_big(i)) { return; }
    let h = atomicLoad(&grid[lists + n + 2u * i]);
    atomicStore(&grid[lists + atomicLoad(&grid[h]) + atomicLoad(&grid[lists + n + 2u * i + 1u])], i);
}
