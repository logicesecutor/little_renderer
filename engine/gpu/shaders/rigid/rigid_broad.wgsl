// rigid_broad.wgsl - used by engine/gpu/rigid_gpu.c (K_BROAD)
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read_write> pairs: array<vec2u>;
@group(0) @binding(3) var<storage, read_write> pc: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> wake: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> F: array<vec4f>;
@group(0) @binding(6) var<storage, read> grid: array<u32>;
/* list 0: balls and boxes, 1: hulls, 2: a body and a fixed mesh */
fn push(l: u32, i: u32, j: u32) {
    let k = atomicAdd(&pc[l], 1u);
    if (k < P.m.z) { pairs[l * P.m.z + k] = vec2u(i, j); }
}
fn asleep(b: Body) -> bool { return b.f.y > 0.5; }
/* (a body jittering in place, quiet after the last step (K_SLEEP, f.w), wakes nobody) */
fn moving(b: Body) -> bool {
    return b.vel.w > 0.0 && b.f.w < 0.5 && (length(b.vel.xyz) >= P.h2.z || length(b.ang.xyz) * body_radius(b) >= P.h2.z);
}
/* bodies i and j (i < j) close enough: WAKE, which sleeping one wakes (a moving body near it,
   rigid.c's wake_bodies); else the pair, into its list */
fn meet(i: u32, j: u32) {
    let a = B[i];
    let b = B[j];
    let d = a.pos.xyz - b.pos.xyz;
    let reach = body_radius(a) + body_radius(b) + pair_margin(a, b, false);
    if (dot(d, d) > reach * reach) { return; }
    if (WAKE) {
        if (asleep(a) && moving(b)) { atomicStore(&wake[i], 1u); }
        else if (asleep(b) && moving(a)) { atomicStore(&wake[j], 1u); }
        return;
    }
    if (a.vel.w == 0.0 && b.vel.w == 0.0) { return; }
    if (a.pos.w > 2.5 && b.pos.w > 2.5) { return; }
    if (a.pos.w > 2.5) { push(2u, j, i); }
    else if (b.pos.w > 2.5) { push(2u, i, j); }
    else { push(select(0u, 1u, a.pos.w > 1.5 || b.pos.w > 1.5), i, j); }
}
fn cell_hash(e: vec3i, H: u32) -> u32 { return ((u32(e.x) * 73856093u) ^ (u32(e.y) * 19349663u) ^ (u32(e.z) * 83492791u)) & (H - 1u); }
fn around(c: vec3i, k: u32) -> vec3i { return c + vec3i(i32(k % 3u) - 1, i32((k / 3u) % 3u) - 1, i32(k / 9u) - 1); }
/* 27 threads per body, one per cell around it (K_GRID; a thread per body went through all
   the bodies near it alone: 1500 big hulls in the room, 10 times slower than a thread per
   pair). The first: its walls (or the air's force on it, WAKE), and the big ones (a big one:
   the big ones after it); each: the bodies in its cell after it in the list (a cell whose
   slot an earlier one shares: none, they were looked at) */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let t = tid(gid, nw);
    let n = P.n.x;
    let i = t / 27u;
    let k = t % 27u;
    if (i >= n) { return; }
    let a = B[i];
    let H = u32(P.q.z);
    let big = body_radius(a) > P.q.y;
    if (k == 0u) {
        if (WAKE) {
            if (asleep(a) && length(F[2u * i].xyz - a.t.yzw) > 0.25 * length(P.g.xyz) / a.f.z) { atomicStore(&wake[i], 1u); }
        } else if (P.n.w != 0u && a.vel.w > 0.0) { push(select(0u, 1u, a.pos.w > 1.5), i, NONE); }
        let nbig = grid[H + 1u];
        for (var q = 0u; q < nbig; q++) {
            let j = grid[H + 2u + q];
            if (j != i && (!big || j > i)) { meet(min(i, j), max(i, j)); }
        }
    }
    if (big) { return; }
    let c = vec3i(floor(a.pos.xyz / P.q.x));
    let h = cell_hash(around(c, k), H);
    for (var q = 0u; q < k; q++) { if (cell_hash(around(c, q), H) == h) { return; } }
    let lists = H + 2u + n;
    for (var q = grid[h]; q < grid[h + 1u]; q++) {
        let j = grid[lists + q];
        if (j > i) { meet(i, j); }
    }
}
