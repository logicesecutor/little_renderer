// trace_build.wgsl - used by engine/gpu/trace_gpu.c (BUILD)
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
