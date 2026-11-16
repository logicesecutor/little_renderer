// trace_ploc.wgsl - used by engine/gpu/trace_gpu.c (PLOC)
struct Node { lo: vec3f, a: u32, hi: vec3f, b: u32 };
struct BP { n: vec4u, m: vec4u, lo: vec4f, ext: vec4f };
@group(0) @binding(0) var<uniform> U: BP;
@group(0) @binding(1) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(2) var<storage, read_write> pl: array<u32>;
@group(0) @binding(3) var<storage, read_write> args: array<u32>;

const R = 16u;
const WG = 256u;
fn par() -> u32 { return U.m.x & 1u; }
fn cl(p: u32, i: u32) -> u32 { return 16u + p * U.n.x + i; }
fn nn_at(i: u32) -> u32 { return 16u + 2u * U.n.x + i; }
fn sc_at(i: u32) -> u32 { return 16u + 3u * U.n.x + i; }
fn nblocks() -> u32 { return (U.n.x + WG - 1u) / WG; }
fn bk_at(b: u32) -> u32 { return 16u + 4u * U.n.x + b; }
fn bm_at(b: u32) -> u32 { return 16u + 4u * U.n.x + nblocks() + b; }
fn area(lo: vec3f, hi: vec3f) -> f32 { let e = hi - lo; return e.x * e.y + e.y * e.z + e.z * e.x; }
fn bx_at(x: u32) -> u32 { return 16u + 4u * U.n.x + 2u * nblocks() + 8u * x; }
fn bx_lo(x: u32) -> vec3f { let o = bx_at(x); return bitcast<vec3f>(vec3u(pl[o], pl[o + 1u], pl[o + 2u])); }
fn bx_hi(x: u32) -> vec3f { let o = bx_at(x) + 4u; return bitcast<vec3f>(vec3u(pl[o], pl[o + 1u], pl[o + 2u])); }
fn put_bx(x: u32, lo: vec3f, hi: vec3f) {
    let o = bx_at(x);
    let l = bitcast<vec3u>(lo);
    let h = bitcast<vec3u>(hi);
    pl[o] = l.x; pl[o + 1u] = l.y; pl[o + 2u] = l.z;
    pl[o + 4u] = h.x; pl[o + 5u] = h.y; pl[o + 6u] = h.z;
}
fn leaf_in(i: u32) {
    let x = U.n.x - 1u + i;
    let nd = nodes[U.n.z + x];
    put_bx(x, nd.lo, nd.hi);
}
fn final_node(x: u32) -> Node {
    let n = U.n.x;
    if (x < n - 1u) { return Node(bx_lo(x), U.n.z + 1u + 2u * x, bx_hi(x), U.n.z + 2u + 2u * x); }
    return Node(bx_lo(x), U.n.w + x - (n - 1u), bx_hi(x), 0x80000001u);
}
fn merge(k: u32, c: u32, d: u32) {
    let lo = min(bx_lo(c), bx_lo(d));
    let hi = max(bx_hi(c), bx_hi(d));
    put_bx(k, lo, hi);
    nodes[U.n.z + 1u + 2u * k] = final_node(c);
    nodes[U.n.z + 2u + 2u * k] = final_node(d);
    if (k == 0u) { nodes[U.n.z] = Node(lo, U.n.z + 1u, hi, U.n.z + 2u); }
}
fn live(nc: u32) -> u32 { return select(nc, 0u, nc <= U.m.w); }

fn set_args(slot: u32, nc: u32) {
    var g = (nc + WG - 1u) / WG;
    if (nc <= U.m.w) { g = 0u; }
    let gx = min(g, 65535u);
    var gy = 1u;
    if (gx > 0u) { gy = (g + gx - 1u) / gx; }
    args[slot * 4u] = gx;
    args[slot * 4u + 1u] = gy;
    args[slot * 4u + 2u] = 1u;
}

fn nearer(d: f32, i: u32, j: u32, bd: f32, bj: u32) -> bool {
    if (bj == 0xffffffffu || d < bd) { return true; }
    if (d > bd) { return false; }
    let mn = min(i, j);
    let bmn = min(i, bj);
    return mn < bmn || (mn == bmn && max(i, j) < max(i, bj));
}

@compute @workgroup_size(256)
fn init(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = gid.x + gid.y * nw.x * WG;
    let n = U.n.x;
    if (i >= n) { return; }
    leaf_in(i);
    pl[cl(0u, i)] = n - 1u + i;
    if (i == 0u) {
        pl[0] = n;
        pl[1] = 0u;
        pl[2] = 0u;
        pl[3] = n;
        set_args(0u, n);
    }
}

var<workgroup> slo: array<vec3f, 288>;
var<workgroup> shi: array<vec3f, 288>;

// the boxes of clusters base - R .. base + 256 + R of list p into slo, shi
fn boxes_in(p: u32, base: u32, nc: u32, lid: u32) {
    for (var l = lid; l < WG + 2u * R; l += WG) {
        let g = i32(base + l) - i32(R);
        if (g >= 0 && g < i32(nc)) {
            let x = pl[cl(p, u32(g))];
            slo[l] = bx_lo(x);
            shi[l] = bx_hi(x);
        }
    }
}

// cluster i's nearest (i = base + lid, its neighbours' boxes in slo, shi)
fn nearest(i: u32, base: u32, nc: u32, lid: u32) -> u32 {
    let lo = slo[lid + R];
    let hi = shi[lid + R];
    let j0 = u32(max(i32(i) - i32(R), 0));
    let j1 = min(i + R, nc - 1u);
    var bd = 0.0f;
    var bj = 0xffffffffu;
    for (var j = j0; j <= j1; j++) {
        if (j == i) { continue; }
        let l = j + R - base;
        let d = area(min(lo, slo[l]), max(hi, shi[l]));
        if (nearer(d, i, j, bd, bj)) {
            bd = d;
            bj = j;
        }
    }
    return bj;
}

@compute @workgroup_size(256)
fn nn(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32, @builtin(num_workgroups) nw: vec3u) {
    let p = par();
    let nc = live(pl[p]);
    let base = (wid.x + wid.y * nw.x) * WG;
    boxes_in(p, base, nc, lid);
    workgroupBarrier();
    let i = base + lid;
    if (i < nc) { pl[nn_at(i)] = nearest(i, base, nc, lid); }
}

fn keep_merge(i: u32, nc: u32) -> vec2u {
    let j = pl[nn_at(i)];
    var mu = false;
    if (j < nc) { mu = pl[nn_at(j)] == i; }
    return vec2u(select(1u, 0u, mu && i > j), select(0u, 1u, mu && i < j));
}

var<workgroup> sk: array<u32, 256>;
var<workgroup> sm: array<u32, 256>;

fn scan2(lid: u32) {
    for (var o = 1u; o < WG; o <<= 1u) {
        var xk = 0u;
        var xm = 0u;
        if (lid >= o) {
            xk = sk[lid - o];
            xm = sm[lid - o];
        }
        workgroupBarrier();
        sk[lid] += xk;
        sm[lid] += xm;
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn mark(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32, @builtin(num_workgroups) nw: vec3u) {
    let nc = live(pl[par()]);
    let blk = wid.x + wid.y * nw.x;
    let i = blk * WG + lid;
    var km = vec2u(0u);
    if (i < nc) { km = keep_merge(i, nc); }
    sk[lid] = km.x;
    sm[lid] = km.y;
    workgroupBarrier();
    scan2(lid);
    if (i < nc) { pl[sc_at(i)] = (sk[lid] - km.x) | ((sm[lid] - km.y) << 9u) | (km.x << 18u) | (km.y << 19u); }
    if (lid == WG - 1u && blk * WG < nc) {
        pl[bk_at(blk)] = sk[lid];
        pl[bm_at(blk)] = sm[lid];
    }
}

@compute @workgroup_size(256)
fn top(@builtin(local_invocation_index) lid: u32) {
    let p = par();
    let nc = pl[p];
    let nbc = (live(nc) + WG - 1u) / WG;
    var ck = 0u;
    var cm = 0u;
    let chunks = (nblocks() + WG - 1u) / WG;
    for (var c = 0u; c < chunks; c++) {
        let b = c * WG + lid;
        var vk = 0u;
        var vm = 0u;
        if (b < nbc) {
            vk = pl[bk_at(b)];
            vm = pl[bm_at(b)];
        }
        sk[lid] = vk;
        sm[lid] = vm;
        workgroupBarrier();
        scan2(lid);
        if (b < nbc) {
            pl[bk_at(b)] = ck + sk[lid] - vk;
            pl[bm_at(b)] = cm + sm[lid] - vm;
        }
        ck += sk[WG - 1u];
        cm += sm[WG - 1u];
        workgroupBarrier();
    }
    if (lid == 0u) {
        let q = 1u - p;
        var next = nc;
        if (nc > U.m.w) {
            next = ck;
            pl[2] = q;
            pl[3] = ck;
        }
        pl[q] = next;
        set_args(q, next);
    }
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let p = par();
    let nc = live(pl[p]);
    let i = gid.x + gid.y * nw.x * WG;
    if (i >= nc) { return; }
    let s = pl[sc_at(i)];
    if (((s >> 18u) & 1u) == 0u) { return; }
    let blk = i / WG;
    let dest = pl[bk_at(blk)] + (s & 0x1ffu);
    let c = pl[cl(p, i)];
    var out = c;
    if (((s >> 19u) & 1u) != 0u) {
        let n = U.n.x;
        let d = pl[cl(p, pl[nn_at(i)])];
        let k = (n - 2u) - (n - nc + pl[bm_at(blk)] + ((s >> 9u) & 0x1ffu));
        merge(k, c, d);
        out = k;
    }
    pl[cl(1u - p, dest)] = out;
}

var<workgroup> wnc: u32;

@compute @workgroup_size(256)
fn small(@builtin(local_invocation_index) lid: u32) {
    let n = U.n.x;
    var p = 0u;
    if (U.m.y == 0u) {
        for (var i = lid; i < n; i += WG) {
            leaf_in(i);
            pl[cl(0u, i)] = n - 1u + i;
        }
        if (lid == 0u) { wnc = n; }
    } else {
        p = pl[2];
        if (lid == 0u) { wnc = pl[3]; }
    }
    storageBarrier();
    loop {
        let nc = workgroupUniformLoad(&wnc);
        if (nc <= 1u) { break; }
        for (var c = 0u; c < (nc + WG - 1u) / WG; c++) {
            workgroupBarrier();
            boxes_in(p, c * WG, nc, lid);
            workgroupBarrier();
            let i = c * WG + lid;
            if (i < nc) { pl[nn_at(i)] = nearest(i, c * WG, nc, lid); }
        }
        storageBarrier();
        let per = (nc + WG - 1u) / WG;
        let s0 = min(lid * per, nc);
        let s1 = min(s0 + per, nc);
        var ck = 0u;
        var cm = 0u;
        for (var i = s0; i < s1; i++) {
            let km = keep_merge(i, nc);
            ck += km.x;
            cm += km.y;
        }
        sk[lid] = ck;
        sm[lid] = cm;
        workgroupBarrier();
        scan2(lid);
        var dk = sk[lid] - ck;
        var dm = sm[lid] - cm;
        let total = sk[WG - 1u];
        let q = 1u - p;
        for (var i = s0; i < s1; i++) {
            let km = keep_merge(i, nc);
            if (km.x == 0u) { continue; }
            let c = pl[cl(p, i)];
            var out = c;
            if (km.y == 1u) {
                let d = pl[cl(p, pl[nn_at(i)])];
                let k = (n - 2u) - (n - nc + dm);
                dm++;
                merge(k, c, d);
                out = k;
            }
            pl[cl(q, dk)] = out;
            dk++;
        }
        storageBarrier();
        if (lid == 0u) { wnc = total; }
        p = q;
    }
}
