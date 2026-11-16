// rigid_meshq.wgsl - used by engine/gpu/rigid_gpu.c (K_MESHQ)
@group(0) @binding(6) var<storage, read> G: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> T: array<vec2u>;
fn gq(i: u32, c: u32) -> u32 { return bitcast<u32>(G[i][c]); }
/* one workgroup per pair: the tree is walked level by level, each level's nodes shared by the
   64 threads (a front of at most 1024 nodes; a thread that cannot add to it walks on alone) */
var<workgroup> front: array<u32, 2048>;
var<workgroup> nnext: atomic<u32>;
var<workgroup> wn: u32;
var<workgroup> wcount: u32;
var<workgroup> wlo: vec3f;
var<workgroup> whi: vec3f;

fn emit(k: u32, first: u32, count: u32) {
    for (var q = 0u; q < count; q++) {
        let e = atomicAdd(&pc[3], 1u);
        if (e < 64u * P.m.z) { T[e] = vec2u(k, first + q); }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lv: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let lid = lv.x;
    let k = wid.x + wid.y * nw.x;
    if (lid == 0u) { wcount = min(atomicLoad(&pc[2]), P.m.z); }
    let count = workgroupUniformLoad(&wcount);
    if (k >= count) { return; }
    let pr = pairs[2u * P.m.z + k];
    let a = B[pr.x];
    let m = B[pr.y];
    let gi = bitcast<u32>(m.t.x);
    let node0 = gq(2u * gi + 1u, 0u);
    let rt = transpose(qmat(m.rot));
    let mg = pair_margin(a, m, false);
    if (lid == 0u) {
        /* the body's box in the mesh's frame: a ball's, or around a hull's (box's) corners */
        var lo = vec3f(1e30);
        var hi = vec3f(-1e30);
        if (a.pos.w < 0.5) {
            let c = rt * (a.pos.xyz - m.pos.xyz);
            lo = c - vec3f(a.half.x);
            hi = c + vec3f(a.half.x);
        } else {
            let A = shape_of(pr.x);
            for (var v = 0u; v < A.nv; v++) {
                let l = rt * (sv(A, v) - m.pos.xyz);
                lo = min(lo, l);
                hi = max(hi, l);
            }
        }
        wlo = lo - vec3f(mg);
        whi = hi + vec3f(mg);
        front[0] = 0u;
        wn = 1u;
        atomicStore(&nnext, 0u);
    }
    var n = workgroupUniformLoad(&wn);
    let lo = wlo;
    let hi = whi;
    var par = 0u;
    for (var level = 0u; level < 64u; level++) {
        if (n == 0u) { break; }
        for (var idx = lid; idx < n; idx += 64u) {
            let nd = node0 + 3u * front[par * 1024u + idx];
            if (any(G[nd].xyz > hi) || any(G[nd + 1u].xyz < lo)) { continue; }
            let cnt = gq(nd + 2u, 1u);
            if (cnt > 0u) { emit(k, gq(nd + 2u, 0u), cnt); continue; }
            let slot = atomicAdd(&nnext, 2u);
            if (slot + 2u <= 1024u) {
                front[(1u - par) * 1024u + slot] = gq(nd, 3u);
                front[(1u - par) * 1024u + slot + 1u] = gq(nd + 1u, 3u);
                continue;
            }
            /* the front is full: this subtree alone */
            var stack: array<u32, 48>;
            var sp = 2u;
            stack[0] = gq(nd, 3u);
            stack[1] = gq(nd + 1u, 3u);
            while (sp > 0u) {
                sp--;
                let nd2 = node0 + 3u * stack[sp];
                if (any(G[nd2].xyz > hi) || any(G[nd2 + 1u].xyz < lo)) { continue; }
                let c2 = gq(nd2 + 2u, 1u);
                if (c2 > 0u) { emit(k, gq(nd2 + 2u, 0u), c2); continue; }
                if (sp + 2u <= 48u) {
                    stack[sp] = gq(nd2, 3u);
                    sp++;
                    stack[sp] = gq(nd2 + 1u, 3u);
                    sp++;
                }
            }
        }
        workgroupBarrier();
        if (lid == 0u) { wn = min(atomicLoad(&nnext), 1024u); atomicStore(&nnext, 0u); }
        n = workgroupUniformLoad(&wn);
        par = 1u - par;
    }
}
