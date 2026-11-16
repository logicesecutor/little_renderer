// rigid_memo.wgsl - used by engine/gpu/rigid_gpu.c (K_MEMO: the hull pairs' kernels)
@group(0) @binding(6) var<storage, read> G: array<vec4f>;
/* what decided each pair last step (rigid.c's memos, P3): two tables, last step's (read) and
   this step's (written, cleared before), each `cap` keys then `cap` entries of 12 words: the
   feature (type, i, j), the pose of b in a's frame at the full test (t; q as x, y, z, w) */
@group(0) @binding(7) var<storage, read_write> memo: array<atomic<u32>>;
/* [0, cap): the feature whose contacts K_HULLC makes for pair k: type, i, j; w = 1 (0: none);
   [cap, 2 cap): x = the pairs K_HEAVY tests in full (K_HULLPRE queues them, pc[4] counts) */
@group(0) @binding(8) var<storage, read_write> feat: array<vec4u>;
fn qmul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}
fn memo_cap() -> u32 { return 2u * P.m.z; }
fn memo_base(t: u32) -> u32 { return t * 13u * memo_cap(); }
fn memo_hash(key: u32) -> u32 { return (key * 2654435761u) >> 8u; }
/* the entry of key in table t: its index, or NONE */
fn memo_find(t: u32, key: u32) -> u32 {
    let cap = memo_cap();
    let base = memo_base(t);
    var h = memo_hash(key) & (cap - 1u);
    for (var n = 0u; n < 32u; n++) {
        let k = atomicLoad(&memo[base + h]);
        if (k == key) { return h; }
        if (k == 0u) { return NONE; }
        h = (h + 1u) & (cap - 1u);
    }
    return NONE;
}
fn memo_word(t: u32, e: u32, w: u32) -> u32 { return atomicLoad(&memo[memo_base(t) + memo_cap() + e * 12u + w]); }
fn memo_put(t: u32, key: u32, ft: vec3u, tr: vec3f, q: vec4f) {
    let cap = memo_cap();
    let base = memo_base(t);
    var h = memo_hash(key) & (cap - 1u);
    for (var n = 0u; n < 32u; n++) {
        let r = atomicCompareExchangeWeak(&memo[base + h], 0u, key);
        if (r.exchanged || r.old_value == key) {
            let d = base + cap + h * 12u;
            atomicStore(&memo[d], ft.x); atomicStore(&memo[d + 1u], ft.y); atomicStore(&memo[d + 2u], ft.z);
            atomicStore(&memo[d + 4u], bitcast<u32>(tr.x)); atomicStore(&memo[d + 5u], bitcast<u32>(tr.y));
            atomicStore(&memo[d + 6u], bitcast<u32>(tr.z));
            atomicStore(&memo[d + 8u], bitcast<u32>(q.x)); atomicStore(&memo[d + 9u], bitcast<u32>(q.y));
            atomicStore(&memo[d + 10u], bitcast<u32>(q.z)); atomicStore(&memo[d + 11u], bitcast<u32>(q.w));
            return;
        }
        h = (h + 1u) & (cap - 1u);
    }
}
/* b's pose in a's frame (t; q) */
fn rel_pose(ia: u32, ib: u32, t: ptr<function, vec3f>, q: ptr<function, vec4f>) {
    let qa = B[ia].rot;
    *q = qmul(vec4f(-qa.xyz, qa.w), B[ib].rot);
    *t = transpose(qmat(qa)) * (B[ib].pos.xyz - B[ia].pos.xyz);
}
fn pair_key(ia: u32, ib: u32) -> u32 { return ((ia << 16u) | ib) + 1u; }
