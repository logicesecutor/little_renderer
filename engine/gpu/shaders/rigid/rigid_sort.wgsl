// rigid_sort.wgsl - used by engine/gpu/rigid_gpu.c (K_SORT)
@group(0) @binding(1) var<storage, read> B: array<Body>;
@group(0) @binding(2) var<storage, read> C: array<Contact>;
@group(0) @binding(3) var<storage, read> off: array<u32>;
@group(0) @binding(4) var<storage, read_write> L: array<u32>;

/* longer lists are left as they are (a body on thousands of contacts: rare, and slow to sort
   in one thread) */
const SORT_MAX = 256u;

/* what an entry of body i's list is, whatever the order the contacts were found in: the other
   body (NONE: the room), then the contact's point on body i in its own frame (la or lb, as
   the narrow phase set them), then its normal */
struct Key { a: vec4u, b: vec3u };
fn key(e: u32) -> Key {
    let c = C[e >> 1u];
    let second = (e & 1u) != 0u;
    let other = select(bitcast<u32>(c.ra.w), bitcast<u32>(c.n.w), second);
    let q = bitcast<vec3u>(select(c.la.xyz, c.lb.xyz, second));
    return Key(vec4u(other, q), bitcast<vec3u>(c.n.xyz));
}
fn lt4(a: vec4u, b: vec4u) -> i32 {      /* -1: a first, 1: b first, 0: the same */
    if (a.x != b.x) { return select(1, -1, a.x < b.x); }
    if (a.y != b.y) { return select(1, -1, a.y < b.y); }
    if (a.z != b.z) { return select(1, -1, a.z < b.z); }
    if (a.w != b.w) { return select(1, -1, a.w < b.w); }
    return 0;
}
fn less(a: Key, b: Key) -> bool {
    let k = lt4(a.a, b.a);
    if (k != 0) { return k < 0; }
    return lt4(vec4u(a.b, 0u), vec4u(b.b, 0u)) < 0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i >= P.n.x) { return; }      /* (a fixed body's too: K_PREP looks up last step's contacts in it) */
    let lo = off[i];
    let hi = off[i + 1u];
    if (hi - lo < 2u || hi - lo > SORT_MAX) { return; }
    for (var q = lo + 1u; q < hi; q++) {                  /* insertion sort: a few to a few dozen */
        let e = L[q];
        let k = key(e);
        var r = q;
        while (r > lo) {
            let f = L[r - 1u];
            if (!less(k, key(f))) { break; }
            L[r] = f;
            r--;
        }
        L[r] = e;
    }
}
