// rigid_acolor.wgsl - used by engine/gpu/rigid_gpu.c (K_ACOLOR; PICK: a round's first half)
@group(0) @binding(1) var<storage, read> C: array<Contact>;
@group(0) @binding(2) var<storage, read> off: array<u32>;
@group(0) @binding(3) var<storage, read> L: array<u32>;
@group(0) @binding(4) var<storage, read_write> COL: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> TENT: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> CTL: array<atomic<u32>>;
fn prio(i: u32) -> u32 {          /* (PCG hash) */
    let s = i * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn first(o: u32, i: u32) -> bool {  /* o comes before i */
    let po = prio(o);
    let pi = prio(i);
    return po > pi || (po == pi && o > i);
}
/* A round of speculative colouring: each uncoloured body picks (PICK) the smallest colour
   none of its coloured neighbours has, nor its uncoloured neighbours that come first picked
   in the round before; then keeps it unless a neighbour that comes first picked the same.
   (The checks read only picks, TENT, which the second half does not change: no races; a
   body's pick is its colour once kept.) */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = tid(gid, nw);
    if (i >= P.n.x || atomicLoad(&COL[i]) != UNCOLOURED) { return; }
    let mine = atomicLoad(&TENT[i]);
    var used = 0u;
    for (var q = off[i]; q < off[i + 1u]; q++) {
        let e = L[q];
        let k = C[e >> 1u];
        let o = select(bitcast<u32>(k.ra.w), bitcast<u32>(k.n.w), (e & 1u) == 1u);
        if (o == NONE) { continue; }
        let oc = atomicLoad(&COL[o]);
        if (oc == FIXED) { continue; }
        let ot = atomicLoad(&TENT[o]);
        if (PICK) {
            if (oc != UNCOLOURED) { used |= 1u << min(oc, 31u); }
            else if (ot < 32u && first(o, i)) { used |= 1u << ot; }
        } else if (ot == mine && first(o, i)) { return; }
    }
    if (PICK) {
        atomicStore(&TENT[i], min(firstTrailingBit(~used), 31u));
    } else {
        atomicStore(&COL[i], mine);
        atomicMax(&CTL[2], mine + 1u);
    }
}
