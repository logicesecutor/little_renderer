// trace_decay.wgsl - used by engine/gpu/trace_gpu.c (DECAY)
/* The photon cache fades (trace.wgsl pc_claim, deposit): one thread per slot (8 words: 3 sums
   of 2 words, the photon count, the cell's key). Its sums and its count times D.k.x. A slot
   whose count has faded to nothing (under 1/256 of a photon) is given back, key and all: a
   cell keeps its slot only while it holds light. (The keys stayed for good: what moves leaves
   a trail of cells, a ball rolling fills its neighbourhood of the table with dead ones in half
   a minute, and new cells there found no slot: their light was dropped.) Only when the next
   slot is free: a cell that took that one because this one was taken is found from its hash
   through this one, and must go on being found (pc_claim and the resolve stop at a free slot),
   so a run of dead cells is given back from its end, a slot a frame. */
struct DC { k: vec4f, n: vec4u };   /* n.x: the slots (a power of 2) */
@group(0) @binding(0) var<uniform> D: DC;
@group(0) @binding(1) var<storage, read_write> grid: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let s = gid.x + gid.y * nw.x * 64u;
    if (s >= D.n.x) { return; }
    let at = s * 8u;
    if (atomicLoad(&grid[at + 7u]) == 0u) { return; }           /* free */
    for (var ch = 0u; ch < 3u; ch++) {
        let hi = atomicLoad(&grid[at + 2u * ch + 1u]);
        let lo = atomicLoad(&grid[at + 2u * ch]);
        if (hi == 0u && lo == 0u) { continue; }
        let v = (f32(hi) * 4294967296.0 + f32(lo)) * D.k.x;
        let h = floor(v / 4294967296.0);
        atomicStore(&grid[at + 2u * ch + 1u], u32(h));
        atomicStore(&grid[at + 2u * ch], u32(min(v - h * 4294967296.0, 4294967295.0)));
    }
    let count = u32(f32(atomicLoad(&grid[at + 6u])) * D.k.x);
    atomicStore(&grid[at + 6u], count);
    /* (the next slot's key through an atomic: its own thread may be giving it back now; either
       answer will do, this slot's turn then comes next frame) */
    if (count == 0u && atomicLoad(&grid[((s + 1u) & (D.n.x - 1u)) * 8u + 7u]) == 0u) {
        for (var k = 0u; k < 8u; k++) { atomicStore(&grid[at + k], 0u); }
    }
}
