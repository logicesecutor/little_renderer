// rigid_args.wgsl - used by engine/gpu/rigid_gpu.c (K_ARGS)
@group(0) @binding(1) var<storage, read_write> pc: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1)
fn main() {
    for (var l = 0u; l < 3u; l++) {
        let n = min(atomicLoad(&pc[l]), P.m.z);
        args[4u * l] = select((n + 63u) / 64u, min(n, 65535u), l == 1u);   /* hulls: a workgroup each */
        if (l == 1u) {                                                     /* hulls' contacts: a thread each */
            args[20] = (min(n, 65535u) + 63u) / 64u;
            args[21] = 1u;
            args[22] = 1u;
        }
        if (l == 2u) {                                                     /* meshes' trees: the same */
            args[16] = min(n, 65535u);
            args[17] = 1u;
            args[18] = 1u;
        }
        args[4u * l + 1u] = 1u;
        args[4u * l + 2u] = 1u;
    }
}
