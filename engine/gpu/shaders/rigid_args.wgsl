// rigid_args.wgsl - used by engine/gpu/rigid_gpu.c (K_ARGS)
@group(0) @binding(1) var<storage, read_write> pc: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1)
fn main() {
    for (var l = 0u; l < 3u; l++) {
        let n = min(atomicLoad(&pc[l]), P.m.z);
        args[4u * l] = (n + 63u) / 64u;
        if (l == 2u) {                       /* the meshes' trees: a workgroup per pair */
            args[16] = min(n, 65535u);
            args[17] = 1u;
            args[18] = 1u;
        }
        args[4u * l + 1u] = 1u;
        args[4u * l + 2u] = 1u;
    }
}
