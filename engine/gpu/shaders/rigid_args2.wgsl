// rigid_args2.wgsl - used by engine/gpu/rigid_gpu.c (K_ARGS2)
@group(0) @binding(1) var<storage, read_write> pc: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1)
fn main() {
    let n = min(atomicLoad(&pc[3]), 64u * P.m.z);
    args[12] = (n + 63u) / 64u;
    args[13] = 1u;
    args[14] = 1u;
}
