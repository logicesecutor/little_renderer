// rigid_args3.wgsl - used by engine/gpu/rigid_gpu.c (K_ARGS3)
@group(0) @binding(1) var<storage, read_write> pc: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> args: array<u32>;
/* K_HEAVY's queue (K_HULLPRE's pc[4]): a workgroup per pair, at args[20] */
@compute @workgroup_size(1)
fn main() {
    args[20] = min(min(atomicLoad(&pc[4]), P.m.z), 65535u);
    args[21] = 1u;
    args[22] = 1u;
}
