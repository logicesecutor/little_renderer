// fluid_solid.wgsl - used by engine/gpu/fluid_gpu.c (SOLID)
fn solid_at(x: i32, y: i32, z: i32) -> bool {
    return x >= 0 && y >= 0 && z >= 0 && x < NX() && y < NY() && z < NZ() && solid[cidx(x, y, z)] != 0u;
}
