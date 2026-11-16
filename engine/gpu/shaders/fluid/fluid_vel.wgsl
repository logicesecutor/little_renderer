// fluid_vel.wgsl - used by engine/gpu/fluid_gpu.c (VEL)
fn vel_at(p: vec3f) -> vec3f {
    let g = (p - P.lo.xyz) * P.a.x;
    return vec3f(s_u(g - vec3f(0.0, 0.5, 0.5), SU()), s_v(g - vec3f(0.5, 0.0, 0.5), SV()), s_w(g - vec3f(0.5, 0.5, 0.0), SW()));
}
