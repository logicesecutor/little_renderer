// rigid_simple.wgsl - used by engine/gpu/rigid_gpu.c (K_SIMPLE)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let k = tid(gid, nw);
    if (k >= min(atomicLoad(&pc[0]), P.m.z)) { return; }
    let pr = pairs[k];
    let i = pr.x;
    let j = pr.y;
    MG = margin_of(i, j);
    if (j == NONE) { walls(i); return; }
    let sa = B[i].pos.w < 0.5;
    let sb = B[j].pos.w < 0.5;
    if (sa && sb) { sphere_sphere(i, j); }
    else if (sa) { sphere_box(i, j); }
    else if (sb) { sphere_box(j, i); }
    else { box_box(i, j); }
}
