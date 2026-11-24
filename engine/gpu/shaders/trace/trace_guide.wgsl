// trace_guide.wgsl - used by engine/gpu/trace_gpu.c (GUIDE_COPY)
// A new image's guide from the rasterizer's G-buffer (raster_gpu.c gpu_raster_guide): each
// pixel's planes 2-4 as trace.wgsl's gbuf would make them, where the camera sees a surface the
// light scatters on (or a lamp, or nothing); a pixel that sees a mirror or glass first is
// marked (plane 3's w = -1), and gbuf then traces its guide through them as before. The
// samples (planes 0 and 1) start at nothing, unless the reprojection that follows writes them.
struct GC {
    o: vec4f, ul: vec4f, h: vec4f, v: vec4f,     /* the camera (trace.wgsl's TP) */
    dims: vec4u,                                 /* width, height; 1 = the samples left to the reprojection */
};
@group(0) @binding(0) var<uniform> C: GC;
/* the G-buffer: the normal (towards the eye) and the distance from the eye; the albedo and
   the kind (0 a surface, 1 a lamp); the record drawn (0 the world, i + 1 instance i; -1
   nothing, -2 a mirror or glass) */
@group(0) @binding(1) var GN: texture_2d<f32>;
@group(0) @binding(2) var GA: texture_2d<f32>;
@group(0) @binding(3) var GI: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> accum: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let W = C.dims.x;
    let N = W * C.dims.y;
    let k = gid.x + gid.y * nw.x * 64u;
    if (k >= N) { return; }
    let xy = vec2i(i32(k % W), i32(k / W));
    let id = textureLoad(GI, xy, 0).r;
    if (C.dims.z == 0u) {                         /* (else the reprojection that follows writes them) */
        accum[k] = vec4f(0.0);
        accum[N + k] = vec4f(0.0);
    }
    if (id < -1.5) {                              /* a mirror or glass: gbuf traces this one */
        accum[3u * N + k] = vec4f(1.0, 1.0, 1.0, -1.0);
        return;
    }
    if (id < 0.0) {                               /* nothing: the sky (trace.wgsl guide's defaults) */
        accum[2u * N + k] = vec4f(0.0, 0.0, 0.0, 1e30);
        accum[3u * N + k] = vec4f(1.0, 1.0, 1.0, 1.0);
        accum[4u * N + k] = vec4f(0.0);
        return;
    }
    let gn = textureLoad(GN, xy, 0);
    let ga = textureLoad(GA, xy, 0);
    let kind = ga.w;
    accum[2u * N + k] = gn;
    /* gpack: the kind plus 4 times the instance + 1 (the record: 0 the world) */
    accum[3u * N + k] = vec4f(ga.xyz, kind + 4.0 * id);
    if (kind < 0.5) {                             /* a surface: where caustics are looked up */
        let pc = C.ul.xyz + C.h.xyz * ((f32(xy.x) + 0.5) / f32(W)) - C.v.xyz * ((f32(xy.y) + 0.5) / f32(C.dims.y));
        accum[4u * N + k] = vec4f(C.o.xyz + normalize(pc - C.o.xyz) * gn.w, 1.0);
    } else {
        accum[4u * N + k] = vec4f(0.0);
    }
}
