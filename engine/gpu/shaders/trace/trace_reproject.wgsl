// trace_reproject.wgsl - used by engine/gpu/trace_gpu.c (REPROJECT)
//
// main: a new image's samples from the previous image's, where each pixel still sees the same
// surface (at most a history of them). With temporal gradients (dims.w = 1) the history is
// each pixel's own, from how much the light changed there (Schied et al. 2018, A-SVGF):
//   gmatch   per tile of 4 x 4 new pixels (TILE), one that saw the same surface before, and the old
//            pixel's last sample (its value, and what its random numbers were: trace.wgsl
//            keeps them in plane 1's z and w); the path tracer then traces that sample again
//            in the new scene (trace.wgsl main, T.txl.w = 1): their difference is the tile's
//            change, lambda = |new - old| / max(new, old)
//   gfilter  lambda over the tiles, 3 a-trous passes (1, 2, 4 tiles apart) weighted by the
//            surfaces' distance and normal; the last into every pixel of its tile (plane 0's x)
//   main     a pixel keeps (1 - lambda) / lambda samples, at most history: all where nothing
//            changed, none where a shadow moved; history_fallback where no tile near had a lambda
// An edit's move (ext.z = 1) may have changed colours, which lambda does not see (it compares
// the light over the albedo): a surface's samples are kept only where its albedo is the same,
// and lamps, the sky and media (no gradient sample of their own) keep history_fallback.
struct RP {
    oo: vec4f, oul: vec4f, oh: vec4f, ov: vec4f,     /* the old camera */
    no: vec4f, nul: vec4f, nh: vec4f, nv: vec4f,     /* the new one */
    dims: vec4u,                                     /* width, height, history, 1 = gradients */
    ext: vec4u,                                      /* the next sample's number (gmatch), gfilter's
                                                        pass (0, 1, 2), 1 = an edit's, history_fallback */
};
@group(0) @binding(0) var<uniform> R: RP;
@group(0) @binding(1) var<storage, read> old: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cur: array<vec4f>;

const TILE = 4u;   /* (trace.wgsl GTILE, trace_gpu.c GRAD_TILE) */
/* An edit's lambda counts 4 times: a drag's steps come faster than the gradients, a sixteenth of
   the pixels, can follow (edtest22, a box dragged for half a second: the old look left 0.25 and
   0.5 s after it began 0.008 and 0.004 of the change, against 0.027 and 0.013 as the physics'
   and 0.017 and 0.010 with a start over once the edits stop) */
const EDIT_LAMBDA = 4.0;

/* where the new pixel k's surface was in the old image (continuous, pixel centres at
   integers; z: its distance from the new eye, w: 1 if in front of the old eye) */
fn old_place(k: u32, g2: vec4f) -> vec4f {
    let W = R.dims.x;
    let H = R.dims.y;
    let x = f32(k % W);
    let y = f32(k / W);
    let d = normalize(R.nul.xyz + R.nh.xyz * ((x + 0.5) / f32(W)) - R.nv.xyz * ((y + 0.5) / f32(H)) - R.no.xyz);
    let far = g2.w > 1e29;
    let p = R.no.xyz + d * select(g2.w, 1e6, far);
    /* into the old camera: the screen is 1 unit in front of the eye, horizontal and vertical
       perpendicular to the view direction c */
    let f = R.oul.xyz - R.oo.xyz;
    let c = f + 0.5 * R.oh.xyz - 0.5 * R.ov.xyz;
    let q0 = p - R.oo.xyz;
    let den = dot(q0, c);
    if (den <= 0.0) { return vec4f(0.0, 0.0, 0.0, 0.0); }
    let q = q0 * (dot(c, c) / den) - f;
    let fx = dot(q, R.oh.xyz) / dot(R.oh.xyz, R.oh.xyz) * f32(W) - 0.5;
    let fy = -dot(q, R.ov.xyz) / dot(R.ov.xyz, R.ov.xyz) * f32(H) - 0.5;
    return vec4f(fx, fy, length(q0), 1.0);
}

/* the old pixel o saw what the new one sees (g2, g3: its guide; dist: from the old eye):
   samples, the same kind and object, the distance within 2% (25% in a medium) plus how fast
   it changes there, the normal within ~25 degrees */
fn same_surface(o: u32, g2: vec4f, g3: vec4f, dist: f32) -> bool {
    let W = R.dims.x;
    let H = R.dims.y;
    let N = W * H;
    if (old[o].w == 0.0) { return false; }
    if (old[3u * N + o].w != g3.w) { return false; }      /* (the same kind, and the same object) */
    let gk = g3.w - 4.0 * floor(g3.w * 0.25);             /* its kind (w also says which object: trace.wgsl, gpack) */
    let o2 = old[2u * N + o];
    if (g2.w <= 1e29) {
        let ux = o % W;
        let uy = o / W;
        let zl = old[2u * N + o - select(0u, 1u, ux > 0u)].w;
        let zr = old[2u * N + o + select(0u, 1u, ux + 1u < W)].w;
        let zu = old[2u * N + o - select(0u, W, uy > 0u)].w;
        let zd = old[2u * N + o + select(0u, W, uy + 1u < H)].w;
        let grad = min(abs(zl - o2.w), abs(zr - o2.w)) + min(abs(zu - o2.w), abs(zd - o2.w));
        if (abs(o2.w - dist) > select(0.02, 0.25, gk == 2.0) * dist + grad) { return false; }
    }
    if (gk == 0.0 && dot(o2.xyz, g2.xyz) < 0.9) { return false; }
    if (R.ext.z != 0u && gk == 0.0) {                     /* an edit's: the same colour too */
        let d = abs(old[3u * N + o].xyz - g3.xyz);
        if (max(d.x, max(d.y, d.z)) > 0.02 + 0.1 * max(g3.x, max(g3.y, g3.z))) { return false; }
    }
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let W = R.dims.x;
    let H = R.dims.y;
    let N = W * H;
    let k = gid.x + gid.y * nw.x * 64u;
    if (k >= N) { return; }
    let g2 = cur[2u * N + k];
    let g3 = cur[3u * N + k];
    /* the history this pixel keeps: with gradients its tile's lambda (gfilter put it here) */
    var hist = f32(R.dims.z);
    if (R.dims.w != 0u) {
        let lam = cur[k].x;
        if (R.ext.z != 0u && g3.w - 4.0 * floor(g3.w * 0.25) != 0.0) {
            hist = f32(R.ext.w);                  /* an edit's: not a surface */
        } else if (lam >= 0.0) {
            /* what was there weighs 1 - lambda against the new sample (A-SVGF's blend): (1 -
               lambda) / lambda samples' worth */
            let l = select(lam, lam * EDIT_LAMBDA, R.ext.z != 0u);
            hist = min(floor((1.0 - min(l, 1.0)) / max(l, 1e-3) + 0.5), f32(R.dims.z));
        } else {
            hist = f32(R.ext.w);
        }
    }
    let at = old_place(k, g2);
    var acc0 = vec4f(0.0);
    var acc1 = vec4f(0.0);
    var nsum = 0.0;
    var wsum = 0.0;
    if (at.w > 0.0) {
        let x0 = floor(at.x);
        let y0 = floor(at.y);
        for (var j = 0; j < 4; j++) {
            let tx = x0 + f32(j & 1);
            let ty = y0 + f32(j >> 1);
            if (tx < 0.0 || ty < 0.0 || tx >= f32(W) || ty >= f32(H)) { continue; }
            let o = u32(ty) * W + u32(tx);
            if (!same_surface(o, g2, g3, at.z)) { continue; }
            let s = old[o];
            let w = (1.0 - abs(at.x - tx)) * (1.0 - abs(at.y - ty));
            acc0 += w * s / s.w;
            acc1 += w * old[N + o] / s.w;
            nsum += w * s.w;
            wsum += w;
        }
    }
    if (wsum < 0.05 || hist < 0.5) {
        cur[k] = vec4f(0.0);
        cur[N + k] = vec4f(0.0);
        return;
    }
    let n = min(nsum / wsum, hist);
    cur[k] = vec4f(acc0.rgb / wsum * n, n);
    /* (z, w: the last sample of this image, none yet) */
    cur[N + k] = vec4f(acc1.xy / wsum * n, 0.0, 0.0);
}

fn hashr(x0: u32) -> u32 {
    var x = x0;
    x ^= x >> 16u; x *= 0x7feb352du;
    x ^= x >> 15u; x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}
fn tiles_x() -> u32 { return (R.dims.x + TILE - 1u) / TILE; }
fn anchor(tx: u32, ty: u32) -> u32 { return ty * TILE * R.dims.x + tx * TILE; }

/* one tile's gradient sample: a pixel of it (from a different one each image) that saw the
   same surface in the old image, the old pixel nearest to where it was, and that pixel's last
   sample. Into the tile's first pixel (its anchor a), for trace.wgsl main (T.txl.w = 1):
     cur[a]     = (the sample's value l, its number's low 16 bits, high 16 bits, the pixel's
                  count when it was taken)
     cur[N + a] = (the new pixel, the old one, the old pixel's offset from where the new
                  one's centre was: x, y); the old one -1: none */
@compute @workgroup_size(64)
fn gmatch(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let W = R.dims.x;
    let H = R.dims.y;
    let N = W * H;
    let t = gid.x + gid.y * nw.x * 64u;
    let tw = tiles_x();
    if (t >= tw * ((H + TILE - 1u) / TILE)) { return; }
    let tx = t % tw;
    let ty = t / tw;
    let a = anchor(tx, ty);
    let first = hashr(t ^ hashr(R.ext.x)) % (TILE * TILE);
    for (var i = 0u; i < TILE * TILE; i++) {
        let c = (first + i) % (TILE * TILE);
        let px = tx * TILE + c % TILE;
        let py = ty * TILE + c / TILE;
        if (px >= W || py >= H) { continue; }
        let k = py * W + px;
        let g2 = cur[2u * N + k];
        let g3 = cur[3u * N + k];
        if (g3.w - 4.0 * floor(g3.w * 0.25) != 0.0) { continue; }   /* a surface only */
        let at = old_place(k, g2);
        if (at.w <= 0.0) { continue; }
        let ox = floor(at.x + 0.5);
        let oy = floor(at.y + 0.5);
        if (ox < 0.0 || oy < 0.0 || ox >= f32(W) || oy >= f32(H)) { continue; }
        let o = u32(oy) * W + u32(ox);
        let last = old[N + o];
        if (last.w < 0.5 || !same_surface(o, g2, g3, at.z)) { continue; }
        /* the old sample's number: its low 24 bits kept, the rest from the next one's */
        let lo = u32(last.w) - 1u;
        var cnt = (R.ext.x & 0xff000000u) | lo;
        if (cnt > R.ext.x) { cnt -= 0x1000000u; }
        cur[a] = vec4f(last.z, f32(cnt & 0xffffu), f32(cnt >> 16u), old[o].w - 1.0);
        cur[N + a] = vec4f(f32(k), f32(o), ox - at.x, oy - at.y);
        return;
    }
    cur[N + a] = vec4f(-1.0, -1.0, 0.0, 0.0);
}

/* lambda over the tiles (pass ext.y: 0 reads cur[N + a].x and writes .y, 1 .y into .z, 2 .z
   into plane 0's x of every pixel of the tile); -1: no lambda near */
fn chan(v: vec4f, c: u32) -> f32 { return select(select(v.z, v.y, c == 1u), v.x, c == 0u); }
@compute @workgroup_size(64)
fn gfilter(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let W = R.dims.x;
    let H = R.dims.y;
    let N = W * H;
    let t = gid.x + gid.y * nw.x * 64u;
    let tw = tiles_x();
    let th = (H + TILE - 1u) / TILE;
    if (t >= tw * th) { return; }
    let tx = i32(t % tw);
    let ty = i32(t / tw);
    let a = anchor(u32(tx), u32(ty));
    let fpass = R.ext.y;
    let step = i32(1u << fpass);
    let ga = cur[2u * N + a];
    var sum = 0.0;
    var wsum = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let bx = tx + dx * step;
            let by = ty + dy * step;
            if (bx < 0 || by < 0 || bx >= i32(tw) || by >= i32(th)) { continue; }
            let b = anchor(u32(bx), u32(by));
            let lam = chan(cur[N + b], fpass);
            if (lam < 0.0) { continue; }
            let gb = cur[2u * N + b];
            var w = select(0.5, 1.0, dx == 0) * select(0.5, 1.0, dy == 0);
            if (ga.w < 1e29 && gb.w < 1e29) {
                w *= exp(-abs(ga.w - gb.w) / (0.05 * ga.w + 1e-3)) * pow(max(dot(ga.xyz, gb.xyz), 0.0), 8.0);
            }
            sum += w * lam;
            wsum += w;
        }
    }
    let out = select(-1.0, sum / wsum, wsum > 1e-4);
    if (fpass < 2u) {
        var v = cur[N + a];
        if (fpass == 0u) { v.y = out; } else { v.z = out; }
        cur[N + a] = v;
        return;
    }
    for (var i = 0u; i < TILE * TILE; i++) {
        let px = u32(tx) * TILE + i % TILE;
        let py = u32(ty) * TILE + i / TILE;
        if (px >= W || py >= H) { continue; }
        cur[py * W + px].x = out;
    }
}
