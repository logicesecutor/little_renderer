// view_fxaa.wgsl - used by engine/gpu/view_gpu.c (FXAA)
/* Anti-aliasing after the composite: FXAA, the "quality" variant of Lottes' FXAA 3.11 (from
   its description). The frame (already sRGB-encoded, as on the canvas) is read back; where
   the contrast across a pixel is high enough, the edge through it is followed both ways to its
   ends, and the pixel is moved (one bilinear read) across the edge by how far along it lies:
   near an end, the stair step is blended most. Thin lines and lone pixels get a blend of
   their neighbours (the sub-pixel part). */
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

const EDGE_MIN = 0.0312;     // darker contrast than this is left alone
const EDGE_REL = 0.125;      // and contrast below this fraction of the brightest neighbour
const SUBPIX = 0.75;         // how much of the sub-pixel blend
const STEPS = 12;

fn px(p: vec2i) -> vec3f {
    let s = vec2i(textureDimensions(src)) - 1;
    return textureLoad(src, clamp(p, vec2i(0), s), 0).rgb;
}
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }
fn lum(p: vec2i) -> f32 { return luma(px(p)); }
// bilinear, at a point in pixels (centres at +0.5): no sampler needed
fn bil(q: vec2f) -> vec3f {
    let t = q - 0.5;
    let b = floor(t);
    let f = t - b;
    let i = vec2i(b);
    let a = mix(px(i), px(i + vec2i(1, 0)), f.x);
    let c = mix(px(i + vec2i(0, 1)), px(i + vec2i(1, 1)), f.x);
    return mix(a, c, f.y);
}
fn step_len(k: i32) -> f32 {
    if (k < 5) { return 1.0; }
    if (k == 5) { return 1.5; }
    if (k < 10) { return 2.0; }
    if (k == 10) { return 4.0; }
    return 8.0;
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let p = vec2i(pos.xy);
    let cM = px(p);
    let M = luma(cM);
    let N = lum(p + vec2i(0, -1));
    let S = lum(p + vec2i(0, 1));
    let W = lum(p + vec2i(-1, 0));
    let E = lum(p + vec2i(1, 0));
    let lo = min(M, min(min(N, S), min(W, E)));
    let hi = max(M, max(max(N, S), max(W, E)));
    let range = hi - lo;
    if (range < max(EDGE_MIN, hi * EDGE_REL)) { return vec4f(cM, 1.0); }

    let NW = lum(p + vec2i(-1, -1));
    let NE = lum(p + vec2i(1, -1));
    let SW = lum(p + vec2i(-1, 1));
    let SE = lum(p + vec2i(1, 1));
    // a horizontal edge changes along y the most
    let eh = abs(NW + SW - 2.0 * W) + 2.0 * abs(N + S - 2.0 * M) + abs(NE + SE - 2.0 * E);
    let ev = abs(NW + NE - 2.0 * N) + 2.0 * abs(W + E - 2.0 * M) + abs(SW + SE - 2.0 * S);
    let horiz = eh >= ev;
    // the side of the edge: the neighbour across it with the bigger difference
    let l1 = select(W, N, horiz);       // before (up or left)
    let l2 = select(E, S, horiz);       // after
    let g1 = l1 - M;
    let g2 = l2 - M;
    let first = abs(g1) >= abs(g2);
    let grad = 0.25 * max(abs(g1), abs(g2));
    let dir = select(1.0, -1.0, first);                   // across the edge, towards that side
    let avg = select(0.5 * (l2 + M), 0.5 * (l1 + M), first);
    let across = select(vec2f(1.0, 0.0), vec2f(0.0, 1.0), horiz);
    let along = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), horiz);

    // follow the edge both ways (on the line between the two rows) until the contrast changes
    let c = pos.xy + across * (0.5 * dir);
    var q1 = c - along;
    var q2 = c + along;
    var e1 = luma(bil(q1)) - avg;
    var e2 = luma(bil(q2)) - avg;
    var done1 = abs(e1) >= grad;
    var done2 = abs(e2) >= grad;
    for (var k = 1; k < STEPS && !(done1 && done2); k++) {
        let s = step_len(k);
        if (!done1) { q1 -= along * s; e1 = luma(bil(q1)) - avg; done1 = abs(e1) >= grad; }
        if (!done2) { q2 += along * s; e2 = luma(bil(q2)) - avg; done2 = abs(e2) >= grad; }
    }
    let d1 = dot(pos.xy - q1, along);
    let d2 = dot(q2 - pos.xy, along);
    let near1 = d1 < d2;
    let dmin = min(d1, d2);
    var off = 0.5 - dmin / (d1 + d2);
    // only towards the end whose contrast goes the way the centre's does
    let e_near = select(e2, e1, near1);
    if ((e_near < 0.0) == (M < avg)) { off = 0.0; }

    // the sub-pixel part: a pixel unlike all its neighbours
    let L = (2.0 * (N + S + W + E) + NW + NE + SW + SE) / 12.0;
    let s1 = clamp(abs(L - M) / range, 0.0, 1.0);
    let s2 = (-2.0 * s1 + 3.0) * s1 * s1;
    off = max(off, s2 * s2 * SUBPIX);

    return vec4f(bil(pos.xy + across * (off * dir)), 1.0);
}
