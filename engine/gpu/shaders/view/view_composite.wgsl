// view_composite.wgsl - used by engine/gpu/view_gpu.c (COMPOSITE)
@group(0) @binding(3) var<storage, read> light: array<f32>;
@group(0) @binding(6) var<storage, read> VS: array<VStat, 2>;

fn grid_tau(k: u32, p: vec3f) -> f32 {
    let v = F.vols[k];
    let n = vec3f(v.ldims.xyz);
    var g = (p - v.lo.xyz) / (v.hi.xyz - v.lo.xyz) * n - 0.5;
    g = clamp(g, vec3f(0.0), n - 1.001);
    let i = vec3u(g);
    let f = g - vec3f(i);
    let sy = v.ldims.x;
    let sz = v.ldims.x * v.ldims.y;
    let o = v.ldims.w + i.z * sz + i.y * sy + i.x;
    let c00 = mix(light[o], light[o + 1u], f.x);
    let c10 = mix(light[o + sy], light[o + sy + 1u], f.x);
    let c01 = mix(light[o + sz], light[o + sz + 1u], f.x);
    let c11 = mix(light[o + sz + sy], light[o + sz + sy + 1u], f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

fn phase_hg(c: f32, g: f32) -> f32 {
    let d = 1.0 + g * g - 2.0 * g * c;
    return (1.0 - g * g) / (4.0 * 3.14159265 * d * sqrt(d));
}

fn pixel_jitter(x: u32, y: u32) -> f32 {
    var h = (x * 0x8da6b343u) ^ (y * 0xd8163841u);
    h ^= h >> 13u; h *= 0x5bd1e995u; h ^= h >> 15u;
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

fn srgb_to_linear(c: vec3f) -> vec3f {
    return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}
fn linear_to_srgb(c0: vec3f) -> vec3f {
    let c = clamp(c0, vec3f(0.0), vec3f(1.0));
    return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

/* the ray from the eye along unit dir, stopped at tmax: the light it collects, and T */
fn march(dir: vec3f, tmax: f32, jit: f32) -> vec4f {
    let eye = F.eye.xyz;
    let inv = safe_inv(dir);
    let nv = u32(F.size.w);
    var seg0 = array<f32, 2>(1.0, 1.0);
    var seg1 = array<f32, 2>(0.0, 0.0);
    var ta = 1e30;
    var tb = 0.0;
    for (var k = 0u; k < nv; k++) {
        if (!vol_active(k)) { continue; }
        let t = ray_box(eye, inv, VS[k].alo.xyz, VS[k].ahi.xyz);
        let a = max(t.x, 0.0);
        let b = min(t.y, tmax);
        if (t.y < t.x || t.y <= 0.0 || b <= a) { continue; }
        seg0[k] = a; seg1[k] = b;
        ta = min(ta, a); tb = max(tb, b);
    }
    var L = vec3f(0.0);
    var T = 1.0;
    if (tb <= ta) { return vec4f(L, T); }
    let max_steps = F.size.z;
    let ds_floor = (tb - ta) / max_steps;
    var t = ta;
    var is_first = true;
    var steps = 0.0;
    let octaves = i32(F.amb.w);
    loop {
        if (t >= tb || T <= 0.003 || steps >= 4.0 * max_steps) { break; }
        steps += 1.0;
        var ds = 1e30;
        for (var k = 0u; k < nv; k++) { if (t >= seg0[k] && t <= seg1[k]) { ds = min(ds, F.vols[k].p.y); } }
        if (ds >= 1e30) { ds = select(1.0, ds_floor, ds_floor > 0.0); }
        ds = max(ds, ds_floor);
        if (is_first) { t += ds * jit; is_first = false; if (t >= tb) { break; } }
        let pos = eye + dir * t;
        var sig = 0.0;
        var src = vec3f(0.0);
        var any_here = false;
        var lamp_done = false;
        var E = vec3f(0.0);
        var cosv = 0.0;
        for (var k = 0u; k < nv; k++) {
            if (t < seg0[k] || t > seg1[k]) { continue; }
            if (brick_majorant(k, pos) <= 0.0) { continue; }
            any_here = true;
            let rho = density(k, pos);
            if (rho <= 0.0) { continue; }
            let v = F.vols[k];
            let sg = v.albedo.w * rho;
            if (!lamp_done) {
                let d = pos - F.lamp_pos.xyz;
                let d2 = dot(d, d);
                let len = sqrt(d2);
                let w = select(vec3f(0.0, -1.0, 0.0), d / len, len > 1e-6);
                var c = 1.0;
                if (F.lamp_n.w > 0.5) { c = dot(F.lamp_n.xyz, w); }
                if (c > 0.0) { E = F.lamp_L.xyz * (F.lamp_pos.w * c / (d2 + F.lamp_pos.w / 3.14159265)); }
                cosv = -dot(w, dir);
                lamp_done = true;
            }
            let tau = grid_tau(k, pos);
            var lamp_term = 0.0;
            var a = 1.0; var b = 1.0; var cc = 1.0;
            for (var o = 0; o < octaves; o++) {
                lamp_term += a * exp(-b * tau) * phase_hg(cosv, v.p.x * cc);
                a *= 0.7; b *= 0.3; cc *= 0.5;
            }
            let amb_t = 0.35 + 0.65 * exp(-0.25 * tau);
            let S = E * (lamp_term * F.lamp_L.w) + F.amb.xyz * amb_t;
            src += v.albedo.xyz * S * sg;
            sig += sg;
        }
        var stp = select(ds * 4.0, ds, any_here);
        for (var k = 0u; k < nv; k++) {
            if (seg0[k] > t && seg0[k] < t + stp && F.vols[k].p.y < stp) { stp = seg0[k] - t + 1e-3; }
        }
        if (sig > 0.0) {
            let st = exp(-sig * stp);
            L += src * (T * (1.0 - st) / sig);
            T *= st;
        }
        t += stp;
    }
    return vec4f(L, T);
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let x = u32(pos.x);
    let y = u32(pos.y);
    let pt = traced(x, y);
    if (pt.w > 0.0) { return vec4f(linear_to_srgb(pt.rgb / pt.w), 1.0); }
    var rgb = frame_rgb(x, y);
    if (F.size.w > 0.5) {
        /* the ray through the pixel center; at view depth D it is at eye + D * du */
        let nx = (2.0 * (f32(x) + 0.5) / F.size.x - 1.0) * F.proj.x;
        let ny = (1.0 - 2.0 * (f32(y) + 0.5) / F.size.y) * F.proj.y;
        let du = F.right.xyz * nx + F.up.xyz * ny - F.back.xyz;
        let dl = length(du);
        var tmax = 1e30;
        let d = frame_depth(x, y);
        if (d < 1.0) {
            let vd = F.proj.w / (F.proj.z + 2.0 * d - 1.0);
            if (vd > 0.0) { tmax = vd * dl; }
        }
        let r = march(du / dl, tmax, pixel_jitter(x, y));
        if (r.w < 1.0) { rgb = linear_to_srgb(r.xyz + srgb_to_linear(rgb) * r.w); }
    }
    return vec4f(rgb, 1.0);
}
