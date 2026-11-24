// raster_shader.wgsl - used by engine/gpu/raster_gpu.c (SHADER)
struct RMat {
    base: vec4f,     /* base colour (a lamp's: its colour at full brightness); w = 1: drawn as it is */
    pbr: vec4f,      /* metallic, roughness, index of refraction, transmission (material_resolved) */
    tex: vec4u,      /* its textures (tex_gpu.c gpu_tex_pack) */
    emis: vec4f,     /* its glow (a MAT_PBR's emission) */
};
/* the lamp (RenderLamp): the quad q + a u + b v, the side it shines to, w = (u x v) / |u x v|^2 */
struct RLamp {
    q: vec4f,        /* w = 1: there is a lamp */
    u: vec4f,        /* w = its area */
    v: vec4f,
    n: vec4f,
    w: vec4f,
    center: vec4f,
    le: vec4f,       /* its radiance */
    k: vec4f,        /* the ambient light: uniform, towards the camera */
};
struct RU {
    vp: mat4x4f,     /* world -> clip, z in -w..w as on the CPU */
    eye: vec4f,      /* w = mode: 0 wireframe, 1 flat, 2 gouraud, 3 phong */
    lamp: RLamp,
    /* the environment it lights with (RenderLamp.env; scene/env.h): its levels in LV */
    envp: vec4f,     /* scale, cos and sin of its turn, 1 = there is one */
    envc: vec4f,     /* a probe's centre, w = 1: lookups corrected by its box */
    envlo: vec4f,
    envhi: vec4f,
    lv: array<vec4u, 6>,   /* level k: its first texel in LV, width, height */
    sh: array<vec4f, 9>,   /* the irradiance's harmonics (E / pi) */
    /* the sky behind everything (Scene.env), in BG */
    sky: vec4f,      /* scale, cos and sin of its turn, 1 = drawn */
    skyd: vec4u,     /* its width and height */
    cam_r: vec4f,    /* the camera's right (w: 1 / the projection's x scale), */
    cam_u: vec4f,    /* up (w: 1 / its y scale), */
    cam_b: vec4f,    /* back */
    vps: vec4f,      /* the viewport's width and height */
    irrn: vec4u,     /* the irradiance grid (IrrGrid, in IG): its probes along x, y, z; w = 1: there is one */
    irrlo: vec4f,
    irrhi: vec4f,
    mats: array<RMat, 64>,
};
@group(0) @binding(0) var<uniform> U: RU;
@group(0) @binding(1) var<storage, read> vpos: array<vec4f>;
@group(0) @binding(2) var<storage, read> vnrm: array<vec4f>;
/* where a draw's triangles are: record 0 the world itself, then one per instance (the
   draw's first instance): object -> world rows, the inverse's rows (normals), material */
struct IX { r0: vec4f, r1: vec4f, r2: vec4f, i0: vec4f, i1: vec4f, i2: vec4f, m: vec4u };
@group(0) @binding(3) var<storage, read> ix: array<IX>;
/* the BSDF's tables (gpu_bsdf_tables) */
struct RT { t: array<vec4f, 528> };
@group(0) @binding(4) var<uniform> TB: RT;
fn bsdf_tab(k: u32) -> f32 { return TB.t[k >> 2u][k & 3u]; }
@group(0) @binding(5) var<storage, read> LV: array<vec4f>;
@group(0) @binding(6) var<storage, read> BG: array<vec4f>;
/* per vertex: texture coordinates, the tangent (octahedral; its handedness is vnrm's w) */
@group(0) @binding(7) var<storage, read> vtex: array<vec4f>;
/* the irradiance grid: per probe 7 vec4 (9 rgb harmonics, then valid) */
@group(0) @binding(8) var<storage, read> IG: array<vec4f>;
/* the textures (gpu/tex_gpu.c) */
@group(0) @binding(9) var TEXA: texture_2d<f32>;
struct TT { e: array<vec4u, 512> };
@group(0) @binding(10) var<uniform> TXT: TT;
fn tex_entry(k: u32) -> vec4u { return TXT.e[k]; }
/* the draws' order: draw instance -> record (raster_gpu.c order_instances: bucketed by mesh
   and level of detail, those out of view left out) */
@group(0) @binding(11) var<storage, read> ORD: array<u32>;

/* ---- the environment (scene/env.c) ---- */
fn env_uv(d: vec3f, c: f32, s: f32) -> vec2f {
    let x = d.x * c - d.z * s;
    let z = d.x * s + d.z * c;
    let u = 0.5 + atan2(x, z) * (0.5 / BSDF_PI);
    return vec2f(u - floor(u), acos(clamp(d.y, -1.0, 1.0)) / BSDF_PI);
}
/* bilinear in a w x h map that starts at texel `first` of LV (bg: of BG) */
fn map_bilerp(first: u32, w: i32, h: i32, uv: vec2f, bg: bool) -> vec3f {
    let x = uv.x * f32(w) - 0.5;
    let y = uv.y * f32(h) - 0.5;
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let ya = u32(clamp(y0, 0, h - 1));
    let yb = u32(clamp(y0 + 1, 0, h - 1));
    let xa = u32(((x0 % w) + w) % w);
    let xb = (xa + 1u) % u32(w);
    let uw = u32(w);
    var a: vec3f; var b: vec3f; var c: vec3f; var e: vec3f;
    if (bg) {
        a = BG[ya * uw + xa].rgb; b = BG[ya * uw + xb].rgb; c = BG[yb * uw + xa].rgb; e = BG[yb * uw + xb].rgb;
    } else {
        a = LV[first + ya * uw + xa].rgb; b = LV[first + ya * uw + xb].rgb;
        c = LV[first + yb * uw + xa].rgb; e = LV[first + yb * uw + xb].rgb;
    }
    return (1.0 - fy) * ((1.0 - fx) * a + fx * b) + fy * ((1.0 - fx) * c + fx * e);
}
fn env_level(d: vec3f, rough: f32) -> vec3f {
    let uv = env_uv(d, U.envp.y, U.envp.z);
    let lod = clamp(rough, 0.0, 1.0) * 5.0;
    let k = min(u32(lod), 4u);
    let f = lod - f32(k);
    let a = map_bilerp(U.lv[k].x, i32(U.lv[k].y), i32(U.lv[k].z), uv, false);
    let b = map_bilerp(U.lv[k + 1u].x, i32(U.lv[k + 1u].y), i32(U.lv[k + 1u].z), uv, false);
    return mix(a, b, f) * U.envp.x;
}
fn env_irradiance(n0: vec3f) -> vec3f {
    let n = vec3f(n0.x * U.envp.y - n0.z * U.envp.z, n0.y, n0.x * U.envp.z + n0.z * U.envp.y);
    var y: array<f32, 9>;
    y[0] = 0.282095;
    y[1] = 0.488603 * n.y; y[2] = 0.488603 * n.z; y[3] = 0.488603 * n.x;
    y[4] = 1.092548 * n.x * n.y; y[5] = 1.092548 * n.y * n.z; y[6] = 0.315392 * (3.0 * n.z * n.z - 1.0);
    y[7] = 1.092548 * n.x * n.z; y[8] = 0.546274 * (n.x * n.x - n.y * n.y);
    var c = vec3f(0.0);
    for (var i = 0; i < 9; i++) { c += U.sh[i].rgb * y[i]; }
    return max(c, vec3f(0.0)) * U.envp.x;
}
/* render.c sh_lum4: the luminance of the environment's harmonics towards n, the first 4 */
fn env_lum4(n0: vec3f) -> f32 {
    let n = vec3f(n0.x * U.envp.y - n0.z * U.envp.z, n0.y, n0.x * U.envp.z + n0.z * U.envp.y);
    let c = U.sh[0].rgb * 0.282095 + (U.sh[1].rgb * n.y + U.sh[2].rgb * n.z + U.sh[3].rgb * n.x) * 0.488603;
    return dot(c, vec3f(0.2126, 0.7152, 0.0722)) * U.envp.x;
}
/* env.c irr_grid_blend: the valid probes around p blended (ok = 0: none near) */
struct IrrSH { c: array<vec3f, 9>, ok: f32 };
fn irr_blend(p: vec3f) -> IrrSH {
    var r: IrrSH;
    r.ok = 0.0;
    if (U.irrn.w == 0u) { return r; }
    let nn = vec3i(U.irrn.xyz);
    var x = (p - U.irrlo.xyz) / (U.irrhi.xyz - U.irrlo.xyz) * vec3f(nn) - vec3f(0.5);
    x = clamp(x, vec3f(0.0), vec3f(nn - vec3i(1)));
    let i0 = min(vec3i(x), max(nn - vec3i(2), vec3i(0)));
    let f = x - vec3f(i0);
    var ws = 0.0;
    for (var corner = 0u; corner < 8u; corner++) {
        let o = vec3i(i32(corner & 1u), i32((corner >> 1u) & 1u), i32((corner >> 2u) & 1u));
        let ii = i0 + o;
        if (any(ii >= nn)) { continue; }
        let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1) * select(1.0 - f.z, f.z, o.z == 1);
        let base = u32((ii.z * nn.y + ii.y) * nn.x + ii.x) * 7u;
        if (!(IG[base + 6u].w > 0.0) || w <= 0.0) { continue; }
        for (var i = 0u; i < 9u; i++) {
            let k = 3u * i;
            r.c[i] += w * vec3f(IG[base + k / 4u][k % 4u], IG[base + (k + 1u) / 4u][(k + 1u) % 4u], IG[base + (k + 2u) / 4u][(k + 2u) % 4u]);
        }
        ws += w;
    }
    if (ws <= 1e-4) { return r; }
    for (var i = 0u; i < 9u; i++) { r.c[i] /= ws; }
    r.ok = 1.0;
    return r;
}
fn irr_eval(b: IrrSH, n: vec3f) -> vec3f {
    var y: array<f32, 9>;
    y[0] = 0.282095;
    y[1] = 0.488603 * n.y; y[2] = 0.488603 * n.z; y[3] = 0.488603 * n.x;
    y[4] = 1.092548 * n.x * n.y; y[5] = 1.092548 * n.y * n.z; y[6] = 0.315392 * (3.0 * n.z * n.z - 1.0);
    y[7] = 1.092548 * n.x * n.z; y[8] = 0.546274 * (n.x * n.x - n.y * n.y);
    var c = vec3f(0.0);
    for (var i = 0u; i < 9u; i++) { c += b.c[i] * y[i]; }
    return max(c, vec3f(0.0));
}
fn env_box_dir(p: vec3f, d: vec3f) -> vec3f {
    if (U.envc.w < 0.5) { return d; }
    var t = 1e30;
    for (var k = 0; k < 3; k++) {
        if (d[k] > 1e-8) { t = min(t, (U.envhi[k] - p[k]) / d[k]); }
        else if (d[k] < -1e-8) { t = min(t, (U.envlo[k] - p[k]) / d[k]); }
    }
    if (!(t > 0.0) || t >= 1e30) { return d; }
    let q = p + d * t - U.envc.xyz;
    let l = length(q);
    return select(d, q / l, l > 1e-6);
}
fn refract_(d: vec3f, n: vec3f, eta: f32) -> vec3f {
    let cos_i = min(dot(-d, n), 1.0);
    let perp = (d + n * cos_i) * eta;
    return perp - n * sqrt(abs(1.0 - dot(perp, perp)));
}
/* render.c through_dir: refracted in, out of a sphere */
fn through_dir(v: vec3f, n: vec3f, eta: f32) -> vec3f {
    let d = -v;
    let c = dot(v, n);
    if (1.0 - eta * eta * (1.0 - c * c) <= 0.0) { return reflect(d, n); }
    let t = normalize(refract_(d, n, eta));
    let n2 = n - t * (2.0 * dot(n, t));
    let c2 = -dot(t, n2);
    if (1.0 - (1.0 / (eta * eta)) * (1.0 - c2 * c2) <= 0.0) { return reflect(t, -n2); }
    return normalize(refract_(t, -n2, 1.0 / eta));
}
fn to_world(k: u32, p: vec3f) -> vec3f {
    let x = ix[k];
    let q = vec4f(p, 1.0);
    return vec3f(dot(x.r0, q), dot(x.r1, q), dot(x.r2, q));
}
fn dir_to_world(k: u32, d: vec3f) -> vec3f {
    let x = ix[k];
    let q = vec4f(d, 0.0);
    return normalize(vec3f(dot(x.r0, q), dot(x.r1, q), dot(x.r2, q)));
}
fn normal_to_world(k: u32, n: vec3f) -> vec3f {
    let x = ix[k];
    return normalize(n.x * x.i0.xyz + n.y * x.i1.xyz + n.z * x.i2.xyz);
}
fn mat_of(k: u32, code: u32) -> u32 { return select(code & 63u, ix[k].m.x & 63u, ix[k].m.x != 0xffffffffu); }

struct VO {
    @builtin(position) pos: vec4f,
    @location(0) wp: vec3f,
    @location(1) n: vec3f,
    @location(2) light: vec3f,
    @location(3) @interpolate(flat) mat: u32,
    @location(4) uv: vec2f,
    @location(5) tg: vec4f,
};

/* render_shade (raster/render.c): the lamp's corner i (q, q + u, q + u + v, q + v) */
fn lamp_corner(i: u32) -> vec3f {
    return U.lamp.q.xyz + select(vec3f(0.0), U.lamp.u.xyz, i == 1u || i == 2u) + select(vec3f(0.0), U.lamp.v.xyz, i >= 2u);
}
/* the lamp's form factor from p (normal n), the lamp clipped at p's horizon; an edge's
   theta / sin theta by Heitz and Hill's fit */
fn edge_theta_sin(x: f32) -> f32 {
    let y = abs(x);
    let t = (0.8543985 + (0.4965155 + 0.0145206 * y) * y) / (3.4175940 + (4.1616724 + y) * y);
    return select(0.5 * inverseSqrt(max(1.0 - x * x, 1e-7)) - t, t, x > 0.0);
}
fn lamp_form_factor(p: vec3f, n: vec3f) -> f32 {
    var o: array<vec3f, 8>;
    var k = 0u;
    for (var i = 0u; i < 4u; i++) {
        let a = lamp_corner(i) - p;
        let b = lamp_corner((i + 1u) & 3u) - p;
        let da = dot(a, n);
        let db = dot(b, n);
        if (da >= 0.0) { o[k] = normalize(a); k++; }
        if ((da >= 0.0) != (db >= 0.0)) { o[k] = normalize(a + (b - a) * (da / (da - db))); k++; }
    }
    if (k < 3u) { return 0.0; }
    var sum = 0.0;
    for (var i = 0u; i < k; i++) {
        let a = o[i];
        let b = o[(i + 1u) % k];
        sum += dot(cross(a, b), n) * edge_theta_sin(dot(a, b));
    }
    return abs(sum) * BSDF_PI;
}
/* the lamp's point nearest the ray from p along r */
fn lamp_nearest(p: vec3f, r: vec3f) -> vec3f {
    let ln = U.lamp.n.xyz;
    let q = U.lamp.q.xyz;
    let dn = dot(r, ln);
    let far = 4.0 * length(U.lamp.center.xyz - p);
    var t = far;
    if (dn < 0.0) { t = min(dot(q - p, ln) / dn, far); }
    t = max(t, 0.0);
    let y = p + r * t;
    let qx = y - ln * dot(y - q, ln) - q;
    let a = clamp(dot(U.lamp.w.xyz, cross(qx, U.lamp.v.xyz)), 0.0, 1.0);
    let b = clamp(dot(U.lamp.w.xyz, cross(U.lamp.u.xyz, qx)), 0.0, 1.0);
    return q + U.lamp.u.xyz * a + U.lamp.v.xyz * b;
}
/* the radiance p (normal n0, turned to the eye), material mi, sends to the eye; a textured
   material at uv (tangent tg, footprint 2^uv_lod: render.c shade_textured) */
fn scene_shade(mi: u32, p: vec3f, n0: vec3f, uv: vec2f, tg: vec4f, uv_lod: f32) -> vec3f {
    let m = U.mats[mi];
    let v = normalize(U.eye.xyz - p);
    var n = n0;
    if (dot(n, v) < 0.0) { n = -n; }
    let ng = n;
    var base = m.base.rgb;
    var metallic = m.pbr.x;
    var rough = m.pbr.y;
    var emis = m.emis.rgb;
    var ao = 1.0;
    if (any(m.tex.xyz != vec3u(0u))) {
        let tm = tex_material(m.tex, base, metallic, rough, emis, uv, uv_lod);
        base = tm.base; metallic = tm.metallic; rough = tm.rough; emis = tm.emission;
        ao = tex_occlusion(m.tex, uv, uv_lod);
        if ((m.tex.z & 255u) != 0u) {
            n = tex_normal(m.tex, uv, uv_lod, n, tg);
            let c = dot(n, v);
            if (c < 1e-3) { n = normalize(n + v * (1e-3 - c)); }
        }
    }
    let cv = max(dot(n, v), 1e-4);
    var b = bsdf_make(base, metallic, rough, m.pbr.z, m.pbr.w, true);
    var c: vec3f;
    if (U.envp.w > 0.5) {
        /* the environment, split-sum (render.c shade_env) */
        let ap = bsdf_albedo_parts(&b, cv);
        let r = select(b.rough, 0.0, b.perfect != 0u);
        let rr = reflect(-v, n);
        let sm = 1.0 - r;
        let rd = normalize(mix(n, rr, sm * (sqrt(sm) + r)));
        var spec = ap.spec * env_level(env_box_dir(p, rd), r);
        let ib = irr_blend(p);
        var diff = env_irradiance(n);
        if (ib.ok > 0.0) {
            /* render.c shade_env: the probe's view scaled by the light around rd here against there */
            let lw = vec3f(0.2126, 0.7152, 0.0722);
            let here = dot(ib.c[0] * 0.282095 + (ib.c[1] * rd.y + ib.c[2] * rd.z + ib.c[3] * rd.x) * 0.488603, lw);
            let there = env_lum4(rd);
            if (there > 1e-6 && here < there) { spec *= max(here, 0.0) / there; }
            diff = irr_eval(ib, n);
        }
        c = spec + ap.diff * diff;
        if ((1.0 - b.metallic) * b.transmission > 0.0) {
            c += ap.trans * env_level(env_box_dir(p, through_dir(v, n, b.eta)), r);
        }
    } else {
        c = bsdf_albedo_uniform(&b, cv) * (U.lamp.k.x + U.lamp.k.y * cv);
    }
    let center = U.lamp.center.xyz;
    let ln = U.lamp.n.xyz;
    c = c * ao + emis;
    if (U.lamp.q.w < 0.5 || dot(p - center, ln) <= 0.0) { return c; }
    /* a bent normal: only the share of the lamp above the surface's own plane (render.c lamp_hz) */
    var hz = 1.0;
    if ((m.tex.z & 255u) != 0u) {
        let tc0 = center - p;
        let d = length(tc0);
        let s = max(sqrt(U.lamp.u.w / BSDF_PI) / d, 1e-3);
        hz = clamp(0.5 + dot(ng, tc0) / (d * 2.0 * s), 0.0, 1.0);
        if (hz <= 0.0) { return c; }
    }
    let base_c = c;
    let ff = lamp_form_factor(p, n);
    if (ff > 0.0) {
        let lc = normalize(center - p);
        c += bsdf_diffuse_lobes(&b, max(dot(n, lc), 0.0)) * U.lamp.le.rgb * ff;
    }
    let area = U.lamp.u.w;
    let tc = center - p;
    let dc2 = dot(tc, tc);
    let om = min(area * max(-dot(ln, tc), 0.0) / (dc2 * sqrt(dc2)), 2.0 * BSDF_PI);
    let lobe = 4.0 * BSDF_PI * b.alpha * b.alpha * cv;
    let w = lobe / (lobe + om);
    var spec = vec3f(0.0);
    if (w > 0.01 && lobe > 32.0 * om) {
        let lc = tc * inverseSqrt(dc2);
        let ck = dot(n, lc);
        if (ck > 0.0) { spec = bsdf_spec_lobe(&b, n, v, lc, b.alpha) * (w * ck * om); }
    } else if (w > 0.01) {
        for (var k = 0u; k < 4u; k++) {
            let xk = U.lamp.q.xyz + U.lamp.u.xyz * select(0.25, 0.75, (k & 1u) != 0u) + U.lamp.v.xyz * select(0.25, 0.75, (k & 2u) != 0u);
            let dk = xk - p;
            let d2 = dot(dk, dk);
            let lk = dk * inverseSqrt(d2);
            let ck = dot(n, lk);
            let cosl = -dot(ln, lk);
            if (ck > 0.0 && cosl > 0.0) { spec += bsdf_spec_lobe(&b, n, v, lk, b.alpha) * (w * ck * cosl * area * 0.25 / d2); }
        }
    }
    if (w < 0.99) {
        let l = normalize(lamp_nearest(p, reflect(-v, n)) - p);
        let cl = dot(n, l);
        if (cl > 0.0) { spec += bsdf_spec_lobe(&b, n, v, l, b.alpha) * ((1.0 - w) * cl * om * lobe / (om + lobe)); }
    }
    return base_c + (c - base_c + spec * U.lamp.le.rgb) * hz;
}
/* WebGPU keeps z in 0..w: (z + w) / 2 is the same depth the CPU z-buffer stores */
fn clip_of(p: vec3f) -> vec4f {
    let c = U.vp * vec4f(p, 1.0);
    return vec4f(c.x, c.y, (c.z + c.w) * 0.5, c.w);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) di: u32) -> VO {
    let k = ORD[di];
    let t = (vi / 3u) * 3u;
    let p0 = to_world(k, vpos[t].xyz);
    let p1 = to_world(k, vpos[t + 1u].xyz);
    let p2 = to_world(k, vpos[t + 2u].xyz);
    let code = u32(vpos[vi].w);
    let mode = u32(U.eye.w);
    let face_n = normalize(cross(p1 - p0, p2 - p0));
    var n = face_n;
    if ((code & 256u) != 0u && mode != 1u) { n = normal_to_world(k, vnrm[vi].xyz); }
    let p = to_world(k, vpos[vi].xyz);
    var o: VO;
    o.pos = clip_of(p);
    o.wp = p;
    o.n = n;
    o.mat = mat_of(k, code);
    o.light = vec3f(0.0);
    let tx = vtex[vi];
    o.uv = tx.xy;
    o.tg = vec4f(dir_to_world(k, oct_decode(tx.zw)), vnrm[vi].w);
    if (U.mats[o.mat].base.w > 0.5) { return o; }
    if (mode == 1u) {
        let flag = (code >> 6u) & 3u;
        var at = (p0 + p1 + p2) * (1.0 / 3.0);
        if (flag == 1u) { at = (p0 + p2) * 0.5; }
        if (flag == 2u) { at = (p0 + p1) * 0.5; }
        let um = (vtex[t].xy + vtex[t + 1u].xy + vtex[t + 2u].xy) * (1.0 / 3.0);
        let t0 = vec4f(dir_to_world(k, oct_decode(vtex[t].zw)), vnrm[t].w);
        o.light = scene_shade(o.mat, at, face_n, um, t0, -1000.0);
    } else if (mode == 2u) {
        o.light = scene_shade(o.mat, p, n, o.uv, o.tg, -1000.0);
    }
    return o;
}

@fragment
fn fs(i: VO) -> @location(0) vec4f {
    /* (derivatives first: they need every pixel of the quad still here) */
    let dx = dpdxFine(i.uv);
    let dy = dpdyFine(i.uv);
    let m = U.mats[i.mat];
    if (m.base.w > 0.5) { return vec4f(linear_to_srgb(m.base.rgb), 1.0); }
    /* flat and gouraud: the vertex stage shaded */
    if (u32(U.eye.w) != 3u) { return vec4f(linear_to_srgb(i.light), 1.0); }
    /* phong: render_shade at every pixel; a texture's level from how uv changes to the next
       pixels (raster.c's derivatives) */
    let mm = max(dot(dx, dx), dot(dy, dy));
    let uv_lod = select(-1000.0, 0.5 * log2(mm), mm > 0.0);
    return vec4f(linear_to_srgb(scene_shade(i.mat, i.wp, normalize(i.n), i.uv, i.tg, uv_lod)), 1.0);
}

/* the sky where nothing was drawn: a triangle over the whole screen at the far depth (1),
   drawn after the scene where the depth is still 1 */
struct SO { @builtin(position) pos: vec4f };
@vertex
fn vs_sky(@builtin(vertex_index) vi: u32) -> SO {
    let x = select(-1.0, 3.0, vi == 1u);
    let y = select(-1.0, 3.0, vi == 2u);
    return SO(vec4f(x, y, 1.0, 1.0));
}
@fragment
fn fs_sky(i: SO) -> @location(0) vec4f {
    let nx = (2.0 * i.pos.x / U.vps.x - 1.0) * U.cam_r.w;
    let ny = (1.0 - 2.0 * i.pos.y / U.vps.y) * U.cam_u.w;
    let d = normalize(U.cam_r.xyz * nx + U.cam_u.xyz * ny - U.cam_b.xyz);
    let c = map_bilerp(0u, i32(U.skyd.x), i32(U.skyd.y), env_uv(d, U.sky.y, U.sky.z), true) * U.sky.x;
    return vec4f(linear_to_srgb(c), 1.0);
}

/* wireframe: the 3 edges of every triangle as lines, 6 vertices per triangle */
struct WO {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) mat: u32,
};
@vertex
fn vs_wire(@builtin(vertex_index) vi: u32, @builtin(instance_index) di: u32) -> WO {
    let k = ORD[di];
    let tri = vi / 6u;
    let e = (vi % 6u) / 2u;
    let corner = (e + (vi & 1u)) % 3u;
    let q = vpos[tri * 3u + corner];
    var o: WO;
    o.pos = clip_of(to_world(k, q.xyz));
    o.mat = mat_of(k, u32(q.w));
    return o;
}
@fragment
fn fs_wire(i: WO) -> @location(0) vec4f { return vec4f(linear_to_srgb(U.mats[i.mat].base.rgb), 1.0); }

/* ---- the path tracer's guide (raster_gpu.c gpu_raster_guide) ----
   What the camera sees first, through each pixel's centre, as trace.wgsl's guide() finds it
   with a ray: the normal (towards the eye, as the textures bend it) and the distance from the
   eye; the albedo, and the kind (0 a surface the light scatters on, 1 a lamp or a surface
   glowing as bright); the record drawn (0 the world, i + 1 instance i), or -2 where the
   surface is a mirror or glass (its guide is traced through it; trace_guide.wgsl). The depth
   reversed and without a far plane (z = near / w in a float buffer: the nearest wins over any
   distance), so that nothing a ray would hit is left out. */
const GUIDE_NEAR = 0.01;
struct VG {
    @builtin(position) pos: vec4f,
    @location(0) wp: vec3f,
    @location(1) n: vec3f,
    @location(2) @interpolate(flat) mat: u32,
    @location(3) uv: vec2f,
    @location(4) tg: vec4f,
    @location(5) @interpolate(flat) rec: u32,
};
@vertex
fn vs_g(@builtin(vertex_index) vi: u32, @builtin(instance_index) di: u32) -> VG {
    let k = ORD[di];
    let code = u32(vpos[vi].w);
    let p = to_world(k, vpos[vi].xyz);
    var n: vec3f;
    if ((code & 256u) != 0u) {
        n = normal_to_world(k, vnrm[vi].xyz);
    } else {
        let t = (vi / 3u) * 3u;
        let p0 = to_world(k, vpos[t].xyz);
        n = normalize(cross(to_world(k, vpos[t + 1u].xyz) - p0, to_world(k, vpos[t + 2u].xyz) - p0));
    }
    let c = U.vp * vec4f(p, 1.0);
    var o: VG;
    o.pos = vec4f(c.x, c.y, GUIDE_NEAR, c.w);
    o.wp = p;
    o.n = n;
    o.mat = mat_of(k, code);
    let tx = vtex[vi];
    o.uv = tx.xy;
    o.tg = vec4f(dir_to_world(k, oct_decode(tx.zw)), vnrm[vi].w);
    o.rec = k;
    return o;
}
struct GO { @location(0) n: vec4f, @location(1) a: vec4f, @location(2) id: vec4f };
@fragment
fn fs_g(i: VG) -> GO {
    let m = U.mats[i.mat];
    let v = normalize(U.eye.xyz - i.wp);
    var n = normalize(i.n);
    if (dot(n, v) < 0.0) { n = -n; }
    var base = m.base.rgb;
    var metallic = m.pbr.x;
    var rough = m.pbr.y;
    var emis = m.emis.rgb;
    /* the textures at their finest level, as the path tracer reads them (trace.wgsl shade_tex) */
    if (any(m.tex.xyz != vec3u(0u))) {
        let tm = tex_material(m.tex, base, metallic, rough, emis, i.uv, -1000.0);
        base = tm.base; metallic = tm.metallic; rough = tm.rough; emis = tm.emission;
        if ((m.tex.z & 255u) != 0u) {
            n = tex_normal(m.tex, i.uv, -1000.0, n, i.tg);
            let c = dot(n, v);
            if (c < 1e-3) { n = normalize(n + v * (1e-3 - c)); }
        }
    }
    var o: GO;
    o.n = vec4f(n, length(i.wp - U.eye.xyz));
    o.id = vec4f(f32(i.rec), 0.0, 0.0, 0.0);
    if (m.base.w > 0.5 || max(emis.r, max(emis.g, emis.b)) >= 1.0) {
        o.a = vec4f(1.0, 1.0, 1.0, 1.0);          /* a lamp (trace.wgsl: kind 3, or glow_bright) */
    } else {
        var b = bsdf_make(base, metallic, rough, m.pbr.z, m.pbr.w, true);
        o.a = vec4f(base, 0.0);
        if (!bsdf_rough_part(&b)) { o.id = vec4f(-2.0, 0.0, 0.0, 0.0); }
    }
    return o;
}
