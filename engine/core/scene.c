#include "core/scene.h"
#include "core/gltf.h"

#include "core/env.h"

#include "core/volume.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- primitives ---------------------------------------------------------------------

Primitive make_sphere(Vec3 center, real radius, int material){
    Primitive p;
    memset(&p, 0, sizeof(p));
    p.type = PRIM_SPHERE;
    p.material = material;
    p.center = center;
    p.radius = radius;
    return p;
}

Primitive make_quad(Vec3 q, Vec3 u, Vec3 v, int material){
    Primitive p;
    memset(&p, 0, sizeof(p));
    p.type = PRIM_QUAD;
    p.material = material;
    p.q = q;
    p.u = u;
    p.v = v;
    Vec3 n = v_cross(u, v);
    p.normal = v_norm(n);
    p.plane_d = v_dot(p.normal, q);
    p.w = v_scale(n, 1.0 / v_dot(n, n));
    p.area = v_len(n);
    return p;
}

static void grow_box(Aabb *b, Vec3 p){
    b->lo = vec3(r_fmin(b->lo.x, p.x), r_fmin(b->lo.y, p.y), r_fmin(b->lo.z, p.z));
    b->hi = vec3(r_fmax(b->hi.x, p.x), r_fmax(b->hi.y, p.y), r_fmax(b->hi.z, p.z));
}

Aabb primitive_bounds(const Primitive *prim){
    Aabb b;
    if (prim->type == PRIM_SPHERE){
        Vec3 r = vec3(prim->radius, prim->radius, prim->radius);
        b.lo = v_sub(prim->center, r);
        b.hi = v_add(prim->center, r);
    } else {
        b.lo = b.hi = prim->q;
        grow_box(&b, v_add(prim->q, prim->u));
        grow_box(&b, v_add(prim->q, prim->v));
        grow_box(&b, v_add(prim->q, v_add(prim->u, prim->v)));
    }
    Vec3 pad = vec3(1e-3, 1e-3, 1e-3);   // a wall is flat: give its box some thickness
    b.lo = v_sub(b.lo, pad);
    b.hi = v_add(b.hi, pad);
    return b;
}

Aabb triangle_bounds(const Triangle *t){
    Aabb b;
    b.lo = b.hi = t->q;
    grow_box(&b, v_add(t->q, t->u));
    grow_box(&b, v_add(t->q, t->v));
    Vec3 pad = vec3(1e-3, 1e-3, 1e-3);   // a flat triangle's box gets some thickness too
    b.lo = v_sub(b.lo, pad);
    b.hi = v_add(b.hi, pad);
    return b;
}

Aabb scene_shape_bounds(const Scene *s, int id){
    return id < s->nprims ? primitive_bounds(&s->prims[id]) : triangle_bounds(&s->tris[id - s->nprims]);
}

// ---- building scenes ----------------------------------------------------------------

int scene_add_material(Scene *s, Material m){
    if (s->nmats >= MAX_MATS){ fprintf(stderr, "too many materials\n"); exit(1); }
    s->mats[s->nmats] = m;
    return s->nmats++;
}

int scene_add_prim(Scene *s, Primitive p){
    if (s->nprims >= s->cap_prims){
        s->cap_prims = s->cap_prims ? s->cap_prims * 2 : 64;
        s->prims = realloc(s->prims, (size_t)s->cap_prims * sizeof(Primitive));
        if (!s->prims){ perror("realloc"); exit(1); }
    }
    s->prims[s->nprims] = p;
    return s->nprims++;
}

static void *grow(void *p, int *cap, int need, size_t elem){
    if (need <= *cap) return p;
    int c = *cap ? *cap : 64;
    while (c < need) c *= 2;
    p = realloc(p, (size_t)c * elem);
    if (!p){ perror("realloc"); exit(1); }
    *cap = c;
    return p;
}

// a triangle with normal indices n0..n2 (or -1)
static int add_tri(Scene *s, Vec3 a, Vec3 b, Vec3 c, const int *n, int material){
    s->tris = grow(s->tris, &s->cap_tris, s->ntris + 1, sizeof(Triangle));
    Triangle *t = &s->tris[s->ntris];
    t->q = a;
    t->u = v_sub(b, a);
    t->v = v_sub(c, a);
    for (int k = 0; k < 3; k++){ t->n[k] = n ? n[k] : -1; t->t[k] = -1; }
    t->material = material;
    return s->ntris++;
}

static int add_normal(Scene *s, Vec3 n){
    s->normals = grow(s->normals, &s->cap_normals, s->nnormals + 1, sizeof(Vec3));
    s->normals[s->nnormals] = v_norm(n);
    return s->nnormals++;
}

int scene_add_triangle(Scene *s, Vec3 a, Vec3 b, Vec3 c, const Vec3 *normals, int material){
    if (!normals) return add_tri(s, a, b, c, NULL, material);
    int n[3];
    for (int k = 0; k < 3; k++) n[k] = add_normal(s, normals[k]);
    return add_tri(s, a, b, c, n, material);
}

int scene_add_mesh(Scene *s, const Mesh *m, int material){
    // the mesh's vertex normals, normalized once, shared by all its triangles
    int base = s->nnormals;
    s->normals = grow(s->normals, &s->cap_normals, s->nnormals + m->nvn, sizeof(Vec3));
    for (int i = 0; i < m->nvn; i++) s->normals[s->nnormals++] = v_norm(m->vn[i]);
    // its texture coordinates (v turned over: OBJ's v = 0 is the bottom, ours the top)
    int tbase = s->nuvs;
    s->uvs = grow(s->uvs, &s->cap_uvs, s->nuvs + m->nvt, sizeof(Vec2));
    s->tangents = realloc(s->tangents, (size_t)(s->cap_uvs ? s->cap_uvs : 1) * sizeof(Vec4));
    if (!s->tangents){ perror("realloc"); exit(1); }
    for (int i = 0; i < m->nvt; i++) s->uvs[s->nuvs++] = vec2(m->vt[i].x, 1 - m->vt[i].y);
    int first = s->ntris;
    s->tris = grow(s->tris, &s->cap_tris, s->ntris + m->nf, sizeof(Triangle));
    for (int i = 0; i < m->nf; i++){
        const Face *f = &m->f[i];
        int has_n = f->n[0] >= 0 && f->n[1] >= 0 && f->n[2] >= 0;
        int n[3] = {base + f->n[0], base + f->n[1], base + f->n[2]};
        int k = add_tri(s, m->v[f->v[0]], m->v[f->v[1]], m->v[f->v[2]], has_n ? n : NULL, material);
        if (f->t[0] >= 0 && f->t[1] >= 0 && f->t[2] >= 0)
            for (int j = 0; j < 3; j++) s->tris[k].t[j] = tbase + f->t[j];
    }
    scene_compute_tangents(s->tris + first, m->nf, s->uvs, s->tangents, tbase, m->nvt);
    return m->nf;
}

void scene_compute_tangents(const Triangle *tris, int ntris, const Vec2 *uvs, Vec4 *tangents, int base, int n){
    if (n <= 0) return;
    Vec3 *tu = calloc((size_t)n, sizeof(Vec3)), *up = calloc((size_t)n, sizeof(Vec3)), *nn = calloc((size_t)n, sizeof(Vec3));
    if (!tu || !up || !nn){ perror("calloc"); exit(1); }
    for (int i = 0; i < ntris; i++){
        const Triangle *t = &tris[i];
        if (t->t[0] < base || t->t[0] >= base + n) continue;
        Vec2 a = uvs[t->t[0]], b = uvs[t->t[1]], c = uvs[t->t[2]];
        real du1 = b.x - a.x, dv1 = b.y - a.y, du2 = c.x - a.x, dv2 = c.y - a.y;
        real det = du1 * dv2 - du2 * dv1;
        if (r_fabs(det) < 1e-12f) continue;
        real r = 1 / det;
        // dP/du and dP/dv from the two edges (P = q + s u + t v in the uv plane)
        Vec3 dpdu = v_scale(v_sub(v_scale(t->u, dv2), v_scale(t->v, dv1)), r);
        Vec3 dpdv = v_scale(v_sub(v_scale(t->v, du1), v_scale(t->u, du2)), r);
        Vec3 fn = v_cross(t->u, t->v);
        for (int k = 0; k < 3; k++){
            int j = t->t[k] - base;
            tu[j] = v_add(tu[j], dpdu);
            up[j] = v_sub(up[j], dpdv);                  // up the image: -dP/dv
            nn[j] = v_add(nn[j], fn);
        }
    }
    for (int j = 0; j < n; j++){
        real l = v_len(tu[j]);
        Vec3 tt = l > 0 ? v_scale(tu[j], 1 / l) : vec3(1, 0, 0);
        real w = v_dot(v_cross(nn[j], tt), up[j]) < 0 ? -1 : 1;
        tangents[base + j] = (Vec4){tt.x, tt.y, tt.z, w};
    }
    free(tu); free(up); free(nn);
}

// numbers no two texture sets share, across scenes (a scene made again at the same address
// with as many textures must not look unchanged to the GPU)
static unsigned next_tex_version(void){
    static unsigned v;
    return ++v;
}

int scene_texture(Scene *s, const char *name, int srgb){
    if (!name || !*name) return 0;
    for (int i = 0; i < s->ntexmaps; i++)
        if (!strcmp(s->texmaps[i].name, name) && s->texmaps[i].srgb == srgb) return i + 1;
    TexMap t;
    size_t len = strlen(name);
    int file = (len > 4 && (!strcmp(name + len - 4, ".png") || !strcmp(name + len - 4, ".jpg"))) || (len > 5 && !strcmp(name + len - 5, ".jpeg"));
    int err = file ? texmap_load(&t, name, srgb) : texmap_procedural(&t, name);
    if (err) return 0;
    if (strlen(name) < sizeof(t.name)) snprintf(t.name, sizeof(t.name), "%s", name);
    s->texmaps = grow(s->texmaps, &s->cap_texmaps, s->ntexmaps + 1, sizeof(TexMap));
    s->texmaps[s->ntexmaps] = t;
    s->tex_version = next_tex_version();
    return ++s->ntexmaps;
}

int scene_add_texmap(Scene *s, const TexMap *t){
    s->texmaps = grow(s->texmaps, &s->cap_texmaps, s->ntexmaps + 1, sizeof(TexMap));
    s->texmaps[s->ntexmaps] = *t;
    s->tex_version = next_tex_version();
    return ++s->ntexmaps;
}

static inline Vec2 uv_scaled(const Material *m, Vec2 uv){
    real k = m->uv_scale > 0 ? m->uv_scale : 1;
    return vec2(uv.x * k, uv.y * k);
}

// texture t at uv for a footprint of 2^uv_lod texture-coordinate units (times the scale)
static Vec4 tex_at(const TexMap *t, const Material *m, Vec2 uv, real uv_lod){
    real lod = MATERIAL_LOD0;
    if (uv_lod > MATERIAL_LOD0 * 0.5f){
        real k = m->uv_scale > 0 ? m->uv_scale : 1;
        lod = uv_lod + r_log2(k * (real)(t->w > t->h ? t->w : t->h));
    }
    return texmap_sample(t, uv.x, uv.y, lod);
}

Material material_at(const Scene *s, const Material *m, Vec2 uv, real uv_lod){
    if (!material_textured(m) || m->type == MAT_LIGHT) return *m;
    Material r = material_resolved(m);
    uv = uv_scaled(m, uv);
    if (m->tex_base > 0 && m->tex_base <= s->ntexmaps){
        Vec4 c = tex_at(&s->texmaps[m->tex_base - 1], m, uv, uv_lod);
        r.albedo = vec3(r.albedo.x * c.x, r.albedo.y * c.y, r.albedo.z * c.z);
    }
    if (m->tex_mr > 0 && m->tex_mr <= s->ntexmaps){
        Vec4 c = tex_at(&s->texmaps[m->tex_mr - 1], m, uv, uv_lod);
        r.roughness *= c.y;
        r.metallic *= c.z;
    }
    if (m->tex_emissive > 0 && m->tex_emissive <= s->ntexmaps){
        Vec4 c = tex_at(&s->texmaps[m->tex_emissive - 1], m, uv, uv_lod);
        r.emission = vec3(r.emission.x * c.x, r.emission.y * c.y, r.emission.z * c.z);
    }
    r.type = MAT_PBR;
    r.tex_base = r.tex_mr = r.tex_normal = r.tex_emissive = r.tex_occlusion = 0;
    return r;
}

real material_occlusion(const Scene *s, const Material *m, Vec2 uv, real uv_lod){
    if (!(m->tex_occlusion > 0 && m->tex_occlusion <= s->ntexmaps)) return 1;
    real k = m->occlusion > 0 ? m->occlusion : 1;
    Vec4 c = tex_at(&s->texmaps[m->tex_occlusion - 1], m, uv_scaled(m, uv), uv_lod);
    return 1 + k * (c.x - 1);
}

Vec3 material_normal(const Scene *s, const Material *m, Vec2 uv, real uv_lod, Vec3 n, Vec4 t){
    if (!(m->tex_normal > 0 && m->tex_normal <= s->ntexmaps)) return n;
    uv = uv_scaled(m, uv);
    Vec4 c = tex_at(&s->texmaps[m->tex_normal - 1], m, uv, uv_lod);
    Vec3 tt = v_sub(vec3(t.x, t.y, t.z), v_scale(n, v_dot(n, vec3(t.x, t.y, t.z))));
    real l = v_len(tt);
    if (!(l > 1e-6f)) return n;
    tt = v_scale(tt, 1 / l);
    Vec3 b = v_scale(v_cross(n, tt), t.w);
    real k = m->normal_scale > 0 ? m->normal_scale : 1;
    Vec3 out = v_add(v_add(v_scale(tt, k * (2 * c.x - 1)), v_scale(b, k * (2 * c.y - 1))), v_scale(n, 2 * c.z - 1));
    real lo = v_len(out);
    return lo > 1e-6f ? v_scale(out, 1 / lo) : n;
}

int scene_load_mesh(Scene *s, const char *path, int material, real size, Vec3 base, real rot_y_deg){
    Mesh *m = mesh_load(path);
    if (!m) return -1;
    mesh_place(m, size, base, rot_y_deg);
    int n = scene_add_mesh(s, m, material);
    mesh_destroy(m);
    return n;
}

static Material diffuse(real r, real g, real b){ return material_diffuse(vec3(r, g, b)); }

// rotate around the vertical (y) axis
static Material mesh_material(MatType t){
    if (t == MAT_MIRROR) return material_mirror(vec3(0.95, 0.95, 0.95));
    if (t == MAT_GLASS) return material_glass(vec3(1, 1, 1), 1.5);
    return diffuse(0.80, 0.55, 0.25);                       // warm orange
}

Material scene_mesh_material_tex(Scene *s, const SceneOptions *opt){
    Material m = scene_mesh_material(opt);
    if (!opt->mesh_tex && !opt->mesh_mr && !opt->mesh_normal) return m;
    if (!(opt->mesh_set & MESH_SET_BASE) && m.type != MAT_GLASS) m.albedo = vec3(1, 1, 1);   // the texture's colour as it is
    int bricks = opt->mesh_tex && !strcmp(opt->mesh_tex, "bricks");
    m.tex_base = scene_texture(s, opt->mesh_tex, 1);
    m.tex_mr = scene_texture(s, opt->mesh_mr ? opt->mesh_mr : (bricks ? "bricks_mr" : NULL), 0);
    m.tex_normal = scene_texture(s, opt->mesh_normal ? opt->mesh_normal : (bricks ? "bricks_normal" : NULL), 0);
    if (m.tex_mr){                             // its factors: the texture's values as they are
        m = material_resolved(&m);
        m.type = MAT_PBR;
        if (!(opt->mesh_set & MESH_SET_METAL)) m.metallic = 1;
        if (!(opt->mesh_set & MESH_SET_ROUGH)) m.roughness = 1;
    }
    m.uv_scale = opt->mesh_uv_scale;
    return m;
}

Material scene_mesh_material(const SceneOptions *opt){
    Material m = mesh_material(opt->mesh_mat);
    if (!opt->mesh_set) return m;
    m = material_resolved(&m);
    m.type = MAT_PBR;
    if (opt->mesh_set & MESH_SET_BASE) m.albedo = opt->mesh_base;
    if (opt->mesh_set & MESH_SET_METAL) m.metallic = opt->mesh_metal;
    if (opt->mesh_set & MESH_SET_ROUGH) m.roughness = opt->mesh_rough;
    if (opt->mesh_set & MESH_SET_IOR) m.ior = opt->mesh_ior;
    if (opt->mesh_set & MESH_SET_TRANS) m.transmission = opt->mesh_trans;
    return m;
}

static Vec3 rotate_y(Vec3 p, real deg){
    real a = deg * PI / 180.0;
    real c = r_cos(a), s = r_sin(a);
    return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

// An axis-aligned box from (0,0,0) to size, rotated around y by deg, then moved by offset.
// Made of 6 quads; we transform the quad corners and edges instead of the rays.
static void add_box(Scene *s, Vec3 size, real deg, Vec3 offset, int mat){
    Vec3 dx = vec3(size.x, 0, 0), dy = vec3(0, size.y, 0), dz = vec3(0, 0, size.z);
    Vec3 corner[6] = { vec3(0, 0, 0), vec3(0, 0, size.z), vec3(0, 0, 0),
                       vec3(size.x, 0, 0), vec3(0, 0, 0), vec3(0, size.y, 0) };
    Vec3 eu[6] = { dx, dx, dz, dz, dx, dx };
    Vec3 ev[6] = { dy, dy, dy, dy, dz, dz };
    for (int i = 0; i < 6; i++){
        Vec3 q = v_add(rotate_y(corner[i], deg), offset);
        scene_add_prim(s, make_quad(q, rotate_y(eu[i], deg), rotate_y(ev[i], deg), mat));
    }
}

// The Cornell box, 555 units on each side, with the dimensions and colors used by
// Peter Shirley's "Ray Tracing: The Next Week" (based on the Cornell measurements).
// The camera looks down +z, so x = 555 shows up on the LEFT of the image.
// SCENE_SKY: a floor under the sky, five spheres of five materials on it
static int open_scene(Scene *s, int spheres){
    // the floor: brick paving (PBR5's textures: colour, normal map, roughness), a repeat
    // every 450 units
    Material floor = material_pbr(vec3(0.85, 0.85, 0.85), 1, 1, 1.5, 0);
    floor.tex_base = scene_texture(s, "bricks", 1);
    floor.tex_normal = scene_texture(s, "bricks_normal", 0);
    floor.tex_mr = scene_texture(s, "bricks_mr", 0);
    floor.uv_scale = 40;
    int ground = scene_add_material(s, floor);
    s->light = -1;
    scene_add_prim(s, make_quad(vec3(-8722, 0, -8722), vec3(0, 0, 18000), vec3(18000, 0, 0), ground));
    Material m[5] = {
        material_mirror(vec3(0.95, 0.95, 0.95)),
        material_pbr(vec3(1.0, 0.78, 0.34), 1, 0.3, 1.5, 0),       // rough gold
        material_glass(vec3(1, 1, 1), 1.5),
        material_pbr(vec3(0.1, 0.25, 0.7), 0, 0.4, 1.5, 0),        // blue plastic
        diffuse(0.8, 0.8, 0.8),
    };
    for (int i = 0; i < 5 && spheres; i++)
        scene_add_prim(s, make_sphere(vec3(278 + (i - 2) * 165.0f, 70, 300), 70, scene_add_material(s, m[i])));
    s->probe_set = 1;
    s->probe_at = vec3(278, 230, 300);
    s->probe_box = (Aabb){vec3(-8722, 0, -8722), vec3(9278, 1e5f, 9278)};
    s->cam_from = vec3(278, 300, -640);
    s->cam_at = vec3(278, 40, 300);
    s->cam_up = vec3(0, 1, 0);
    s->cam_vfov_deg = 45.0;
    return 0;
}

// the sky of opt (SceneOptions.sky)
static int scene_sky(Scene *s, const SceneOptions *opt){
    const char *sky = opt->sky ? opt->sky : (opt->kind == SCENE_SKY ? "procedural" : NULL);
    if (!sky) return 0;
    Env *e = strcmp(sky, "procedural") == 0 ? env_procedural_sky(1024, 35, -30) : env_load_hdr(sky);
    if (!e){ fprintf(stderr, "could not make the sky %s\n", sky); return 1; }
    e->scale = opt->sky_scale > 0 ? opt->sky_scale : 1;
    env_set_rot(e, opt->sky_rot * (real)PI / 180);
    if (s->env) env_free(s->env);
    s->env = e;
    return 0;
}

int scene_cornell_box(Scene *s, const SceneOptions *opt){
    if (opt->kind == SCENE_SKY){
        if (open_scene(s, !opt->gltf_path) || scene_sky(s, opt)) return 1;
        if (opt->gltf_path && scene_load_gltf(s, opt->gltf_path, 330, vec3(278, 0, 300), 180 + opt->rot_deg) < 0) return 1;
        return 0;
    }
    int red   = scene_add_material(s, diffuse(0.65, 0.05, 0.05));
    int white = scene_add_material(s, diffuse(0.73, 0.73, 0.73));
    int green = scene_add_material(s, diffuse(0.12, 0.45, 0.15));
    int light = scene_add_material(s, material_light(vec3(15, 15, 15)));

    scene_add_prim(s, make_quad(vec3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red));     // left wall
    scene_add_prim(s, make_quad(vec3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), green));     // right wall
    scene_add_prim(s, make_quad(vec3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), white));     // floor
    scene_add_prim(s, make_quad(vec3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), white)); // ceiling
    scene_add_prim(s, make_quad(vec3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), white));   // back wall

    s->probe_set = 1;
    s->probe_at = vec3(278, 278, 278);
    s->probe_box = (Aabb){vec3(0, 0, 0), vec3(555, 555, 555)};

    // ceiling light, 1 unit below the ceiling. u x v points DOWN (-y): that is the side that emits.
    s->light = scene_add_prim(s, make_quad(vec3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), light));

    if (opt->kind == SCENE_BOXES){
        add_box(s, vec3(165, 330, 165), 15.0, vec3(265, 0, 295), white);   // tall box, back left
        add_box(s, vec3(165, 165, 165), -18.0, vec3(130, 0, 65), white);   // short box, front right
    } else if (opt->kind == SCENE_SPHERES){
        Material mirror = material_mirror(vec3(0.95, 0.95, 0.95));
        Material glass  = material_glass(vec3(1, 1, 1), 1.5);
        scene_add_prim(s, make_sphere(vec3(370, 100, 370), 100, scene_add_material(s, mirror)));
        scene_add_prim(s, make_sphere(vec3(185, 100, 170), 100, scene_add_material(s, glass)));
    } else if (opt->kind == SCENE_MESH && opt->gltf_path){
        // the glTF model, as the .obj one would be (its front, +z, to the camera)
        if (scene_load_gltf(s, opt->gltf_path, 330.0, vec3(278, 0, 278), 180 + opt->rot_deg) < 0) return 1;
    } else if (opt->kind == SCENE_MESH && opt->instances <= 0){
        // largest side 330 units, standing on the floor in the middle of the box
        int n = scene_load_mesh(s, opt->obj_path, scene_add_material(s, scene_mesh_material_tex(s, opt)), 330.0,
                                vec3(278, 0, 278), opt->rot_deg);
        if (n < 0){
            fprintf(stderr, "could not load %s\n", opt->obj_path);
            return 1;
        }
    }
    if (opt->kind == SCENE_SHOWCASE){
        // one model of each material, each its own mesh and instance: the dragon at the back,
        // the cube in front on the right, the torus standing up in front on the left (turned
        // up about x, lifted by its radius); the cloud (below) floats above them all
        static const struct { const char *path; real size, x, y, z, rot, up; } m[3] = {
            {"models/dragon_recon/dragon_vrip.ply", 300, 278, 0, 390, -35, 0},
            {"models/cube.obj", 105, 135, 0, 170, 30, 0},
            {"models/torus.obj", 160, 405, 81, 190, 25, 90},
        };
        Material mats[3] = {mesh_material(MAT_MIRROR), diffuse(0.25, 0.45, 0.75), mesh_material(MAT_GLASS)};
        // the cube's place takes a glTF model when it is there (PBR6: make gltf-download):
        // Khronos's damaged helmet, textured and glowing, turned to the camera
        static const char *helmet = "models/gltf/DamagedHelmet.glb";
        FILE *hf = fopen(helmet, "rb");
        int gltf = hf != NULL;
        if (hf) fclose(hf);
        for (int i = 0; i < 3; i++){
            if (i == 1 && gltf){
                int ok = scene_load_gltf(s, helmet, 130, vec3(140, 0, 165), 180 + 25) >= 0;
                if (ok) continue;
                gltf = 0;
            }
            int mesh = scene_load_mesh_object(s, m[i].path, scene_add_material(s, mats[i]), m[i].size);
            if (mesh < 0){
                fprintf(stderr, "could not load %s (run from the repository's folder; make models)\n", m[i].path);
                return 1;
            }
            Quat q = quat_mul(quat_axis_angle(vec3(0, 1, 0), m[i].rot * (real)PI / 180),
                              quat_axis_angle(vec3(1, 0, 0), m[i].up * (real)PI / 180));
            scene_add_instance(s, mesh, xform_make(vec3(m[i].x, m[i].y, m[i].z), quat_to_mat3(q).m, 1), -1);
        }
        printf("showcase: a mirror dragon (%d triangles), %s, a glass torus\n", s->meshes[0].ntris,
               gltf ? "a glTF model (models/gltf/DamagedHelmet.glb)" : "a diffuse cube");
    }
    if (opt->instances > 0){
        // one mesh, opt->instances instances of it in a row on the floor, each turned a little
        // more (SCENE_SPHERES: in the free corners first, beside the spheres; SCENE_MESH: one,
        // as its model)
        int n = opt->instances;
        int sph = opt->kind == SCENE_SPHERES;
        real size = sph ? 180.0f : 330.0f;
        if (size > 520.0f / n) size = 520.0f / n;
        int mesh = scene_load_mesh_object(s, opt->obj_path, scene_add_material(s, scene_mesh_material_tex(s, opt)), size);
        if (mesh < 0){
            fprintf(stderr, "could not load %s\n", opt->obj_path);
            return 1;
        }
        static const real corner[4][2] = {{420, 140}, {140, 420}, {90, 110}, {460, 470}};
        for (int i = 0; i < n; i++){
            Vec3 at = vec3(555.0f * (i + 0.5f) / n, 0, 278.0f);
            if (sph) at = i < 4 ? vec3(corner[i][0], 0, corner[i][1]) : vec3(555.0f * (i + 0.5f) / n, 0, 500.0f);
            scene_add_instance(s, mesh, xform_rot_y(at, opt->rot_deg + 35.0f * i, 1), -1);
        }
    }

    // the cloud: floating in the middle of the room, below the lamp, so its shadow falls
    // on the floor and its top is lit from above
    if (opt->cloud || opt->kind == SCENE_CLOUD || opt->kind == SCENE_SHOWCASE){
        CloudParams cp = opt->cloud ? *opt->cloud : cloud_default_params();
        Aabb b = {vec3(70, 150, 110), vec3(485, 440, 450)};
        if (opt->kind == SCENE_SHOWCASE) b = (Aabb){vec3(110, 330, 190), vec3(470, 505, 500)};   // above the dragon
        Volume *v = volume_new_cloud(b, &cp);
        if (!v){ fprintf(stderr, "out of memory for the cloud\n"); return 1; }
        scene_add_volume(s, v);
    }
    // fog: a uniform medium filling the room. The grid reaches 30 units past the walls, so
    // the fade of the interpolation at the grid's border stays outside the room.
    if (opt->fog > 0){
        Aabb b = {vec3(-30, -30, -30), vec3(585, 585, 585)};
        Volume *v = volume_create(16, 16, 16, b);
        if (!v){ fprintf(stderr, "out of memory for the fog\n"); return 1; }
        for (int i = 0; i < 16 * 16 * 16; i++) v->density[i] = 1;
        v->sigma_t = opt->fog;
        v->albedo = vec3(0.95, 0.95, 0.95);   // haze: scatters most, absorbs a little
        v->g = 0.4;                           // aerosols scatter mostly forward
        volume_update(v);
        scene_add_volume(s, v);
    }

    s->cam_from = vec3(278, 278, -800);
    s->cam_at = vec3(278, 278, 0);
    s->cam_up = vec3(0, 1, 0);
    s->cam_vfov_deg = 40.0;
    return scene_sky(s, opt);
}

Vec3 scene_light_center(const Scene *s){
    if (s->light < 0 || s->light >= s->nprims) return vec3(0, 1000, 0);
    const Primitive *l = &s->prims[s->light];
    if (l->type == PRIM_SPHERE) return l->center;
    return v_add(l->q, v_scale(v_add(l->u, l->v), 0.5));
}

int scene_add_volume(Scene *s, Volume *v){
    if (s->nvolumes >= MAX_VOLUMES){
        fprintf(stderr, "too many media (max %d)\n", MAX_VOLUMES);
        volume_free(v);
        return -1;
    }
    s->volumes[s->nvolumes] = v;
    return s->nvolumes++;
}

// ---- instances ---------------------------------------------------------------------------

Xform xform_identity(void){
    Xform x;
    memset(&x, 0, sizeof(x));
    x.m[0][0] = x.m[1][1] = x.m[2][2] = 1;
    return x;
}

Xform xform_make(Vec3 t, const real r[3][3], real scale){
    Xform x;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) x.m[i][j] = r[i][j] * scale;
    x.m[0][3] = t.x; x.m[1][3] = t.y; x.m[2][3] = t.z;
    return x;
}

Xform xform_rot_y(Vec3 t, real deg, real scale){
    real a = deg * (real)PI / 180, c = r_cos(a), sn = r_sin(a);
    real r[3][3] = {{c, 0, sn}, {0, 1, 0}, {-sn, 0, c}};
    return xform_make(t, r, scale);
}

Xform xform_inverse(const Xform *x){
    const real (*m)[4] = x->m;
    real det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    real id = det != 0 ? 1 / det : 0;
    Xform r;
    r.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id;
    r.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id;
    r.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id;
    r.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id;
    r.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id;
    r.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id;
    r.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id;
    r.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id;
    r.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id;
    for (int i = 0; i < 3; i++)
        r.m[i][3] = -(r.m[i][0] * m[0][3] + r.m[i][1] * m[1][3] + r.m[i][2] * m[2][3]);
    return r;
}

Aabb xform_box(const Xform *x, Aabb b){
    Aabb o = {vec3(1e30f, 1e30f, 1e30f), vec3(-1e30f, -1e30f, -1e30f)};
    for (int c = 0; c < 8; c++){
        Vec3 p = vec3(c & 1 ? b.hi.x : b.lo.x, c & 2 ? b.hi.y : b.lo.y, c & 4 ? b.hi.z : b.lo.z);
        grow_box(&o, xform_point(x, p));
    }
    return o;
}

static void mesh_bounds(SceneMesh *m){
    m->box = (Aabb){vec3(1e30f, 1e30f, 1e30f), vec3(-1e30f, -1e30f, -1e30f)};
    for (int i = 0; i < m->ntris; i++){
        const Triangle *t = &m->tris[i];
        grow_box(&m->box, t->q);
        grow_box(&m->box, v_add(t->q, t->u));
        grow_box(&m->box, v_add(t->q, t->v));
    }
}

int scene_add_mesh_object(Scene *s, const Mesh *src, int material){
    s->meshes = grow(s->meshes, &s->cap_meshes, s->nmeshes + 1, sizeof(SceneMesh));
    SceneMesh *m = &s->meshes[s->nmeshes];
    memset(m, 0, sizeof(*m));
    m->normals = malloc((size_t)(src->nvn ? src->nvn : 1) * sizeof(Vec3));
    m->tris = malloc((size_t)(src->nf ? src->nf : 1) * sizeof(Triangle));
    if (!m->normals || !m->tris){ perror("malloc"); exit(1); }
    for (int i = 0; i < src->nvn; i++) m->normals[i] = v_norm(src->vn[i]);
    m->nnormals = src->nvn;
    for (int i = 0; i < src->nf; i++){
        const Face *f = &src->f[i];
        Triangle *t = &m->tris[i];
        Vec3 a = src->v[f->v[0]], b = src->v[f->v[1]], c = src->v[f->v[2]];
        t->q = a; t->u = v_sub(b, a); t->v = v_sub(c, a);
        int has_n = f->n[0] >= 0 && f->n[1] >= 0 && f->n[2] >= 0;
        int has_t = f->t[0] >= 0 && f->t[1] >= 0 && f->t[2] >= 0;
        for (int k = 0; k < 3; k++){ t->n[k] = has_n ? f->n[k] : -1; t->t[k] = has_t ? f->t[k] : -1; }
        t->material = material;
    }
    m->ntris = src->nf;
    if (src->nvt > 0){                       // texture coordinates (v turned over) and tangents
        m->uvs = malloc((size_t)src->nvt * sizeof(Vec2));
        m->tangents = malloc((size_t)src->nvt * sizeof(Vec4));
        if (!m->uvs || !m->tangents){ perror("malloc"); exit(1); }
        for (int i = 0; i < src->nvt; i++) m->uvs[i] = vec2(src->vt[i].x, 1 - src->vt[i].y);
        m->nuvs = src->nvt;
        scene_compute_tangents(m->tris, m->ntris, m->uvs, m->tangents, 0, m->nuvs);
    }
    m->version = 1;
    mesh_bounds(m);
    return s->nmeshes++;
}

int scene_add_mesh_arrays(Scene *s, Triangle *tris, int ntris, Vec3 *normals, int nnormals,
                          Vec2 *uvs, Vec4 *tangents, int nuvs){
    s->meshes = grow(s->meshes, &s->cap_meshes, s->nmeshes + 1, sizeof(SceneMesh));
    SceneMesh *m = &s->meshes[s->nmeshes];
    memset(m, 0, sizeof(*m));
    m->tris = tris; m->ntris = ntris;
    m->normals = normals; m->nnormals = nnormals;
    if (uvs && nuvs > 0){
        m->uvs = uvs;
        m->nuvs = nuvs;
        m->tangents = tangents;
        if (!m->tangents){
            m->tangents = malloc((size_t)nuvs * sizeof(Vec4));
            if (!m->tangents){ perror("malloc"); exit(1); }
            scene_compute_tangents(tris, ntris, uvs, m->tangents, 0, nuvs);
        }
    } else { free(uvs); free(tangents); }
    m->version = 1;
    mesh_bounds(m);
    return s->nmeshes++;
}

int scene_load_mesh_object(Scene *s, const char *path, int material, real size){
    Mesh *m = mesh_load(path);
    if (!m) return -1;
    mesh_place(m, size, vec3(0, 0, 0), 0);
    int id = scene_add_mesh_object(s, m, material);
    mesh_destroy(m);
    return id;
}

int scene_add_instance(Scene *s, int mesh, Xform xf, int material){
    s->insts = grow(s->insts, &s->cap_insts, s->ninsts + 1, sizeof(Instance));
    Instance *in = &s->insts[s->ninsts];
    in->mesh = mesh;
    in->material = material;
    s->ninsts++;
    scene_move_instance(s, s->ninsts - 1, xf);
    return s->ninsts - 1;
}

void scene_move_instance(Scene *s, int i, Xform xf){
    Instance *in = &s->insts[i];
    in->xf = xf;
    in->inv = xform_inverse(&xf);
    if (in->mesh >= 0) in->box = xform_box(&xf, s->meshes[in->mesh].box);
    else if (in->mesh == INST_BOX) in->box = xform_box(&xf, (Aabb){vec3(-1, -1, -1), vec3(1, 1, 1)});
    else {                                    // a sphere: its centre +- its radius
        Vec3 c = vec3(xf.m[0][3], xf.m[1][3], xf.m[2][3]);
        real r = 0;
        for (int j = 0; j < 3; j++) r = r_fmax(r, v_len(vec3(xf.m[0][j], xf.m[1][j], xf.m[2][j])));
        in->box = (Aabb){v_sub(c, vec3(r, r, r)), v_add(c, vec3(r, r, r))};
    }
    s->inst_version++;
}

Primitive unit_box_face(int f, int material){
    Vec3 e[3] = {vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1)};
    int k = f / 2;
    real side = f & 1 ? 1 : -1;
    Vec3 u = v_scale(e[(k + 1) % 3], 2), v = v_scale(e[(k + 2) % 3], 2);
    Vec3 q = v_sub(v_scale(e[k], side), v_scale(v_add(u, v), 0.5f));
    return side > 0 ? make_quad(q, u, v, material) : make_quad(q, v, u, material);
}

Primitive instance_shape_prim(const Instance *in, int part){
    const Xform *x = &in->xf;
    if (in->mesh == INST_SPHERE){
        real r = 0;
        for (int j = 0; j < 3; j++) r = r_fmax(r, v_len(vec3(x->m[0][j], x->m[1][j], x->m[2][j])));
        return make_sphere(vec3(x->m[0][3], x->m[1][3], x->m[2][3]), r, in->material);
    }
    Primitive f = unit_box_face(part, in->material);
    return make_quad(xform_point(x, f.q), xform_dir(x, f.u), xform_dir(x, f.v), in->material);
}

Xform xform_axes(Mat3 r, Vec3 scale, Vec3 t){
    Xform x;
    real sc[3] = {scale.x, scale.y, scale.z};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) x.m[i][j] = r.m[i][j] * sc[j];
    x.m[0][3] = t.x; x.m[1][3] = t.y; x.m[2][3] = t.z;
    return x;
}

void scene_mesh_changed(Scene *s, int mesh){
    SceneMesh *m = &s->meshes[mesh];
    mesh_bounds(m);
    m->version++;
    for (int i = 0; i < s->ninsts; i++)
        if (s->insts[i].mesh == mesh) s->insts[i].box = xform_box(&s->insts[i].xf, m->box);
}

void scene_free(Scene *s){
    for (int i = 0; i < s->nmeshes; i++){
        SceneMesh *m = &s->meshes[i];
        if (m->bvh && m->bvh_free) m->bvh_free(m->bvh);
        free(m->tris);
        free(m->normals);
        free(m->uvs);
        free(m->tangents);
    }
    free(s->meshes);
    s->meshes = NULL;
    s->nmeshes = s->cap_meshes = 0;
    free(s->insts);
    s->insts = NULL;
    s->ninsts = s->cap_insts = 0;
    for (int i = 0; i < s->nvolumes; i++) volume_free(s->volumes[i]);
    s->nvolumes = 0;
    env_free(s->env);
    env_free(s->probe);
    s->env = s->probe = NULL;
    irr_grid_free(s->irr);
    s->irr = NULL;
    if (s->accel && s->accel_free) s->accel_free(s->accel);
    s->accel = NULL;
    s->accel_free = NULL;
    free(s->prims);
    s->prims = NULL;
    s->nprims = s->cap_prims = 0;
    free(s->tris);
    s->tris = NULL;
    s->ntris = s->cap_tris = 0;
    free(s->normals);
    s->normals = NULL;
    s->nnormals = s->cap_normals = 0;
    free(s->uvs);
    free(s->tangents);
    s->uvs = NULL;
    s->tangents = NULL;
    s->nuvs = s->cap_uvs = 0;
    for (int i = 0; i < s->ntexmaps; i++) texmap_free(&s->texmaps[i]);
    free(s->texmaps);
    s->texmaps = NULL;
    s->ntexmaps = s->cap_texmaps = 0;
}
