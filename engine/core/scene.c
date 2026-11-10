#include "core/scene.h"

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
    for (int k = 0; k < 3; k++) t->n[k] = n ? n[k] : -1;
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
    s->tris = grow(s->tris, &s->cap_tris, s->ntris + m->nf, sizeof(Triangle));
    for (int i = 0; i < m->nf; i++){
        const Face *f = &m->f[i];
        int has_n = f->n[0] >= 0 && f->n[1] >= 0 && f->n[2] >= 0;
        int n[3] = {base + f->n[0], base + f->n[1], base + f->n[2]};
        add_tri(s, m->v[f->v[0]], m->v[f->v[1]], m->v[f->v[2]], has_n ? n : NULL, material);
    }
    return m->nf;
}

int scene_load_mesh(Scene *s, const char *path, int material, real size, Vec3 base, real rot_y_deg){
    Mesh *m = mesh_load(path);
    if (!m) return -1;
    mesh_place(m, size, base, rot_y_deg);
    int n = scene_add_mesh(s, m, material);
    mesh_destroy(m);
    return n;
}

static Material diffuse(real r, real g, real b){
    Material m = {MAT_DIFFUSE, vec3(r, g, b), vec3(0, 0, 0), 1.0};
    return m;
}

// rotate around the vertical (y) axis
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
int scene_cornell_box(Scene *s, const SceneOptions *opt){
    int red   = scene_add_material(s, diffuse(0.65, 0.05, 0.05));
    int white = scene_add_material(s, diffuse(0.73, 0.73, 0.73));
    int green = scene_add_material(s, diffuse(0.12, 0.45, 0.15));
    Material lm = {MAT_LIGHT, vec3(0, 0, 0), vec3(15, 15, 15), 1.0};
    int light = scene_add_material(s, lm);

    scene_add_prim(s, make_quad(vec3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red));     // left wall
    scene_add_prim(s, make_quad(vec3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), green));     // right wall
    scene_add_prim(s, make_quad(vec3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), white));     // floor
    scene_add_prim(s, make_quad(vec3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), white)); // ceiling
    scene_add_prim(s, make_quad(vec3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), white));   // back wall

    // ceiling light, 1 unit below the ceiling. u x v points DOWN (-y): that is the side that emits.
    s->light = scene_add_prim(s, make_quad(vec3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), light));

    if (opt->kind == SCENE_BOXES){
        add_box(s, vec3(165, 330, 165), 15.0, vec3(265, 0, 295), white);   // tall box, back left
        add_box(s, vec3(165, 165, 165), -18.0, vec3(130, 0, 65), white);   // short box, front right
    } else if (opt->kind == SCENE_SPHERES){
        Material mirror = {MAT_MIRROR, vec3(0.95, 0.95, 0.95), vec3(0, 0, 0), 1.0};
        Material glass  = {MAT_GLASS,  vec3(1, 1, 1),          vec3(0, 0, 0), 1.5};
        scene_add_prim(s, make_sphere(vec3(370, 100, 370), 100, scene_add_material(s, mirror)));
        scene_add_prim(s, make_sphere(vec3(185, 100, 170), 100, scene_add_material(s, glass)));
    } else if (opt->kind == SCENE_MESH){
        Material m = diffuse(0.80, 0.55, 0.25);                       // warm orange
        if (opt->mesh_mat == MAT_MIRROR) m = (Material){MAT_MIRROR, vec3(0.95, 0.95, 0.95), vec3(0, 0, 0), 1.0};
        if (opt->mesh_mat == MAT_GLASS)  m = (Material){MAT_GLASS,  vec3(1, 1, 1),          vec3(0, 0, 0), 1.5};
        // largest side 330 units, standing on the floor in the middle of the box
        int n = scene_load_mesh(s, opt->obj_path, scene_add_material(s, m), 330.0, vec3(278, 0, 278), opt->rot_deg);
        if (n < 0){
            fprintf(stderr, "could not load %s\n", opt->obj_path);
            return 1;
        }
    }

    // the cloud: floating in the middle of the room, below the lamp, so its shadow falls
    // on the floor and its top is lit from above
    if (opt->cloud || opt->kind == SCENE_CLOUD){
        CloudParams cp = opt->cloud ? *opt->cloud : cloud_default_params();
        Aabb b = {vec3(70, 150, 110), vec3(485, 440, 450)};
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
    return 0;
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

void scene_free(Scene *s){
    for (int i = 0; i < s->nvolumes; i++) volume_free(s->volumes[i]);
    s->nvolumes = 0;
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
}
