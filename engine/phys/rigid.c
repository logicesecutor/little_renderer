#include "phys/rigid.h"

#include "core/rng.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONTACT_SLOP 0.5f        // overlap (units) left alone, so resting contacts do not jitter
#define BAUMGARTE 0.2f           // fraction of the remaining overlap removed per step
#define BOUNCE_THRESHOLD 60.0f   // approach speeds below this (units / s) do not bounce
#define SPECULATIVE_MIN 0.1f     // the smallest margin (units) of a pair's speculative contacts
#define MAX_PUSH 300.0f          // the fastest an overlap is pushed out (units / s; Box2D: 3 m/s)
#define MAX_HULL_VERTS 64        // corners of a hull (rigid_geom_hull)
#define MAX_HULL_FACES 128

// ---- bodies -----------------------------------------------------------------------------

void rigid_init(RigidWorld *w){
    memset(w, 0, sizeof(*w));
    w->gravity = vec3(0, -981, 0);
    w->wall_restitution = 0.4f;
    w->wall_friction = 0.5f;
    w->iterations = 16;
    w->linear_damping = 0.0f;
    w->angular_damping = 0.1f;
    w->warm_start = 1;
}

void rigid_free(RigidWorld *w){
    for (int i = 0; i < w->ngeoms; i++){
        RigidGeom *g = w->geoms[i];
        convex_free(&g->hull);
        mesh_tree_free(&g->tree);
        free(g->occ);
        free(g);
    }
    free(w->geoms);
    free(w->bodies);
    free(w->contacts);
    free(w->prev);
    memset(w, 0, sizeof(*w));
}

void rigid_set_room(RigidWorld *w, Aabb room){
    w->room = room;
    w->has_room = 1;
}

static int add_body(RigidWorld *w, Body b){
    if (w->nbodies >= w->cap){
        w->cap = w->cap ? w->cap * 2 : 16;
        w->bodies = realloc(w->bodies, (size_t)w->cap * sizeof(Body));
        if (!w->bodies){ perror("realloc"); exit(1); }
    }
    w->bodies[w->nbodies] = b;
    return w->nbodies++;
}

static Body body_base(Vec3 pos, real volume, real density){
    Body b;
    memset(&b, 0, sizeof(b));
    b.pos = pos;
    b.rot = quat_identity();
    b.volume = volume;
    b.mass = volume * density;
    b.inv_mass = b.mass > 0 ? 1.0f / b.mass : 0;
    b.restitution = 0.4f;
    b.friction = 0.5f;
    b.prim = -1;
    b.inst = -1;
    b.dirty = 1;
    return b;
}

int rigid_add_sphere(RigidWorld *w, Vec3 center, real radius, real density){
    Body b = body_base(center, 4.0f / 3.0f * (real)PI * radius * radius * radius, density);
    b.shape = BODY_SPHERE;
    b.half = vec3(radius, radius, radius);
    real i = b.mass > 0 ? 1.0f / (0.4f * b.mass * radius * radius) : 0;   // I = 2/5 m r^2
    b.inv_inertia = vec3(i, i, i);
    return add_body(w, b);
}

int rigid_add_box(RigidWorld *w, Vec3 center, Vec3 half, Quat rot, real density){
    Body b = body_base(center, 8.0f * half.x * half.y * half.z, density);
    b.shape = BODY_BOX;
    b.half = half;
    b.rot = quat_norm(rot);
    // I_x = m/12 * ((2 hy)^2 + (2 hz)^2) = m/3 * (hy^2 + hz^2), and so on
    if (b.mass > 0){
        b.inv_inertia = vec3(3.0f / (b.mass * (half.y * half.y + half.z * half.z)),
                             3.0f / (b.mass * (half.x * half.x + half.z * half.z)),
                             3.0f / (b.mass * (half.x * half.x + half.y * half.y)));
    }
    return add_body(w, b);
}

// ---- mesh bodies -------------------------------------------------------------------------

// the eigenvectors (columns of v, a rotation) and eigenvalues of a symmetric matrix (Jacobi)
static void eigen3(Mat3 a, Mat3 *v, Vec3 *e){
    *v = mat3_identity();
    for (int sweep = 0; sweep < 32; sweep++){
        real off = r_fabs(a.m[0][1]) + r_fabs(a.m[0][2]) + r_fabs(a.m[1][2]);
        if (off < 1e-12f * (r_fabs(a.m[0][0]) + r_fabs(a.m[1][1]) + r_fabs(a.m[2][2]))) break;
        for (int p = 0; p < 2; p++)
            for (int q = p + 1; q < 3; q++){
                if (a.m[p][q] == 0) continue;
                real th = (a.m[q][q] - a.m[p][p]) / (2 * a.m[p][q]);
                real t = (th >= 0 ? 1 : -1) / (r_fabs(th) + r_sqrt(th * th + 1));
                real c = 1 / r_sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < 3; k++){          // a = J^T a J
                    real akp = a.m[k][p], akq = a.m[k][q];
                    a.m[k][p] = c * akp - s * akq;
                    a.m[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; k++){
                    real apk = a.m[p][k], aqk = a.m[q][k];
                    a.m[p][k] = c * apk - s * aqk;
                    a.m[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; k++){
                    real vkp = v->m[k][p], vkq = v->m[k][q];
                    v->m[k][p] = c * vkp - s * vkq;
                    v->m[k][q] = s * vkp + c * vkq;
                }
            }
    }
    *e = vec3(a.m[0][0], a.m[1][1], a.m[2][2]);
    // a rotation, not a reflection
    if (v_dot(mat3_col(*v, 0), v_cross(mat3_col(*v, 1), mat3_col(*v, 2))) < 0)
        for (int k = 0; k < 3; k++) v->m[k][2] = -v->m[k][2];
}

static Quat mat3_to_quat(Mat3 m){
    real tr = m.m[0][0] + m.m[1][1] + m.m[2][2];
    Quat q;
    if (tr > 0){
        real s = r_sqrt(tr + 1) * 2;
        q = quat(0.25f * s, (m.m[2][1] - m.m[1][2]) / s, (m.m[0][2] - m.m[2][0]) / s, (m.m[1][0] - m.m[0][1]) / s);
    } else if (m.m[0][0] > m.m[1][1] && m.m[0][0] > m.m[2][2]){
        real s = r_sqrt(1 + m.m[0][0] - m.m[1][1] - m.m[2][2]) * 2;
        q = quat((m.m[2][1] - m.m[1][2]) / s, 0.25f * s, (m.m[0][1] + m.m[1][0]) / s, (m.m[0][2] + m.m[2][0]) / s);
    } else if (m.m[1][1] > m.m[2][2]){
        real s = r_sqrt(1 + m.m[1][1] - m.m[0][0] - m.m[2][2]) * 2;
        q = quat((m.m[0][2] - m.m[2][0]) / s, (m.m[0][1] + m.m[1][0]) / s, 0.25f * s, (m.m[1][2] + m.m[2][1]) / s);
    } else {
        real s = r_sqrt(1 + m.m[2][2] - m.m[0][0] - m.m[1][1]) * 2;
        q = quat((m.m[1][0] - m.m[0][1]) / s, (m.m[0][2] + m.m[2][0]) / s, (m.m[1][2] + m.m[2][1]) / s, 0.25f * s);
    }
    return quat_norm(q);
}

static int add_geom(RigidWorld *w, RigidGeom *g){
    if (w->ngeoms >= w->cap_geoms){
        int cap = w->cap_geoms ? w->cap_geoms * 2 : 8;
        RigidGeom **ng = realloc(w->geoms, (size_t)cap * sizeof(RigidGeom *));
        if (!ng) return -1;
        w->geoms = ng;
        w->cap_geoms = cap;
    }
    w->geoms[w->ngeoms] = g;
    return w->ngeoms++;
}

static void free_geom(RigidGeom *g){
    convex_free(&g->hull);
    mesh_tree_free(&g->tree);
    free(g->occ);
    free(g);
}

int rigid_geom_hull(RigidWorld *w, const Triangle *tris, int ntris, int max_verts){
    if (ntris <= 0) return -1;
    Vec3 *pts = malloc((size_t)ntris * 3 * sizeof(Vec3));
    RigidGeom *g = calloc(1, sizeof(RigidGeom));
    if (!pts || !g){ free(pts); free(g); return -1; }
    for (int i = 0; i < ntris; i++){
        pts[3 * i] = tris[i].q;
        pts[3 * i + 1] = v_add(tris[i].q, tris[i].u);
        pts[3 * i + 2] = v_add(tris[i].q, tris[i].v);
    }
    if (max_verts <= 0 || max_verts > MAX_HULL_VERTS) max_verts = max_verts <= 0 ? 48 : MAX_HULL_VERTS;
    int ok = convex_hull(pts, ntris * 3, max_verts, &g->hull);
    free(pts);
    if (!ok){ free_geom(g); return -1; }
    g->kind = BODY_HULL;
    Mat3 inertia;
    convex_mass(&g->hull, &g->volume, &g->com, &inertia);
    eigen3(inertia, &g->frame, &g->inertia);
    // the hull into its body frame: p' = F^T (p - com)
    Mat3 ft = mat3_transpose(g->frame);
    Vec3 h = vec3(0, 0, 0);
    for (int i = 0; i < g->hull.nv; i++){
        Vec3 p = mat3_mul_vec(ft, v_sub(g->hull.v[i], g->com));
        g->hull.v[i] = p;
        h = vec3(r_fmax(h.x, r_fabs(p.x)), r_fmax(h.y, r_fabs(p.y)), r_fmax(h.z, r_fabs(p.z)));
    }
    for (int f = 0; f < g->hull.nf; f++){
        g->hull.f[f].d -= v_dot(g->hull.f[f].n, g->com);
        g->hull.f[f].n = mat3_mul_vec(ft, g->hull.f[f].n);
    }
    g->half = v_add(h, vec3(1e-3f, 1e-3f, 1e-3f));
    g->occ = malloc(OCC_WORDS * sizeof(uint32_t));
    if (!g->occ || !(g->volume > 0)){ free_geom(g); return -1; }
    occupancy_convex(&g->hull, g->half, g->occ);
    int id = add_geom(w, g);
    if (id < 0) free_geom(g);
    return id;
}

int rigid_geom_mesh(RigidWorld *w, const Triangle *tris, int ntris){
    if (ntris <= 0) return -1;
    RigidGeom *g = calloc(1, sizeof(RigidGeom));
    if (!g) return -1;
    g->kind = BODY_MESH;
    Aabb b = triangle_bounds(&tris[0]);
    for (int i = 1; i < ntris; i++){
        Aabb t = triangle_bounds(&tris[i]);
        b.lo = vec3(r_fmin(b.lo.x, t.lo.x), r_fmin(b.lo.y, t.lo.y), r_fmin(b.lo.z, t.lo.z));
        b.hi = vec3(r_fmax(b.hi.x, t.hi.x), r_fmax(b.hi.y, t.hi.y), r_fmax(b.hi.z, t.hi.z));
    }
    g->com = v_scale(v_add(b.lo, b.hi), 0.5f);
    g->frame = mat3_identity();
    g->half = v_add(v_scale(v_sub(b.hi, b.lo), 0.5f), vec3(1e-3f, 1e-3f, 1e-3f));
    g->occ = malloc(OCC_WORDS * sizeof(uint32_t));
    if (!g->occ || !mesh_tree_build(&g->tree, tris, ntris, g->com)){ free_geom(g); return -1; }
    occupancy_mesh(&g->tree, g->half, g->occ);
    int id = add_geom(w, g);
    if (id < 0) free_geom(g);
    return id;
}

static Mat3 xform_rot(Xform x){
    Mat3 r;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r.m[i][j] = x.m[i][j];
    return r;
}

// the body's pose from the mesh's placement: R = X F, pos = X(com)
static void place_body(Body *b, const RigidGeom *g, Xform place){
    b->rot = mat3_to_quat(mat3_mul(xform_rot(place), g->frame));
    b->pos = xform_point(&place, g->com);
}

int rigid_add_hull_body(RigidWorld *w, int geom, Xform place, real density){
    if (geom < 0 || geom >= w->ngeoms || w->geoms[geom]->kind != BODY_HULL) return -1;
    const RigidGeom *g = w->geoms[geom];
    Body b = body_base(vec3(0, 0, 0), g->volume, density);
    b.shape = BODY_HULL;
    b.g = g;
    b.half = g->half;
    if (b.mass > 0) b.inv_inertia = vec3(1 / (density * g->inertia.x), 1 / (density * g->inertia.y), 1 / (density * g->inertia.z));
    place_body(&b, g, place);
    return add_body(w, b);
}

int rigid_add_mesh_collider(RigidWorld *w, int geom, Xform place){
    if (geom < 0 || geom >= w->ngeoms || w->geoms[geom]->kind != BODY_MESH) return -1;
    const RigidGeom *g = w->geoms[geom];
    Body b = body_base(vec3(0, 0, 0), 0, 0);
    b.shape = BODY_MESH;
    b.g = g;
    b.half = g->half;
    place_body(&b, g, place);
    return add_body(w, b);
}

Xform body_mesh_xform(const Body *b){
    Mat3 m = mat3_identity();
    Vec3 com = vec3(0, 0, 0);
    if (b->g){ m = mat3_transpose(b->g->frame); com = b->g->com; }
    m = mat3_mul(quat_to_mat3(b->rot), m);                 // world = pos + R F^T (p - com)
    Vec3 t = v_sub(b->pos, mat3_mul_vec(m, com));
    return xform_make(t, m.m, 1);
}

Mat3 body_inv_inertia_world(const Body *b){
    Mat3 r = quat_to_mat3(b->rot);
    return mat3_mul(mat3_mul(r, mat3_diag(b->inv_inertia)), mat3_transpose(r));
}

Vec3 body_point_velocity(const Body *b, Vec3 p){
    return v_add(b->vel, v_cross(b->angvel, v_sub(p, b->pos)));
}

int body_contains(const Body *b, Vec3 p){
    Vec3 d = v_sub(p, b->pos);
    if (b->shape == BODY_SPHERE) return v_dot(d, d) <= b->half.x * b->half.x;
    Mat3 r = quat_to_mat3(b->rot);
    if (b->g) return occupancy_at(b->g->occ, b->half, mat3_mul_vec(mat3_transpose(r), d));
    for (int k = 0; k < 3; k++){
        real c = v_dot(mat3_col(r, k), d);
        real h = k == 0 ? b->half.x : (k == 1 ? b->half.y : b->half.z);
        if (c < -h || c > h) return 0;
    }
    return 1;
}

void body_apply_impulse(Body *b, Vec3 j, Vec3 p){
    if (b->inv_mass == 0) return;
    b->dirty = 1;
    b->vel = v_add(b->vel, v_scale(j, b->inv_mass));
    b->angvel = v_add(b->angvel, mat3_mul_vec(body_inv_inertia_world(b), v_cross(v_sub(p, b->pos), j)));
}

// ---- contacts -----------------------------------------------------------------------------

static void add_contact(RigidWorld *w, int a, int b, Vec3 p, Vec3 n, real depth){
    if (w->ncontacts >= w->cap_contacts){
        w->cap_contacts = w->cap_contacts ? w->cap_contacts * 2 : 64;
        w->contacts = realloc(w->contacts, (size_t)w->cap_contacts * sizeof(Contact));
        if (!w->contacts){ perror("realloc"); exit(1); }
    }
    Contact *c = &w->contacts[w->ncontacts++];
    memset(c, 0, sizeof(*c));
    c->a = a; c->b = b; c->p = p; c->n = n; c->depth = depth;
    // the point in each body's frame (a wall's: in the world), to follow the bodies through
    // the substeps without looking for the contacts again
    c->depth0 = depth;
    c->la = mat3_mul_vec(mat3_transpose(quat_to_mat3(w->bodies[a].rot)), v_sub(p, w->bodies[a].pos));
    c->lb = b >= 0 ? mat3_mul_vec(mat3_transpose(quat_to_mat3(w->bodies[b].rot)), v_sub(p, w->bodies[b].pos)) : p;
}

static real comp(Vec3 v, int k){ return k == 0 ? v.x : (k == 1 ? v.y : v.z); }

// the 8 corners of a box in world space
static void box_corners(const Body *b, Vec3 *out){
    Mat3 r = quat_to_mat3(b->rot);
    for (int i = 0; i < 8; i++){
        Vec3 l = vec3((i & 1) ? b->half.x : -b->half.x, (i & 2) ? b->half.y : -b->half.y,
                      (i & 4) ? b->half.z : -b->half.z);
        out[i] = v_add(b->pos, mat3_mul_vec(r, l));
    }
}

// ---- boxes and hulls as placed polyhedra (phys/convex.h) ----
typedef struct {
    WorldConvex wc;
    Convex box;              // a box: the unit box's topology, its corners and planes
    Vec3 bv[8];
    ConvexFace bf[6];
    Vec3 v[MAX_HULL_VERTS], n[MAX_HULL_FACES];
    real d[MAX_HULL_FACES];
} Placed;

static const Convex *unit_box(void){
    static Convex c;
    static int made;
    if (!made){ convex_box(vec3(1, 1, 1), &c); made = 1; }
    return &c;
}

static void place(const Body *b, Placed *p){
    const Convex *c;
    if (b->shape == BODY_BOX){
        const Convex *u = unit_box();
        p->box = *u;
        for (int i = 0; i < u->nv; i++) p->bv[i] = v_mul(u->v[i], b->half);
        for (int f = 0; f < u->nf; f++){
            p->bf[f] = u->f[f];
            p->bf[f].d = r_fabs(v_dot(u->f[f].n, b->half));   // its normals are the axes
        }
        p->box.v = p->bv;
        p->box.f = p->bf;
        c = &p->box;
    } else c = &b->g->hull;
    convex_place(&p->wc, c, b->pos, quat_to_mat3(b->rot), p->v, p->n, p->d);
}

// The room: six walls, each a plane dot(n, x) = off with n pointing into the room.
static void collide_walls(RigidWorld *w, int ia, real m){
    const Body *b = &w->bodies[ia];
    const Aabb *r = &w->room;
    Vec3 n[6] = {vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0), vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1)};
    real off[6] = {r->lo.x, -r->hi.x, r->lo.y, -r->hi.y, r->lo.z, -r->hi.z};
    if (b->shape == BODY_SPHERE){
        for (int k = 0; k < 6; k++){
            real d = v_dot(n[k], b->pos) - off[k] - b->half.x;
            if (d < m) add_contact(w, ia, -1, v_sub(b->pos, v_scale(n[k], b->half.x)), n[k], -d);
        }
    } else if (b->shape == BODY_BOX){
        Vec3 c[8];
        box_corners(b, c);
        for (int k = 0; k < 6; k++)
            for (int i = 0; i < 8; i++){
                real d = v_dot(n[k], c[i]) - off[k];
                if (d < m) add_contact(w, ia, -1, c[i], n[k], -d);
            }
    } else if (b->shape == BODY_HULL){
        // the hull's corners below each wall (or within the margin), at most 4 per wall
        const real mg = m;
        Placed *p = malloc(sizeof(Placed));
        if (!p) return;
        place(b, p);
        for (int k = 0; k < 6; k++){
            ConvexContact c[MAX_HULL_VERTS];
            int m = 0;
            for (int i = 0; i < p->wc.c->nv; i++){
                real d = v_dot(n[k], p->v[i]) - off[k];
                if (d < mg){ c[m].p = p->v[i]; c[m].n = n[k]; c[m].depth = -d; m++; }
            }
            m = convex_reduce(c, m, 4);
            for (int i = 0; i < m; i++) add_contact(w, ia, -1, c[i].p, c[i].n, c[i].depth);
        }
        free(p);
    }
}

// contacts found for (b, a) turned into contacts for (a, b)
static void flip_contacts(RigidWorld *w, int first, int a, int b){
    for (int k = first; k < w->ncontacts; k++){
        Contact *c = &w->contacts[k];
        c->a = a; c->b = b; c->n = v_neg(c->n);
    }
}

// a sphere (ia) against a box or hull (ib)
static void collide_sphere_convex(RigidWorld *w, int ia, int ib, real m){
    Placed *p = malloc(sizeof(Placed));
    if (!p) return;
    place(&w->bodies[ib], p);
    ConvexContact c;
    const Body *s = &w->bodies[ia];
    if (convex_collide_sphere(s->pos, s->half.x, &p->wc, m, &c)) add_contact(w, ia, ib, c.p, c.n, c.depth);
    free(p);
}

// boxes and hulls
static void collide_convex(RigidWorld *w, int ia, int ib, real mg){
    Placed *p = malloc(2 * sizeof(Placed));
    if (!p) return;
    place(&w->bodies[ia], &p[0]);
    place(&w->bodies[ib], &p[1]);
    ConvexContact c[CONVEX_MAX_CONTACTS];
    int m = convex_collide(&p[0].wc, &p[1].wc, mg, c);
    for (int k = 0; k < m; k++) add_contact(w, ia, ib, c[k].p, c[k].n, c[k].depth);
    free(p);
}

// a body (ia) against a fixed mesh (im): the mesh's triangles near the body, each
// against it, at most 4 contacts kept
typedef struct { int *tri; int n, cap; } TriList;
static void tri_list_add(void *user, int t){
    TriList *l = user;
    if (l->n >= l->cap){
        int cap = l->cap ? l->cap * 2 : 64;
        int *nt = realloc(l->tri, (size_t)cap * sizeof(int));
        if (!nt) return;
        l->tri = nt;
        l->cap = cap;
    }
    l->tri[l->n++] = t;
}

#define MESH_CONTACTS 256

static void collide_mesh(RigidWorld *w, int ia, int im, real mg){
    const Body *a = &w->bodies[ia], *m = &w->bodies[im];
    const RigidGeom *g = m->g;
    Mat3 r = quat_to_mat3(m->rot), rt = mat3_transpose(r);
    // the body's box in the mesh's frame: a sphere's, or around its corners there
    Aabb q;
    Placed *p = NULL;
    if (a->shape == BODY_SPHERE){
        real rad = a->half.x + mg;
        Vec3 c = mat3_mul_vec(rt, v_sub(a->pos, m->pos));
        q = (Aabb){v_sub(c, vec3(rad, rad, rad)), v_add(c, vec3(rad, rad, rad))};
    } else {
        p = malloc(sizeof(Placed));
        if (!p) return;
        place(a, p);
        q.lo = vec3(1e30f, 1e30f, 1e30f);
        q.hi = vec3(-1e30f, -1e30f, -1e30f);
        for (int i = 0; i < p->wc.c->nv; i++){
            Vec3 l = mat3_mul_vec(rt, v_sub(p->v[i], m->pos));
            q.lo = vec3(r_fmin(q.lo.x, l.x), r_fmin(q.lo.y, l.y), r_fmin(q.lo.z, l.z));
            q.hi = vec3(r_fmax(q.hi.x, l.x), r_fmax(q.hi.y, l.y), r_fmax(q.hi.z, l.z));
        }
    }
    if (a->shape != BODY_SPHERE){ q.lo = v_sub(q.lo, vec3(mg, mg, mg)); q.hi = v_add(q.hi, vec3(mg, mg, mg)); }   // (a sphere's: in rad)
    TriList list = {NULL, 0, 0};
    mesh_tree_query(&g->tree, q, tri_list_add, &list);
    if (list.n == 0){ free(list.tri); free(p); return; }
    ConvexContact *all = malloc(MESH_CONTACTS * sizeof(ConvexContact));
    int n = 0;
    for (int k = 0; all && k < list.n && n + CONVEX_MAX_CONTACTS <= MESH_CONTACTS; k++){
        const Vec3 *tl = g->tree.tri + 3 * (size_t)list.tri[k];
        Vec3 t[3] = {v_add(m->pos, mat3_mul_vec(r, tl[0])), v_add(m->pos, mat3_mul_vec(r, tl[1])),
                     v_add(m->pos, mat3_mul_vec(r, tl[2]))};
        if (a->shape == BODY_SPHERE) n += triangle_collide_sphere(a->pos, a->half.x, t, mg, all + n);
        else n += convex_collide_triangle(&p->wc, t, mg, all + n);
    }
    if (all){
        n = convex_reduce(all, n, 4);
        for (int k = 0; k < n; k++) add_contact(w, ia, im, all[k].p, all[k].n, all[k].depth);
    }
    free(all);
    free(p);
    free(list.tri);
}

static void collide_sphere_sphere(RigidWorld *w, int ia, int ib, real m);
static void collide_sphere_box(RigidWorld *w, int ia, int ib, real m);
static void collide_box_box(RigidWorld *w, int ia, int ib, real m);

// (flipped contacts: their anchors swap too)
static void flip_anchors(RigidWorld *w, int first){
    for (int k = first; k < w->ncontacts; k++){ Contact *c = &w->contacts[k]; Vec3 t = c->la; c->la = c->lb; c->lb = t; }
}

static void collide_pair(RigidWorld *w, int i, int j, real m){
    BodyShape a = w->bodies[i].shape, b = w->bodies[j].shape;
    int first = w->ncontacts;
    if (a == BODY_MESH || b == BODY_MESH){
        if (a == BODY_MESH && b == BODY_MESH) return;
        if (a == BODY_MESH){ collide_mesh(w, j, i, m); flip_contacts(w, first, i, j); flip_anchors(w, first); }
        else collide_mesh(w, i, j, m);
    } else if (a == BODY_SPHERE && b == BODY_SPHERE) collide_sphere_sphere(w, i, j, m);
    else if (a == BODY_BOX && b == BODY_BOX) collide_box_box(w, i, j, m);
    else if (a == BODY_SPHERE && b == BODY_BOX) collide_sphere_box(w, i, j, m);
    else if (b == BODY_SPHERE && a == BODY_BOX){ collide_sphere_box(w, j, i, m); flip_contacts(w, first, i, j); flip_anchors(w, first); }
    else if (a == BODY_SPHERE) collide_sphere_convex(w, i, j, m);
    else if (b == BODY_SPHERE){ collide_sphere_convex(w, j, i, m); flip_contacts(w, first, i, j); flip_anchors(w, first); }
    else collide_convex(w, i, j, m);
}

static void collide_sphere_sphere(RigidWorld *w, int ia, int ib, real m){
    const Body *a = &w->bodies[ia], *b = &w->bodies[ib];
    Vec3 d = v_sub(a->pos, b->pos);
    real r = a->half.x + b->half.x, l2 = v_dot(d, d);
    if (l2 >= (r + m) * (r + m)) return;
    real l = r_sqrt(l2);
    Vec3 n = l > 1e-6f ? v_scale(d, 1.0f / l) : vec3(0, 1, 0);
    real depth = r - l;
    add_contact(w, ia, ib, v_add(b->pos, v_scale(n, b->half.x - 0.5f * depth)), n, depth);
}

// sphere a against box b
static void collide_sphere_box(RigidWorld *w, int ia, int ib, real m){
    const Body *s = &w->bodies[ia], *b = &w->bodies[ib];
    Mat3 r = quat_to_mat3(b->rot);
    Vec3 d = v_sub(s->pos, b->pos);
    Vec3 c = vec3(v_dot(mat3_col(r, 0), d), v_dot(mat3_col(r, 1), d), v_dot(mat3_col(r, 2), d));   // box frame
    Vec3 q = vec3(r_fmax(-b->half.x, r_fmin(b->half.x, c.x)), r_fmax(-b->half.y, r_fmin(b->half.y, c.y)),
                  r_fmax(-b->half.z, r_fmin(b->half.z, c.z)));
    Vec3 diff = v_sub(c, q);
    real l2 = v_dot(diff, diff), rad = s->half.x;
    Vec3 nl;
    real depth;
    if (l2 > 1e-12f){                                   // center outside the box
        if (l2 >= (rad + m) * (rad + m)) return;
        real l = r_sqrt(l2);
        nl = v_scale(diff, 1.0f / l);
        depth = rad - l;
    } else {                                            // center inside: leave by the nearest face
        int k = 0;
        real best = 1e30f;
        for (int i = 0; i < 3; i++){
            real g = comp(b->half, i) - r_fabs(comp(c, i));
            if (g < best){ best = g; k = i; }
        }
        real sg = comp(c, k) < 0 ? -1.0f : 1.0f;
        nl = vec3(k == 0 ? sg : 0, k == 1 ? sg : 0, k == 2 ? sg : 0);
        depth = rad + best;
    }
    Vec3 n = mat3_mul_vec(r, nl);
    Vec3 p = v_add(b->pos, mat3_mul_vec(r, q));
    add_contact(w, ia, ib, p, n, depth);
}

// Separating axis test for two boxes: they overlap only if their projections overlap on all
// 15 candidate axes (3 face normals of each, 9 edge cross products). The axis with the
// smallest overlap is the contact normal. Then:
//   face axis: the face of that box ("reference") and the most opposed face of the other
//     ("incident"); the incident face is clipped to the sides of the reference face, and the
//     points of the result below the reference face are the contacts (up to 8, typically 4:
//     a box resting on a box gets its 4 corners, or the corners of the overlap).
//   edge axis: one contact between the closest points of the two edges.

// the 4 corners (world) of face (axis k, side s = +-1) of box b
static void box_face(const Body *b, Mat3 r, int k, real s, Vec3 *out){
    int i = (k + 1) % 3, j = (k + 2) % 3;
    Vec3 c = v_add(b->pos, v_scale(mat3_col(r, k), s * comp(b->half, k)));
    Vec3 u = v_scale(mat3_col(r, i), comp(b->half, i)), v = v_scale(mat3_col(r, j), comp(b->half, j));
    out[0] = v_add(c, v_add(u, v));
    out[1] = v_add(c, v_sub(u, v));
    out[2] = v_sub(c, v_add(u, v));
    out[3] = v_sub(c, v_sub(u, v));
}

// Sutherland-Hodgman: keep the part of polygon `in` (n points) where dot(p, d) <= off
static int clip(const Vec3 *in, int n, Vec3 d, real off, Vec3 *out){
    int m = 0;
    for (int i = 0; i < n; i++){
        Vec3 p = in[i], q = in[(i + 1) % n];
        real dp = v_dot(p, d) - off, dq = v_dot(q, d) - off;
        if (dp <= 0) out[m++] = p;
        if ((dp < 0 && dq > 0) || (dp > 0 && dq < 0)) out[m++] = v_add(p, v_scale(v_sub(q, p), dp / (dp - dq)));
    }
    return m;
}

// closest points of the segments p0 + s d0 and p1 + t d1 (s, t in [-1, 1])
static void closest_segments(Vec3 p0, Vec3 d0, Vec3 p1, Vec3 d1, Vec3 *c0, Vec3 *c1){
    Vec3 r = v_sub(p0, p1);
    real a = v_dot(d0, d0), e = v_dot(d1, d1), f = v_dot(d1, r), c = v_dot(d0, r), b = v_dot(d0, d1);
    real den = a * e - b * b, s = 0, t;
    if (den > 1e-8f) s = r_fmax(-1, r_fmin(1, (b * f - c * e) / den));
    t = (b * s + f) / e;
    if (t < -1 || t > 1){
        t = r_fmax(-1, r_fmin(1, t));
        s = r_fmax(-1, r_fmin(1, (t * b - c) / a));
    }
    *c0 = v_add(p0, v_scale(d0, s));
    *c1 = v_add(p1, v_scale(d1, t));
}

static void collide_box_box(RigidWorld *w, int ia, int ib, real m){
    const Body *a = &w->bodies[ia], *b = &w->bodies[ib];
    Mat3 ra = quat_to_mat3(a->rot), rb = quat_to_mat3(b->rot);
    Vec3 ax[3] = {mat3_col(ra, 0), mat3_col(ra, 1), mat3_col(ra, 2)};
    Vec3 bx[3] = {mat3_col(rb, 0), mat3_col(rb, 1), mat3_col(rb, 2)};
    Vec3 t = v_sub(a->pos, b->pos);
    real best = 1e30f, depth = 0;
    Vec3 bn = vec3(0, 1, 0);
    int best_axis = -1;
    for (int i = 0; i < 15; i++){
        Vec3 L;
        if (i < 3) L = ax[i];
        else if (i < 6) L = bx[i - 3];
        else L = v_cross(ax[(i - 6) / 3], bx[(i - 6) % 3]);
        real ll = v_dot(L, L);
        if (ll < 1e-6f) continue;                       // parallel edges: no new axis
        L = v_scale(L, 1.0f / r_sqrt(ll));
        real pa = a->half.x * r_fabs(v_dot(ax[0], L)) + a->half.y * r_fabs(v_dot(ax[1], L)) + a->half.z * r_fabs(v_dot(ax[2], L));
        real pb = b->half.x * r_fabs(v_dot(bx[0], L)) + b->half.y * r_fabs(v_dot(bx[1], L)) + b->half.z * r_fabs(v_dot(bx[2], L));
        real dist = v_dot(t, L);
        real overlap = pa + pb - r_fabs(dist);
        if (overlap < -m) return;                       // separating axis found (beyond the margin)
        // prefer face axes (an edge axis must be clearly better), and a's faces over b's,
        // so that the choice does not flip between steps for a resting contact
        real score = i < 3 ? overlap : (i < 6 ? overlap * 1.02f + 0.01f : overlap * 1.1f + 0.1f);
        if (score < best){
            best = score;
            depth = overlap;
            bn = dist < 0 ? v_neg(L) : L;               // from b towards a
            best_axis = i;
        }
    }
    if (best_axis < 0) return;

    if (best_axis < 6){
        // reference box: the one whose face it is. nref = its face's outward normal, which
        // points towards the incident box.
        int ref_is_a = best_axis < 3;
        const Body *R = ref_is_a ? a : b, *I = ref_is_a ? b : a;
        Mat3 rr = ref_is_a ? ra : rb, ri = ref_is_a ? rb : ra;
        Vec3 nref = ref_is_a ? v_neg(bn) : bn;
        int k = best_axis % 3;
        real sk = v_dot(mat3_col(rr, k), nref) > 0 ? 1.0f : -1.0f;
        // incident face: the face of I most opposed to nref
        int ki = 0;
        real most = 0;
        for (int q = 0; q < 3; q++){
            real d = v_dot(mat3_col(ri, q), nref);
            if (r_fabs(d) > r_fabs(most)){ most = d; ki = q; }
        }
        Vec3 poly[16], tmp[16];
        box_face(I, ri, ki, most > 0 ? -1.0f : 1.0f, poly);
        int n = 4;
        // clip against the 4 sides of the reference face
        for (int q = 1; q <= 2 && n > 0; q++){
            int j = (k + q) % 3;
            Vec3 dj = mat3_col(rr, j);
            real hj = comp(R->half, j), cj = v_dot(dj, R->pos);
            n = clip(poly, n, dj, cj + hj, tmp);
            n = clip(tmp, n, v_neg(dj), -(cj - hj), poly);
        }
        // keep the points below the reference face
        real face = v_dot(nref, R->pos) + comp(R->half, k);
        (void)sk;
        int added = 0;
        for (int q = 0; q < n; q++){
            real sep = v_dot(nref, poly[q]) - face;
            if (sep > m) continue;
            // the contact point halfway between the incident point and the reference face
            Vec3 p = v_sub(poly[q], v_scale(nref, 0.5f * sep));
            add_contact(w, ia, ib, p, bn, -sep);
            added++;
        }
        if (added) return;
    }
    // edge against edge: the edges of a and b along the two axes of the cross product,
    // the ones furthest towards each other
    int ea = best_axis >= 6 ? (best_axis - 6) / 3 : 0, eb = best_axis >= 6 ? (best_axis - 6) % 3 : 0;
    Vec3 pa = a->pos, pb = b->pos;
    for (int q = 0; q < 3; q++){
        if (q != ea) pa = v_add(pa, v_scale(ax[q], (v_dot(ax[q], bn) > 0 ? -1.0f : 1.0f) * comp(a->half, q)));
        if (q != eb) pb = v_add(pb, v_scale(bx[q], (v_dot(bx[q], bn) > 0 ? 1.0f : -1.0f) * comp(b->half, q)));
    }
    Vec3 c0, c1;
    closest_segments(pa, v_scale(ax[ea], comp(a->half, ea)), pb, v_scale(bx[eb], comp(b->half, eb)), &c0, &c1);
    add_contact(w, ia, ib, v_scale(v_add(c0, c1), 0.5f), bn, depth);
}

// How far apart two bodies may be and still get a contact: as far as they can close in a
// step (their speed towards each other along the line between their centres, the spin of
// their outermost points; against a wall, their whole speed), plus a little. Contacts are
// found once per step (rigid_steps), so those that will touch during its substeps must be
// there from the start: "speculative" contacts (Catto, Box2D v3), whose gap the solver lets
// close and no more.
static real body_radius(const Body *b){ return b->shape == BODY_SPHERE ? b->half.x : v_len(b->half); }
// At most half their sizes: a body that moves farther than that in a step would need
// continuous collision detection anyway, and the cap keeps a burst of fast bodies from
// pairing everything with everything.
static real pair_margin(const Body *a, const Body *b, real dt){
    real ra = body_radius(a), rb = b ? body_radius(b) : 0;
    real spin = v_len(a->angvel) * ra + (b ? v_len(b->angvel) * rb : 0);
    real closing = v_len(a->vel);
    if (b){
        Vec3 dv = v_sub(a->vel, b->vel), d = v_sub(a->pos, b->pos);
        real l = v_len(d);
        closing = l > 0 ? r_fmax(-v_dot(dv, d) / l, 0) : v_len(dv);
    }
    return r_fmin(SPECULATIVE_MIN + dt * (closing + spin), SPECULATIVE_MIN + 0.5f * (ra + rb));
}

static void find_contacts(RigidWorld *w, real dt){
    w->ncontacts = 0;
    for (int i = 0; i < w->nbodies; i++){
        Body *a = &w->bodies[i];
        if (w->has_room && a->inv_mass > 0) collide_walls(w, i, pair_margin(a, NULL, dt));
        for (int j = i + 1; j < w->nbodies; j++){
            Body *b = &w->bodies[j];
            if (a->inv_mass == 0 && b->inv_mass == 0) continue;
            // quick reject with bounding spheres
            real m = pair_margin(a, b, dt);
            real r = body_radius(a) + body_radius(b) + m;
            Vec3 d = v_sub(a->pos, b->pos);
            if (v_dot(d, d) > r * r) continue;
            collide_pair(w, i, j, m);
        }
    }
}

// ---- the solver: sequential impulses (Catto, "Iterative Dynamics with Temporal Coherence") --
//
// At a contact the bodies must not keep approaching: the relative velocity along the normal
// must become >= 0 (or >= a small push-out "bias" that removes overlap; or, for a gap, no
// faster than closes it within the substep). The bounce comes once per step, after.
// An impulse j along n changes it by j / m_eff, where the effective mass m_eff combines both
// bodies' mass and rotational inertia at that point. Solving one contact disturbs the
// others, so all are visited many times (iterations); the accumulated impulse is clamped
// to >= 0 (contacts push, never pull). Friction works the same way along two tangents, its
// impulse clamped to +- friction * normal impulse (Coulomb's law).

static real eff_mass(const Body *a, const Body *b, Mat3 ia, Mat3 ib, Vec3 ra, Vec3 rb, Vec3 d){
    real k = a->inv_mass + (b ? b->inv_mass : 0);
    k += v_dot(d, v_cross(mat3_mul_vec(ia, v_cross(ra, d)), ra));
    if (b) k += v_dot(d, v_cross(mat3_mul_vec(ib, v_cross(rb, d)), rb));
    return k > 0 ? 1.0f / k : 0;
}

static Vec3 rel_velocity(const Body *a, const Body *b, Vec3 ra, Vec3 rb){
    Vec3 va = v_add(a->vel, v_cross(a->angvel, ra));
    Vec3 vb = b ? v_add(b->vel, v_cross(b->angvel, rb)) : vec3(0, 0, 0);
    return v_sub(va, vb);
}

static void apply(Body *a, Body *b, Mat3 ia, Mat3 ib, Vec3 ra, Vec3 rb, Vec3 j){
    a->vel = v_add(a->vel, v_scale(j, a->inv_mass));
    a->angvel = v_add(a->angvel, mat3_mul_vec(ia, v_cross(ra, j)));
    if (b){
        b->vel = v_sub(b->vel, v_scale(j, b->inv_mass));
        b->angvel = v_sub(b->angvel, mat3_mul_vec(ib, v_cross(rb, j)));
    }
}

// One step of dt in `substeps` substeps (rigid.h). The contacts are found once, at the start
// (with margins: pair_margin), and followed through the substeps by their points in each
// body's frame: in each substep a contact's depth is its depth when found minus how far its
// two points have moved apart along the normal since. Looking for contacts is the costly
// part (the hulls' separating axes, the meshes' triangles), so this does it once instead of
// once per substep; the solver itself still runs every substep, stiff as before.
void rigid_steps(RigidWorld *w, real dt, int substeps, const Vec3 *forces){
    if (dt <= 0) return;
    if (substeps < 1) substeps = 1;
    const real h = dt / substeps;
    // the contacts (the previous step's are kept for warm starting)
    { Contact *t = w->prev; w->prev = w->contacts; w->contacts = t;
      int tc = w->cap_prev; w->cap_prev = w->cap_contacts; w->cap_contacts = tc;
      w->nprev = w->ncontacts; w->ncontacts = 0; }
    find_contacts(w, dt);
    for (int k = 0; k < w->ncontacts; k++){
        Contact *c = &w->contacts[k];
        onb_from_normal(c->n, &c->t1, &c->t2);
        // Warm start: the same contact last step (same bodies, within 1 unit) needed some
        // impulse to stay put; it starts from that, and the iterations only correct the change.
        // A resting stack then converges in a few steps instead of sinking and jittering.
        if (!w->warm_start) continue;
        for (int q = 0; q < w->nprev; q++){
            const Contact *o = &w->prev[q];
            if (o->a != c->a || o->b != c->b) continue;
            Vec3 d = v_sub(o->p, c->p);
            if (v_dot(d, d) > 1.0f || v_dot(o->n, c->n) < 0.95f) continue;
            c->jn = o->jn; c->jt1 = o->jt1; c->jt2 = o->jt2;
            // stopped at the surface by a speculative contact in the last substep, without a
            // bounce yet: it still owes one, from its speed then
            if (o->jn > 0) c->vn0 = o->vn0;
            break;
        }
    }
    Mat3 *iw = malloc((size_t)(w->nbodies > 0 ? w->nbodies : 1) * sizeof(Mat3));
    int *act = malloc((size_t)(w->ncontacts > 0 ? w->ncontacts : 1) * sizeof(int));
    if (!iw || !act){ perror("malloc"); exit(1); }
    Mat3 zero = {{{0}}};

    for (int s = 0; s < substeps; s++){
        // 1. forces -> velocities (the external ones act through the whole step)
        for (int i = 0; i < w->nbodies; i++){
            Body *b = &w->bodies[i];
            if (b->inv_mass == 0) continue;
            Vec3 f = forces ? v_add(b->force, forces[i]) : b->force;
            b->vel = v_add(b->vel, v_scale(v_add(w->gravity, v_scale(f, b->inv_mass)), h));
            b->angvel = v_add(b->angvel, mat3_mul_vec(body_inv_inertia_world(b), v_scale(b->torque, h)));
            b->vel = v_scale(b->vel, 1.0f / (1.0f + w->linear_damping * h));
            b->angvel = v_scale(b->angvel, 1.0f / (1.0f + w->angular_damping * h));
        }
        for (int i = 0; i < w->nbodies; i++) iw[i] = body_inv_inertia_world(&w->bodies[i]);

        // 2. each contact where its bodies are now: arms, depth, effective masses, target
        int nact = 0;
        for (int k = 0; k < w->ncontacts; k++){
            Contact *c = &w->contacts[k];
            Body *a = &w->bodies[c->a], *b = c->b >= 0 ? &w->bodies[c->b] : NULL;
            Mat3 ia = iw[c->a], ib = b ? iw[c->b] : zero;
            c->ra = quat_rotate(a->rot, c->la);
            c->rb = b ? quat_rotate(b->rot, c->lb) : vec3(0, 0, 0);
            Vec3 pa = v_add(a->pos, c->ra), pb = b ? v_add(b->pos, c->rb) : c->lb;
            c->depth = c->depth0 - v_dot(v_sub(pa, pb), c->n);
            c->mass_n = eff_mass(a, b, ia, ib, c->ra, c->rb, c->n);
            c->mass_t1 = eff_mass(a, b, ia, ib, c->ra, c->rb, c->t1);
            c->mass_t2 = eff_mass(a, b, ia, ib, c->ra, c->rb, c->t2);
            real vn = v_dot(rel_velocity(a, b, c->ra, c->rb), c->n);
            if (s == 0) c->vn0 = r_fmin(c->vn0, vn);
            // Apart (a speculative contact): they may approach until they touch, no closer.
            // Touching: the overlap beyond the slop is pushed out a little each substep, and
            // if they approached fast when the step began, they bounce. That speed, not the
            // one now: a speculative contact has already stopped them at the surface. The push
            // is capped: a light body squeezed deep into a wall by heavy ones was shot out ever
            // faster, until the numbers overflowed. (A gap
            // that only will close in this substep does not bounce: in a pile such bounces
            // fed on each other until it blew up.) A contact bounces once (vn0 = 0 after).
            real e = b ? r_fmax(a->restitution, b->restitution) : r_fmax(a->restitution, w->wall_restitution);
            c->mu = b ? r_sqrt(a->friction * b->friction) : r_sqrt(a->friction * w->wall_friction);
            real bounce = c->depth > -SPECULATIVE_MIN && c->vn0 < -BOUNCE_THRESHOLD ? -e * c->vn0 : 0;
            c->bias = bounce > 0 ? bounce
                    : (c->depth < 0 ? c->depth / h : r_fmin(BAUMGARTE / h * r_fmax(c->depth - CONTACT_SLOP, 0), MAX_PUSH));
            // the impulse it had (last step's, or this step's last substep's), applied first
            if (bounce > 0){ c->jn = c->jt1 = c->jt2 = 0; c->vn0 = 0; act[nact++] = k; continue; }
            if (c->jn != 0 || c->jt1 != 0 || c->jt2 != 0)
                apply(a, b, ia, ib, c->ra, c->rb,
                      v_add(v_scale(c->n, c->jn), v_add(v_scale(c->t1, c->jt1), v_scale(c->t2, c->jt2))));
            // The iterations skip the gaps that cannot close in this substep, even twice as
            // fast (most of a pile's speculative contacts): should one close all the same
            // (pushed by another), the next substep finds it touching and pushes it out.
            if (c->depth > -SPECULATIVE_MIN || c->jn > 0 || -c->depth < 2 * h * r_fmax(-vn, 0) + SPECULATIVE_MIN)
                act[nact++] = k;
        }

        // 3. iterate
        for (int it = 0; it < w->iterations; it++){
            for (int q = 0; q < nact; q++){
                Contact *c = &w->contacts[act[q]];
                Body *a = &w->bodies[c->a], *b = c->b >= 0 ? &w->bodies[c->b] : NULL;
                Mat3 ia = iw[c->a], ib = b ? iw[c->b] : zero;

                Vec3 dv = rel_velocity(a, b, c->ra, c->rb);
                real dj = c->mass_n * (c->bias - v_dot(dv, c->n));
                real jn = r_fmax(c->jn + dj, 0);
                dj = jn - c->jn;
                c->jn = jn;
                if (dj != 0) apply(a, b, ia, ib, c->ra, c->rb, v_scale(c->n, dj));
                // no push, no friction: nothing more to do (a gap, most of the time)
                if (c->jn == 0 && c->jt1 == 0 && c->jt2 == 0) continue;

                dv = rel_velocity(a, b, c->ra, c->rb);
                real lim = c->mu * c->jn;
                real jt = r_fmax(-lim, r_fmin(lim, c->jt1 - c->mass_t1 * v_dot(dv, c->t1)));
                apply(a, b, ia, ib, c->ra, c->rb, v_scale(c->t1, jt - c->jt1));
                c->jt1 = jt;
                dv = rel_velocity(a, b, c->ra, c->rb);
                jt = r_fmax(-lim, r_fmin(lim, c->jt2 - c->mass_t2 * v_dot(dv, c->t2)));
                apply(a, b, ia, ib, c->ra, c->rb, v_scale(c->t2, jt - c->jt2));
                c->jt2 = jt;
            }
        }

        // 4. velocities -> positions
        for (int i = 0; i < w->nbodies; i++){
            Body *b = &w->bodies[i];
            if (b->inv_mass == 0) continue;
            b->pos = v_add(b->pos, v_scale(b->vel, h));
            b->rot = quat_integrate(b->rot, b->angvel, h);
        }
    }

    free(iw);
    free(act);
    for (int i = 0; i < w->nbodies; i++) w->bodies[i].force = w->bodies[i].torque = vec3(0, 0, 0);
}

void rigid_step(RigidWorld *w, real dt){ rigid_steps(w, dt, 1, NULL); }

real rigid_kinetic_energy(const RigidWorld *w){
    real e = 0;
    for (int i = 0; i < w->nbodies; i++){
        const Body *b = &w->bodies[i];
        if (b->inv_mass == 0) continue;
        Mat3 r = quat_to_mat3(b->rot);
        Vec3 wl = mat3_mul_vec(mat3_transpose(r), b->angvel);   // body frame
        real rot = 0;
        if (b->inv_inertia.x > 0) rot += wl.x * wl.x / b->inv_inertia.x;
        if (b->inv_inertia.y > 0) rot += wl.y * wl.y / b->inv_inertia.y;
        if (b->inv_inertia.z > 0) rot += wl.z * wl.z / b->inv_inertia.z;
        e += 0.5f * b->mass * v_dot(b->vel, b->vel) + 0.5f * rot;
    }
    return e;
}

Vec3 rigid_momentum(const RigidWorld *w){
    Vec3 p = vec3(0, 0, 0);
    for (int i = 0; i < w->nbodies; i++) p = v_add(p, v_scale(w->bodies[i].vel, w->bodies[i].mass));
    return p;
}
