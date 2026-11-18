#include "phys/rigid.h"
#include "base/mem.h"

#include "base/rng.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CONTACT_SLOP 0.5f        // overlap (units) left alone, so resting contacts do not jitter
#define BAUMGARTE 0.2f           // fraction of the remaining overlap removed per step
#define BOUNCE_THRESHOLD 60.0f   // approach speeds below this (units / s) do not bounce
#define SPECULATIVE_MIN 0.1f     // the smallest margin (units) of a pair's speculative contacts
#define MAX_PUSH 300.0f          // the fastest an overlap is pushed out (units / s; Box2D: 3 m/s)
#define SLEEP_SPEED 5.0f         // slower than this (units / s, the outermost point too) is still
#define SLEEP_TIME 0.5f          // still this long (s): asleep
#define WAKE_SPEED 20.0f         // a body faster than this wakes the sleeping ones it comes near
#define CCD_FRACTION 0.5f        // moving more than this times its inner radius in a step: swept (rigid_ccd)
#define SLOW_SPEED 60.0f         // slower than this, and staying near one place and turn...
#define ANCHOR_MOVE 2.5f         // ... no farther than this (units, the outermost point too) ...
#define ANCHOR_TIME 1.5f         // ... this long (s): asleep too (a rocking body in a pile)
#define SETTLE_DAMPING 2.0f      // a slow body (SLOW_SPEED) resting on something: speed lost per second
#define MAX_HULL_VERTS 64        // corners of a hull (rigid_geom_hull)
#define MAX_HULL_FACES 128

// ---- bodies -----------------------------------------------------------------------------

// ---- what decided each pair of hulls last step (P3) ---------------------------------------
//
// Testing two hulls is the costly part of a step (phys/convex.c: every face of each against
// the other's corners, every pair of edges). Each pair's result is kept for the next step
// (convex_collide_cached): the axis that separated them, or the feature of least overlap
// with the pose of b in a's frame when it was found. Next step a separating axis is tried
// first (most pairs whose bounding spheres meet are apart, and stay apart along the same
// axis), and a touching pair that has moved less than MEMO_MOVE (of the smaller body's
// size) and MEMO_TURN since that full test takes its contacts from the same feature.
// Two tables: last step's (read) and this step's (written), so that pairs no longer met
// drop out by themselves. Open addressing, keys (a, b) + 1.

#define MEMO_MOVE 0.01f          // of the smaller bounding radius
#define MEMO_TURN 0.99996f       // cos of half the angle: about 1 degree

typedef struct { uint64_t key; ConvexFeature ft; Vec3 t; Quat q; } PairMemo;
struct PairMemos { PairMemo *e[2]; size_t cap; int cur; size_t used; };

static void memos_free(RigidWorld *w){
    if (!w->memos) return;
    free(w->memos->e[0]);
    free(w->memos->e[1]);
    free(w->memos);
    w->memos = NULL;
}

static size_t memo_slot(uint64_t key, size_t cap){ return (size_t)((key * 0x9E3779B97F4A7C15ull) >> 20) & (cap - 1); }

static PairMemo *memo_find(PairMemo *t, size_t cap, uint64_t key){
    for (size_t i = memo_slot(key, cap);; i = (i + 1) & (cap - 1)){
        if (t[i].key == key) return &t[i];
        if (t[i].key == 0) return NULL;
    }
}
static void memo_put(PairMemo *t, size_t cap, const PairMemo *m){
    for (size_t i = memo_slot(m->key, cap);; i = (i + 1) & (cap - 1))
        if (t[i].key == 0 || t[i].key == m->key){ t[i] = *m; return; }
}
// While the contacts are found on several threads (find_contacts), each thread's memos go
// to a list of its own (memo_sink), put into the table afterwards in the order one thread
// would have put them: the same table, the same contacts next step.
typedef struct { PairMemo *e; int n, cap; } MemoList;
static _Thread_local MemoList *memo_sink;
static void memo_write(struct PairMemos *mm, const PairMemo *m){
    if (memo_sink){
        if (memo_sink->n == memo_sink->cap){
            int cap = memo_sink->cap ? 2 * memo_sink->cap : 64;
            PairMemo *e = realloc(memo_sink->e, (size_t)cap * sizeof(PairMemo));
            if (!e) return;
            memo_sink->e = e;
            memo_sink->cap = cap;
        }
        memo_sink->e[memo_sink->n++] = *m;
        return;
    }
    if (mm->used * 2 < mm->cap){ memo_put(mm->e[mm->cur ^ 1], mm->cap, m); mm->used++; }
}

// at the start of a step: last step's table becomes the one read, a clear one is written
// (twice as large when last step filled a quarter of it)
static void memos_step(RigidWorld *w){
    struct PairMemos *m = w->memos;
    if (!m){
        m = w->memos = calloc(1, sizeof(*m));
        if (!m) return;
        m->cap = 1024;
        m->e[0] = calloc(m->cap, sizeof(PairMemo));
        m->e[1] = calloc(m->cap, sizeof(PairMemo));
        if (!m->e[0] || !m->e[1]){ memos_free(w); return; }
        return;
    }
    if (m->used * 4 > m->cap){
        size_t cap = m->cap * 4;
        PairMemo *r = calloc(cap, sizeof(PairMemo)), *wr = calloc(cap, sizeof(PairMemo));
        if (!r || !wr){ free(r); free(wr); memos_free(w); return; }
        const PairMemo *last = m->e[m->cur ^ 1];      // (the table written last step)
        for (size_t i = 0; i < m->cap; i++) if (last[i].key) memo_put(r, cap, &last[i]);
        free(m->e[0]);
        free(m->e[1]);
        m->e[0] = r;
        m->e[1] = wr;
        m->cur = 0;
        m->cap = cap;
    } else {
        m->cur ^= 1;
        memset(m->e[m->cur ^ 1], 0, m->cap * sizeof(PairMemo));
    }
    m->used = 0;
}

void rigid_init(RigidWorld *w){
    memset(w, 0, sizeof(*w));
    w->gravity = vec3(0, -981, 0);
    w->wall_restitution = 0.4f;
    w->wall_friction = 0.5f;
    w->iterations = 10;
    w->linear_damping = 0.0f;
    w->angular_damping = 0.1f;
    w->warm_start = 1;
    w->sleep = 1;
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
    memos_free(w);
    memset(w, 0, sizeof(*w));
}

void rigid_set_room(RigidWorld *w, Aabb room){
    w->room = room;
    w->has_room = 1;
}

static int add_body(RigidWorld *w, Body b){
    w->bodies = lr_grow(w->bodies, sizeof(Body), &w->cap, w->nbodies + 1, 16);
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
    w->contacts = lr_grow(w->contacts, sizeof(Contact), &w->cap_contacts, w->ncontacts + 1, 64);
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
        Placed pl, *p = &pl;         // (on the stack: a few KB, not a malloc per body and step)
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
    Placed pl, *p = &pl;
    place(&w->bodies[ib], p);
    ConvexContact c;
    const Body *s = &w->bodies[ia];
    if (convex_collide_sphere(s->pos, s->half.x, &p->wc, m, &c)) add_contact(w, ia, ib, c.p, c.n, c.depth);
}

// boxes and hulls
static real body_radius(const Body *b);

static void collide_convex(RigidWorld *w, int ia, int ib, real mg){
    Placed p[2];                 // (on the stack: 2 x a few KB, not a malloc per pair and step)
    const Body *a = &w->bodies[ia], *b = &w->bodies[ib];
    place(a, &p[0]);
    place(b, &p[1]);
    // what decided this pair last step, and b in a's frame now (P3: the memos above)
    struct PairMemos *mm = w->memos;
    PairMemo memo = {((uint64_t)ia << 32 | (uint64_t)ib) + 1, {CONVEX_NONE, 0, 0, 0}, vec3(0, 0, 0), quat_identity()};
    const PairMemo *old = mm ? memo_find(mm->e[mm->cur], mm->cap, memo.key) : NULL;
    if (old) memo = *old;
    Quat qa = a->rot, ca = quat(qa.w, -qa.x, -qa.y, -qa.z);
    Quat q = quat_mul(ca, b->rot);
    Vec3 t = quat_rotate(ca, v_sub(b->pos, a->pos));
    real rmin = r_fmin(body_radius(a), body_radius(b));
    int reuse = old && v_len(v_sub(t, memo.t)) < MEMO_MOVE * rmin &&
                r_fabs(q.w * memo.q.w + q.x * memo.q.x + q.y * memo.q.y + q.z * memo.q.z) > MEMO_TURN;
    ConvexContact c[CONVEX_MAX_CONTACTS];
    int m = convex_collide_cached(&p[0].wc, &p[1].wc, mg, &memo.ft, reuse, c);
    if (memo.ft.full){ memo.t = t; memo.q = q; }
    if (mm) memo_write(mm, &memo);
    for (int k = 0; k < m; k++) add_contact(w, ia, ib, c[k].p, c[k].n, c[k].depth);
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
// The closing speed is the whole relative speed, not only its part along the line between
// the centres: a box flying past a wall's middle at 500 units/s closes on its face at that
// speed, though the line to the wall's centre is almost square to its path (the part along
// it let the box through a 4-unit slab). Faster than the cap allows, rigid_ccd takes over.
static real pair_margin(const Body *a, const Body *b, real dt){
    real ra = body_radius(a), rb = b ? body_radius(b) : 0;
    real spin = v_len(a->angvel) * ra + (b ? v_len(b->angvel) * rb : 0);
    real closing = b ? v_len(v_sub(a->vel, b->vel)) : v_len(a->vel);
    return r_fmin(SPECULATIVE_MIN + dt * (closing + spin), SPECULATIVE_MIN + 0.5f * (ra + rb));
}

// ---- the broad phase: sort and sweep ---------------------------------------------------------
//
// Which pairs of bodies can touch within the step: each body's bounding sphere, grown by as
// far as it can move in the step (its speed and the spin of its outermost point), as an
// interval on one axis, the one along which the bodies spread most (a tall pile: the
// vertical). The intervals sorted by their start, a sweep keeps those still open: each
// body pairs with the open ones its interval meets. The pairs are then visited in the order
// the brute force of every pair (i < j) had, so the contacts, and the solver's results, are
// the same as before, only found in O(n log n + pairs) instead of O(n^2).

typedef struct { real lo, hi, rad; int i; } SweepEnd;   // (rad: the bounding sphere's radius, grown)
typedef struct { int i, j; } BodyPair;

static int cmp_sweep(const void *pa, const void *pb){
    const SweepEnd *a = pa, *b = pb;
    if (a->lo != b->lo) return a->lo < b->lo ? -1 : 1;
    return a->i - b->i;
}
static int cmp_pair(const void *pa, const void *pb){
    const BodyPair *a = pa, *b = pb;
    return a->i != b->i ? a->i - b->i : a->j - b->j;
}

// the pairs (i < j) whose grown bounds meet on the sweep axis, sorted; their number, -1 if
// out of memory (then the caller tests every pair)
static int broad_phase(const RigidWorld *w, real dt, BodyPair **out){
    int n = w->nbodies;
    *out = NULL;
    if (n < 2) return 0;
    SweepEnd *e = malloc((size_t)n * sizeof(SweepEnd));
    int *open = malloc((size_t)n * sizeof(int));
    if (!e || !open){ free(e); free(open); return -1; }
    // the axis along which the centres spread most
    Vec3 mean = vec3(0, 0, 0), var = vec3(0, 0, 0);
    for (int i = 0; i < n; i++) mean = v_add(mean, w->bodies[i].pos);
    mean = v_scale(mean, 1.0f / n);
    for (int i = 0; i < n; i++){ Vec3 d = v_sub(w->bodies[i].pos, mean); var = v_add(var, v_mul(d, d)); }
    int axis = var.x >= var.y && var.x >= var.z ? 0 : (var.y >= var.z ? 1 : 2);
    for (int i = 0; i < n; i++){
        const Body *b = &w->bodies[i];
        real r = body_radius(b);
        // >= half of any pair margin of this body (pair_margin: |va - vb| <= |va| + |vb|)
        real grow = 0.5f * SPECULATIVE_MIN + dt * (v_len(b->vel) + v_len(b->angvel) * r);
        real c = comp(b->pos, axis);
        e[i] = (SweepEnd){c - r - grow, c + r + grow, r + grow, i};
    }
    qsort(e, (size_t)n, sizeof(SweepEnd), cmp_sweep);
    int cap = 4 * n, np = 0, nopen = 0;
    BodyPair *p = malloc((size_t)cap * sizeof(BodyPair));
    if (!p){ free(e); free(open); return -1; }
    for (int k = 0; k < n; k++){
        int keep = 0;
        for (int q = 0; q < nopen; q++)                   // close the intervals that ended
            if (e[open[q]].hi >= e[k].lo) open[keep++] = open[q];
        nopen = keep;
        const Body *a = &w->bodies[e[k].i];
        const real ra = e[k].rad;                         // its bounding sphere, grown as its interval
        for (int q = 0; q < nopen; q++){
            int i = e[k].i, j = e[open[q]].i;
            if (a->inv_mass == 0 && w->bodies[j].inv_mass == 0) continue;   // two fixed bodies
            // their grown spheres apart: test_pair would drop the pair (its margin is no more
            // than the two growths), so it is not listed at all. On one axis alone a pile on
            // the floor pairs each body with every body of its slab: 100 times the pairs
            // that touch, and sorting them was half of a step
            Vec3 d = v_sub(a->pos, w->bodies[j].pos);
            real rr = (ra + e[open[q]].rad) * 1.0001f + 1e-4f;
            if (v_dot(d, d) > rr * rr) continue;
            if (np == cap){
                BodyPair *t = realloc(p, (size_t)cap * 2 * sizeof(BodyPair));
                if (!t){ free(p); free(e); free(open); return -1; }
                p = t;
                cap *= 2;
            }
            p[np++] = i < j ? (BodyPair){i, j} : (BodyPair){j, i};
        }
        open[nopen++] = k;
    }
    free(e);
    free(open);
    qsort(p, (size_t)np, sizeof(BodyPair), cmp_pair);
    *out = p;
    return np;
}

// a pair the broad phase gave: its bounding spheres, then its shapes
static void test_pair(RigidWorld *w, int i, int j, real dt){
    Body *a = &w->bodies[i], *b = &w->bodies[j];
    if (a->inv_mass == 0 && b->inv_mass == 0) return;
    real m = pair_margin(a, b, dt);
    real r = body_radius(a) + body_radius(b) + m;
    Vec3 d = v_sub(a->pos, b->pos);
    if (v_dot(d, d) > r * r) return;
    collide_pair(w, i, j, m);
}

// bodies [i0, i1): their walls and their pairs (from pairs[q]), into w's contacts
static void contacts_of(RigidWorld *w, int i0, int i1, const BodyPair *pairs, int np, int q, real dt){
    for (int i = i0; i < i1; i++){
        Body *a = &w->bodies[i];
        if (w->has_room && a->inv_mass > 0) collide_walls(w, i, pair_margin(a, NULL, dt));
        if (np < 0){                                  // (no memory for the pairs: all of them)
            for (int j = i + 1; j < w->nbodies; j++) test_pair(w, i, j, dt);
            continue;
        }
        for (; q < np && pairs[q].i == i; q++) test_pair(w, i, pairs[q].j, dt);
    }
}

// The narrow phase on all the CPU's cores (OpenMP): the bodies in runs of about the same
// number of pairs, each run's contacts into a world of its own (a copy of w with its own
// contact list; the bodies, the walls and the shapes only read), then the lists put
// together in the runs' order, and the memos written in it: the very contacts, in the very
// order, one thread finds. Few bodies: one thread (the runs cost more than they save).
typedef struct { int i0, i1, q; Contact *c; int n, cap; MemoList memos; } ContactRun;

static void find_contacts(RigidWorld *w, real dt){
    w->ncontacts = 0;
    BodyPair *pairs;
    int np = broad_phase(w, dt, &pairs);
    int nruns = 1;
#ifdef _OPENMP
    if (np > 256 && w->nbodies > 64) nruns = 4 * omp_get_max_threads();
#endif
    ContactRun *runs = nruns > 1 ? calloc((size_t)nruns, sizeof(ContactRun)) : NULL;
    if (!runs){ contacts_of(w, 0, w->nbodies, pairs, np, 0, dt); free(pairs); return; }
    // runs of bodies with about np / nruns pairs each (and a body each at least)
    int per = (np + w->nbodies + nruns - 1) / nruns, i = 0, q = 0;
    for (int r = 0; r < nruns; r++){
        runs[r].i0 = i;
        runs[r].q = q;
        int load = 0;
        while (i < w->nbodies && (load < per || r == nruns - 1)){
            int q0 = q;
            while (q < np && pairs[q].i == i) q++;
            load += 1 + (q - q0);
            i++;
        }
        runs[r].i1 = i;
    }
    (void)unit_box();                 // (made once, before the threads share it)
    #pragma omp parallel for schedule(dynamic, 1)
    for (int r = 0; r < nruns; r++){
        RigidWorld tw = *w;
        tw.contacts = NULL;
        tw.ncontacts = tw.cap_contacts = 0;
        memo_sink = &runs[r].memos;
        contacts_of(&tw, runs[r].i0, runs[r].i1, pairs, np, runs[r].q, dt);
        memo_sink = NULL;
        runs[r].c = tw.contacts;
        runs[r].n = tw.ncontacts;
        runs[r].cap = tw.cap_contacts;
    }
    int total = 0;
    for (int r = 0; r < nruns; r++) total += runs[r].n;
    if (total > w->cap_contacts){
        Contact *c = realloc(w->contacts, (size_t)total * sizeof(Contact));
        if (!c) lr_oom("rigid");
        w->contacts = c;
        w->cap_contacts = total;
    }
    for (int r = 0; r < nruns; r++){
        if (runs[r].n) memcpy(w->contacts + w->ncontacts, runs[r].c, (size_t)runs[r].n * sizeof(Contact));
        w->ncontacts += runs[r].n;
        for (int k = 0; k < runs[r].memos.n && w->memos; k++) memo_write(w->memos, &runs[r].memos.e[k]);
        free(runs[r].c);
        free(runs[r].memos.e);
    }
    free(runs);
    free(pairs);
}

// ---- sleeping --------------------------------------------------------------------------------
//
// A body that has been still (SLEEP_SPEED) for SLEEP_TIME falls asleep: its velocity is set
// to 0 and it is no longer moved. To the bodies still awake it is fixed, so they rest on it;
// pairs of sleeping bodies (and their walls) are not even tested. A resting pile then costs
// little more than its awake edge. It wakes when a body faster than WAKE_SPEED comes near it
// (within the contact margin), when the program changes it (Body.dirty, body_apply_impulse,
// a velocity set), or when the external force on it changes by a quarter of its weight (the
// air's push). Each body decides alone (as ODE's auto-disable), not with the island of all it
// touches (Box2D, PhysX): in a pile of hulls some body always jitters a little, which kept
// whole islands, or every body next to one, awake; once the still ones sleep, they hold the
// others still, and the pile falls asleep from the bottom up. The gap between SLEEP_SPEED and
// WAKE_SPEED keeps a jittering neighbour from waking them again.

static int body_still(const Body *b){
    return v_len(b->vel) < SLEEP_SPEED && v_len(b->angvel) * body_radius(b) < SLEEP_SPEED;
}
static void wake(Body *b){ b->sleeping = 0; b->sleep_time = 0; b->anchor_time = 0; }

// at the start of a step: wake the sleeping bodies that must move; returns how many still sleep
static int wake_bodies(RigidWorld *w, real dt, const Vec3 *forces){
    int asleep = 0;
    real wake_force = 0.25f * v_len(w->gravity);
    for (int i = 0; i < w->nbodies; i++){
        Body *b = &w->bodies[i];
        int changed = b->dirty;
        b->dirty = 0;                         // (the CPU's steps: no GPU copy to resend)
        if (!b->sleeping) continue;
        Vec3 f = forces ? v_add(b->force, forces[i]) : b->force;
        if (changed || v_dot(b->vel, b->vel) > 0 || v_dot(b->angvel, b->angvel) > 0 ||
            v_dot(b->torque, b->torque) > 0 || v_len(v_sub(f, b->sleep_force)) > wake_force * b->mass) wake(b);
        else asleep++;
    }
    if (!asleep) return 0;
    for (int i = 0; i < w->nbodies; i++){
        const Body *a = &w->bodies[i];
        if (a->inv_mass == 0 || a->sleeping || a->anchor_time >= 0.5f * ANCHOR_TIME) continue;   // (jittering in place)
        if (v_len(a->vel) < WAKE_SPEED && v_len(a->angvel) * body_radius(a) < WAKE_SPEED) continue;
        for (int j = 0; j < w->nbodies; j++){
            Body *b = &w->bodies[j];
            if (!b->sleeping) continue;
            real r = body_radius(a) + body_radius(b) + pair_margin(a, b, dt);
            Vec3 d = v_sub(a->pos, b->pos);
            if (v_dot(d, d) <= r * r){ wake(b); asleep--; }
        }
    }
    return asleep;
}

// At the end of a step: the bodies still long enough fall asleep. So do those that, slow,
// stay where they are: a body rocking in a pile, or creeping a hair's breadth back and forth,
// is never still at any moment, but it goes nowhere. While it is slower than SLOW_SPEED, its
// place and turn are kept (the anchor); if it stays within ANCHOR_MOVE of them for
// ANCHOR_TIME, it sleeps. A slow body that does get somewhere (a ball rolling across the
// floor) moves its anchor along and stays awake.
static int body_slow(const Body *b){
    return v_len(b->vel) < SLOW_SPEED && v_len(b->angvel) * body_radius(b) < SLOW_SPEED;
}
// still, or going nowhere for a while (half the time that puts it to sleep)
static int body_quiet(const Body *b){ return body_still(b) || b->anchor_time >= 0.5f * ANCHOR_TIME; }
static int near_anchor(const Body *b){
    Quat q = b->anchor_rot;
    real c = r_fabs(q.w * b->rot.w + q.x * b->rot.x + q.y * b->rot.y + q.z * b->rot.z);
    real turn = 2 * r_acos(r_fmin(c, 1.0f));                // the angle between the two turns
    return v_len(v_sub(b->pos, b->anchor_pos)) < ANCHOR_MOVE && turn * body_radius(b) < ANCHOR_MOVE;
}
// A body does not fall asleep while one it touches is awake and not quiet (not still, nor
// going nowhere): asleep it would be a fixed body under it, and a pile that is still
// settling, held up by some bodies asleep and pressing on others awake, slid off them in
// avalanches that woke it all again. (The pile still falls asleep from the bottom up: the
// bodies at its bottom, quiet first, hold the others still.)
static void fall_asleep(RigidWorld *w, real dt, const Vec3 *forces){
    char *busy = calloc((size_t)(w->nbodies > 0 ? w->nbodies : 1), 1);   // touches a moving body
    if (busy) for (int k = 0; k < w->ncontacts; k++){
        const Contact *c = &w->contacts[k];
        if (c->b < 0 || (c->jn <= 0 && c->depth0 <= 0)) continue;   // a wall, or not touching
        const Body *a = &w->bodies[c->a], *b = &w->bodies[c->b];
        if (!a->sleeping && a->inv_mass > 0 && !body_quiet(a)) busy[c->b] = 1;
        if (!b->sleeping && b->inv_mass > 0 && !body_quiet(b)) busy[c->a] = 1;
    }
    for (int i = 0; i < w->nbodies; i++){
        Body *b = &w->bodies[i];
        if (b->inv_mass == 0 || b->sleeping) continue;
        b->sleep_time = body_still(b) ? b->sleep_time + dt : 0;
        if (body_slow(b) && b->anchor_time > 0 && near_anchor(b)) b->anchor_time += dt;
        else if (body_slow(b)){ b->anchor_pos = b->pos; b->anchor_rot = b->rot; b->anchor_time = dt; }
        else b->anchor_time = 0;
        // (going nowhere three times as long: asleep whatever its neighbours do)
        if ((b->sleep_time < SLEEP_TIME && b->anchor_time < ANCHOR_TIME) ||
            (busy && busy[i] && b->anchor_time < 3 * ANCHOR_TIME)) continue;
        b->anchor_time = 0;
        b->sleeping = 1;
        b->vel = b->angvel = vec3(0, 0, 0);
        b->sleep_force = forces ? v_add(b->force, forces[i]) : b->force;
    }
    free(busy);
}

int rigid_sleeping(const RigidWorld *w){
    int n = 0;
    for (int i = 0; i < w->nbodies; i++) n += w->bodies[i].sleeping;
    return n;
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

// (a fixed or sleeping body, inv_mass 0, is left alone)
static void apply(Body *a, Body *b, Mat3 ia, Mat3 ib, Vec3 ra, Vec3 rb, Vec3 j){
    if (a->inv_mass > 0){
        a->vel = v_add(a->vel, v_scale(j, a->inv_mass));
        a->angvel = v_add(a->angvel, mat3_mul_vec(ia, v_cross(ra, j)));
    }
    if (b && b->inv_mass > 0){
        b->vel = v_sub(b->vel, v_scale(j, b->inv_mass));
        b->angvel = v_sub(b->angvel, mat3_mul_vec(ib, v_cross(rb, j)));
    }
}

// A contact's three directions (the normal, two tangents) as the iterations use them, made
// once per substep (the bodies do not turn within it): the speed of a's point against b's
// along d is d . (va - vb) + (ra x d) . wa - (rb x d) . wb, and an impulse j d changes the
// spins by j I^-1 (r x d). So a row is 4 dot products to read and 4 scaled additions to
// apply, where it was 4 cross products and 2 matrix products at every visit (10 passes and
// one more in each of 4 substeps).
typedef struct { Vec3 d, rad, rbd, iad, ibd; } SolveRow;   // d; ra x d, rb x d; Ia^-1 (ra x d), Ib^-1 (rb x d)
typedef struct { SolveRow n, t1, t2; } SolveRows;

static void solve_row_make(SolveRow *r, Vec3 d, Vec3 ra, Vec3 rb, const Mat3 *ia, const Mat3 *ib){
    r->d = d;
    r->rad = v_cross(ra, d);
    r->iad = mat3_mul_vec(*ia, r->rad);
    r->rbd = ib ? v_cross(rb, d) : vec3(0, 0, 0);
    r->ibd = ib ? mat3_mul_vec(*ib, r->rbd) : vec3(0, 0, 0);
}
static inline real row_speed(const Body *a, const Body *b, const SolveRow *r){
    real v = v_dot(r->d, a->vel) + v_dot(r->rad, a->angvel);
    if (b) v -= v_dot(r->d, b->vel) + v_dot(r->rbd, b->angvel);
    return v;
}
// (a fixed or sleeping body, inv_mass 0, is left alone)
static inline void row_apply(Body *a, Body *b, const SolveRow *r, real j){
    if (a->inv_mass > 0){
        a->vel = v_add(a->vel, v_scale(r->d, j * a->inv_mass));
        a->angvel = v_add(a->angvel, v_scale(r->iad, j));
    }
    if (b && b->inv_mass > 0){
        b->vel = v_sub(b->vel, v_scale(r->d, j * b->inv_mass));
        b->angvel = v_sub(b->angvel, v_scale(r->ibd, j));
    }
}

// One contact towards its target speed along the normal (bias), friction after it
static void solve_contact(RigidWorld *w, const SolveRows *rows, Contact *c, real bias){
    Body *a = &w->bodies[c->a], *b = c->b >= 0 ? &w->bodies[c->b] : NULL;

    real dj = c->mass_n * (bias - row_speed(a, b, &rows->n));
    real jn = r_fmax(c->jn + dj, 0);
    dj = jn - c->jn;
    c->jn = jn;
    if (dj != 0) row_apply(a, b, &rows->n, dj);
    // no push, no friction: nothing more to do (a gap, most of the time)
    if (c->jn == 0 && c->jt1 == 0 && c->jt2 == 0) return;

    real lim = c->mu * c->jn;
    real jt = r_fmax(-lim, r_fmin(lim, c->jt1 - c->mass_t1 * row_speed(a, b, &rows->t1)));
    row_apply(a, b, &rows->t1, jt - c->jt1);
    c->jt1 = jt;
    jt = r_fmax(-lim, r_fmin(lim, c->jt2 - c->mass_t2 * row_speed(a, b, &rows->t2)));
    row_apply(a, b, &rows->t2, jt - c->jt2);
    c->jt2 = jt;
}

// one pass over the active contacts (act), each towards c->bias, or (bias) bias[k]
static void solve_pass(RigidWorld *w, const SolveRows *rows, const int *act, int nact, const real *bias){
    for (int q = 0; q < nact; q++){
        Contact *c = &w->contacts[act[q]];
        solve_contact(w, &rows[act[q]], c, bias ? bias[act[q]] : c->bias);
    }
}

// ---- continuous collision: fast bodies swept ---------------------------------------------
//
// The speculative contacts stop a pair that can close within a step, up to half their sizes
// (pair_margin): a body faster than that could step over a thin wall or a small body between
// two steps. After the step, each body that moved more than CCD_FRACTION of its inner radius
// (the largest sphere inside it) is swept: that sphere along its path, from where it was to
// where it is, against the room's walls and every other body (at its place now, grown by the
// sphere's radius: a sphere, a box or a hull grown, a fixed mesh's triangles). At the first
// hit it is moved back to just before it, keeping its velocity: the next step's speculative
// contact then stops or bounces it at the surface (motion clamping, as Bullet does). What it
// misses: the turn within a step, and a fast body that only grazes a corner with its edge.

// the radius of the largest sphere about the centre of mass inside the body
static real body_inner_radius(const Body *b){
    if (b->shape == BODY_SPHERE) return b->half.x;
    if (b->shape == BODY_BOX) return r_fmin(b->half.x, r_fmin(b->half.y, b->half.z));
    if (b->shape == BODY_HULL && b->g){
        real r = 1e30f;
        for (int k = 0; k < b->g->hull.nf; k++) r = r_fmin(r, b->g->hull.f[k].d);
        return r > 0 && r < 1e30f ? r : r_fmin(b->half.x, r_fmin(b->half.y, b->half.z));
    }
    return 0;
}

// p + t d, t in [0, *t): where it enters the sphere (c, rad); 1 and *t = that t if it does
// (a start inside is no hit: the contacts have it)
static int sweep_sphere(Vec3 p, Vec3 d, Vec3 c, real rad, real *t){
    Vec3 m = v_sub(p, c);
    real a = v_dot(d, d), b = v_dot(m, d), cc = v_dot(m, m) - rad * rad;
    if (cc <= 0 || a <= 0 || b >= 0) return 0;
    real disc = b * b - a * cc;
    if (disc < 0) return 0;
    real tt = (-b - r_sqrt(disc)) / a;
    if (tt < 0 || tt >= *t) return 0;
    *t = tt;
    return 1;
}

// the same against the convex solid n_k . x <= off_k + grow (k < nf)
static int sweep_planes(Vec3 p, Vec3 d, const Vec3 *n, const real *off, int nf, real grow, real *t){
    real tin = 0, tout = 1;
    int outside = 0;
    for (int k = 0; k < nf; k++){
        real dist = v_dot(n[k], p) - off[k] - grow, den = v_dot(n[k], d);
        if (dist > 0) outside = 1;
        if (den == 0){ if (dist > 0) return 0; continue; }
        real tk = -dist / den;
        if (den < 0) tin = r_fmax(tin, tk); else tout = r_fmin(tout, tk);
        if (tin > tout) return 0;
    }
    if (!outside || tin >= *t) return 0;
    *t = tin;
    return 1;
}

// the swept sphere (p + t d, radius rad) against body o, at opos with its turn now
static int sweep_body(const Body *o, Vec3 opos, Vec3 p, Vec3 d, real rad, real *t){
    if (o->shape == BODY_SPHERE) return sweep_sphere(p, d, opos, o->half.x + rad, t);
    Mat3 rt = mat3_transpose(quat_to_mat3(o->rot));
    Vec3 lp = mat3_mul_vec(rt, v_sub(p, opos)), ld = mat3_mul_vec(rt, d);    // its frame
    if (o->shape == BODY_BOX){
        Vec3 n[6] = {vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0), vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1)};
        real off[6] = {o->half.x, o->half.x, o->half.y, o->half.y, o->half.z, o->half.z};
        return sweep_planes(lp, ld, n, off, 6, rad, t);
    }
    if (o->shape == BODY_HULL){
        const Convex *c = &o->g->hull;
        Vec3 n[MAX_HULL_FACES];
        real off[MAX_HULL_FACES];
        int nf = c->nf < MAX_HULL_FACES ? c->nf : MAX_HULL_FACES;
        for (int k = 0; k < nf; k++){ n[k] = c->f[k].n; off[k] = c->f[k].d; }
        return sweep_planes(lp, ld, n, off, nf, rad, t);
    }
    // a fixed mesh: its triangles near the path, each as a plane grown by rad, hit inside it
    TriList list = {NULL, 0, 0};
    Vec3 e = v_add(lp, ld);
    Aabb q = {vec3(r_fmin(lp.x, e.x) - rad, r_fmin(lp.y, e.y) - rad, r_fmin(lp.z, e.z) - rad),
              vec3(r_fmax(lp.x, e.x) + rad, r_fmax(lp.y, e.y) + rad, r_fmax(lp.z, e.z) + rad)};
    mesh_tree_query(&o->g->tree, q, tri_list_add, &list);
    int hit = 0;
    for (int k = 0; k < list.n; k++){
        const Vec3 *v = o->g->tree.tri + 3 * (size_t)list.tri[k];
        Vec3 nn = v_cross(v_sub(v[1], v[0]), v_sub(v[2], v[0]));
        real l = v_len(nn);
        if (l <= 0) continue;
        nn = v_scale(nn, 1.0f / l);
        real s = v_dot(nn, v_sub(lp, v[0]));
        if (r_fabs(s) <= rad) continue;                  // touching it already: the contacts' job
        if (s < 0){ nn = v_neg(nn); s = -s; }            // the side it comes from
        real dn = v_dot(nn, ld);
        if (dn >= 0) continue;                           // moving away
        real tt = (s - rad) / -dn;
        if (tt < 0 || tt >= *t) continue;
        Vec3 h = v_sub(v_add(lp, v_scale(ld, tt)), v_scale(nn, rad));   // on the triangle's plane
        Vec3 c0 = v_cross(v_sub(v[1], v[0]), v_sub(h, v[0])), c1 = v_cross(v_sub(v[2], v[1]), v_sub(h, v[1]));
        Vec3 c2 = v_cross(v_sub(v[0], v[2]), v_sub(h, v[2]));
        if (v_dot(c0, nn) < 0 || v_dot(c1, nn) < 0 || v_dot(c2, nn) < 0) continue;
        *t = tt;
        hit = 1;
    }
    free(list.tri);
    return hit;
}

// after a step: the bodies that moved fast since `start` (their positions then) swept, each
// against the others as they moved too: from where both were, along the difference of
// their paths (two bodies falling side by side do not meet). Each body against the others
// where the step left them (on all threads), then the bodies put where they stopped.
static void rigid_ccd(RigidWorld *w, const Vec3 *start){
    Vec3 *to = malloc((size_t)(w->nbodies > 0 ? w->nbodies : 1) * sizeof(Vec3));
    char *moved = calloc((size_t)(w->nbodies > 0 ? w->nbodies : 1), 1);
    if (!to || !moved){ free(to); free(moved); return; }
    #pragma omp parallel for schedule(dynamic, 16) if (w->nbodies >= 256)
    for (int i = 0; i < w->nbodies; i++){
        Body *b = &w->bodies[i];
        if (b->inv_mass == 0) continue;
        real rad = body_inner_radius(b);
        Vec3 p = start[i], d = v_sub(b->pos, p);
        real len = v_len(d);
        if (rad <= 0 || len <= CCD_FRACTION * rad) continue;
        real t = 1;
        int hit = 0;
        if (w->has_room){                                // the walls: n . x >= off + rad inside
            Vec3 n[6] = {vec3(-1, 0, 0), vec3(1, 0, 0), vec3(0, -1, 0), vec3(0, 1, 0), vec3(0, 0, -1), vec3(0, 0, 1)};
            real off[6] = {-w->room.lo.x, w->room.hi.x, -w->room.lo.y, w->room.hi.y, -w->room.lo.z, w->room.hi.z};
            for (int k = 0; k < 6; k++){                 // (each wall alone: the room is not a solid)
                real d0 = off[k] - rad - v_dot(n[k], p), dn = v_dot(n[k], d);
                if (d0 <= 0 || dn <= 0) continue;        // already at it, or moving away
                real tt = d0 / dn;
                if (tt < t){ t = tt; hit = 1; }
            }
        }
        for (int j = 0; j < w->nbodies; j++){
            if (j == i) continue;
            const Body *o = &w->bodies[j];
            Vec3 dr = v_sub(d, v_sub(o->pos, start[j]));   // the path relative to o's
            real l2 = v_dot(dr, dr);
            if (l2 <= 0) continue;
            // the path near its bounding sphere?
            Vec3 m = v_sub(start[j], p);
            real s = r_fmax(0, r_fmin(1, v_dot(m, dr) / l2));
            Vec3 q = v_sub(m, v_scale(dr, s));
            real reach = body_radius(o) + rad;
            if (v_dot(q, q) > reach * reach) continue;
            hit |= sweep_body(o, start[j], p, dr, rad, &t);
        }
        if (!hit) continue;
        // just before the hit: a little short of it, never behind the start
        real back = r_fmin(t, (SPECULATIVE_MIN + 0.05f * rad) / len);
        to[i] = v_add(p, v_scale(d, t - back));
        moved[i] = 1;
    }
    for (int i = 0; i < w->nbodies; i++) if (moved[i]) w->bodies[i].pos = to[i];
    free(to);
    free(moved);
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
    // the sleeping bodies (those that stay asleep) are fixed for this step: inv_mass 0
    int nsleep = w->sleep ? wake_bodies(w, dt, forces) : 0;
    real *sleep_m = nsleep ? malloc((size_t)w->nbodies * sizeof(real)) : NULL;
    if (sleep_m) for (int i = 0; i < w->nbodies; i++){
        Body *b = &w->bodies[i];
        sleep_m[i] = b->inv_mass;
        if (b->sleeping) b->inv_mass = 0;
    }
    // the contacts (the previous step's are kept for warm starting)
    { Contact *t = w->prev; w->prev = w->contacts; w->contacts = t;
      int tc = w->cap_prev; w->cap_prev = w->cap_contacts; w->cap_contacts = tc;
      w->nprev = w->ncontacts; w->ncontacts = 0; }
    memos_step(w);
    find_contacts(w, dt);
    // last step's contacts of each body a: they were found body by body (find_contacts), so
    // those of a are together, from prev_first[a] (-1: none) to the next body's
    int *prev_first = w->warm_start && w->nprev > 0 ? malloc((size_t)(w->nbodies + 1) * sizeof(int)) : NULL;
    if (prev_first){
        for (int i = 0; i <= w->nbodies; i++) prev_first[i] = -1;
        for (int q = w->nprev - 1; q >= 0; q--)
            if (w->prev[q].a >= 0 && w->prev[q].a < w->nbodies) prev_first[w->prev[q].a] = q;
    }
    for (int k = 0; k < w->ncontacts; k++){
        Contact *c = &w->contacts[k];
        onb_from_normal(c->n, &c->t1, &c->t2);
        // Warm start: the same contact last step (same bodies, within 1 unit) needed some
        // impulse to stay put; it starts from that, and the iterations only correct the change.
        // A resting stack then converges in a few steps instead of sinking and jittering.
        if (!w->warm_start) continue;
        int q0 = prev_first ? prev_first[c->a] : 0;
        if (q0 < 0) continue;
        for (int q = q0; q < w->nprev && (!prev_first || w->prev[q].a == c->a); q++){
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
    free(prev_first);
    Vec3 *start = malloc((size_t)(w->nbodies > 0 ? w->nbodies : 1) * sizeof(Vec3));   // for rigid_ccd
    if (start) for (int i = 0; i < w->nbodies; i++) start[i] = w->bodies[i].pos;
    // the bodies touching something (to settle them: step 1)
    char *resting = calloc((size_t)(w->nbodies > 0 ? w->nbodies : 1), 1);
    if (resting) for (int k = 0; k < w->ncontacts; k++){
        const Contact *c = &w->contacts[k];
        if (c->depth0 <= -SPECULATIVE_MIN) continue;
        resting[c->a] = 1;
        if (c->b >= 0) resting[c->b] = 1;
    }
    Mat3 *iw = malloc((size_t)(w->nbodies > 0 ? w->nbodies : 1) * sizeof(Mat3));
    size_t nc = (size_t)(w->ncontacts > 0 ? w->ncontacts : 1);
    int *act = malloc(nc * sizeof(int));
    char *bouncing = malloc(nc), *on = malloc(nc);
    real *relax = malloc(nc * sizeof(real));   // step 5
    SolveRows *rows = malloc(nc * sizeof(SolveRows));
    if (!iw || !act || !bouncing || !on || !relax || !rows) lr_oom("rigid");
    Mat3 zero = {{{0}}};
    // The bodies' and the contacts' own work on all the CPU's cores (OpenMP), when there is
    // enough of it; the iterations stay on one: each contact needs the ones before it (a
    // solver by colours on all cores, tried, lost more to the colours' barriers than it won)
    const int par_bodies = w->nbodies >= 4096, par_contacts = w->ncontacts >= 1024;
    (void)par_bodies; (void)par_contacts;

    for (int s = 0; s < substeps; s++){
        // 1. forces -> velocities (the external ones act through the whole step)
        #pragma omp parallel for if (par_bodies)
        for (int i = 0; i < w->nbodies; i++){
            Body *b = &w->bodies[i];
            // the world inverse inertia, from the rotation this substep starts with: made once
            // here for the torque and for the contacts below (2. and the iterations)
            iw[i] = b->inv_mass > 0 ? body_inv_inertia_world(b) : zero;
            if (b->inv_mass == 0) continue;
            Vec3 f = forces ? v_add(b->force, forces[i]) : b->force;
            b->vel = v_add(b->vel, v_scale(v_add(w->gravity, v_scale(f, b->inv_mass)), h));
            b->angvel = v_add(b->angvel, mat3_mul_vec(iw[i], v_scale(b->torque, h)));
            // A slow body resting on something loses its speed a little faster: the jitter of a
            // pile, a hull rocking on its rim for ever, die out instead of keeping it awake
            // (as Bullet's "additional damping")
            real settle = resting && resting[i] && body_slow(b) ? SETTLE_DAMPING : 0;
            b->vel = v_scale(b->vel, 1.0f / (1.0f + (w->linear_damping + settle) * h));
            b->angvel = v_scale(b->angvel, 1.0f / (1.0f + (w->angular_damping + settle) * h));
        }
        // 2. each contact where its bodies are now: arms, depth, effective masses, target
        // (on all threads: each contact alone); then, on one, the warm start applied
        #pragma omp parallel for if (par_contacts)
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
            solve_row_make(&rows[k].n, c->n, c->ra, c->rb, &iw[c->a], b ? &iw[c->b] : NULL);
            solve_row_make(&rows[k].t1, c->t1, c->ra, c->rb, &iw[c->a], b ? &iw[c->b] : NULL);
            solve_row_make(&rows[k].t2, c->t2, c->ra, c->rb, &iw[c->a], b ? &iw[c->b] : NULL);
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
            bouncing[k] = bounce > 0;
            c->bias = bounce > 0 ? bounce
                    : (c->depth < 0 ? c->depth / h : r_fmin(BAUMGARTE / h * r_fmax(c->depth - CONTACT_SLOP, 0), MAX_PUSH));
            if (bounce > 0){ c->jn = c->jt1 = c->jt2 = 0; c->vn0 = 0; on[k] = 1; continue; }
            // The iterations skip the gaps that cannot close in this substep, even twice as
            // fast (most of a pile's speculative contacts): should one close all the same
            // (pushed by another), the next substep finds it touching and pushes it out.
            on[k] = c->depth > -SPECULATIVE_MIN || c->jn > 0 || -c->depth < 2 * h * r_fmax(-vn, 0) + SPECULATIVE_MIN;
        }
        int nact = 0;
        for (int k = 0; k < w->ncontacts; k++){
            Contact *c = &w->contacts[k];
            if (on[k]) act[nact++] = k;
            // the impulse it had (last step's, or this step's last substep's), applied first
            if (bouncing[k] || (c->jn == 0 && c->jt1 == 0 && c->jt2 == 0)) continue;
            Body *a = &w->bodies[c->a], *b = c->b >= 0 ? &w->bodies[c->b] : NULL;
            apply(a, b, iw[c->a], b ? iw[c->b] : zero, c->ra, c->rb,
                  v_add(v_scale(c->n, c->jn), v_add(v_scale(c->t1, c->jt1), v_scale(c->t2, c->jt2))));
        }
        // 3. iterate
        for (int it = 0; it < w->iterations; it++) solve_pass(w, rows, act, nact, NULL);

        // 4. velocities -> positions
        #pragma omp parallel for if (par_bodies)
        for (int i = 0; i < w->nbodies; i++){
            Body *b = &w->bodies[i];
            if (b->inv_mass == 0) continue;
            b->pos = v_add(b->pos, v_scale(b->vel, h));
            b->rot = quat_integrate(b->rot, b->angvel, h);
        }

        // 5. relax (Box2D v3's "soft step"): the push out of an overlap was a velocity that
        // has now moved the bodies; it is taken back, so that it does not stay in them as
        // motion (in a pile it kept bodies rocking and never asleep). One pass towards the
        // speeds the contacts allow without a push: a gap may still close, a bounce keeps
        // its speed, a touching contact stops approaching and no more.
        #pragma omp parallel for if (par_contacts)
        for (int q = 0; q < nact; q++){
            int k = act[q];
            const Contact *c = &w->contacts[k];
            const Body *a = &w->bodies[c->a], *b = c->b >= 0 ? &w->bodies[c->b] : NULL;
            Vec3 pa = v_add(a->pos, c->ra), pb = b ? v_add(b->pos, c->rb) : c->lb;
            real depth = c->depth0 - v_dot(v_sub(pa, pb), c->n);   // now
            relax[k] = bouncing[k] ? c->bias : (depth < 0 ? depth / h : 0);
        }
        solve_pass(w, rows, act, nact, relax);
    }

    if (start) rigid_ccd(w, start);       // (the sleeping still fixed: inv_mass 0)
    free(start);
    free(relax);
    free(bouncing);
    free(on);
    free(resting);
    if (sleep_m){ for (int i = 0; i < w->nbodies; i++) w->bodies[i].inv_mass = sleep_m[i]; free(sleep_m); }
    if (w->sleep) fall_asleep(w, dt, forces);
    free(iw);
    free(rows);
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
