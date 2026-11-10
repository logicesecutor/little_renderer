#include "phys/rigid.h"

#include "core/rng.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONTACT_SLOP 0.5f        // overlap (units) left alone, so resting contacts do not jitter
#define BAUMGARTE 0.2f           // fraction of the remaining overlap removed per step
#define BOUNCE_THRESHOLD 60.0f   // approach speeds below this (units / s) do not bounce

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
    for (int k = 0; k < 3; k++){
        real c = v_dot(mat3_col(r, k), d);
        real h = k == 0 ? b->half.x : (k == 1 ? b->half.y : b->half.z);
        if (c < -h || c > h) return 0;
    }
    return 1;
}

void body_apply_impulse(Body *b, Vec3 j, Vec3 p){
    if (b->inv_mass == 0) return;
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

// The room: six walls, each a plane dot(n, x) = off with n pointing into the room.
static void collide_walls(RigidWorld *w, int ia){
    const Body *b = &w->bodies[ia];
    const Aabb *r = &w->room;
    Vec3 n[6] = {vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0), vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1)};
    real off[6] = {r->lo.x, -r->hi.x, r->lo.y, -r->hi.y, r->lo.z, -r->hi.z};
    if (b->shape == BODY_SPHERE){
        for (int k = 0; k < 6; k++){
            real d = v_dot(n[k], b->pos) - off[k] - b->half.x;
            if (d < 0) add_contact(w, ia, -1, v_sub(b->pos, v_scale(n[k], b->half.x)), n[k], -d);
        }
    } else {
        Vec3 c[8];
        box_corners(b, c);
        for (int k = 0; k < 6; k++)
            for (int i = 0; i < 8; i++){
                real d = v_dot(n[k], c[i]) - off[k];
                if (d < 0) add_contact(w, ia, -1, c[i], n[k], -d);
            }
    }
}

static void collide_sphere_sphere(RigidWorld *w, int ia, int ib){
    const Body *a = &w->bodies[ia], *b = &w->bodies[ib];
    Vec3 d = v_sub(a->pos, b->pos);
    real r = a->half.x + b->half.x, l2 = v_dot(d, d);
    if (l2 >= r * r) return;
    real l = r_sqrt(l2);
    Vec3 n = l > 1e-6f ? v_scale(d, 1.0f / l) : vec3(0, 1, 0);
    real depth = r - l;
    add_contact(w, ia, ib, v_add(b->pos, v_scale(n, b->half.x - 0.5f * depth)), n, depth);
}

// sphere a against box b
static void collide_sphere_box(RigidWorld *w, int ia, int ib){
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
        if (l2 >= rad * rad) return;
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

static void collide_box_box(RigidWorld *w, int ia, int ib){
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
        if (overlap < 0) return;                        // separating axis found
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
            if (sep > 0) continue;
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

static void find_contacts(RigidWorld *w){
    w->ncontacts = 0;
    for (int i = 0; i < w->nbodies; i++){
        Body *a = &w->bodies[i];
        if (w->has_room && a->inv_mass > 0) collide_walls(w, i);
        for (int j = i + 1; j < w->nbodies; j++){
            Body *b = &w->bodies[j];
            if (a->inv_mass == 0 && b->inv_mass == 0) continue;
            // quick reject with bounding spheres
            real ra = v_len(a->half), rb = v_len(b->half);
            Vec3 d = v_sub(a->pos, b->pos);
            if (v_dot(d, d) > (ra + rb) * (ra + rb)) continue;
            if (a->shape == BODY_SPHERE && b->shape == BODY_SPHERE) collide_sphere_sphere(w, i, j);
            else if (a->shape == BODY_SPHERE) collide_sphere_box(w, i, j);
            else if (b->shape == BODY_SPHERE){
                // sphere j against box i, then flip the normal so it points from j to i
                int first = w->ncontacts;
                collide_sphere_box(w, j, i);
                for (int k = first; k < w->ncontacts; k++){
                    Contact *c = &w->contacts[k];
                    c->a = i; c->b = j; c->n = v_neg(c->n);
                }
            } else collide_box_box(w, i, j);
        }
    }
}

// ---- the solver: sequential impulses (Catto, "Iterative Dynamics with Temporal Coherence") --
//
// At a contact the bodies must not keep approaching: the relative velocity along the normal
// must become >= 0 (or >= a small push-out "bias" that removes overlap, plus the bounce).
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

void rigid_step(RigidWorld *w, real dt){
    if (dt <= 0) return;
    // 1. forces -> velocities
    for (int i = 0; i < w->nbodies; i++){
        Body *b = &w->bodies[i];
        if (b->inv_mass == 0){ b->force = b->torque = vec3(0, 0, 0); continue; }
        b->vel = v_add(b->vel, v_scale(v_add(w->gravity, v_scale(b->force, b->inv_mass)), dt));
        b->angvel = v_add(b->angvel, mat3_mul_vec(body_inv_inertia_world(b), v_scale(b->torque, dt)));
        b->vel = v_scale(b->vel, 1.0f / (1.0f + w->linear_damping * dt));
        b->angvel = v_scale(b->angvel, 1.0f / (1.0f + w->angular_damping * dt));
        b->force = b->torque = vec3(0, 0, 0);
    }

    // 2. contacts (the previous step's are kept for warm starting)
    { Contact *t = w->prev; w->prev = w->contacts; w->contacts = t;
      int tc = w->cap_prev; w->cap_prev = w->cap_contacts; w->cap_contacts = tc;
      w->nprev = w->ncontacts; w->ncontacts = 0; }
    find_contacts(w);
    Mat3 *iw = malloc((size_t)(w->nbodies > 0 ? w->nbodies : 1) * sizeof(Mat3));
    if (!iw){ perror("malloc"); exit(1); }
    for (int i = 0; i < w->nbodies; i++) iw[i] = body_inv_inertia_world(&w->bodies[i]);
    Mat3 zero = {{{0}}};

    for (int k = 0; k < w->ncontacts; k++){
        Contact *c = &w->contacts[k];
        Body *a = &w->bodies[c->a], *b = c->b >= 0 ? &w->bodies[c->b] : NULL;
        Mat3 ia = iw[c->a], ib = b ? iw[c->b] : zero;
        c->ra = v_sub(c->p, a->pos);
        c->rb = b ? v_sub(c->p, b->pos) : vec3(0, 0, 0);
        onb_from_normal(c->n, &c->t1, &c->t2);
        c->mass_n = eff_mass(a, b, ia, ib, c->ra, c->rb, c->n);
        c->mass_t1 = eff_mass(a, b, ia, ib, c->ra, c->rb, c->t1);
        c->mass_t2 = eff_mass(a, b, ia, ib, c->ra, c->rb, c->t2);
        real vn = v_dot(rel_velocity(a, b, c->ra, c->rb), c->n);
        real e = b ? r_fmax(a->restitution, b->restitution) : r_fmax(a->restitution, w->wall_restitution);
        real bounce = vn < -BOUNCE_THRESHOLD ? -e * vn : 0;
        real push = BAUMGARTE / dt * r_fmax(c->depth - CONTACT_SLOP, 0);
        c->bias = r_fmax(bounce, push);
        // Warm start: the same contact last step (same bodies, within 1 unit) needed some
        // impulse to stay put; apply it now, and the iterations only correct the change.
        // A resting stack then converges in a few steps instead of sinking and jittering.
        if (w->warm_start && bounce == 0){
            for (int q = 0; q < w->nprev; q++){
                const Contact *o = &w->prev[q];
                if (o->a != c->a || o->b != c->b) continue;
                Vec3 d = v_sub(o->p, c->p);
                if (v_dot(d, d) > 1.0f || v_dot(o->n, c->n) < 0.95f) continue;
                c->jn = o->jn; c->jt1 = o->jt1; c->jt2 = o->jt2;
                apply(a, b, ia, ib, c->ra, c->rb,
                      v_add(v_scale(c->n, c->jn), v_add(v_scale(c->t1, c->jt1), v_scale(c->t2, c->jt2))));
                break;
            }
        }
    }

    // 3. iterate
    for (int it = 0; it < w->iterations; it++){
        for (int k = 0; k < w->ncontacts; k++){
            Contact *c = &w->contacts[k];
            Body *a = &w->bodies[c->a], *b = c->b >= 0 ? &w->bodies[c->b] : NULL;
            Mat3 ia = iw[c->a], ib = b ? iw[c->b] : zero;
            real mu = b ? r_sqrt(a->friction * b->friction) : r_sqrt(a->friction * w->wall_friction);

            Vec3 dv = rel_velocity(a, b, c->ra, c->rb);
            real dj = c->mass_n * (c->bias - v_dot(dv, c->n));
            real jn = r_fmax(c->jn + dj, 0);
            dj = jn - c->jn;
            c->jn = jn;
            apply(a, b, ia, ib, c->ra, c->rb, v_scale(c->n, dj));

            dv = rel_velocity(a, b, c->ra, c->rb);
            real lim = mu * c->jn;
            real jt = r_fmax(-lim, r_fmin(lim, c->jt1 - c->mass_t1 * v_dot(dv, c->t1)));
            apply(a, b, ia, ib, c->ra, c->rb, v_scale(c->t1, jt - c->jt1));
            c->jt1 = jt;
            dv = rel_velocity(a, b, c->ra, c->rb);
            jt = r_fmax(-lim, r_fmin(lim, c->jt2 - c->mass_t2 * v_dot(dv, c->t2)));
            apply(a, b, ia, ib, c->ra, c->rb, v_scale(c->t2, jt - c->jt2));
            c->jt2 = jt;
        }
    }
    free(iw);

    // 4. velocities -> positions
    for (int i = 0; i < w->nbodies; i++){
        Body *b = &w->bodies[i];
        if (b->inv_mass == 0) continue;
        b->pos = v_add(b->pos, v_scale(b->vel, dt));
        b->rot = quat_integrate(b->rot, b->angvel, dt);
    }
}

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
