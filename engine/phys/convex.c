#include "phys/convex.h"

#include "core/rng.h"   // onb_from_normal

#include <stdlib.h>
#include <string.h>

// ---- the hull --------------------------------------------------------------------------------

typedef struct { int v[3]; Vec3 n; real d; } HullTri;

static int hull_tri(HullTri *t, const Vec3 *p, int a, int b, int c){
    t->v[0] = a; t->v[1] = b; t->v[2] = c;
    Vec3 n = v_cross(v_sub(p[b], p[a]), v_sub(p[c], p[a]));
    real l = v_len(n);
    if (l <= 0) return 0;
    t->n = v_scale(n, 1 / l);
    t->d = v_dot(t->n, p[a]);
    return 1;
}

// the points furthest along many directions (spread over the sphere): the hull's corners;
// at most max of them, spread out (each next one the furthest from those kept)
static int support_points(const Vec3 *pts, int n, int max, Vec3 *out){
    int dirs = 6 * max, m = 0;
    Vec3 *cand = malloc((size_t)dirs * sizeof(Vec3));
    if (!cand) return 0;
    for (int k = 0; k < dirs; k++){
        // a Fibonacci spiral: evenly spread directions
        real z = 1 - (2 * k + 1) / (real)dirs, r = r_sqrt(r_fmax(0, 1 - z * z));
        real a = (real)k * 2.39996323f;
        Vec3 d = vec3(r * r_cos(a), r * r_sin(a), z);
        int best = 0;
        real bd = v_dot(pts[0], d);
        for (int i = 1; i < n; i++){ real s = v_dot(pts[i], d); if (s > bd){ bd = s; best = i; } }
        int dup = 0;
        for (int j = 0; j < m && !dup; j++){ Vec3 e = v_sub(cand[j], pts[best]); dup = v_dot(e, e) < 1e-10f; }
        if (!dup) cand[m++] = pts[best];
    }
    if (m <= max){ memcpy(out, cand, (size_t)m * sizeof(Vec3)); free(cand); return m; }
    real *dist = malloc((size_t)m * sizeof(real));
    if (!dist){ free(cand); return 0; }
    out[0] = cand[0];
    for (int j = 0; j < m; j++) dist[j] = v_len(v_sub(cand[j], out[0]));
    for (int k = 1; k < max; k++){
        int far = 0;
        for (int j = 1; j < m; j++) if (dist[j] > dist[far]) far = j;
        out[k] = cand[far];
        for (int j = 0; j < m; j++) dist[j] = r_fmin(dist[j], v_len(v_sub(cand[j], out[k])));
    }
    free(dist);
    free(cand);
    return max;
}

int convex_hull(const Vec3 *pts, int n, int max_verts, Convex *out){
    memset(out, 0, sizeof(*out));
    if (n < 4) return 0;
    if (max_verts < 8) max_verts = 8;
    Vec3 *p = malloc((size_t)max_verts * sizeof(Vec3));
    if (!p) return 0;
    int np = support_points(pts, n, max_verts, p);
    if (np < 4){ free(p); return 0; }
    Vec3 lo = p[0], hi = p[0];
    for (int i = 1; i < np; i++){ lo = vec3(r_fmin(lo.x, p[i].x), r_fmin(lo.y, p[i].y), r_fmin(lo.z, p[i].z));
                                  hi = vec3(r_fmax(hi.x, p[i].x), r_fmax(hi.y, p[i].y), r_fmax(hi.z, p[i].z)); }
    real size = v_len(v_sub(hi, lo)), eps = size * 1e-5f;
    // a first tetrahedron: 4 points far apart
    int i0 = 0, i1 = -1, i2 = -1, i3 = -1;
    real best = 0;
    for (int i = 0; i < np; i++){ real d = v_len(v_sub(p[i], p[i0])); if (d > best){ best = d; i1 = i; } }
    best = 0;
    for (int i = 0; i < np && i1 >= 0; i++){
        real d = v_len(v_cross(v_sub(p[i], p[i0]), v_sub(p[i1], p[i0])));
        if (d > best){ best = d; i2 = i; }
    }
    best = 0;
    if (i2 >= 0){
        Vec3 nn = v_norm(v_cross(v_sub(p[i1], p[i0]), v_sub(p[i2], p[i0])));
        for (int i = 0; i < np; i++){ real d = r_fabs(v_dot(nn, v_sub(p[i], p[i0]))); if (d > best){ best = d; i3 = i; } }
    }
    if (i1 < 0 || i2 < 0 || i3 < 0 || best < eps * 10){ free(p); return 0; }
    int cap = 8 * max_verts + 16;
    HullTri *t = malloc((size_t)cap * sizeof(HullTri)), *nt = malloc((size_t)cap * sizeof(HullTri));
    int (*hz)[2] = malloc((size_t)cap * 3 * sizeof(int[2]));
    char *vis = malloc((size_t)cap);
    if (!t || !nt || !hz || !vis){ free(p); free(t); free(nt); free(hz); free(vis); return 0; }
    Vec3 inner = v_scale(v_add(v_add(p[i0], p[i1]), v_add(p[i2], p[i3])), 0.25f);
    int tet[4][3] = {{i0, i1, i2}, {i0, i1, i3}, {i0, i2, i3}, {i1, i2, i3}};
    int nf = 0;
    for (int k = 0; k < 4; k++){
        hull_tri(&t[nf], p, tet[k][0], tet[k][1], tet[k][2]);
        if (v_dot(t[nf].n, inner) > t[nf].d){ int s = tet[k][1]; tet[k][1] = tet[k][2]; tet[k][2] = s;
                                               hull_tri(&t[nf], p, tet[k][0], tet[k][1], tet[k][2]); }
        nf++;
    }
    for (int q = 0; q < np; q++){
        if (q == i0 || q == i1 || q == i2 || q == i3) continue;
        int any = 0;
        for (int f = 0; f < nf; f++){ vis[f] = v_dot(t[f].n, p[q]) - t[f].d > eps; any |= vis[f]; }
        if (!any) continue;
        // the horizon: edges of visible faces whose other face is not visible
        int nh = 0;
        for (int f = 0; f < nf; f++){
            if (!vis[f]) continue;
            for (int k = 0; k < 3; k++){
                int a = t[f].v[k], b = t[f].v[(k + 1) % 3], twin_visible = 0;
                for (int g = 0; g < nf && !twin_visible; g++){
                    if (!vis[g] || g == f) continue;
                    for (int j = 0; j < 3; j++) if (t[g].v[j] == b && t[g].v[(j + 1) % 3] == a){ twin_visible = 1; break; }
                }
                if (!twin_visible && nh < cap * 3){ hz[nh][0] = a; hz[nh][1] = b; nh++; }
            }
        }
        int m = 0;
        for (int f = 0; f < nf; f++) if (!vis[f]) nt[m++] = t[f];
        for (int h = 0; h < nh && m < cap; h++) if (hull_tri(&nt[m], p, hz[h][0], hz[h][1], q)) m++;
        HullTri *sw = t; t = nt; nt = sw;
        nf = m;
    }
    // coplanar triangles become one face: their corners in order around the normal
    int *group = malloc((size_t)nf * sizeof(int));
    int *used = calloc((size_t)np, sizeof(int)), *remap = malloc((size_t)np * sizeof(int));
    out->f = malloc((size_t)nf * sizeof(ConvexFace));
    out->fv = malloc((size_t)nf * 3 * sizeof(int) + (size_t)np * sizeof(int) * 2);
    if (!group || !used || !remap || !out->f || !out->fv){
        free(group); free(used); free(remap); free(p); free(t); free(nt); free(hz); free(vis); convex_free(out); return 0;
    }
    for (int f = 0; f < nf; f++) group[f] = -1;
    int nfv = 0;
    for (int f = 0; f < nf; f++){
        if (group[f] >= 0) continue;
        int g = out->nf++;
        int ids[256], nid = 0;
        for (int h = f; h < nf; h++){
            if (group[h] >= 0) continue;
            if (h != f && (v_dot(t[h].n, t[f].n) < 1 - 1e-4f || r_fabs(t[h].d - t[f].d) > eps * 20)) continue;
            group[h] = g;
            for (int k = 0; k < 3; k++){
                int v = t[h].v[k], dup = 0;
                for (int j = 0; j < nid; j++) dup |= ids[j] == v;
                if (!dup && nid < 256) ids[nid++] = v;
            }
        }
        Vec3 c = vec3(0, 0, 0);
        for (int j = 0; j < nid; j++) c = v_add(c, p[ids[j]]);
        c = v_scale(c, 1.0f / nid);
        Vec3 t1, t2;
        onb_from_normal(t[f].n, &t1, &t2);
        real ang[256];
        for (int j = 0; j < nid; j++){ Vec3 d = v_sub(p[ids[j]], c); ang[j] = r_atan2(v_dot(d, t2), v_dot(d, t1)); }
        for (int j = 1; j < nid; j++)                       // by angle: counter-clockwise seen from outside
            for (int k = j; k > 0 && ang[k] < ang[k - 1]; k--){
                real a = ang[k]; ang[k] = ang[k - 1]; ang[k - 1] = a;
                int b = ids[k]; ids[k] = ids[k - 1]; ids[k - 1] = b;
            }
        // t1 x t2 = n? then increasing angle is counter-clockwise around n
        if (v_dot(v_cross(t1, t2), t[f].n) < 0)
            for (int j = 0; j < nid / 2; j++){ int b = ids[j]; ids[j] = ids[nid - 1 - j]; ids[nid - 1 - j] = b; }
        out->f[g].n = t[f].n;
        out->f[g].d = t[f].d;
        out->f[g].first = nfv;
        out->f[g].count = nid;
        for (int j = 0; j < nid; j++){ out->fv[nfv++] = ids[j]; used[ids[j]] = 1; }
    }
    // only the corners the faces use
    out->v = malloc((size_t)np * sizeof(Vec3));
    for (int i = 0; i < np; i++) if (used[i]){ remap[i] = out->nv; out->v[out->nv++] = p[i]; } else remap[i] = -1;
    for (int k = 0; k < nfv; k++) out->fv[k] = remap[out->fv[k]];
    // the edges: each directed edge of a face, and the face of its reverse
    int nv = out->nv;
    int *owner = malloc((size_t)nv * nv * sizeof(int));
    out->e = malloc((size_t)nfv * sizeof(ConvexEdge));
    if (!owner || !out->e){ free(owner); free(group); free(used); free(remap); free(p); free(t); free(nt); free(hz); free(vis);
                            convex_free(out); return 0; }
    for (int k = 0; k < nv * nv; k++) owner[k] = -1;
    for (int g = 0; g < out->nf; g++)
        for (int j = 0; j < out->f[g].count; j++){
            int a = out->fv[out->f[g].first + j], b = out->fv[out->f[g].first + (j + 1) % out->f[g].count];
            owner[a * nv + b] = g;
        }
    for (int a = 0; a < nv; a++)
        for (int b = a + 1; b < nv; b++){
            if (owner[a * nv + b] < 0 || owner[b * nv + a] < 0) continue;
            ConvexEdge *e = &out->e[out->ne++];
            e->a = a; e->b = b; e->fa = owner[a * nv + b]; e->fb = owner[b * nv + a];
        }
    free(owner); free(group); free(used); free(remap); free(p); free(t); free(nt); free(hz); free(vis);
    return out->nf >= 4;
}

int convex_box(Vec3 half, Convex *out){
    Vec3 c[8];
    for (int i = 0; i < 8; i++) c[i] = vec3((i & 1) ? half.x : -half.x, (i & 2) ? half.y : -half.y, (i & 4) ? half.z : -half.z);
    return convex_hull(c, 8, 8, out);
}

void convex_free(Convex *c){
    free(c->v); free(c->f); free(c->fv); free(c->e);
    memset(c, 0, sizeof(*c));
}

void convex_mass(const Convex *c, real *volume, Vec3 *com, Mat3 *inertia){
    Vec3 r = vec3(0, 0, 0);
    for (int i = 0; i < c->nv; i++) r = v_add(r, c->v[i]);
    r = v_scale(r, 1.0f / (c->nv > 0 ? c->nv : 1));
    real vol = 0;
    Vec3 m1 = vec3(0, 0, 0);
    real C[3][3] = {{0}};
    for (int f = 0; f < c->nf; f++){
        const ConvexFace *F = &c->f[f];
        Vec3 a = v_sub(c->v[c->fv[F->first]], r);
        for (int j = 1; j + 1 < F->count; j++){
            Vec3 b = v_sub(c->v[c->fv[F->first + j]], r), d = v_sub(c->v[c->fv[F->first + j + 1]], r);
            real det = v_dot(a, v_cross(b, d));
            vol += det / 6;
            Vec3 s = v_add(v_add(a, b), d);
            m1 = v_add(m1, v_scale(s, det / 24));
            real va[3] = {a.x, a.y, a.z}, vb[3] = {b.x, b.y, b.z}, vd[3] = {d.x, d.y, d.z}, vs[3] = {s.x, s.y, s.z};
            for (int i = 0; i < 3; i++)
                for (int k = 0; k < 3; k++)
                    C[i][k] += det / 120 * (va[i] * va[k] + vb[i] * vb[k] + vd[i] * vd[k] + vs[i] * vs[k]);
        }
    }
    Vec3 cm = vol > 0 ? v_scale(m1, 1 / vol) : vec3(0, 0, 0);
    real g[3] = {cm.x, cm.y, cm.z};
    for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) C[i][k] -= vol * g[i] * g[k];
    real tr = C[0][0] + C[1][1] + C[2][2];
    for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) inertia->m[i][k] = (i == k ? tr : 0) - C[i][k];
    *volume = vol;
    *com = v_add(r, cm);
}

int convex_contains(const Convex *c, Vec3 p){
    for (int f = 0; f < c->nf; f++) if (v_dot(c->f[f].n, p) > c->f[f].d) return 0;
    return 1;
}

void convex_place(WorldConvex *w, const Convex *c, Vec3 pos, Mat3 rot, Vec3 *v, Vec3 *n, real *d){
    w->c = c; w->v = v; w->n = n; w->d = d;
    Vec3 s = vec3(0, 0, 0);
    for (int i = 0; i < c->nv; i++){ v[i] = v_add(pos, mat3_mul_vec(rot, c->v[i])); s = v_add(s, v[i]); }
    w->center = v_scale(s, 1.0f / (c->nv > 0 ? c->nv : 1));
    for (int f = 0; f < c->nf; f++){ n[f] = mat3_mul_vec(rot, c->f[f].n); d[f] = c->f[f].d + v_dot(n[f], pos); }
}

// ---- contacts --------------------------------------------------------------------------------

#define MAXPOLY 160

// Sutherland-Hodgman: the part of polygon `in` where dot(p, d) <= off
static int clip_poly(const Vec3 *in, int n, Vec3 d, real off, Vec3 *out){
    int m = 0;
    for (int i = 0; i < n && m < MAXPOLY - 1; i++){
        Vec3 p = in[i], q = in[(i + 1) % n];
        real dp = v_dot(p, d) - off, dq = v_dot(q, d) - off;
        if (dp <= 0) out[m++] = p;
        if ((dp < 0 && dq > 0) || (dp > 0 && dq < 0)) out[m++] = v_add(p, v_scale(v_sub(q, p), dp / (dp - dq)));
    }
    return m;
}

// the incident polygon clipped to the sides of the reference face (loop `ref`, counter-
// clockwise around its outward normal nr, plane nr . x = dr); the points below it are the
// contacts, with normal `cn` (from b to a)
static int face_contacts(const Vec3 *ref, int nref, Vec3 nr, real dr, const Vec3 *inc, int ninc, Vec3 cn, ConvexContact *out){
    Vec3 a[MAXPOLY], b[MAXPOLY];
    int n = ninc < MAXPOLY ? ninc : MAXPOLY;
    memcpy(a, inc, (size_t)n * sizeof(Vec3));
    for (int k = 0; k < nref && n > 0; k++){
        Vec3 p = ref[k], q = ref[(k + 1) % nref];
        Vec3 side = v_cross(v_sub(q, p), nr);
        n = clip_poly(a, n, side, v_dot(side, p), b);
        memcpy(a, b, (size_t)n * sizeof(Vec3));
    }
    ConvexContact c[MAXPOLY];
    int m = 0;
    for (int k = 0; k < n; k++){
        real depth = dr - v_dot(nr, a[k]);
        if (depth < 0) continue;
        c[m].p = v_add(a[k], v_scale(nr, 0.5f * depth));
        c[m].n = cn;
        c[m].depth = depth;
        m++;
    }
    m = convex_reduce(c, m, CONVEX_MAX_CONTACTS);
    memcpy(out, c, (size_t)m * sizeof(ConvexContact));
    return m;
}

// closest points of segments p0-p1 and q0-q1
static void closest_seg(Vec3 p0, Vec3 p1, Vec3 q0, Vec3 q1, Vec3 *c0, Vec3 *c1){
    Vec3 d1 = v_sub(p1, p0), d2 = v_sub(q1, q0), r = v_sub(p0, q0);
    real a = v_dot(d1, d1), e = v_dot(d2, d2), f = v_dot(d2, r), c = v_dot(d1, r), b = v_dot(d1, d2);
    real s = 0, t = 0, den = a * e - b * b;
    if (den > 1e-12f * a * e) s = r_fmax(0, r_fmin(1, (b * f - c * e) / den));
    t = e > 0 ? (b * s + f) / e : 0;
    if (t < 0){ t = 0; s = a > 0 ? r_fmax(0, r_fmin(1, -c / a)) : 0; }
    else if (t > 1){ t = 1; s = a > 0 ? r_fmax(0, r_fmin(1, (b - c) / a)) : 0; }
    *c0 = v_add(p0, v_scale(d1, s));
    *c1 = v_add(q0, v_scale(d2, t));
}

static void face_loop(const WorldConvex *w, int f, Vec3 *out){
    const ConvexFace *F = &w->c->f[f];
    for (int j = 0; j < F->count; j++) out[j] = w->v[w->c->fv[F->first + j]];
}

// the face of w most opposed to direction n
static int incident_face(const WorldConvex *w, Vec3 n){
    int best = 0;
    real bd = 1e30f;
    for (int f = 0; f < w->c->nf; f++){ real d = v_dot(w->n[f], n); if (d < bd){ bd = d; best = f; } }
    return best;
}

static real min_along(const Vec3 *v, int n, Vec3 d){
    real m = 1e30f;
    for (int i = 0; i < n; i++){ real s = v_dot(v[i], d); if (s < m) m = s; }
    return m;
}

// faces preferred: an edge axis only when clearly better, a's faces before b's (so the
// choice does not flip between steps for a resting contact)
#define PREFER_B(sa, sb) ((sa) < 1.02f * (sb) - 0.01f)
#define PREFER_EDGE(sf, se) ((sf) < 1.1f * (se) - 0.1f)

int convex_collide(const WorldConvex *A, const WorldConvex *B, ConvexContact *out){
    const Convex *ca = A->c, *cb = B->c;
    real sa = -1e30f, sb = -1e30f, se = -1e30f;
    int fa = -1, fb = -1, ea = -1, eb = -1;
    Vec3 le = vec3(0, 1, 0);
    for (int f = 0; f < ca->nf; f++){
        real s = min_along(B->v, cb->nv, A->n[f]) - A->d[f];
        if (s > 0) return 0;
        if (s > sa){ sa = s; fa = f; }
    }
    for (int f = 0; f < cb->nf; f++){
        real s = min_along(A->v, ca->nv, B->n[f]) - B->d[f];
        if (s > 0) return 0;
        if (s > sb){ sb = s; fb = f; }
    }
    for (int i = 0; i < ca->ne; i++){
        const ConvexEdge *E = &ca->e[i];
        Vec3 a = A->n[E->fa], b = A->n[E->fb];
        Vec3 pa = A->v[E->a], da = v_sub(A->v[E->b], pa);
        Vec3 bxa = v_cross(b, a);
        for (int j = 0; j < cb->ne; j++){
            const ConvexEdge *G = &cb->e[j];
            Vec3 c = v_neg(B->n[G->fa]), d = v_neg(B->n[G->fb]);
            real cba = v_dot(c, bxa), dba = v_dot(d, bxa);
            if (cba * dba >= 0) continue;
            Vec3 dxc = v_cross(d, c);
            real adc = v_dot(a, dxc), bdc = v_dot(b, dxc);
            if (adc * bdc >= 0 || cba * bdc <= 0) continue;   // not a face of the Minkowski difference
            Vec3 pb = B->v[G->a], db = v_sub(B->v[G->b], pb);
            Vec3 L = v_cross(da, db);
            real ll = v_len(L);
            if (ll < 1e-6f * v_len(da) * v_len(db)) continue;
            L = v_scale(L, 1 / ll);
            if (v_dot(L, v_sub(pa, A->center)) < 0) L = v_neg(L);
            real s = v_dot(L, v_sub(pb, pa));
            if (s > 0) return 0;
            if (s > se){ se = s; ea = i; eb = j; le = L; }
        }
    }
    real sf = PREFER_B(sa, sb) ? sb : sa;
    if (ea >= 0 && PREFER_EDGE(sf, se)){
        const ConvexEdge *E = &ca->e[ea], *G = &cb->e[eb];
        Vec3 c0, c1;
        closest_seg(A->v[E->a], A->v[E->b], B->v[G->a], B->v[G->b], &c0, &c1);
        out[0].p = v_scale(v_add(c0, c1), 0.5f);
        out[0].n = v_neg(le);
        out[0].depth = -se;
        return 1;
    }
    Vec3 ref[MAXPOLY], inc[MAXPOLY];
    if (PREFER_B(sa, sb)){                     // b's face
        int fi = incident_face(A, B->n[fb]);
        face_loop(B, fb, ref);
        face_loop(A, fi, inc);
        return face_contacts(ref, cb->f[fb].count, B->n[fb], B->d[fb], inc, ca->f[fi].count, B->n[fb], out);
    }
    int fi = incident_face(B, A->n[fa]);
    face_loop(A, fa, ref);
    face_loop(B, fi, inc);
    return face_contacts(ref, ca->f[fa].count, A->n[fa], A->d[fa], inc, cb->f[fi].count, v_neg(A->n[fa]), out);
}

int convex_collide_triangle(const WorldConvex *A, const Vec3 t[3], ConvexContact *out){
    const Convex *ca = A->c;
    Vec3 nt = v_cross(v_sub(t[1], t[0]), v_sub(t[2], t[0]));
    real ln = v_len(nt);
    if (ln <= 0) return 0;
    nt = v_scale(nt, 1 / ln);
    real dt = v_dot(nt, t[0]);
    // the triangle's face, both sides (a mesh need not be closed)
    real sp = min_along(A->v, ca->nv, nt) - dt, sm = min_along(A->v, ca->nv, v_neg(nt)) + dt;
    real st = sp > sm ? sp : sm;
    Vec3 ns = sp > sm ? nt : v_neg(nt);   // pointing towards a
    if (st > 0) return 0;
    real sa = -1e30f, se = -1e30f;
    int fa = -1, ea = -1, ek = -1;
    Vec3 le = nt;
    for (int f = 0; f < ca->nf; f++){
        real s = min_along(t, 3, A->n[f]) - A->d[f];
        if (s > 0) return 0;
        if (s > sa){ sa = s; fa = f; }
    }
    // a's edges against the triangle's: only a's edges on its outline seen along the
    // triangle's edge (where a's support along the axis is that edge)
    for (int k = 0; k < 3; k++){
        Vec3 e = v_sub(t[(k + 1) % 3], t[k]);
        for (int i = 0; i < ca->ne; i++){
            const ConvexEdge *E = &ca->e[i];
            Vec3 na = A->n[E->fa], nb = A->n[E->fb];
            if (v_dot(na, e) * v_dot(nb, e) >= 0) continue;
            Vec3 pa = A->v[E->a], h = v_sub(A->v[E->b], pa);
            Vec3 L = v_cross(h, e);
            real ll = v_len(L);
            if (ll < 1e-6f * v_len(h) * v_len(e)) continue;
            L = v_scale(L, 1 / ll);
            if (v_dot(L, v_add(na, nb)) < 0) L = v_neg(L);
            real s = min_along(t, 3, L) - v_dot(L, pa);
            if (s > 0) return 0;
            if (s > se){ se = s; ea = i; ek = k; le = L; }
        }
    }
    real sf = PREFER_B(sa, st) ? st : sa;
    if (ea >= 0 && PREFER_EDGE(sf, se)){
        const ConvexEdge *E = &ca->e[ea];
        Vec3 c0, c1;
        closest_seg(A->v[E->a], A->v[E->b], t[ek], t[(ek + 1) % 3], &c0, &c1);
        out[0].p = v_scale(v_add(c0, c1), 0.5f);
        out[0].n = v_neg(le);
        out[0].depth = -se;
        return 1;
    }
    Vec3 ref[MAXPOLY], inc[MAXPOLY];
    if (PREFER_B(sa, st)){                     // the triangle's face
        Vec3 tl[3] = {t[0], t[1], t[2]};
        if (v_dot(ns, nt) < 0){ tl[1] = t[2]; tl[2] = t[1]; }
        int fi = incident_face(A, ns);
        face_loop(A, fi, inc);
        return face_contacts(tl, 3, ns, v_dot(ns, t[0]), inc, ca->f[fi].count, ns, out);
    }
    face_loop(A, fa, ref);
    return face_contacts(ref, ca->f[fa].count, A->n[fa], A->d[fa], t, 3, v_neg(A->n[fa]), out);
}

// the point of triangle a b c closest to p (Ericson, Real-Time Collision Detection 5.1.5)
static Vec3 closest_on_triangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c){
    Vec3 ab = v_sub(b, a), ac = v_sub(c, a), ap = v_sub(p, a);
    real d1 = v_dot(ab, ap), d2 = v_dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return a;
    Vec3 bp = v_sub(p, b);
    real d3 = v_dot(ab, bp), d4 = v_dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return b;
    real vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return v_add(a, v_scale(ab, d1 / (d1 - d3)));
    Vec3 cp = v_sub(p, c);
    real d5 = v_dot(ab, cp), d6 = v_dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return c;
    real vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return v_add(a, v_scale(ac, d2 / (d2 - d6)));
    real va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
        return v_add(b, v_scale(v_sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    real den = 1 / (va + vb + vc);
    return v_add(a, v_add(v_scale(ab, vb * den), v_scale(ac, vc * den)));
}

int triangle_collide_sphere(Vec3 c, real r, const Vec3 t[3], ConvexContact *out){
    Vec3 q = closest_on_triangle(c, t[0], t[1], t[2]);
    Vec3 d = v_sub(c, q);
    real l2 = v_dot(d, d);
    if (l2 >= r * r) return 0;
    real l = r_sqrt(l2);
    Vec3 n;
    if (l > 1e-4f) n = v_scale(d, 1 / l);
    else {
        n = v_norm(v_cross(v_sub(t[1], t[0]), v_sub(t[2], t[0])));
        l = 0;
    }
    out[0].p = q; out[0].n = n; out[0].depth = r - l;
    return 1;
}

int convex_collide_sphere(Vec3 c, real r, const WorldConvex *B, ConvexContact *out){
    const Convex *cb = B->c;
    real dmax = -1e30f;
    int fm = 0;
    for (int f = 0; f < cb->nf; f++){ real s = v_dot(B->n[f], c) - B->d[f]; if (s > dmax){ dmax = s; fm = f; } }
    if (dmax > r) return 0;
    if (dmax <= 0){                              // the centre is inside: out by the nearest face
        out[0].n = B->n[fm];
        out[0].depth = r - dmax;
        out[0].p = v_sub(c, v_scale(B->n[fm], dmax));
        return 1;
    }
    // outside: the closest point of the faces that face the centre
    real best = 1e30f;
    Vec3 bq = c;
    for (int f = 0; f < cb->nf; f++){
        if (v_dot(B->n[f], c) - B->d[f] <= 0) continue;
        const ConvexFace *F = &cb->f[f];
        Vec3 a = B->v[cb->fv[F->first]];
        for (int j = 1; j + 1 < F->count; j++){
            Vec3 q = closest_on_triangle(c, a, B->v[cb->fv[F->first + j]], B->v[cb->fv[F->first + j + 1]]);
            real d = v_len(v_sub(c, q));
            if (d < best){ best = d; bq = q; }
        }
    }
    if (best >= r) return 0;
    out[0].n = best > 1e-5f ? v_scale(v_sub(c, bq), 1 / best) : B->n[fm];
    out[0].depth = r - best;
    out[0].p = bq;
    return 1;
}

int convex_reduce(ConvexContact *c, int n, int keep){
    if (n <= keep || keep < 1) return n;
    int pick[8], np = 0;
    int i0 = 0;
    for (int i = 1; i < n; i++) if (c[i].depth > c[i0].depth) i0 = i;
    pick[np++] = i0;
    if (keep >= 2){
        int i1 = -1; real bd = -1;
        for (int i = 0; i < n; i++){ real d = v_len(v_sub(c[i].p, c[i0].p)); if (d > bd){ bd = d; i1 = i; } }
        if (i1 >= 0 && i1 != i0) pick[np++] = i1;
    }
    if (keep >= 3 && np == 2){
        int i2 = -1; real ba = -1;
        for (int i = 0; i < n; i++){
            real a = v_len(v_cross(v_sub(c[i].p, c[pick[0]].p), v_sub(c[i].p, c[pick[1]].p)));
            if (a > ba){ ba = a; i2 = i; }
        }
        if (i2 >= 0 && ba > 0) pick[np++] = i2;
    }
    if (keep >= 4 && np == 3){
        // the point that adds the most area to the triangle (outside of one of its edges)
        int i3 = -1; real ba = 0;
        Vec3 nn = v_cross(v_sub(c[pick[1]].p, c[pick[0]].p), v_sub(c[pick[2]].p, c[pick[0]].p));
        for (int i = 0; i < n; i++){
            for (int k = 0; k < 3; k++){
                Vec3 p = c[pick[k]].p, q = c[pick[(k + 1) % 3]].p;
                real a = -v_dot(v_cross(v_sub(q, p), v_sub(c[i].p, p)), nn);
                if (a > ba){ ba = a; i3 = i; }
            }
        }
        if (i3 >= 0) pick[np++] = i3;
    }
    ConvexContact t[8];
    for (int k = 0; k < np; k++) t[k] = c[pick[k]];
    memcpy(c, t, (size_t)np * sizeof(ConvexContact));
    return np;
}

// ---- a BVH of triangles ------------------------------------------------------------------------

static Aabb tri_box(const Vec3 *t){
    Aabb b = {t[0], t[0]};
    for (int k = 1; k < 3; k++){
        b.lo = vec3(r_fmin(b.lo.x, t[k].x), r_fmin(b.lo.y, t[k].y), r_fmin(b.lo.z, t[k].z));
        b.hi = vec3(r_fmax(b.hi.x, t[k].x), r_fmax(b.hi.y, t[k].y), r_fmax(b.hi.z, t[k].z));
    }
    return b;
}

static Aabb box_join(Aabb a, Aabb b){
    return (Aabb){vec3(r_fmin(a.lo.x, b.lo.x), r_fmin(a.lo.y, b.lo.y), r_fmin(a.lo.z, b.lo.z)),
                  vec3(r_fmax(a.hi.x, b.hi.x), r_fmax(a.hi.y, b.hi.y), r_fmax(a.hi.z, b.hi.z))};
}

static real tri_axis(const MeshTree *t, int i, int axis){
    const Vec3 *v = t->tri + 3 * (size_t)i;
    real s = 0;
    for (int k = 0; k < 3; k++) s += axis == 0 ? v[k].x : (axis == 1 ? v[k].y : v[k].z);
    return s;
}

static int tree_node(MeshTree *t, int first, int count){
    int id = t->nnodes++;
    MeshNode *nd = &t->nodes[id];
    Aabb b = tri_box(t->tri + 3 * (size_t)t->order[first]);
    for (int i = 1; i < count; i++) b = box_join(b, tri_box(t->tri + 3 * (size_t)t->order[first + i]));
    nd->box = b;
    if (count <= 4){ nd->first = first; nd->count = count; nd->left = nd->right = -1; return id; }
    Vec3 e = v_sub(b.hi, b.lo);
    int axis = e.x > e.y ? (e.x > e.z ? 0 : 2) : (e.y > e.z ? 1 : 2);
    // median split (a quickselect on the centres)
    int lo = first, hi = first + count - 1, mid = first + count / 2;
    while (lo < hi){
        real pv = tri_axis(t, t->order[(lo + hi) / 2], axis);
        int i = lo, j = hi;
        while (i <= j){
            while (tri_axis(t, t->order[i], axis) < pv) i++;
            while (tri_axis(t, t->order[j], axis) > pv) j--;
            if (i <= j){ int s = t->order[i]; t->order[i] = t->order[j]; t->order[j] = s; i++; j--; }
        }
        if (mid <= j) hi = j; else if (mid >= i) lo = i; else break;
    }
    nd->count = 0;
    int l = tree_node(t, first, mid - first);
    int r = tree_node(t, mid, first + count - mid);
    t->nodes[id].left = l;
    t->nodes[id].right = r;
    return id;
}

int mesh_tree_build(MeshTree *t, const Triangle *tris, int n, Vec3 offset){
    memset(t, 0, sizeof(*t));
    if (n <= 0) return 0;
    t->tri = malloc((size_t)n * 3 * sizeof(Vec3));
    t->order = malloc((size_t)n * sizeof(int));
    t->nodes = malloc((size_t)n * 2 * sizeof(MeshNode));
    if (!t->tri || !t->order || !t->nodes){ mesh_tree_free(t); return 0; }
    for (int i = 0; i < n; i++){
        t->tri[3 * (size_t)i] = v_sub(tris[i].q, offset);
        t->tri[3 * (size_t)i + 1] = v_sub(v_add(tris[i].q, tris[i].u), offset);
        t->tri[3 * (size_t)i + 2] = v_sub(v_add(tris[i].q, tris[i].v), offset);
        t->order[i] = i;
    }
    t->ntris = n;
    tree_node(t, 0, n);
    return 1;
}

void mesh_tree_free(MeshTree *t){
    free(t->tri); free(t->order); free(t->nodes);
    memset(t, 0, sizeof(*t));
}

static int boxes_meet(Aabb a, Aabb b){
    return a.lo.x <= b.hi.x && a.hi.x >= b.lo.x && a.lo.y <= b.hi.y && a.hi.y >= b.lo.y && a.lo.z <= b.hi.z && a.hi.z >= b.lo.z;
}

void mesh_tree_query(const MeshTree *t, Aabb box, void (*fn)(void *user, int tri), void *user){
    if (!t->nnodes) return;
    int stack[128], sp = 0;
    stack[sp++] = 0;
    while (sp > 0){
        const MeshNode *nd = &t->nodes[stack[--sp]];
        if (!boxes_meet(nd->box, box)) continue;
        if (nd->count > 0){
            for (int i = 0; i < nd->count; i++){
                int k = t->order[nd->first + i];
                if (boxes_meet(tri_box(t->tri + 3 * (size_t)k), box)) fn(user, k);
            }
        } else if (sp + 2 <= 128){
            stack[sp++] = nd->left;
            stack[sp++] = nd->right;
        }
    }
}

// ---- occupancy ------------------------------------------------------------------------------

static Vec3 occ_center(Vec3 half, int x, int y, int z){
    return vec3(-half.x + (x + 0.5f) * 2 * half.x / OCC_RES, -half.y + (y + 0.5f) * 2 * half.y / OCC_RES,
                -half.z + (z + 0.5f) * 2 * half.z / OCC_RES);
}

void occupancy_convex(const Convex *c, Vec3 half, uint32_t *bits){
    memset(bits, 0, OCC_WORDS * sizeof(uint32_t));
    for (int z = 0; z < OCC_RES; z++)
        for (int y = 0; y < OCC_RES; y++)
            for (int x = 0; x < OCC_RES; x++)
                if (convex_contains(c, occ_center(half, x, y, z))){
                    int i = (z * OCC_RES + y) * OCC_RES + x;
                    bits[i >> 5] |= 1u << (i & 31);
                }
}

typedef struct { const MeshTree *t; int axis; real u, v; real *hits; int nhits, cap; } RowQuery;

static real cmp3(Vec3 p, int k){ return k == 0 ? p.x : (k == 1 ? p.y : p.z); }

// where the line (u, v) along `axis` crosses triangle i, if it does
static void row_hit(void *user, int i){
    RowQuery *q = user;
    const Vec3 *t = q->t->tri + 3 * (size_t)i;
    int a = q->axis, b = (a + 1) % 3, c = (a + 2) % 3;
    real x0 = cmp3(t[0], b) - q->u, y0 = cmp3(t[0], c) - q->v;
    real x1 = cmp3(t[1], b) - q->u, y1 = cmp3(t[1], c) - q->v;
    real x2 = cmp3(t[2], b) - q->u, y2 = cmp3(t[2], c) - q->v;
    // the signs of the three edge functions at the line: all the same = inside
    real e0 = x0 * y1 - y0 * x1, e1 = x1 * y2 - y1 * x2, e2 = x2 * y0 - y2 * x0;
    if (!((e0 > 0 && e1 > 0 && e2 > 0) || (e0 < 0 && e1 < 0 && e2 < 0))) return;
    real s = e0 + e1 + e2;
    real h = (e1 * cmp3(t[0], a) + e2 * cmp3(t[1], a) + e0 * cmp3(t[2], a)) / s;
    if (q->nhits < q->cap) q->hits[q->nhits++] = h;
}

static int cmp_real(const void *a, const void *b){
    real x = *(const real *)a, y = *(const real *)b;
    return x < y ? -1 : (x > y);
}

void occupancy_mesh(const MeshTree *t, Vec3 half, uint32_t *bits){
    memset(bits, 0, OCC_WORDS * sizeof(uint32_t));
    unsigned char *votes = calloc((size_t)OCC_RES * OCC_RES * OCC_RES, 1);
    real *hits = malloc(4096 * sizeof(real));
    if (!votes || !hits){ free(votes); free(hits); return; }
    real hv[3] = {half.x, half.y, half.z};
    for (int a = 0; a < 3; a++){
        int b = (a + 1) % 3, c = (a + 2) % 3;
        for (int j = 0; j < OCC_RES; j++)
            for (int i = 0; i < OCC_RES; i++){
                RowQuery q = {t, a, -hv[b] + (i + 0.5f) * 2 * hv[b] / OCC_RES, -hv[c] + (j + 0.5f) * 2 * hv[c] / OCC_RES, hits, 0, 4096};
                real lo[3], hi[3];
                lo[a] = -1e30f; hi[a] = 1e30f;
                lo[b] = hi[b] = q.u; lo[c] = hi[c] = q.v;
                Aabb line = {vec3(lo[0], lo[1], lo[2]), vec3(hi[0], hi[1], hi[2])};
                mesh_tree_query(t, line, row_hit, &q);
                if (q.nhits < 2) continue;
                qsort(hits, (size_t)q.nhits, sizeof(real), cmp_real);
                int k = 0;
                for (int s = 0; s < OCC_RES; s++){
                    real at = -hv[a] + (s + 0.5f) * 2 * hv[a] / OCC_RES;
                    while (k < q.nhits && hits[k] < at) k++;
                    if (!(k & 1)) continue;
                    int cell[3];
                    cell[a] = s; cell[b] = i; cell[c] = j;
                    votes[(cell[2] * OCC_RES + cell[1]) * OCC_RES + cell[0]]++;
                }
            }
    }
    for (int i = 0; i < OCC_RES * OCC_RES * OCC_RES; i++) if (votes[i] >= 2) bits[i >> 5] |= 1u << (i & 31);
    free(votes);
    free(hits);
}

int occupancy_at(const uint32_t *bits, Vec3 half, Vec3 l){
    int x = (int)r_floor((l.x + half.x) / (2 * half.x) * OCC_RES);
    int y = (int)r_floor((l.y + half.y) / (2 * half.y) * OCC_RES);
    int z = (int)r_floor((l.z + half.z) / (2 * half.z) * OCC_RES);
    if (x < 0 || y < 0 || z < 0 || x >= OCC_RES || y >= OCC_RES || z >= OCC_RES) return 0;
    int i = (z * OCC_RES + y) * OCC_RES + x;
    return (int)((bits[i >> 5] >> (i & 31)) & 1u);
}
