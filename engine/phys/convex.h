// convex.h - physics: the geometry behind mesh bodies.
//
//   Convex      a convex polyhedron: vertices, faces (planes and their vertex loops) and
//               edges (with the two faces they join). Made by convex_hull from a mesh's
//               vertices (at most max_verts of them: the hull of the points that are
//               furthest in many directions), or convex_box.
//   mass        volume, centre of mass and inertia tensor of a polyhedron (density 1).
//   SAT         two polyhedra (or a polyhedron and a triangle) collide by the separating
//               axis test (faces of each, pairs of edges, these pruned on the Gauss map as
//               in Gregorius, "The Separating Axis Test between Convex Polyhedra", GDC 2013):
//               the axis of least overlap gives the normal; a face axis gives the incident
//               face clipped against the reference face's sides (up to 4 points kept), an
//               edge axis the closest points of the two edges.
//   MeshTree    a BVH of triangles, for a static mesh collider (box queries).
//   occupancy   which cells of a grid over a shape are inside it: for the air (fluid.c and
//               the GPU's), which only needs to know which cells are solid.
#ifndef LR_CONVEX_H
#define LR_CONVEX_H

#include <stdint.h>

#include "core/lrmath.h"
#include "core/scene.h"   // Aabb, Triangle

typedef struct { Vec3 n; real d; int first, count; } ConvexFace;   // n . x = d (n outward); loop in fv
typedef struct { int a, b, fa, fb; } ConvexEdge;                   // its vertices, the faces it joins

typedef struct {
    int nv, nf, ne;
    Vec3 *v;
    ConvexFace *f;
    int *fv;
    ConvexEdge *e;
} Convex;

// the hull of n points, from at most max_verts of them (0 if they are flat or too few)
int convex_hull(const Vec3 *pts, int n, int max_verts, Convex *out);
// the box [-half, half]
int convex_box(Vec3 half, Convex *out);
void convex_free(Convex *c);
// with density 1: the volume, the centre of mass and the inertia tensor about it
void convex_mass(const Convex *c, real *volume, Vec3 *com, Mat3 *inertia);
int convex_contains(const Convex *c, Vec3 p);   // in its own frame

// a polyhedron placed in the world: v and the planes moved, the topology shared
typedef struct {
    const Convex *c;
    Vec3 *v, *n;             // world vertices, world face normals
    real *d;                 // world face offsets
    Vec3 center;
} WorldConvex;
// (fills the arrays given: c->nv vertices, c->nf normals and offsets)
void convex_place(WorldConvex *w, const Convex *c, Vec3 pos, Mat3 rot, Vec3 *v, Vec3 *n, real *d);

typedef struct { Vec3 p, n; real depth; } ConvexContact;   // n points from b towards a
#define CONVEX_MAX_CONTACTS 4

// The contacts of two shapes closer than `margin` (0 = only touching ones): depth < 0 is a
// gap (a speculative contact, phys/rigid.c: the solver lets the gap close, no more).
// a against b; the contacts found (0 = apart)
int convex_collide(const WorldConvex *a, const WorldConvex *b, real margin, ConvexContact *out);
// a against the triangle t (world): two-sided
int convex_collide_triangle(const WorldConvex *a, const Vec3 t[3], real margin, ConvexContact *out);
// a sphere against a polyhedron (1 contact or 0)
int convex_collide_sphere(Vec3 center, real radius, const WorldConvex *b, real margin, ConvexContact *out);
// a sphere against a triangle (two-sided)
int triangle_collide_sphere(Vec3 center, real radius, const Vec3 t[3], real margin, ConvexContact *out);
// keeps at most `keep` of n contacts: the deepest, then those spreading the most area
int convex_reduce(ConvexContact *c, int n, int keep);

// ---- a BVH of triangles (static mesh colliders) ----
typedef struct { Aabb box; int left, right, first, count; } MeshNode;   // leaf: count > 0
typedef struct {
    Vec3 *tri;               // 3 corners per triangle (its frame)
    int ntris;
    int *order;              // triangles by leaf
    MeshNode *nodes;
    int nnodes;
} MeshTree;
int mesh_tree_build(MeshTree *t, const Triangle *tris, int n, Vec3 offset);   // corners - offset
void mesh_tree_free(MeshTree *t);
// the triangles whose box meets `box` (its frame): calls fn(user, triangle index)
void mesh_tree_query(const MeshTree *t, Aabb box, void (*fn)(void *user, int tri), void *user);

// ---- occupancy: res^3 cells over [-half, half], 1 bit each ----
#define OCC_RES 48
#define OCC_WORDS ((OCC_RES * OCC_RES * OCC_RES + 31) / 32)
void occupancy_convex(const Convex *c, Vec3 half, uint32_t *bits);
// inside a mesh: crossings counted along x, y and z for each cell, the majority decides
// (so a mesh with small holes still has an inside)
void occupancy_mesh(const MeshTree *t, Vec3 half, uint32_t *bits);
int occupancy_at(const uint32_t *bits, Vec3 half, Vec3 local);

#endif
