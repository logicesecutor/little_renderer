// mesh.h - indexed triangle meshes and the .obj / ASCII .ply loaders, shared by both
// backends. The rasterizer draws a Mesh directly (render_mesh); a scene turns it into
// triangle primitives (scene_add_mesh).
#ifndef LR_MESH_H
#define LR_MESH_H

#include "core/lrmath.h"

// One triangle. Each corner stores indices into the mesh arrays (-1 = not present).
typedef struct {
    int v[3];   // position
    int t[3];   // texture coordinate
    int n[3];   // normal
} Face;

typedef struct {
    Vec3 *v;   int nv;
    Vec2 *vt;  int nvt;
    Vec3 *vn;  int nvn;
    Face *f;   int nf;
} Mesh;

// Reads .obj (v / vt / vn / f, polygons, negative indices) or ASCII .ply (vertex x y z +
// faces), picked by the file extension. Polygons are split into triangles. Files without
// normals get smooth normals (mesh_compute_normals). NULL on failure.
Mesh *mesh_load(const char *path);
void mesh_destroy(Mesh *m);

// Smooth per-vertex normals: every triangle adds its (b - a) x (c - a) to its 3 vertices.
// The sums are area-weighted and NOT normalized; every user normalizes them.
void mesh_compute_normals(Mesh *m);

// center on the origin and scale to fit in [-1, 1] (what the rasterizer demos use)
void mesh_normalize(Mesh *m);

// turn by rot_y_deg around the vertical axis, scale so the largest side is `size`, center
// on base.x / base.z and put the lowest point at base.y (how a model is put into a scene)
void mesh_place(Mesh *m, real size, Vec3 base, real rot_y_deg);

#endif
