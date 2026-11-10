// mesh.h - triangle meshes and a small Wavefront .obj loader.
#ifndef MESH_H
#define MESH_H

#include "vec.h"

// One triangle. Each corner stores indices into the mesh arrays (-1 = not present).
typedef struct {
    int v[3];   // position
    int t[3];   // texture coordinate
    int n[3];   // normal
} Face;

typedef struct {
    vec3 *v;   int nv;
    vec2 *vt;  int nvt;
    vec3 *vn;  int nvn;
    Face *f;   int nf;
} Mesh;

Mesh *mesh_load_obj(const char *path);   // NULL on failure. Polygons are split into triangles.
void mesh_destroy(Mesh *m);
void mesh_normalize(Mesh *m);            // center on the origin and scale to fit in [-1, 1]
void mesh_compute_normals(Mesh *m);      // smooth per-vertex normals from the triangles

#endif
