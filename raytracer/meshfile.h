// meshfile.h - reads triangle meshes (.obj or ASCII .ply) and adds them to a scene.
#ifndef MESHFILE_H
#define MESHFILE_H

#include "scene.h"

// The file type is picked from the extension (.obj / .ply).
// The mesh is turned by rot_y_deg around the vertical axis, scaled so its largest side is
// `size`, centered on base.x / base.z, and its lowest point is put at base.y.
// Meshes without normals get smooth normals computed from their triangles.
// Returns the number of triangles added, or -1 on failure (with a message on stderr).
int mesh_load(Scene *s, const char *path, int material, double size, Vec3 base, double rot_y_deg);

#endif
