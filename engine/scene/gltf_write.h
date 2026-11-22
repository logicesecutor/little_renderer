// gltf_write.h - a scene written as glTF 2.0 binary (.glb), by the engine's own code: what
// scene/gltf.h reads, back the other way (docs/PBR.md, PBR6).
//
// A writer collects what goes into the file, then writes it:
// - geometry: a SceneMesh's triangles (positions, normals, texture coordinates; one glTF
//   primitive per material), or one of the two shapes (INST_BOX, INST_SPHERE) as triangles;
// - meshes: a geometry with its triangles' own materials, or all of them of one material
//   (an instance's override): meshes of one geometry share its data;
// - materials: the metallic-roughness model, KHR_materials_transmission, KHR_materials_ior and
//   KHR_materials_emissive_strength where they differ from glTF's defaults; textures (base
//   colour, metallic-roughness, normal, emission, occlusion) as PNG images in the file;
// - nodes: a name, a mesh, a transform, and the JSON of their extensions if any; groups of
//   them (a node with children);
// - extensions of the file (its root): their JSON, and the names it uses.
// Everything is scaled by `unit` (the file in metres: 0.01 for a scene in centimetres).
// The writer knows nothing of what the extensions say: whoever writes them builds their
// JSON (the web editor: KHR_physics_rigid_bodies, KHR_collision_shapes, its own LR_scene).
#ifndef LR_GLTF_WRITE_H
#define LR_GLTF_WRITE_H

#include "scene/scene.h"

typedef struct GltfWriter GltfWriter;

GltfWriter *gltf_writer_new(const Scene *s, real unit);
void gltf_writer_free(GltfWriter *w);

// a geometry: mesh >= 0 a SceneMesh of the scene, or INST_BOX / INST_SPHERE; -1 on failure
int gltf_add_geometry(GltfWriter *w, int mesh);
// a mesh of a geometry: material -1: its triangles' own; else that scene material on all of
// them (a shape needs one). Returns its index, -1 on failure.
int gltf_add_mesh(GltfWriter *w, int geometry, int material);
// a node: name (NULL: none), mesh (-1: none), xf (the scene's units; NULL: none), ext: the
// text inside its "extensions": {...} (NULL: none), extras: inside "extras": {...} (NULL: none).
// Returns its index.
int gltf_add_node(GltfWriter *w, const char *name, int mesh, const Xform *xf, const char *ext, const char *extras);
// a node of nodes added before (their transforms stay theirs: the group has none), e.g. an
// object of several parts; they are no longer in the scene itself, the group is. Its index.
int gltf_add_group(GltfWriter *w, const char *name, const int *children, int n, const char *ext, const char *extras);
// an extension of the file: its name in extensionsUsed (required: also extensionsRequired),
// and json: the text of its value at the root's "extensions" (NULL: used by nodes only)
void gltf_use_extension(GltfWriter *w, const char *name, const char *json, int required);

// the file (every node in the scene, in the order added); 0 = written
int gltf_write_glb(GltfWriter *w, const char *path);

#endif
