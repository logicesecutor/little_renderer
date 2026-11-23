// gltf.h - glTF 2.0 models (Khronos), read by the engine's own code (docs/PBR.md, PBR6).
//
// `.gltf` (JSON, its buffers and images in files next to it or in data: URIs) and `.glb`
// (one binary file). What is read:
// - meshes: triangles (and strips and fans, turned into triangles) with positions, normals,
//   texture coordinates 0 and tangents (computed when missing), indexed or not, every
//   component type (KHR_mesh_quantization's integers too);
// - the node tree of the default scene with its transforms (matrix, or translation /
//   rotation / scale): each node with a mesh becomes an instance of that mesh;
// - materials, the metallic-roughness model: base colour, metallic and roughness (factors and
//   texture), normal map (with its scale), occlusion (with its strength), emission (factor,
//   texture, KHR_materials_emissive_strength), KHR_materials_ior, KHR_materials_transmission
//   (the factor); the textures' wrap modes;
// - images: PNG and JPEG (formats/png.c, formats/jpeg.c), from files, data: URIs or the buffers.
// Not read: animations and skins (the rest pose is drawn), cameras, lights, morph targets,
// sparse accessors, alpha masking and blending (drawn opaque), texture coordinates 1 and
// KHR_texture_transform; Draco or meshopt compression are refused.
#ifndef LR_GLTF_H
#define LR_GLTF_H

#include "scene/scene.h"

// Puts the model into the scene, as mesh_place does with a .obj: turned by rot_y_deg about the
// vertical axis, scaled so its largest side is `size` (size < 0: at its own size times -size,
// so -100 keeps glTF's metres in units of 1 cm), centred on base.x / base.z with its lowest
// point at base.y. Its materials are added to the scene's (at most MAX_MATS in all:
// the rest share the last). Returns the instances added, -1 on failure (a message on stderr).
// The scene's acceleration structure must be (re)built afterwards.
int scene_load_gltf(Scene *s, const char *path, real size, Vec3 base, real rot_y_deg);
// The file's nodes where the file puts them (no fitting, no moving), scaled by `unit` (100:
// glTF's metres in units of 1 cm), e.g. to open again a scene saved with scene/gltf_write.h.
// node_of (or NULL): *node_of gets a new array (free it), the file's node of each instance
// added, in order. Returns the instances added, -1 on failure.
int scene_load_gltf_placed(Scene *s, const char *path, real unit, int **node_of);

#endif
