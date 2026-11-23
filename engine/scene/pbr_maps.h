// pbr_maps.h - a PBR material's maps as artists have them, made into the ones the renderers
// read (docs/PBR.md, PBR7). Every pipeline (the CPU and GPU path tracers, both rasterizers)
// reads glTF's set: base colour, metal-roughness in one (green, blue), a normal map, the
// occlusion, the emission (Material.tex_*). Texture sets often come otherwise: a bump (height)
// map, roughness and metallic apart, a normal map whose green points down (DirectX's). These
// make glTF's from them, once, in C: the native programs and the web's see the same texels.
#ifndef LR_PBR_MAPS_H
#define LR_PBR_MAPS_H

#include "scene/texture.h"
#include "scene/scene.h"

// A normal map from a height map (its grey: white high), in the renderers' tangent space (red
// along u, green up the image). strength: how steep, for any size of map: a step from black to
// white over one texel of a 256-wide map tilts the normal by atan(strength / 2) (a wider map's
// texel step tilts as much over the same part of the image). Its wrap modes are the height's.
// 0 = made
int pbr_bump_to_normal(TexMap *out, const TexMap *height, real strength);
// Two normal maps as one: detail's on top of base's, as if the detail were carved in the
// surface base makes (reoriented normal mapping, Barre-Brisebois and Hill 2012: a flat detail
// leaves base as it is, a flat base gives the detail). Sizes: the larger's (the other read
// bilinearly). 0 = made
int pbr_combine_normals(TexMap *out, const TexMap *base, const TexMap *detail);
// a normal map with its green turned over (DirectX's, green down, into the renderers' up)
int pbr_flip_green(TexMap *out, const TexMap *normal);
// Roughness and metallic maps (grey) as glTF's one: roughness in green, metallic in blue
// (red and alpha white). Either may be NULL: its channel is then white (the factor alone).
// Sizes: the larger's. 0 = made, 1 = neither given or no memory
int pbr_pack_mr(TexMap *out, const TexMap *rough, const TexMap *metal);

// A material's maps as they come: each a texture of the scene (1 + an index into
// Scene.texmaps, 0: none). normal: tangent space (green up, or down with flip_green); bump: a
// height map (grey), bump_strength as pbr_bump_to_normal's; rough, metal: grey; mr: glTF's
// packed metal-roughness (used when rough and metal are not given); ao: occlusion (red);
// emissive: the glow's colour.
typedef struct {
    int base, normal, bump, rough, metal, mr, ao, emissive;
    real bump_strength;
    int flip_green;
} PbrMapSet;
// m's textures from them: base, emissive, ao as they are; the normal map made from the normal
// map (turned over), the bump map, or both (the bump on top); roughness and metallic packed.
// What is made goes into the scene: a new texture, or in place of *made_normal / *made_mr
// (1 + an index, 0: none yet; NULL: always new), which say where it went (0: nothing made).
// 0 = done, 1 = a map could not be made (that texture left out)
int pbr_apply(Scene *s, Material *m, const PbrMapSet *ms, int *made_normal, int *made_mr);

#endif
