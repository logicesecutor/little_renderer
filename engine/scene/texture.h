// texture.h - a texture is just another pixel array we read from.
#ifndef LR_TEXTURE_H
#define LR_TEXTURE_H

#include "base/lrmath.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int w, h;
    uint32_t *px;   // 0x00RRGGBB, row 0 = top of the image
} Texture;

Texture *texture_load_png(const char *path);   // any PNG (formats/png.h), NULL on failure
Texture *texture_checker(int size, int cells); // procedural checkerboard
void texture_destroy(Texture *t);

// u, v in [0, 1] with v = 0 at the BOTTOM (OBJ convention). Wraps around outside [0, 1].
// Returns RGB as floats in [0, 1]. Nearest-neighbour sampling.
Vec3 texture_sample(const Texture *t, real u, real v);

// ---- the materials' textures (docs/PBR.md, PBR5) ---------------------------------------
// A TexMap is an RGBA8 image with its mip levels (each half the previous one, down to
// 1 x 1, made when it is created: 2 x 2 averages, in linear light for colour). Texture
// coordinates: u across, v DOWN (v = 0 the image's top row, glTF's convention; OBJ's
// v = 0 is the bottom: its loader turns them over), repeating outside [0, 1] (or clamped,
// or mirrored: wrap_u, wrap_v).
#define TEXMAP_LEVELS 16
typedef struct {
    int w, h, levels;                  // level 0's size, how many levels
    int lw[TEXMAP_LEVELS], lh[TEXMAP_LEVELS];
    size_t at[TEXMAP_LEVELS];          // each level's first texel in px
    uint32_t *px;                      // r | g << 8 | b << 16 | a << 24, rows top to bottom
    int srgb;                          // colour: the texels are sRGB, decoded when read
    int wrap_u, wrap_v;                // outside 0..1 (glTF's samplers): TEXMAP_REPEAT, _CLAMP, _MIRROR
    char name[64];                     // where it came from (a file, or a procedural one)
    uint64_t hash;                     // of what it shows, for a picture the scene holds once (a file's, a
                                       // model's: scene_share_texmap); 0: not shared (made here, may be made again)
} TexMap;
enum { TEXMAP_REPEAT, TEXMAP_CLAMP, TEXMAP_MIRROR };

// from w x h RGBA8 pixels (copied); 0 on success
int texmap_make(TexMap *t, const unsigned char *rgba, int w, int h, int srgb);
// a PNG file (any kind: grey, alpha, palette...), as RGBA; 0 on success
int texmap_load_png(TexMap *t, const char *path, int srgb);
// a PNG or JPEG file (told apart by their first bytes), or the same in memory (a glTF
// model's images); 0 on success
int texmap_load(TexMap *t, const char *path, int srgb);
int texmap_decode(TexMap *t, const unsigned char *data, size_t size, int srgb);
// Made here, 512 x 512: "checker" (colour), "bricks" (colour), "bricks_normal" (normal
// map), "bricks_mr" (roughness in green, metallic in blue, as glTF), "plate_mr" (a
// worn metal: metallic and roughness both vary). 0 on success, 1 for an unknown name.
int texmap_procedural(TexMap *t, const char *name);
void texmap_free(TexMap *t);

// texel (x, y) of a level (wrapped), decoded: linear 0..1
Vec4 texmap_texel(const TexMap *t, int level, int x, int y);
// bilinear in one level
Vec4 texmap_bilinear(const TexMap *t, int level, real u, real v);
// trilinear: lod = the level (fractional; <= 0: bilinear in level 0)
Vec4 texmap_sample(const TexMap *t, real u, real v, real lod);
// the level for a pixel whose texture coordinates change by (dudx, dvdx) and (dudy,
// dvdy) to the next pixel: log2 of the longer side of its footprint, in texels
real texmap_lod(const TexMap *t, real dudx, real dvdx, real dudy, real dvdy);
// sRGB-encoded 0..1 -> linear, and the table the decoding uses (256 entries)
const float *texmap_srgb_table(void);

#endif
