// pbr_maps.c - see pbr_maps.h. Each map is read at level 0 (bilinearly where the sizes differ)
// and the result made as a TexMap of its own (its mip levels made from it, not filtered here).
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "scene/pbr_maps.h"

static unsigned char to8(real v){
    if (!(v > 0)) return 0;
    if (v >= 1) return 255;
    return (unsigned char)(v * 255 + 0.5f);
}
// texel (x, y) of a w x h result, from map t (the same size: exactly; otherwise bilinearly)
static Vec4 at(const TexMap *t, int x, int y, int w, int h){
    if (t->w == w && t->h == h) return texmap_texel(t, 0, x, y);
    return texmap_bilinear(t, 0, (x + 0.5f) / w, (y + 0.5f) / h);
}
static real grey(Vec4 c){ return (c.x + c.y + c.z) / 3; }
static int finish(TexMap *out, unsigned char *rgba, int w, int h, const TexMap *like){
    int r = texmap_make(out, rgba, w, h, 0);
    free(rgba);
    if (!r && like){ out->wrap_u = like->wrap_u; out->wrap_v = like->wrap_v; }
    return r;
}

int pbr_bump_to_normal(TexMap *out, const TexMap *height, real strength){
    memset(out, 0, sizeof(*out));
    if (!height || !height->px) return 1;
    const int w = height->w, h = height->h;
    unsigned char *rgba = malloc((size_t)w * h * 4);
    if (!rgba) return 1;
    // the slope per texel, times the map's side over 256 (each way its own): one look
    // whatever its size
    const real kx = strength * (real)w / 256, ky = strength * (real)h / 256;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++){
            // (texmap_texel wraps: the map's edges meet as it repeats)
            real hr = grey(texmap_texel(height, 0, x + 1, y)), hl = grey(texmap_texel(height, 0, x - 1, y));
            real hu = grey(texmap_texel(height, 0, x, y - 1)), hd = grey(texmap_texel(height, 0, x, y + 1));
            real gx = kx * (hr - hl) / 2, gy = ky * (hu - hd) / 2;        // (up the image: the row above)
            real l = r_sqrt(gx * gx + gy * gy + 1);
            unsigned char *o = rgba + 4 * ((size_t)y * w + x);
            o[0] = to8((-gx / l) * 0.5f + 0.5f);
            o[1] = to8((-gy / l) * 0.5f + 0.5f);
            o[2] = to8((1 / l) * 0.5f + 0.5f);
            o[3] = 255;
        }
    return finish(out, rgba, w, h, height);
}

int pbr_combine_normals(TexMap *out, const TexMap *base, const TexMap *detail){
    memset(out, 0, sizeof(*out));
    if (!base || !detail || !base->px || !detail->px) return 1;
    const int w = base->w > detail->w ? base->w : detail->w, h = base->h > detail->h ? base->h : detail->h;
    unsigned char *rgba = malloc((size_t)w * h * 4);
    if (!rgba) return 1;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++){
            Vec4 a = at(base, x, y, w, h), b = at(detail, x, y, w, h);
            // t = base in [-1, 1] with z + 1; u = detail turned over in x, y; r = t (t.u) / t.z - u
            real tx = 2 * a.x - 1, ty = 2 * a.y - 1, tz = 2 * a.z;
            real ux = 1 - 2 * b.x, uy = 1 - 2 * b.y, uz = 2 * b.z - 1;
            real d = tx * ux + ty * uy + tz * uz;
            real rx = tx * d / tz - ux, ry = ty * d / tz - uy, rz = tz * d / tz - uz;
            if (!(tz > 1e-4f)){ rx = 2 * b.x - 1; ry = 2 * b.y - 1; rz = 2 * b.z - 1; }   // (a base pointing sideways: the detail)
            real l = r_sqrt(rx * rx + ry * ry + rz * rz);
            if (!(l > 1e-6f)){ rx = 0; ry = 0; rz = 1; l = 1; }
            unsigned char *o = rgba + 4 * ((size_t)y * w + x);
            o[0] = to8(rx / l * 0.5f + 0.5f);
            o[1] = to8(ry / l * 0.5f + 0.5f);
            o[2] = to8(rz / l * 0.5f + 0.5f);
            o[3] = 255;
        }
    return finish(out, rgba, w, h, base);
}

int pbr_flip_green(TexMap *out, const TexMap *normal){
    memset(out, 0, sizeof(*out));
    if (!normal || !normal->px) return 1;
    const int w = normal->w, h = normal->h;
    unsigned char *rgba = malloc((size_t)w * h * 4);
    if (!rgba) return 1;
    for (size_t i = 0; i < (size_t)w * h; i++){
        uint32_t p = normal->px[i];
        rgba[4 * i] = p & 255;
        rgba[4 * i + 1] = (unsigned char)(255 - (p >> 8 & 255));
        rgba[4 * i + 2] = p >> 16 & 255;
        rgba[4 * i + 3] = 255;
    }
    return finish(out, rgba, w, h, normal);
}

int pbr_pack_mr(TexMap *out, const TexMap *rough, const TexMap *metal){
    memset(out, 0, sizeof(*out));
    if ((!rough || !rough->px) && (!metal || !metal->px)) return 1;
    if (rough && !rough->px) rough = NULL;
    if (metal && !metal->px) metal = NULL;
    int w = 0, h = 0;
    if (rough){ w = rough->w; h = rough->h; }
    if (metal){ if (metal->w > w) w = metal->w; if (metal->h > h) h = metal->h; }
    unsigned char *rgba = malloc((size_t)w * h * 4);
    if (!rgba) return 1;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++){
            unsigned char *o = rgba + 4 * ((size_t)y * w + x);
            o[0] = 255;
            o[1] = rough ? to8(grey(at(rough, x, y, w, h))) : 255;
            o[2] = metal ? to8(grey(at(metal, x, y, w, h))) : 255;
            o[3] = 255;
        }
    return finish(out, rgba, w, h, rough ? rough : metal);
}

// t (made) into the scene: in place of *slot if it holds one, else new; 1 + its index
static int put(Scene *s, TexMap *t, int *slot){
    int k = slot && *slot > 0 && *slot <= s->ntexmaps ? scene_replace_texmap(s, *slot, t) : scene_add_texmap(s, t);
    if (slot) *slot = k;
    return k;
}
static const TexMap *tex_of(const Scene *s, int k){ return k > 0 && k <= s->ntexmaps ? &s->texmaps[k - 1] : NULL; }

int pbr_apply(Scene *s, Material *m, const PbrMapSet *ms, int *made_normal, int *made_mr){
    int bad = 0;
    m->tex_base = tex_of(s, ms->base) ? ms->base : 0;
    m->tex_emissive = tex_of(s, ms->emissive) ? ms->emissive : 0;
    m->tex_occlusion = tex_of(s, ms->ao) ? ms->ao : 0;
    // the normal: as it is, turned over, from the bump, or the bump on top of it
    const TexMap *nm = tex_of(s, ms->normal), *bm = tex_of(s, ms->bump);
    m->tex_normal = 0;
    if (nm && !bm && !ms->flip_green){
        m->tex_normal = ms->normal;
    } else if (nm || bm){
        TexMap flipped, fromb, out;
        int ok = 1;
        memset(&flipped, 0, sizeof(flipped));
        memset(&fromb, 0, sizeof(fromb));
        if (nm && ms->flip_green){ ok = pbr_flip_green(&flipped, nm) == 0; nm = ok ? &flipped : NULL; }
        if (ok && bm) ok = pbr_bump_to_normal(&fromb, bm, ms->bump_strength) == 0;
        if (ok){
            if (nm && bm){ ok = pbr_combine_normals(&out, nm, &fromb) == 0; texmap_free(&fromb); if (nm == &flipped) texmap_free(&flipped); }
            else if (bm) out = fromb;
            else out = flipped;
        }
        if (ok){
            snprintf(out.name, sizeof(out.name), "%s", bm ? "normal + bump (made)" : "normal, green turned over (made)");
            m->tex_normal = put(s, &out, made_normal);
        } else {
            texmap_free(&fromb);
            if (nm == &flipped) texmap_free(&flipped);
            bad = 1;
        }
    }
    // roughness and metallic: apart (packed here), or glTF's packed map
    const TexMap *rm = tex_of(s, ms->rough), *mm = tex_of(s, ms->metal);
    m->tex_mr = 0;
    if (rm || mm){
        TexMap out;
        if (pbr_pack_mr(&out, rm, mm) == 0){
            snprintf(out.name, sizeof(out.name), "roughness + metallic (made)");
            m->tex_mr = put(s, &out, made_mr);
        } else bad = 1;
    } else if (tex_of(s, ms->mr)) m->tex_mr = ms->mr;
    return bad;
}
