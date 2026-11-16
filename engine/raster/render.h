// render.h - the rasterizer pipeline: triangles -> world -> camera -> clip -> screen -> pixels.
// Two entry points: render_mesh draws one Mesh with a directional light (the classic
// "model viewer"), render_scene draws a whole Scene (the same one the path tracer uses).
#ifndef LR_RENDER_H
#define LR_RENDER_H

#include "scene/image.h"
#include "base/lrmath.h"
#include "scene/mesh.h"
#include "scene/scene.h"
#include "scene/texture.h"

typedef enum {
    MODE_WIREFRAME,   // only the triangle edges (Bresenham lines)
    MODE_FLAT,        // one brightness per triangle
    MODE_GOURAUD,     // brightness computed per vertex, interpolated
    MODE_PHONG,       // normal interpolated, lighting computed per pixel (+ specular)
    MODE_TEXTURED,    // like PHONG but the base color comes from a texture (render_mesh only)
    MODE_COUNT
} RenderMode;

typedef struct {
    Mat4 model, view, proj;
    Vec3 eye;              // camera position in world space (for specular)
    Vec3 light_dir;        // render_mesh: unit vector pointing TOWARDS the light, world space
    Vec3 base_color;       // render_mesh: surface color when there is no texture
    RenderMode mode;
    const Texture *texture;
    int cull_backfaces;
} RenderParams;

// Camera 3.2 units away on +Z looking at the origin, 45 degree lens, light up-right-front.
RenderParams render_default_params(int width, int height);

// A camera like scene/camera.h's (vertical field of view in degrees), phong, no culling.
RenderParams render_params_camera(Vec3 from, Vec3 at, Vec3 up, real vfov_deg, int width, int height);

// Draws the mesh into fb. Does NOT clear it: call fb_clear first.
void render_mesh(Framebuffer *fb, const Mesh *mesh, const RenderParams *p);

// Draws every primitive of the scene (spheres are tessellated, quads become 2 triangles),
// and the sky (Scene.env) where nothing was drawn.
// The materials' BSDF (scene/brdf.h) lit by the scene's lamp and an ambient light
// (render_shade); output is sRGB-encoded like the path tracer's. Uses p->view, p->proj,
// p->eye, p->mode (TEXTURED acts as PHONG) and p->cull_backfaces. Does NOT clear fb.
void render_scene(Framebuffer *fb, const Scene *s, const RenderParams *p);

// ---- what render_scene draws, piece by piece ------------------------------------------
// For other rasterizers of the same Scene (the GPU's, gpu/raster_gpu.c): the same
// triangles, the same lamp, the same material model, so that they draw the same image.

// the ambient light standing in for the light that bounces around the room (a rasterizer
// has none): a uniform environment of this radiance, and more of it towards the camera (the
// faces the camera sees head-on are lit by what is behind it)
#define RENDER_SCENE_AMBIENT 0.06f
#define RENDER_SCENE_FILL 0.025f
#define RENDER_SPHERE_STACKS 16
#define RENDER_SPHERE_SLICES 32
// a sphere is a latitude / longitude grid; every cell is 2 triangles, 1 at the poles
#define RENDER_SPHERE_TRIS (2 * RENDER_SPHERE_STACKS * RENDER_SPHERE_SLICES - 2 * RENDER_SPHERE_SLICES)

// triangles a primitive is drawn with: 2 for a quad, RENDER_SPHERE_TRIS for a sphere
int render_prim_tri_count(const Primitive *p);
// Triangle `sub` of a primitive: its world-space corners and, if wn is not NULL, its vertex
// normals (*has_n = 0 when it has none: use the face normal) and the point where flat
// shading is evaluated (a quad's center, so that both halves get one color).
void render_prim_triangle(const Primitive *p, int sub, Vec3 *wp, Vec3 *wn, int *has_n, Vec3 *flat_at);
// its corners' texture coordinates and tangents (PBR5; as rt/intersect.c's hit_texcoords:
// a sphere's longitude and latitude, unwrapped across the seam so they interpolate; a
// quad's (a, b))
void render_prim_texcoords(const Primitive *p, int sub, Vec2 *uv, Vec4 *tg);
// a mesh triangle's (no texture coordinates: its barycentrics' corners), the tangents
// turned by xf if not NULL (an instance's)
void render_tri_texcoords(const Triangle *t, const Vec2 *uvs, const Vec4 *tangents, int nuvs, const Xform *xf, Vec2 *uv, Vec4 *tg);
// the point light standing in for the scene's lamp
Vec3 render_scene_light(const Scene *s);
// The lamp as the rasterizers light with it (PBR3): the quad q + a u + b v (a, b in 0..1),
// the side it shines to (n), w = (u x v) / |u x v|^2 (a point's a, b), its radiance; and
// the ambient light (RENDER_SCENE_AMBIENT, _FILL; LR_RASTER_AMBIENT=0 turns it off, to
// compare with the path tracer's direct light, -d 1)
// and the environment it lights with besides (PBR4; NULL: the ambient light): the scene's
// light probe, else its sky
typedef struct {
    int on; Vec3 q, u, v, n, w, center, le; real area, ambient, fill;
    const struct Env *env;
    const struct IrrGrid *irr;   // with the probe: its irradiance grid (the diffuse light), or NULL
} RenderLamp;
RenderLamp render_scene_lamp(const Scene *s);
// The radiance point p (normal n, turned to the eye) sends to the eye: the material's BSDF
// lit by the lamp - its diffuse-like lobes by the lamp's exact form factor (no shadow), its
// microfacet lobe by the lamp's centre or 4 quarters when the lobe is wider than the lamp,
// by the lamp's point nearest the reflected ray over the lobe's own solid angle 4 π α² cos
// when it is narrower (a mirror shows the lamp itself) - and by the environment (split-sum:
// the specular part from env_level around the reflected ray, the diffuse part from the
// irradiance, the transmitted part around the ray through the object) or, without one, the
// ambient light (bsdf_albedo_uniform). The rasterizer tabulates per material what this
// recomputes.
Vec3 render_shade(const RenderLamp *L, const Material *m, Vec3 p, Vec3 n, Vec3 eye);
// a material drawn as it is (constant: a lamp, its colour at full brightness), or shaded
// (render_shade; base: its base colour, for the wireframe)
typedef struct { Vec3 base; int constant; } RenderMaterial;
RenderMaterial render_scene_material(const Material *m);

const char *render_mode_name(RenderMode mode);
int render_mode_from_name(const char *name);   // -1 if unknown

#endif
