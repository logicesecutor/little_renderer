// render.h - the rasterizer pipeline: triangles -> world -> camera -> clip -> screen -> pixels.
// Two entry points: render_mesh draws one Mesh with a directional light (the classic
// "model viewer"), render_scene draws a whole Scene (the same one the path tracer uses).
#ifndef LR_RENDER_H
#define LR_RENDER_H

#include "core/image.h"
#include "core/lrmath.h"
#include "core/mesh.h"
#include "core/scene.h"
#include "core/texture.h"

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

// A camera like core/camera.h's (vertical field of view in degrees), phong, no culling.
RenderParams render_params_camera(Vec3 from, Vec3 at, Vec3 up, real vfov_deg, int width, int height);

// Draws the mesh into fb. Does NOT clear it: call fb_clear first.
void render_mesh(Framebuffer *fb, const Mesh *mesh, const RenderParams *p);

// Draws every primitive of the scene (spheres are tessellated, quads become 2 triangles).
// Colors come from the materials, light from a point light at the center of the scene's
// light quad; output is sRGB-encoded like the path tracer's. Uses p->view, p->proj,
// p->eye, p->mode (TEXTURED acts as PHONG) and p->cull_backfaces. Does NOT clear fb.
void render_scene(Framebuffer *fb, const Scene *s, const RenderParams *p);

const char *render_mode_name(RenderMode mode);
int render_mode_from_name(const char *name);   // -1 if unknown

#endif
