// renderer.h - the whole pipeline: mesh -> world -> camera -> clip -> screen -> pixels.
#ifndef RENDERER_H
#define RENDERER_H

#include "framebuffer.h"
#include "mesh.h"
#include "texture.h"
#include "vec.h"

typedef enum {
    MODE_WIREFRAME,   // only the triangle edges (Bresenham lines)
    MODE_FLAT,        // one brightness per triangle
    MODE_GOURAUD,     // brightness computed per vertex, interpolated
    MODE_PHONG,       // normal interpolated, lighting computed per pixel (+ specular)
    MODE_TEXTURED,    // like PHONG but the base color comes from a texture
    MODE_COUNT
} RenderMode;

typedef struct {
    mat4 model, view, proj;
    vec3 eye;              // camera position in world space (for specular)
    vec3 light_dir;        // unit vector pointing TOWARDS the light, in world space
    vec3 base_color;       // surface color when there is no texture
    RenderMode mode;
    const Texture *texture;
    int cull_backfaces;
} RenderParams;

// Camera 3.2 units away on +Z looking at the origin, 45 degree lens, light up-right-front.
RenderParams render_default_params(int width, int height);

// Draws the mesh into fb. Does NOT clear it: call fb_clear first.
void render_mesh(Framebuffer *fb, const Mesh *mesh, const RenderParams *p);

const char *render_mode_name(RenderMode mode);

#endif
