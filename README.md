# little_renderer

A software renderer in plain C: no graphics library, no dependencies except libm.
Everything is "fill an array of pixels"; showing that array is a separate, tiny step.

## Build and run

    make models       # writes cube/sphere/torus .obj into models/ (needs python3, only once)
    make run          # builds ./main and writes one image per milestone into out/*.ppm
    make viewer       # real-time window through plain Xlib (needs libx11-dev)
    ./viewer [model.obj] [texture.ppm]    # keys: 1-5 shading, space pause, c culling, Esc quit
    make SAN=1 main   # same, with address + undefined-behaviour sanitizers

`.ppm` files open in GIMP, Preview, `eog`, VS Code extensions, etc.

## Files

| file | job |
|---|---|
| `vec.h` | vec2/3/4, mat4 (translate, rotate, look-at, perspective). Header only. |
| `framebuffer.c/.h` | the pixel array + z-buffer, `set_pixel`, PPM output |
| `raster.c/.h` | Bresenham line, barycentric triangle, z-test, perspective-correct interpolation |
| `mesh.c/.h` | `.obj` loader (v / vt / vn / polygons / negative indices), normalize, smooth normals |
| `texture.c/.h` | PPM texture loader, procedural checkerboard, nearest sampling |
| `renderer.c/.h` | the pipeline: model -> world -> view -> projection -> near clip -> viewport -> shaders |
| `main.c` | headless demo, one image per milestone |
| `viewer.c` | X11 window + spin loop + fps counter |
| `tools/gen_models.py` | generates the test models |
| `old/main_v0_first_line.c` | your original first version (the Bresenham test) |

## Milestones and where to see them

| # | what | output / code |
|---|---|---|
| 0 | hello pixels | `out/0_gradient.ppm`, `framebuffer.c` |
| 1 | Bresenham lines | `out/1_lines.ppm`, `draw_line` in `raster.c` |
| 2 | wireframe OBJ | `out/2_wireframe.ppm`, `mesh.c` |
| 3 | filled triangles (edge functions) | `out/3_triangles_zbuffer.ppm`, `draw_triangle` |
| 4 | flat shading + back-face culling | `out/4_flat.ppm`, `out/4_flat_no_culling.ppm` |
| 5 | z-buffer | `out/3_triangles_zbuffer.ppm`, `out/5_depth.ppm` |
| 6 | real camera (model/view/projection) | `vec.h`, `render_mesh` in `renderer.c` |
| 7 | real time | `viewer.c` |
| 8 | Gouraud, Phong, textures, perspective-correct UVs | `out/8_*.ppm` |
| 9 | near-plane clipping, shaders as function pointers | `out/9_near_clipping.ppm`, `FragShader` |

Not done on purpose: top-left fill rule (harmless for opaque triangles with a z-test),
multithreading, far-plane clipping (pixels beyond it fail the depth test).

## Conventions worth remembering

- Framebuffer origin is the **top-left**, y grows downward. The viewport transform flips Y once.
- Counter-clockwise triangles (as seen from outside) are front faces, like OBJ files.
- Colors are `0x00RRGGBB`. If the viewer shows red and blue swapped, swap them in `rgb8`.
- Light direction is the unit vector pointing *towards* the light.

## Ray tracer

`raytracer/` is a separate program: a Cornell box **path tracer** (global illumination,
soft shadows, mirror and glass) that writes PPM images. `cd raytracer && make run`.
It also loads `.obj` / ASCII `.ply` meshes with an octree: `make dragon` renders the
Stanford dragon (871k triangles; model © Stanford Computer Graphics Laboratory,
non-commercial use, downloaded into `models/dragon_recon/`).
The full explanation of its code and maths is in `raytracer/RAYTRACER.md`.
`raytracer/rtview` is the same engine in a window you can fly through (WASD + mouse).

**Every command, option, key and constant of both renderers is listed in `USAGE.md`.**
