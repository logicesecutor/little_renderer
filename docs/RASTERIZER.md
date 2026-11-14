# The rasterizer (`engine/raster`)

A software rasterizer in plain C: no graphics library, no dependencies except `libm`.
Everything is "fill an array of pixels"; showing that array is a separate, tiny step
(`platform/window.c`). It is one of the engine's two backends: it draws the same `Scene`
as the path tracer (RAYTRACER.md), in milliseconds instead of seconds, with local lighting
only. Commands and options: [USAGE.md](USAGE.md). Using it as a library: [SDK.md](SDK.md).

```sh
make models                              # cube/sphere/torus .obj into models/ (uv or python3, once)
make demo                                # bin/raster_milestones: one image per milestone into out/*.png
bin/render -backend raster               # the Cornell box, rasterized -> out/cornell_boxes_raster.png
bin/view -backend raster                 # fly through it in real time (V cycles the shading)
```

## Files

| file | job |
|---|---|
| `engine/core/lrmath.h` | `real`, Vec2/3/4, Mat4 (rotate, look-at, perspective). Header only, shared with the path tracer. |
| `engine/core/image.c` (+ `.h`) | `Framebuffer`: the pixel array + z-buffer, `fb_set_pixel`, PNG output, sRGB packing |
| `engine/core/mesh.c` (+ `.h`) | `.obj` / ASCII `.ply` loader (v / vt / vn / polygons / negative indices), normalize, vertex normals |
| `engine/core/texture.c` (+ `.h`) | PNG texture loader (`core/png.c`), procedural checkerboard, nearest sampling |
| `engine/raster/raster.c` (+ `.h`) | Bresenham line, barycentric triangle, z-test, perspective-correct interpolation |
| `engine/raster/render.c` (+ `.h`) | the pipeline: model -> world -> view -> projection -> near clip -> viewport -> shaders. `render_mesh` draws one mesh, `render_scene` a whole `Scene` |
| `engine/raster/volume_march.c` (+ `.h`) | clouds and fog: raymarched over the finished frame, stopped by its z-buffer (VOLUMES.md 7.1) |
| `examples/raster_milestones.c` | one image per milestone (uses `render_mesh`) |
| `examples/hello_raster.c` | the smallest program that rasterizes a scene |
| `apps/render.c`, `apps/view.c` | `-backend raster` (offline) and the real-time viewer |

## Milestones and where to see them

| # | what | output / code |
|---|---|---|
| 0 | hello pixels | `out/0_gradient.png`, `core/image.c` |
| 1 | Bresenham lines | `out/1_lines.png`, `draw_line` in `raster.c` |
| 2 | wireframe OBJ | `out/2_wireframe.png`, `core/mesh.c` |
| 3 | filled triangles (edge functions) | `out/3_triangles_zbuffer.png`, `draw_triangle` |
| 4 | flat shading + back-face culling | `out/4_flat.png`, `out/4_flat_no_culling.png` |
| 5 | z-buffer | `out/3_triangles_zbuffer.png`, `out/5_depth.png` |
| 6 | real camera (model/view/projection) | `core/lrmath.h`, `render_mesh` in `render.c` |
| 7 | real time | `bin/view -backend raster` |
| 8 | Gouraud, Phong, textures, perspective-correct UVs | `out/8_*.png` |
| 9 | near-plane clipping, shaders as function pointers | `out/9_near_clipping.png`, `FragShader` |
| 10 | whole scenes: the path tracer's `Scene` as triangles | `render_scene`, `out/cornell_*_raster.png` |

## Drawing a `Scene` (`render_scene`)

The path tracer's scene is made of spheres, quads and triangles with materials. The
rasterizer only knows triangles, so `render_scene` converts on the fly, every frame:

- **triangles** go through as they are (with their vertex normals if they have them);
- **quads** become 2 triangles;
- **spheres** are tessellated into a 16 × 32 latitude / longitude grid, with the exact
  sphere normal at every vertex.

Then the same second half of the pipeline as `render_mesh` (`draw_clip_triangle`: near
clip, perspective divide, viewport, cull, rasterize) draws them.

**Lighting.** A rasterizer sees one triangle at a time, so it cannot know what is between
a point and the lamp (shadows), or what light bounces off the walls (color bleeding).
Those are exactly what the path tracer computes. `render_scene` uses *local* lighting:

- a **point light** at the center of the scene's light quad, moved half the lamp's size
  towards the side it shines to (so it also lights the ceiling around it a little);
- `brightness = 0.08 (ambient) + 0.70 · max(n·l, 0) (lamp) + 0.22 · (n·v) (fill)`. The
  *fill* term is a weak light from the camera: it stands in for the bounced light that
  makes the faces the lamp does not see grey instead of black;
- **two-sided** surfaces: the normal is turned towards the camera first, because the
  scene's quads have no "outside" (the walls are seen from inside the box);
- materials: **diffuse** = its albedo, **mirror** = a darker base + a sharp Blinn-Phong
  highlight, **glass** = an even darker base + a sharper highlight, **light** = its color at
  full brightness;
- colors are linear like the path tracer's, and are sRGB-encoded at the end
  (`image_pack_srgb`), so both backends show the same wall colors.

Shading modes: `phong` (per pixel), `gouraud` (per vertex), `flat` (per triangle; a quad's
two halves are shaded at the quad's center so they match), `wireframe`.

## Multithreading (`draw_triangles` in `render.c`)

`render_mesh` and `render_scene` use every core (OpenMP, `-threads n` to limit it). Both
describe their triangles as a `TriJob` (how many, the corners of triangle *i*, the vertex
stage of triangle *i*) and hand it to `draw_triangles`, which works in two parallel passes:

1. **Binning**, in parallel over the triangles: transform only the 3 corners, clip,
   project, cull, and work out which rows of pixels the triangle can touch. The screen is
   cut into horizontal **bands** of 16 rows (`RASTER_BAND_ROWS`), and the triangle's index
   is added to the list of every band it touches. Triangles behind the camera, off screen,
   culled, or so small that they cover no pixel center are dropped here (on the dragon at
   640 × 640, about half of the 871k triangles).
2. **Drawing**, in parallel over the bands: each band is drawn by one thread, which runs the
   full vertex stage for the triangles in its list and fills only its own rows
   (`draw_triangle_rows`, `draw_line_rows` in `raster.c`).

Why bands, and why this order:

- **No locks.** Every pixel (color and depth) belongs to exactly one band, so no two
  threads ever write the same memory.
- **Same image for any number of threads.** Each band draws its triangles in their
  original order. Where two triangles have exactly the same depth, the z-buffer keeps the
  first one drawn, exactly as a single thread would. The output is bit-for-bit identical
  to the single-threaded rasterizer, for 1, 3, 5 or 8 threads (checked on 36 images: the
  milestones and every scene in every shading mode).
- **Pass 1 must never lose a triangle.** It uses the same clip, projection and cull
  arithmetic as pass 2, and the row range it computes (pixel centers covered, or the
  rounded ends of a wireframe line) gets a 1/64-pixel safety margin.
- **Cheap enough to be worth it.** A triangle crossing a band border runs its vertex stage
  once per band; with exact row ranges that is only a few percent extra on the dragon.
  The binning pass also avoids `fminf` / `floorf` calls (library calls on a default build)
  in favour of plain comparisons, which halved its cost.

**Speed** (640 × 640, best of 5, Intel i5-8250U: 4 cores / 8 threads):

| scene | 1 thread | 2 | 4 | 8 | single-threaded rasterizer before |
|---|---|---|---|---|---|
| Cornell box | 42 ms | 26 ms | 11 ms | **9.3 ms** | 35 ms |
| spheres | 39 ms | 24 ms | 15 ms | **11 ms** | |
| torus | 38 ms | 22 ms | 13 ms | **11 ms** | |
| Stanford dragon (871k triangles) | 276 ms | 143 ms | 98 ms | **78 ms** | 250 ms |

One thread is a little slower than before (the binning pass is extra work); from 2 threads
up it is faster, and with all 8 about 3–4× faster. Past 4 threads the gain is small: the
machine has 4 physical cores, and the dragon is limited by memory traffic (every frame
reads 871k primitives).

Volumes (the cloud, `Scene.volume`) are not drawn by the rasterizer yet; VOLUMES.md
section 7.1 describes how a raymarching pass would add them.

Not done on purpose: top-left fill rule (harmless for opaque triangles with a z-test),
far-plane clipping (pixels beyond it fail the depth test), shadows (a
shadow map would be the next step), shadows of clouds on the surfaces.

## Clouds and fog (`raster/volume_march.c`)

Triangles cannot show a cloud: it has no surface. After `render_scene`,
`raster_draw_volumes(rv, fb, scene, &params, &march)` marches one ray per pixel (or per
2×2 block) through the scene's media, from the eye to the surface the z-buffer holds, and
blends the light the media scatter towards the eye over the pixel. The depth is turned
back into a distance by inverting the projection: `NDC z = 2 depth − 1`, view depth
`= B / (A + NDC z)` with `A`, `B` the projection matrix's `m[2][2]`, `m[2][3]`, and the
distance along the ray is that depth times the length of the ray direction whose view-space
z is −1. The light reaching each point comes from a precomputed light grid (optical depth
to the lamp), kept in `rv` between frames. How it works in detail, numbers and checks:
VOLUMES.md section 7.1.

```c
RasterVolumes *rv = raster_volumes_create();        // once
MarchParams mp = march_default_params();            // 128 steps, light grid 48, full resolution
render_scene(fb, &scene, &p);
raster_draw_volumes(rv, fb, &scene, &p, &mp);       // every frame
raster_volumes_free(rv);
```

## Precision

`real` is `float` by default (`make DOUBLE=1` for double, see RAYTRACER.md section 3). The
rasterizer was written for float, and the float build reproduces its earlier images bit for
bit (all milestone images and the textured cube).

## Conventions worth remembering

- Framebuffer origin is the **top-left**, y grows downward. The viewport transform flips Y once.
- Counter-clockwise triangles (as seen from outside) are front faces, like OBJ files.
  `render_scene` does not cull (`render_params_camera` sets `cull_backfaces = 0`).
- Colors are `0x00RRGGBB`. If the window shows red and blue swapped, swap them in `rgb8`.
- `render_mesh`'s light direction is the unit vector pointing *towards* the light.
