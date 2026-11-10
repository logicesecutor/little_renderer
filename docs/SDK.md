# Using little_renderer as an SDK

The engine is one static library with one umbrella header. Your program includes `lr.h`
and links `liblr.a`; that is all. The two apps in `apps/` and the four programs in
`examples/` are built exactly this way, so they double as reference code.

| build | library | binaries | `real` is |
|---|---|---|---|
| `make` | `build/float/liblr.a` | `bin/` | `float` (default: faster, half the memory) |
| `make DOUBLE=1` | `build/double/liblr.a` | `bin/double/` | `double` (compiled with `-DLR_DOUBLE`) |

Include path `-Iengine`, header `lr.h`, link `-lm -lX11` (X11 only matters if you open a
window, but the library contains the window code, so link it anyway), and `-fopenmp`:
the library itself uses OpenMP (the rasterizer and the option parser), so it must be
linked, unless the library was built with `make SINGLE=1`.

---

## 1. Build and link

```sh
make lib                       # build/float/liblr.a
```

Compile your own program against it (from the top folder):

```sh
gcc -O2 -fopenmp -fsingle-precision-constant -Iengine my_app.c build/float/liblr.a -lm -lX11 -o my_app
gcc -O2 -fopenmp -DLR_DOUBLE                 -Iengine my_app.c build/double/liblr.a -lm -lX11 -o my_app
```

Or drop your file into `examples/`: the `Makefile` builds every `examples/NAME.c` into
`bin/NAME` with the right flags (`make bin/NAME`). Put the `.a` after your `.c` on the
command line: the linker reads libraries left to right and only takes what earlier files
need.

Rules of thumb:
- **Same precision as the library.** `-DLR_DOUBLE` changes the size of every struct, so
  an app compiled without it must not link the double library (and the other way round).
  `-fsingle-precision-constant` is optional for apps; it makes constants like `0.5` float
  so float math does not silently become double.
- **A `Scene` must start zeroed**: declare it `static`, or `memset` it to 0.
- **Paths are relative to where you run the program**: the examples load `models/...` and
  write `out/...`, so run them from the top folder.

---

## 2. The layers

| layer | what is in it | headers |
|---|---|---|
| `core/` | everything both backends share | `lrmath.h`, `rng.h`, `image.h`, `texture.h`, `camera.h`, `mesh.h`, `scene.h` |
| `raster/` | the rasterizer | `raster.h`, `render.h` |
| `rt/` | the path tracer | `intersect.h`, `octree.h`, `bvh.h`, `accel.h`, `tracer.h` |
| `platform/` | the window | `window.h` |
| `app/` | the command line of `render` and `view` (not in `lr.h`) | `app/options.h` |

The usual program: build a `Scene` (core), then draw it with `render_scene` (raster) or
`trace_path` (rt), or both, and save the result or show it in a window.

---

## 3. A scene

```c
#include "lr.h"

static Scene scene;                                      // zeroed
SceneOptions so = {.kind = SCENE_BOXES};                // the rest zero: no model, no cloud, no fog
scene_cornell_box(&scene, &so);                          // the classic Cornell box
```

Or your own (`examples/custom_scene.c`):

```c
static Scene s;
Material white = {MAT_DIFFUSE, vec3(0.75, 0.75, 0.75), vec3(0, 0, 0), 1.0};  // type, albedo, emission, ior
Material lamp  = {MAT_LIGHT,   vec3(0, 0, 0), vec3(12, 12, 12), 1.0};
Material glass = {MAT_GLASS,   vec3(1, 1, 1), vec3(0, 0, 0), 1.5};
int w = scene_add_material(&s, white), l = scene_add_material(&s, lamp), g = scene_add_material(&s, glass);

scene_add_prim(&s, make_quad(vec3(-600, 0, -600), vec3(0, 0, 1200), vec3(1200, 0, 0), w));  // floor
s.light = scene_add_prim(&s, make_quad(vec3(-150, 500, -150), vec3(300, 0, 0), vec3(0, 0, 300), l));
scene_add_prim(&s, make_sphere(vec3(0, 100, 0), 100, g));
scene_add_triangle(&s, vec3(300, 0, 0), vec3(400, 0, 0), vec3(350, 120, 0), NULL, w);  // a single triangle
scene_load_mesh(&s, "models/torus.obj", w, 200.0, vec3(250, 0, 0), 30.0);   // optional: a mesh

s.cam_from = vec3(0, 220, -650); s.cam_at = vec3(0, 110, 0); s.cam_up = vec3(0, 1, 0); s.cam_vfov_deg = 40;
```

`scene_load_mesh(scene, path, material, size, base, rot_y_deg)` reads an `.obj` or ASCII
`.ply`, turns it, scales its largest side to `size` and stands it on `base`. To do it in
steps: `Mesh *m = mesh_load(path); mesh_place(m, size, base, rot); scene_add_mesh(&s, m, mat);
mesh_destroy(m);`.

Things the engine expects from a hand-made scene:
- **Exactly one light**, a quad with a `MAT_LIGHT` material, whose index you store in
  `s.light`. It emits on the side `u × v` points to. The path tracer's direct light
  sampling and the rasterizer's point light both use only this quad. A second `MAT_LIGHT`
  object would be visible to the camera and in mirrors and glass, but it would not light
  diffuse surfaces.
- At most `MAX_MATS` (16) materials; primitives are unlimited (the array grows).
- Units are arbitrary, but the path tracer's ray offsets (`RT_EPS` in `rt/tracer.c`)
  assume scenes roughly hundreds of units across (the Cornell box is 555).

### A cloud (or smoke, fog: any medium)

```c
CloudParams cp = cloud_default_params();               // shape, resolution and material
cp.seed = 7; cp.sigma_t = 0.1;                          // another cloud, a bit denser
Aabb where = {vec3(-200, 150, -100), vec3(200, 350, 200)};
scene_add_volume(&s, volume_new_cloud(where, &cp));     // the scene owns it now
```

Or fill a grid yourself (smoke from a simulation, a measured volume):

```c
Volume *v = volume_create(nx, ny, nz, bounds);          // zeros, white-cloud material
v->density[(z * ny + y) * nx + x] = ...;                // 0 = empty, ~1 = dense
v->sigma_t = 0.05; v->albedo = vec3(0.9, 0.9, 0.9); v->g = 0.5;
volume_update(v);                                        // after every change: majorants + active box
scene_add_volume(&s, v);
```

The path tracer picks it up automatically (`trace_path` checks `scene->volumes`); the
rasterizer draws it when asked, after `render_scene`, with `raster_draw_volumes`
(`raster/volume_march.h`, RASTERIZER.md). Up to `MAX_VOLUMES` (8) media per scene; they may overlap
each other (their extinctions add) and the surfaces. How it is
rendered and what the parameters mean: [VOLUMES.md](VOLUMES.md).

---

## 4. Path tracing it (`examples/hello_rt.c`)

```c
AccelParams ap = accel_default_params();                 // octree depth, BVH leaf, bins, split
scene_set_accel(&scene, ACCEL_BVH, &ap);                 // or ACCEL_OCTREE / ACCEL_NONE

Camera cam = camera_make(scene.cam_from, scene.cam_at, scene.cam_up, scene.cam_vfov_deg, (real)W / H);
Image img = image_create(W, H);
for each pixel (x, y):
    Rng rng; rng_seed(&rng, y * W + x, 1);
    for each sample: sum += trace_path(&scene, camera_ray(&cam, (x + rng_real(&rng)) / W,
                                                               (y + rng_real(&rng)) / H), &rng, 50);
    img.px[y * W + x] = sum / samples;
image_write_ppm(&img, "out/hello_rt.ppm");
image_free(&img); scene_free(&scene);
```

- Call `scene_set_accel` after the last primitive is added; call it again if you add
  more. Without it, every ray tests every primitive (fine for a few dozen).
- **Threads:** `trace_path` only **reads** the scene, so any number of threads may call
  it at once, as long as each has its own `Rng` (the examples seed one per pixel, which also
  makes the image independent of the thread count). Do not call `scene_set_accel` or
  `scene_add_prim` while other threads are tracing.
- `trace_path` returns linear radiance; `image_write_ppm` clamps and sRGB-encodes it.
- **Caustics** (as `bin/render` does it): path trace with `TraceParams.skip_caustics = 1`,
  average the samples into an `Image`, then
  `light_trace_caustics(&scene, &cam, W, H, (long long)W * H * SPP, &params, seed, img.px)`
  adds the paths the path tracer left out. Without the second call, leave `skip_caustics`
  at 0, or the caustics are missing.

---

## 5. Rasterizing it (`examples/hello_raster.c`)

```c
Framebuffer *fb = fb_create(W, H);
fb_clear(fb, rgb8(0, 0, 0));                            // also clears the z-buffer
RenderParams p = render_params_camera(scene.cam_from, scene.cam_at, scene.cam_up,
                                      scene.cam_vfov_deg, W, H);   // same camera as the path tracer
p.mode = MODE_PHONG;                                    // or MODE_GOURAUD, MODE_FLAT, MODE_WIREFRAME
render_scene(fb, &scene, &p);
fb_save_ppm(fb, "out/hello_raster.ppm");
fb_destroy(fb);
```

`render_scene` needs no acceleration structure. It and `render_mesh` use all cores
(OpenMP; `omp_set_num_threads` or `OMP_NUM_THREADS` to limit them) and return when the
frame is done; the result is the same for any number of threads. Do not draw into the same
`Framebuffer` from two threads at once, and do not change the scene while it is drawn.
For a single mesh with a directional
light (the classic model viewer), use `render_mesh` (`examples/raster_milestones.c`):

```c
Mesh *mesh = mesh_load("models/torus.obj");             // vertex normals computed if missing
mesh_normalize(mesh);                                   // center, fit in [-1, 1]
RenderParams p = render_default_params(W, H);           // camera at z = 3.2, 45° lens, light
p.model = mat4_mul(mat4_rotate_x(0.45f), mat4_rotate_y(0.6f));
render_mesh(fb, mesh, &p);
```

- **Textures** (`render_mesh` only): `p.texture = texture_checker(256, 16);` or
  `texture_load_ppm("x.ppm")`, then `MODE_TEXTURED`.
- **Your own camera:** set `p.eye`, `p.view = mat4_look_at(eye, target, up)` and
  `p.proj = mat4_perspective(fov_rad, aspect, near, far)`.
- **Your own shading:** `draw_triangle` takes any `FragShader`, a function
  `uint32_t f(const real *vary, const void *ctx)`. `examples/raster_milestones.c`
  (`demo_triangles`) shows one that colors triangles from their vertex values.

---

## 6. A window (`platform/window.h`)

```c
LrWindow *win = window_open(W, H, "my app");            // NULL if there is no display
for (;;){
    LrEvent ev;
    while (window_poll(win, &ev)){                       // never blocks
        if (ev.type == LR_CLOSE || (ev.type == LR_KEYDOWN && ev.key == LR_KEY_ESCAPE)) goto done;
        // LR_KEYDOWN / LR_KEYUP: ev.key = 'w', '1', LR_KEY_LEFT, LR_KEY_SHIFT ...
        // LR_BUTTONDOWN / LR_BUTTONUP / LR_MOTION: ev.button, ev.x, ev.y;  LR_WHEEL: ev.wheel = +1 / -1
    }
    fb_clear(fb, 0); render_scene(fb, &scene, &p);      // or path trace into an Image and convert
    window_present(win, fb);
}
done: window_close(win);
```

To show a path traced `Image`, convert each pixel with `image_pack_srgb` into
`fb->color` (call `srgb_lut_init()` once first); `apps/view.c` does this progressively.

---

## 7. API reference

| header | what you get |
|---|---|
| `core/lrmath.h` | `real`, `REAL_MAX`, `PI`, `r_sqrt`/`r_sin`/`r_pow`/... (float or double versions), `Vec2`/`Vec3`/`Vec4`, `Mat4`, `Mat3` and `Quat` (`mat3_*`, `quat_axis_angle`, `quat_mul`, `quat_rotate`, `quat_to_mat3`, `quat_integrate`), `vec3()`, `v_add`, `v_sub`, `v_mul` (per channel), `v_scale`, `v_neg`, `v_dot`, `v_cross`, `v_len`, `v_norm`, `v_max`, `v_reflect`, `v_refract`, `mat4_identity`, `mat4_mul`, `mat4_mul_vec4`, `mat4_mul_point`, `mat4_mul_dir`, `mat4_rotate_x`, `mat4_rotate_y`, `mat4_look_at`, `mat4_perspective` |
| `core/rng.h` | `Rng`, `rng_seed(rng, seed, stream)`, `rng_u32`, `rng_real` (in [0,1)), `onb_from_normal`, `sample_cosine_hemisphere` |
| `core/image.h` | `Framebuffer` (`color` 0x00RRGGBB + `depth`), `rgb8`, `rgbf`, `fb_create`, `fb_destroy`, `fb_clear`, `fb_set_pixel`, `fb_show_depth`, `fb_save_ppm`; `Image` (linear `Vec3` pixels), `image_create`, `image_free`, `image_write_ppm` (sRGB); `image_srgb8`, `srgb_lut_init`, `image_pack_srgb` |
| `core/texture.h` | `Texture`, `texture_load_ppm`, `texture_checker`, `texture_sample`, `texture_destroy` |
| `core/camera.h` | `Ray`, `Camera`, `camera_make(from, at, up, vfov_deg, aspect)`, `camera_ray(cam, s, t)` with s, t in [0,1] from the top-left |
| `core/mesh.h` | `Mesh`, `Face`, `mesh_load` (`.obj` / ASCII `.ply`), `mesh_destroy`, `mesh_compute_normals`, `mesh_normalize` (fit in [-1,1]), `mesh_place` (scale, turn, stand on a point) |
| `core/noise.h` | `Noise`, `noise_init(seed)`, `noise_perlin`, `noise_worley`, `noise_fbm`, `noise_billow` |
| `core/volume.h` | `Volume` (density grid + `sigma_t`, `albedo`, `g`), `volume_create`, `volume_free`, `volume_update`, `volume_density` (trilinear), `volume_voxel_center`, `phase_hg`; `CloudParams`, `cloud_default_params`, `Cloud`, `cloud_init`, `cloud_density`, `volume_make_cloud`, `volume_new_cloud`, `volume_add_cloud` (a cloud into part of a bigger grid); `Volume.version` (changes at every `volume_update`) |
| `core/scene.h` | `Material`, `MatType` (`MAT_DIFFUSE`, `MAT_MIRROR`, `MAT_GLASS`, `MAT_LIGHT`), `Primitive`, `Aabb`, `make_sphere`, `make_quad`, `primitive_bounds`, `Triangle`, `triangle_bounds`, `scene_add_triangle`, `scene_num_shapes`, `scene_shape_bounds`, `scene_shape_material`, `Scene`, `SceneOptions`, `scene_cornell_box`, `scene_add_material`, `scene_add_prim`, `scene_add_mesh`, `scene_load_mesh`, `scene_light_center`, `scene_add_volume`, `scene_free` |
| `raster/raster.h` | `draw_line` (Bresenham), `draw_triangle` (edge functions, z-test, perspective-correct varyings), `draw_line_rows` / `draw_triangle_rows` (the same, limited to some rows, so threads can share a framebuffer), `ScreenVert`, `FragShader`, the `V_*` varying slots |
| `raster/render.h` | `RenderMode`, `RenderParams`, `render_default_params`, `render_params_camera`, `render_mesh`, `render_scene`, `render_mode_name`, `render_mode_from_name` |
| `raster/volume_march.h` | `MarchParams` (`max_steps`, `step_voxels`, `light_res`, `ambient`, `octaves`, `brightness`, `scale`), `march_default_params`, `RasterVolumes`, `raster_volumes_create`, `raster_draw_volumes(rv, fb, scene, params, march)`, `raster_volumes_timing`, `raster_volumes_free` |
| `rt/intersect.h` | `Hit` (shading normal `n`, geometric normal `ng`), `aabb_hit` (slab test), `intersect_primitive`, `intersect_shape` |
| `rt/octree.h`, `rt/bvh.h` | the two acceleration structures (normally used through `scene_set_accel`): `octree_build`, `octree_intersect`, `octree_auto_depth`, `bvh_build`, `bvh_intersect` |
| `rt/accel.h` | `AccelKind`, `AccelParams`, `accel_default_params`, `scene_set_accel`, `scene_accel_kind`, `accel_name`, `scene_accel_stats`, `scene_intersect`, `scene_occluded` |
| `rt/medium.h` | `medium_sample_distance` (delta tracking), `medium_transmittance` (ratio tracking), both on the majorant grid; `scene_medium_collision`, `scene_medium_transmittance` (all media of a scene); `medium_sample_phase` (Henyey-Greenstein) |
| `rt/tracer.h` | `TraceParams` (`max_depth`, `mis`, `skip_caustics`), `trace_default_params`, `trace_path_ex(scene, ray, rng, params)`, `trace_path(scene, ray, rng, max_bounces)` (defaults: MIS on): linear RGB radiance of one path; `light_trace_caustics(scene, cam, w, h, photons, params, seed, image)`: adds the light-traced caustics (RAYTRACER.md 5.14) |
| `platform/window.h` | `LrWindow`, `LrEvent`, `window_open`, `window_poll`, `window_present`, `window_set_title`, `window_close`, the `LR_KEY_*` codes |
| `phys/rigid.h` | `Body`, `BodyShape`, `Contact`, `RigidWorld`, `rigid_init`, `rigid_free`, `rigid_set_room`, `rigid_add_sphere`, `rigid_add_box`, `rigid_step`, `body_inv_inertia_world`, `body_point_velocity`, `body_contains`, `body_apply_impulse`, `rigid_kinetic_energy`, `rigid_momentum` |
| `phys/fluid.h` | `Fluid` (MAC grid, parameters: `wind`, `wind_rate`, `buoyancy`, `vorticity`, `dissipation`, `pressure_cycles`, `open`), `FLUID_OPEN_*`, `fluid_create(bounds, cell, density_volume)`, `fluid_free`, `fluid_step`, `fluid_velocity`, `fluid_add_impulse`, `fluid_add_expansion`, `fluid_clear_solids`, `fluid_kinetic_energy`, `fluid_total_density` |
| `phys/poisson.h` | `Poisson`, `poisson_init`, `poisson_setup`, `poisson_solve` (multigrid V-cycles), `poisson_residual`, `poisson_free` |
| `phys/world.h` | `PhysWorld` (a `RigidWorld` + a `Fluid`, coupled), `phys_init`, `phys_free`, `phys_step`, `phys_blast`, `phys_air_force`, `phys_air_velocity_around`, `phys_add_to_scene`, `phys_sync_scene` |
| `app/options.h` | the command-line layer of `render` / `view` / `sim`: `options_parse`, `options_setup_scene`, `options_usage`, `options_print_command`. Include it yourself to give your app the same options. |
| `app/frame.h`, `app/simdemo.h` | `frame_render_rt` (a whole path traced image); the physics demos (`sim_setup`, `sim_step`, ...) |

How each part works: [RAYTRACER.md](RAYTRACER.md), [RASTERIZER.md](RASTERIZER.md),
[VOLUMES.md](VOLUMES.md) and [PHYSICS.md](PHYSICS.md) (which also shows how to use the
physics in your program, as `examples/hello_physics.c` does).

---

## 8. Not included (yet)

- No shared library (`.so`) and no `make install`. Copy `engine/` (headers) and the `.a`
  file, or add the repository as a git submodule.
- Linux/X11 only for the window (`platform/window.c`, the one file to port). The rest is
  plain C, but the mesh loader uses POSIX `getline`, which native Windows compilers lack.
  Under WSL2 everything runs unchanged (on Windows 11 the window appears directly through
  WSLg).
