# The path tracer (`engine/rt`)

The engine has two backends that draw the same `Scene`: a **rasterizer** (see
RASTERIZER.md) and this physically based **path tracer**. Both are plain C, with no
libraries except `libm` (and optional OpenMP to use every CPU core). The path tracer
renders the classic **Cornell box**:

- `-scene boxes`: the original Cornell box, with a tall box and a short box.
- `-scene spheres`: the same room with a mirror sphere and a glass sphere.
- `-scene mesh`: any triangle model from an `.obj` or ASCII `.ply` file inside the box,
  accelerated by a **BVH** (section 5.12) or an **octree** (section 5.11), both written
  from scratch. It renders the 871,414-triangle **Stanford dragon** comfortably.
- `-scene cloud` (or `-cloud on` with any scene): a **volumetric cloud**, a participating
  medium rendered with delta tracking, ratio tracking and multiple scattering. It has its
  own document: **VOLUMES.md**.

Two programs use it: `render` writes a PNG (`-backend rt`, the default), and `view`
(section 5.13) is a window you can fly through: the rasterizer draws while you move, and
the path tracer refines the image when you stop.
**Every command, option, key and tunable constant is listed in `USAGE.md`.**

The image shows everything a path tracer gets "for free": soft shadows from the area
light, **color bleeding** (red and green light from the walls tinting the boxes, floor and
ceiling), reflections and refraction, and the **caustic** (light focused by the glass ball
onto the floor).

---

## 1. Build and run

Run these from the top folder of the repository:

```sh
make                  # build/float/liblr.a, bin/render, bin/view, the examples (gcc -O2, OpenMP)
make DOUBLE=1         # the same in double precision, into bin/double/
bin/view              # the real-time window (section 5.13)
make run              # quick previews: out/cornell_boxes.png, _spheres.png, _mesh.png (+ _raster)
make final            # slow, clean 600x600 / 1024 samples per pixel versions
make bench            # none vs. octree vs. BVH on the torus, compares the images
make dragon-download  # fetches the Stanford dragon (11 MB) into models/dragon_recon/
make dragon           # renders it in the box: out/cornell_dragon.png and _glass.png
make view-dragon      # flies around it in the viewer
make bench-dragon     # octree vs. BVH on the dragon
make SINGLE=1         # build without OpenMP (one thread)
make SAN=1            # build with the address + undefined-behaviour sanitizers
```

The most used options (`render` and `view` take the same ones):

```
bin/render -n 600 -s 512 -d 50 -scene mesh -obj model.ply -mat glass -rot -35 \
           -cam 278,278,-800 -look 278,278,0 -fov 40 -accel bvh -clamp 4 -o out/x.png
```

The full list, with defaults, tips, the viewer keys and the constants you can change in
the code, is in **`USAGE.md`**. `-h` prints it too. `make models`
creates `models/torus.obj` (2304 triangles) and `models/torus_dense.obj` (36,864).

**float or double.** The engine computes in `float` by default and in `double` with
`make DOUBLE=1` (section 3, "Precision"). Measured on the development machine (8 threads,
256x256, 256 spp):

| scene | float | double | float is faster by |
|---|---|---|---|
| boxes | 6.9 s | 8.1 s | 14% |
| spheres, `-clamp 4` | 3.6 s | 4.2 s | 14% |
| torus (2,310 primitives, BVH) | 8.8 s | 11.8 s | 25% |
| glass dragon `_res3` (48k triangles, BVH), `-clamp 4` | 21.3 s | 27.6 s | 23% |

The images of the two builds differ only by noise (on average less than 0.3 out of 255
per channel for boxes, spheres and torus).

The measurements below were made earlier with the double build (8 threads):

| render | time |
|---|---|
| boxes, 400x400, 128 spp | 9.8 s |
| spheres, 400x400, 128 spp | 4.5 s |
| boxes, 600x600, 1024 spp | 173 s |
| spheres, 600x600, 1024 spp | 85 s |
| mesh (torus, 2,304 triangles, octree), 600x600, 1024 spp | 590 s |
| mesh (dense torus, 36,864 triangles, glass, octree), 400x400, 256 spp | 110 s |
| Stanford dragon (871,414 triangles), 600x600, 512 spp (`make dragon`) | 411 s |
| Stanford dragon, glass, `-clamp 4`, 600x600, 512 spp | 529 s |

The glass torus is very noisy at 256 spp for the same reason as the glass sphere
(section 7); add `-clamp 4` or many more samples.

The spheres scene is faster because it has fewer primitives (2 spheres instead of
12 box faces), and more paths escape through the open front of the box.

(The mesh renders above used the octree; with the BVH they are about twice as fast.)

**Brute force vs. octree vs. BVH** (double build, 8 threads; the images are bit-for-bit
identical in every row):

| scene | primitives | none | octree | BVH |
|---|---|---|---|---|
| boxes, 400x400, 128 spp | 18 | **7.0 s** | 12.0 s | 9.6 s |
| spheres, 400x400, 128 spp | 8 | **3.7 s** | 4.8 s | 7.7 s |
| torus, 200x200, 16 spp (`make bench`) | 2,310 | 23.2 s | 0.7 s | **0.3 s** |
| dense torus, 200x200, 32 spp | 36,870 | (hours) | 1.6 s | **0.7 s** |
| dragon, 300x300, 32 spp (`make bench-dragon`) | 871,420 | (weeks) | 4.9 s + 0.6 s build | **2.7 s** + 3.7 s build |

Brute force grows linearly with the number of triangles; both trees grow roughly with
its logarithm. The BVH traces about twice as fast as the octree but takes longer to
build (section 5.12). For the original box with only 18 large quads, walking any tree
costs more than it saves, so `-accel auto` (the default) uses the BVH from 64 primitives
up and brute force below that.

---

## 2. The idea in one paragraph

A rasterizer (`engine/raster`, see RASTERIZER.md) asks: *"for each triangle, which pixels does it
cover?"* A ray tracer turns this around: *"for each pixel, what does the camera see in
that direction?"* A **path tracer** answers that question physically. From the camera it
follows a ray into the scene; when the ray hits a surface it bounces in a random direction
that the material allows, again and again, collecting the light it meets along the way.
One such random path is a very noisy guess of the pixel's color. The **average of many
paths** converges to the correct answer, the solution of the *rendering equation*. That
is why the image starts out grainy and gets cleaner with more samples.

---

## 3. Code structure

The path tracer is the `rt/` layer of the engine. It builds on `core/`, which it shares
with the rasterizer (the whole layout is in README.md and SDK.md):

```
engine/
├── core/                shared by both backends
│   ├── lrmath.h         `real` (float or double), Vec3 (points, directions, RGB), Mat4, v_* functions
│   ├── rng.h            PCG32 random numbers, orthonormal basis, cosine-weighted hemisphere sampling
│   ├── camera.h         Ray, pinhole camera: pixel coordinates -> primary ray (header only)
│   ├── mesh.h/c         .obj and ASCII .ply readers, vertex normals, placing a model
│   ├── scene.h/c        materials, primitives (sphere | quad | triangle), Scene, the Cornell box
│   ├── image.h/c        Image (linear float pixels) -> sRGB -> PNG; also the rasterizer's Framebuffer
│   └── png.h/c          PNG writer and reader, with deflate / inflate (no library)
├── rt/                  the path tracer
│   ├── intersect.h/c    Hit, ray-box slab test, sphere / quad / triangle intersection
│   ├── octree.h/c       octree build and front-to-back traversal (closest hit and any hit)
│   ├── bvh.h/c          BVH build (binned SAH or middle split) and stack-based traversal
│   ├── accel.h/c        builds none / octree / BVH for a scene; scene_intersect, scene_occluded
│   └── tracer.h/c       trace_path(): direct light sampling, glass / mirror logic
├── raster/              the rasterizer (RASTERIZER.md)
├── platform/window.h/c  the X11 window (the viewer)
└── app/options.h/c      command-line options of the two programs
apps/
├── render.c             `bin/render`: pixel loop (OpenMP), anti-aliasing, timing, PNG output
└── view.c               `bin/view`: camera controls, raster preview + progressive path tracing
examples/hello_rt.c      the smallest program that path traces
```

Dependencies only point downward, so each layer can be understood on its own
(`make check-layers` verifies it):

```
   apps/render.c     apps/view.c ──► platform/window.c (X11)     examples/hello_rt.c
         │                │                                            │
         └───────┬────────┴────────────────────────────────────────────┘
                 ▼   app/options.c (command line), then the library, liblr.a
   rt/tracer.c ──► rt/accel.c ──┬──► rt/bvh.c ────┐
                                └──► rt/octree.c ─┴──► rt/intersect.c
                 ▼                                         ▼
   core/scene.c ──► core/mesh.c       core/camera.h      core/rng.h      core/image.c
                         └──────────────► core/lrmath.h ◄───────────────┘
```

`rt/` never includes `raster/` or the reverse: the two backends only meet in `core/`
(the scene they both draw) and in the apps.

### What happens for one pixel

```
apps/render.c   for each of the spp samples:
                  pick a random point inside the pixel            (anti-aliasing)
core/camera.h     build the ray from the camera through that point
rt/tracer.c       trace_path(): loop over bounces
rt/accel.c          closest hit: walk the BVH or the octree (or test every primitive)
rt/bvh.c, octree.c    visit only the boxes / cells the ray passes through, near to far
rt/intersect.c          sphere / quad / triangle intersection tests
rt/tracer.c         material: diffuse -> sample the lamp + bounce randomly
                              mirror  -> reflect
                              glass   -> reflect or refract (Fresnel)
                              light   -> add its emission, stop
                    Russian roulette may stop the path
apps/render.c   average the samples -> linear RGB pixel
rt/tracer.c     light_trace_caustics adds the caustics, traced from the lamp (section 5.14)
core/image.c    clamp, sRGB encode, write 3 bytes
```

### File by file

**`core/lrmath.h`**: `Vec3 {real x, y, z}` is used for positions, directions *and*
colors. `v_mul` multiplies component-wise, which is how colors are filtered
("red wall × white light = red light"). `v_reflect` and `v_refract` implement the mirror
and Snell laws (section 5.6). The same header has the matrices the rasterizer needs.

**Precision.** `real` is `float` unless the engine is built with `make DOUBLE=1`
(`-DLR_DOUBLE`); every math call goes through a macro (`r_sqrt`, `r_pow`, ...) that picks
`sqrtf` or `sqrt`. Float has 24 bits of mantissa, about 7 digits: in a box 555 units wide,
one float step is about 6·10⁻⁵ units. A new ray starting on a surface must therefore skip
a little more distance to avoid hitting that same surface again ("shadow acne", section
5.2): `tracer.c` uses `RT_EPS = 2e-3` and stops shadow rays `RT_SHADOW_MARGIN = 2e-2`
before the lamp in float, `1e-4` and `1e-3` in double. With these values the float images
show no acne and match the double ones up to noise. The double build reproduces the
earlier double-only version of this path tracer bit for bit.

One visible difference: a ray that hits the shared edge of two triangles may find both at
*exactly* the same distance in float. The one tested first wins, and the BVH, the octree
and brute force test them in different orders. So in float a handful of pixels can differ
between `-accel` choices (`make bench`: 4 pixels out of 40,000); in double they are
identical.

**`core/rng.h`**: the random number generator and the sampling routines built on it:
- `rng_u32` / `rng_real`: PCG32, a small, fast, statistically good generator. `rng_real`
  gives a number in `[0, 1)`: 32 random bits in double, 24 in float (more would round up
  to 1.0).
- `onb_from_normal`: builds two tangent vectors around a normal (an orthonormal basis), so a
  direction generated "around +z" can be rotated to "around the surface normal".
- `sample_cosine_hemisphere`: random bounce directions for diffuse surfaces (section 5.4).

**`core/camera.h`**: `Ray {o, d}` and a pinhole camera. A virtual screen 1 unit in front
of the eye, with its size set by the vertical field of view. `camera_ray(s, t)` returns
the unit ray through the screen point `(s, t)`.

**`core/scene.h/c`**:
- `Material`: type (`DIFFUSE`, `MIRROR`, `GLASS`, `LIGHT`), `albedo` (the fraction of
  light reflected, per channel), `emission`, `ior` (index of refraction).
- Two kinds of geometry. A `Primitive` is a sphere (`center`, `radius`) or a **quad** (a
  parallelogram `q + a·u + b·v`, with `a, b` in `[0, 1]`); `make_quad` precomputes the
  normal, the plane equation, the area and the helper vector `w` used by the
  intersection. Mesh triangles are a separate, compact `Triangle`: corners `q`, `q + u`,
  `q + v` and three indices into one shared array of vertex normals (`Scene.normals`), 52
  bytes in float instead of the ~130 a general primitive needs. The geometric normal is
  not stored: it is computed for the hits that are kept. On the 871k-triangle dragon this
  took the peak memory from 192 MB to 129 MB.
- Together they are numbered as **shapes**: ids `0 .. nprims−1` are the primitives, then
  the triangles. The trees, the hit records (`Hit.prim`) and the rasterizer use shape ids;
  `scene_shape_bounds`, `scene_shape_material` and `intersect_shape` work on any shape.
  `Aabb {lo, hi}` is an axis-aligned box; flat shapes get a 0.001 padding so their box is
  never zero-thick.
- `Scene`: the primitives, the triangles and their normals, a fixed array of materials,
  the index of the light, the camera, the participating media (`volumes`, VOLUMES.md), and
  a slot for the acceleration structure (`accel`, owned by `rt/`).
- `scene_cornell_box` builds the room (section 6). `add_box` makes a rotated box from
  6 quads by rotating the quad corners and edges directly. For `-scene mesh` it calls
  `scene_load_mesh` (`mesh_load` + `mesh_place` + `scene_add_mesh`).

The scene is a description only: it does not know rays or pixels. That is why the
rasterizer can draw it too (`render_scene` in `raster/render.c`).

**`core/mesh.h/c`**: `mesh_load` picks the reader from the file extension.
- **OBJ** (`read_obj`): `v` (positions), `vt` (texture coordinates, for the rasterizer),
  `vn` (normals) and `f` (faces) lines. Negative (relative) indices work.
- **PLY** (`read_ply`), ASCII only, which is what the Stanford scans use. The header says
  how many vertices and faces follow and in which order the vertex properties come
  (`x y z`, sometimes plus `confidence`, `intensity`...). Only x, y, z are kept. Every face
  line is `count i0 i1 i2 ...`.
- Faces with more than 3 corners are split into a fan of triangles. Faces that point at
  missing vertices are dropped.
- **Smooth normals**: if the file has none (PLY scans never do), every triangle adds its
  un-normalized cross product `(b − a) × (c − a)` to its three vertices. Its length is
  twice the triangle's area, so big triangles weigh more. Normalizing the sums (done by
  `scene_add_mesh`, once per vertex) gives area-weighted vertex normals, and the 871k-triangle dragon looks
  smooth instead of faceted.
- `mesh_place` turns the model by `-rot` around the vertical axis, scales it so its largest
  side is 330 units, centers it in the box and stands it on the floor.

**`rt/intersect.h/c`**:
- `Hit {t, p, n, front_face, prim}`.
- `intersect_shape` (any shape id) and `intersect_primitive` return the hit with
  `tmin < t < tmax` (section 5.2). `Hit` holds the shading normal `n` and the geometric
  normal `ng` (they differ on smooth meshes; light tracing needs both, section 5.14).
- `set_hit` flips the normal so it always faces the incoming ray, and remembers in
  `front_face` whether it was flipped. Glass needs that to know if the ray is entering or
  leaving, and the lamp uses it to emit on one side only.
- `aabb_hit`, the ray-box test both trees use (section 5.11).

**`rt/accel.h/c`**: the bridge between a scene and the trees.
- `scene_set_accel(scene, kind, params)` builds an octree or a BVH (or nothing) over the
  scene's primitives and stores it in `scene->accel`; `scene_free` frees it.
  `scene_accel_stats` prints its statistics.
- `scene_intersect` finds the closest hit. It delegates to `bvh_intersect` or
  `octree_intersect`. Without a tree it tests every primitive and keeps the closest hit,
  shrinking `tmax` each time.
- `scene_occluded` answers the cheaper question a shadow ray asks: *is there anything at
  all* between here and the light? It can stop at the first hit.

**`rt/octree.h/c`** and **`rt/bvh.h/c`**: the two acceleration structures, explained in
sections 5.11 and 5.12.

**`rt/tracer.h/c`**: the heart of the path tracer, explained in section 5.
`trace_path_ex(scene, ray, rng, params)` takes a `TraceParams` (`max_depth`, `mis`,
`skip_caustics`); `trace_path(scene, ray, rng, max_depth)` uses the defaults (MIS on).
`light_trace_caustics` is the light tracer of section 5.14.

**`rt/medium.h/c`**: participating media (clouds, fog), see VOLUMES.md.

**`core/image.h/c`**: the image is stored in **linear** radiance (`Vec3` per pixel). Only on
output is each value clamped to `[0, 1]` and converted with the sRGB transfer curve
(section 5.9). The output is a PNG, written by `core/png.c` (below).

**`core/png.h/c`**: PNG files without a library. Writing: each row gets the PNG filter that
leaves the smallest bytes (none, sub, up, average or Paeth: each predicts a byte from its
neighbours and stores the difference), then deflate compresses the rows: LZ77 (a repeated
run of bytes becomes a back reference, found through hash chains of 3-byte prefixes) and
the fixed Huffman codes. Reading: the whole format (every colour type and bit depth,
palettes, transparency, interlacing) with an inflate that decodes all three kinds of
deflate blocks; the textures and, later, glTF files use it. Every chunk carries a CRC-32,
the compressed stream an Adler-32: both are checked.

**`app/options.h/c`**: one parser for both programs (`-h` prints the list), so every option
means the same in `render` and `view`. `options_setup_scene` builds the scene, applies
`-cam`/`-look`/`-fov`, picks the acceleration structure (`-accel auto` = BVH from 64
primitives up) and sets the thread count. `options_print_command` writes the `render`
command line that reproduces a view, which is what key **C** in the viewer uses.

**`apps/render.c`**: parses options, builds the scene and camera, and loops over pixels. Each
pixel gets **its own seeded random generator**, so the image is bit-for-bit identical for any
number of threads. `#pragma omp parallel for schedule(dynamic, 1)` hands out one row at a
time to each core. Rows near the boxes and spheres cost more than rows of empty wall, and
dynamic scheduling keeps all cores busy. The pragmas sit inside `#ifdef _OPENMP`, so the
same file compiles with or without OpenMP. With `-backend raster` it calls the
rasterizer instead.

---

## 4. Coordinate system and conventions

- Right-handed world, **y up**. The box spans `0..555` on every axis, with the open
  side towards **−z**.
- The camera sits at `(278, 278, −800)` and looks at `(278, 278, 0)` with a 40° vertical field
  of view. Looking down +z, the camera's right vector is **−x**, so the wall at `x = 555`
  appears on the **left** of the image. That is why the red wall is placed at `x = 555`,
  giving the original Cornell layout: red left, green right.
- Normals in `Hit` always face the incoming ray. A quad's *geometric* normal is
  `normalize(u × v)`. For the lamp it points down (−y), the only side that emits.
- Colors are linear RGB until the very last step.

---

## 5. Method

### 5.1 Primary rays and anti-aliasing

For pixel `(x, y)` and each sample, `apps/render.c` picks `u = (x + ξ₁)/N`, `v = (y + ξ₂)/N`
with random `ξ₁, ξ₂ ∈ [0, 1)`. Averaging rays spread uniformly over the pixel is a box
filter, so edges come out smooth without any extra code. This is the same Monte Carlo
averaging that removes the lighting noise.

Camera basis (`core/camera.h`):

```
w = normalize(from − at)      (backwards)
u = normalize(up × w)         (right)
v = w × u                     (up)
screen height = 2·tan(fov/2),  width = aspect · height
ray(s, t) = normalize(upper_left + s·horizontal − t·vertical − origin)
```

### 5.2 Ray-shape intersection

A ray is `P(t) = o + t·d`. Every test returns the smallest `t` in `(tmin, tmax)`.

**Sphere.** `|P(t) − c|² = r²` is a quadratic in `t`. With `h = d·(c − o)`,
`a = d·d` and `c' = |c − o|² − r²`:

$$t = \frac{h \mp \sqrt{h^2 - a\,c'}}{a}$$

A negative discriminant means a miss. The nearer root is tried first, then the farther one:
that is the case when the ray starts *inside* the sphere, as refracted rays in glass do.
The outward normal is `(P − c)/r`.

**Quad.** First intersect the plane `n·P = D`:

$$t = \frac{D - n\cdot o}{n \cdot d}$$

(with `n·d ≈ 0` the ray is parallel and misses). Then write the hit point relative to the
corner, `p = P − q`, in the quad's own coordinates `p = a·u + b·v`. With the precomputed
`w = (u × v)/|u × v|²`:

$$a = w \cdot (p \times v), \qquad b = w \cdot (u \times p)$$

The point is on the quad if `0 ≤ a ≤ 1` and `0 ≤ b ≤ 1`. The same code handles walls,
the lamp and box faces.

**Triangle (Möller-Trumbore).** A point inside the triangle is `q + b₁·u + b₂·v` with
`b₁, b₂ ≥ 0` and `b₁ + b₂ ≤ 1` (barycentric coordinates). Setting it equal to the ray,
`o + t·d = q + b₁u + b₂v`, gives a 3×3 linear system in `(t, b₁, b₂)`. Cramer's rule solves
it with two cross products:

```
p = d × v      det = u · p          (det ≈ 0: ray parallel to the triangle)
s = o − q      b₁ = (s · p) / det
k = s × u      b₂ = (d · k) / det      t = (v · k) / det
```

Each barycentric test can reject early, before the next quantity is computed.
`front_face` uses the true geometric normal `u × v`. With vertex normals, the shading
normal is `normalize((1 − b₁ − b₂)·n0 + b₁·n1 + b₂·n2)`, which makes a faceted model look
smooth. It is turned to the side the ray came from, like `h->n`, so a triangle
that is in the file twice with opposite winding shades the same either way.

**Self-intersection (shadow acne).** A new ray starts exactly on a surface. Rounding
error can make it hit that same surface at `t ≈ 0`, which shows up as dark speckles. Every
search therefore starts at `tmin = RT_EPS` (`1e-4` in double, `2e-3` in float, see
section 3, "Precision").

### 5.3 The rendering equation and Monte Carlo

The light leaving a point `x` towards direction `ωo` is

$$L_o(x,\omega_o) = L_e(x,\omega_o) + \int_{\Omega} f_r(x,\omega_i,\omega_o)\; L_i(x,\omega_i)\; \cos\theta_i \; d\omega_i$$

The first term is the light emitted (non-zero only on the lamp). The second is the light
arriving from every direction `ωi` of the hemisphere, weighted by the material's **BRDF**
`f_r` and by `cos θi` (light hitting at a grazing angle is spread over more area). The
incoming light `L_i` is itself the outgoing light of some other point, so the equation is
recursive. That recursion is exactly "the ray bounces".

The integral has no closed form, so it is estimated with **Monte Carlo**: draw a random
direction `ωi` with probability density `p(ωi)` and use

$$\int g(\omega)\,d\omega \;\approx\; \frac{g(\omega_i)}{p(\omega_i)}$$

This single-sample estimate is correct *on average* (unbiased), but noisy. Averaging `N`
of them shrinks the noise like `1/√N`: 4× more samples halves the grain. That is why the
1024 spp image is only about 3× cleaner than the 128 spp one, despite 8× the work.

### 5.4 Diffuse surfaces and importance sampling

The walls and boxes are **Lambertian**: they scatter light equally in all directions.
Their BRDF is a constant:

$$f_r = \frac{\text{albedo}}{\pi}$$

(the `1/π` makes the surface reflect exactly `albedo` of the incoming energy, never more).

Directions could be picked uniformly over the hemisphere, but the integrand contains
`cos θ`, so directions near the normal matter more. `sample_cosine_hemisphere` picks
directions with density `p(ω) = cos θ / π` (Malley's method: pick a uniform point on the
unit disk, lift it onto the hemisphere). Then the Monte Carlo weight simplifies beautifully:

$$\frac{f_r \cos\theta}{p} = \frac{(\text{albedo}/\pi)\cos\theta}{\cos\theta/\pi} = \text{albedo}$$

So in the code a diffuse bounce is just `throughput *= albedo`. Choosing samples shaped
like the integrand is called **importance sampling**, and it lowers the noise for free.

### 5.5 Next event estimation (direct light sampling)

The lamp is small: 130×105 units on a 555-unit ceiling. A randomly bouncing ray hits it
rarely, so a pure "bounce until you hit the light" tracer is extremely noisy. Instead, at
every diffuse hit, `sample_direct_light` **aims a ray at the light on purpose**:

1. Pick a uniform random point `x_l` on the light quad (probability `1/area` per unit area).
2. Compute direction `ωi`, distance `d`, and the two cosines: `cos θs` at the surface and
   `cos θl` at the light (the lamp only emits downward, so `cos θl ≤ 0` means no light).
3. Cast a **shadow ray**. If anything is hit before the light, the point is in shadow.
4. The integral is over *directions*, but we sampled *area*. A small patch of light
   `dA` at distance `d`, tilted by `θl`, covers the solid angle `dω = cos θl · dA / d²`.
   Substituting gives the estimate

$$L_{direct} = \frac{\text{albedo}}{\pi}\; L_e\; \cos\theta_s \;\frac{\cos\theta_l \cdot \text{Area}}{d^2}$$

That is exactly the last line of `sample_direct_light`. The soft shadows come from the
fact that different samples pick different points of the light: a point that sees part
of the lamp is in **penumbra**.

**Avoiding double counting.** The light now reaches diffuse surfaces in two ways: through
the explicit sample, and if the random bounce happens to hit the lamp. Counting both fully
would make the image too bright. What a lamp hit counts for depends on the previous vertex
(`emit` in `trace_path_ex`):

- camera rays and **mirror / glass** bounces (`EMIT_FULL`): those directions are fixed by
  the material, so the light could not have been sampled explicitly, and hitting the lamp
  counts fully;
- after a **diffuse** bounce without MIS (`-mis off`, `EMIT_NONE`): the light was already
  added by next event estimation, so a lamp hit just ends the path;
- after a diffuse bounce with MIS (the default, `EMIT_MIS`): both count, with weights.

### 5.5.1 Multiple importance sampling (MIS)

Next event estimation is excellent for a small, distant lamp, and poor where the lamp is
large as seen from the point (the walls and ceiling right next to it, fog around it): the
random point on the lamp is then often at a grazing angle or very close, and a few samples
carry huge weights. The cosine-distributed bounce has the opposite strengths. **MIS**
(Veach & Guibas 1995) uses both for the same light and weights each sample by how likely
*each* technique was to produce it, with the **power heuristic**:

$$w_{light} = \frac{p_{light}^2}{p_{light}^2 + p_{bsdf}^2}, \qquad w_{bsdf} = \frac{p_{bsdf}^2}{p_{light}^2 + p_{bsdf}^2}$$

both as densities per unit solid angle of the direction to the lamp point:
`p_light = d² / (cos θ_l · Area)` (section 5.5's change of variables, reversed) and
`p_bsdf = cos θ_s / π` (or the phase function in a medium). The two weights add up to 1
for every path, so the sum stays unbiased; each path is counted mostly by the technique
that samples it well. `sample_direct_light` multiplies by `w_light`; a bounce that hits the
lamp adds `Le · w_bsdf`.

Measured (`-mis off` vs on, same samples, error against a 3072-sample reference): the
spheres scene with fog −28% RMSE; the plain boxes no change (its lamp is small and far
from everything, where light sampling already wins). The cost is negligible.

### 5.6 Mirror and glass

(Since PBR1 these, and the diffuse surfaces of 5.4, are presets of one material model with
a microfacet BSDF: section 5.15. Roughness 0 gives exactly the mirror and glass below,
with the Fresnel reflectance of the metal and the glass.)

These are *specular* materials: for an incoming direction there is only one outgoing
direction (or two, for glass), so no hemisphere sampling is needed.

**Mirror:** `r = d − 2(d·n)n`, and `throughput *= albedo` (0.95: real mirrors absorb a
little).

**Glass** (index of refraction 1.5):
- `front_face` says whether the ray is entering (`η = 1/1.5`) or leaving (`η = 1.5`).
- **Snell's law** `sin θt = η sin θi` gives the refracted direction (`v_refract`: the part
  of the direction parallel to the surface scales by `η`, then the normal part is chosen to
  keep unit length).
- If `η sin θi > 1` there is no refracted ray: **total internal reflection**.
- Otherwise a real surface reflects a fraction `F` of the light and transmits `1 − F`
  (**Fresnel**). `F` comes from **Schlick's approximation**:

$$F(\theta) = R_0 + (1 - R_0)(1 - \cos\theta)^5, \qquad R_0 = \left(\frac{1-\eta}{1+\eta}\right)^2$$

  Instead of splitting the path in two, the tracer **reflects with probability F and
  refracts otherwise**. On average that is the correct mix, and every path stays a single
  chain of rays. That is why the glass ball shows a faint reflection of the room at
  grazing angles.

The bright spot under the glass ball is a **caustic**: lamp light focused by the sphere.
It appears automatically, because paths from the floor can go through the glass, and after a
specular bounce hitting the lamp counts (`emit = EMIT_FULL`).

### 5.7 Throughput and the bounce loop

`trace_path` does not use recursion. It carries two colors:

- `radiance`: the light collected so far,
- `throughput`: how much of any light found *from now on* reaches the camera. It is the
  product of the albedos (and Monte Carlo weights) of all bounces so far.

```
radiance = 0, throughput = 1, emit = FULL
loop:
    hit = closest intersection; none -> stop (background is black)
    if hit is the lamp:  if front side: radiance += throughput * Le * (1, 0 or the MIS weight, by emit)
                         stop
    if depth >= max_depth: stop
    diffuse: radiance += throughput * direct_light(hit)
             new direction = cosine sample; throughput *= albedo; emit = MIS (or NONE)
    mirror:  new direction = reflect;       throughput *= albedo; emit = FULL
    glass:   reflect or refract (Fresnel);  throughput *= albedo; emit = FULL
    Russian roulette (below)
    ray = (hit point, new direction)
```

### 5.8 Russian roulette

Each bounce multiplies `throughput` by an albedo below 1, so late bounces contribute less
and less, but tracing them costs as much as the first. After `RR_START = 3` bounces
the path survives with probability

$$p = \min(\max(\text{throughput}_{r,g,b}),\ 0.95)$$

and survivors are divided by `p`. The expected value is unchanged, so the image stays
unbiased, while dim paths stop early. `max_depth` (50) is only a safety net; Russian
roulette ends almost all paths long before that.

### 5.9 From radiance to pixels

The tracer computes **linear** radiance: twice the number means twice the light. Monitors
expect **sRGB**-encoded values, which are roughly `value^(1/2.2)`. Writing linear values
directly makes the image look much too dark and contrasty. `core/image.c` therefore:

1. clamps to `[0, 1]` (the lamp, with radiance 15, simply becomes white),
2. applies the exact sRGB curve: `12.92·c` below 0.0031308, else `1.055·c^(1/2.4) − 0.055`,
3. writes `round(255·c)`.

NaNs are dropped in `apps/render.c` (a single NaN sample would poison the pixel average), and
`image_srgb8` maps any remaining NaN to 0.

### 5.10 Random numbers

`core/rng.h` uses **PCG32** (O'Neill 2014): a 64-bit linear congruential step, whose output
is scrambled by a xorshift and a data-dependent rotation. It is tiny and fast, and far better than
`rand()`, which is also not thread-safe. Each pixel seeds its own generator from its index
(`seed` and `stream` both derived from the pixel number). That gives:

- **determinism**: the same command produces the same image;
- **thread independence**: no shared state, no locks, and the result does not depend on
  which thread rendered which row.

**The sampler** (`core/sampler.h`, `-sampler sobol`, the default): the numbers a path uses
most come from a low-discrepancy sequence instead, the same on the CPU and the GPU
(`gpu/trace_gpu.c` has the functions in WGSL, with the same results):

- **Dimension pairs:** pair 0 is the camera jitter; each of the first 4 bounces takes
  three pairs: the lamp point, the new direction (or the phase function's), and the
  Russian roulette plus the glass choice. Deeper bounces, and the variable number of
  numbers a medium takes while tracking, come from the PCG32 generator.
- **The sequence:** each pair is the 2D Sobol sequence, a (0,2)-sequence (every 2^k
  consecutive points fall one per cell of a 2^(k/2) × 2^(k/2) grid). It is Owen-scrambled
  with Burley's hash (2020), and the order of its points is shuffled the same way. That is
  random, yet still stratified, and every pair is scrambled differently.
- **Blue noise:** the pixels of a 64×64 tile share the sequence, each shifted by its value
  in a blue-noise mask (a Cranley-Patterson rotation, Georgiev and Fajardo 2016).
  Neighbouring pixels thus get well separated samples, and the error is fine-grained.
- **The mask** is made at startup by void-and-cluster (Ulichney 1993): points are added
  one by one in the largest gap, measured by a wrapping Gaussian. It takes ~50 ms.
- **The sample index:** a pixel's sample number on the CPU. On the GPU, a counter that
  starts each image at a multiple of 2^16 and goes on counting while things move, so the
  samples a pixel keeps from frame to frame (WEB.md 2.5) never repeat.

Measured (512×512 against 4096 spp; RMSE, plain random → sampler): the box 13.9 → 12.2
at 4 spp, 4.8 → 3.5 at 64 (1.35×); the glass spheres and the cloud 1.04–1.06×. Those two
are dominated by caustic fireflies and by tracking in the cloud, which the sequence does
not cover. With direct light only, at 1 spp, the share of the error at low frequencies
(below 1/8 of the sampling rate) falls from 4–5% to 0.5–2%: the noise is blue. In the
viewer while moving: spheres 13.0 → 9.5, cloud 12.5 → 8.2 (denoised 8.0 → 7.8 and
6.2 → 5.5).

`-sampler random` brings back plain PCG32, drawing in exactly the old order:
`-sampler random -mis off -caustics pt` gives the images of before bit for bit (checked
against the previous version).

### 5.11 The octree

**The problem.** Without acceleration every ray, including every shadow ray, tests every
primitive. The torus has 2,304 triangles, so each ray does 2,310 intersection tests, even
though it only passes near a handful of them. The cost grows linearly with the model.

**The idea.** Put the whole scene in one box and split it into 8 equal sub-boxes
(**octants**), then split those again, and so on. Each final cell (a **leaf**) remembers
which primitives touch it. A ray then walks only through the cells it actually crosses,
and only tests the primitives stored there.

```
      root (the whole box)
      ├── octant 0 (x low,  y low,  z low)  -> split again ...
      ├── octant 1 (x high, y low,  z low)  -> leaf: triangles 12, 13, 87
      ├── ...
      └── octant 7 (x high, y high, z high) -> leaf: empty
```

**Building** (`octree_build`, `build`):

1. The root box is the union of all primitive boxes (`primitive_bounds`).
2. A node with more than `OCT_LEAF_SIZE = 16` primitives, and depth below the maximum
   depth, is split at its center into 8 children. Child `c` uses bit 0 of `c` for the x
   half, bit 1 for y and bit 2 for z.

   The maximum depth depends on the mesh (`octree_auto_depth`). A surface fills an octree
   roughly like a 2D sheet, so each extra level gives about 4× more useful leaves, not
   8×. To end with leaves of about 16 primitives the tree needs about
   `log₄(n / 16)` levels, plus 2 for headroom, clamped to `[6, 14]`. That gives 6 for
   the torus (2,310), 8 for the dense torus (36,870) and 10 for the dragon (871,420).
   `-octdepth` overrides it.
3. Each primitive is handed to **every** child whose box overlaps its bounding box. A
   triangle crossing a split plane is therefore referenced by several leaves. This is the
   price of an octree: on the torus each primitive appears in about 4.5 leaves.
4. If a split would give *every* child *all* the primitives (they overlap everything),
   splitting is pointless and the node becomes a leaf. A node limit
   (`OCT_MAX_NODES`) protects against pathological inputs.

Memory layout: all nodes sit in one array, and the 8 children of a node are consecutive,
so a node only stores `first_child` (−1 for a leaf). Leaves store a range
`[prim_start, prim_start + prim_count)` into one shared array of primitive indices. Indices
are used rather than pointers because the arrays grow (and move) with `realloc` during
the build.

**Ray-box test (slab method).** Along each axis, the ray is between the box's two planes
for `t` in `[(lo − o)/d, (hi − o)/d]` (swapped if `d < 0`). The ray is inside the box where
all three intervals overlap:

$$t_{enter} = \max(t_{x0}, t_{y0}, t_{z0}, t_{min}), \qquad t_{exit} = \min(t_{x1}, t_{y1}, t_{z1}, t_{max})$$

and it hits the box if `t_enter ≤ t_exit`. `1/d` is computed once per ray. When a direction
component is 0, `1/d = ±∞` is legal IEEE arithmetic, and `fmin`/`fmax` quietly skip the NaN
that `0 · ∞` can produce.

**Traversal** (`visit`):

1. At an inner node, test the 8 children's boxes, drop empty leaves and missed boxes, and
   sort the rest by `t_enter`, **nearest first** (insertion sort of at most 8 entries).
2. Visit them in that order. At a leaf, intersect its primitives, shrinking `best`
   (the closest hit so far) as hits are found.
3. **Early exit:** if the next child starts beyond `best` (`t_enter > best`), nothing in
   it, or in any later child, can be closer, so stop. For most rays this means only
   the first few cells along the ray are ever looked at.

The early exit stays correct even though a primitive can live in several cells, and even
if a hit lies outside the cell that found it. Every hit found is a real hit, the code keeps
the minimum, and a cell is only skipped when it *starts* beyond the best hit. That is why
the octree images are bit-for-bit identical to brute force in the double build
(`make DOUBLE=1 bench` checks it; for float see section 3, "Precision").

**Shadow rays** use the same traversal with `any_hit = 1`: the first hit anywhere before
the light is enough to say "in shadow", so the walk stops immediately.

**Tuning.** I measured several settings on the dense torus (36,870 primitives) and the
torus (2,310), at 200x200 with 32 spp:

| max depth | leaf size | nodes | refs / primitive | dense torus | torus |
|---|---|---|---|---|---|
| 10 | 8 | 117,745 | 9.80 | 2.5 s | 2.1 s |
| 8 | 8 | 109,233 | 9.22 | 2.6 s | 2.6 s |
| 8 | 16 | 34,393 | 4.46 | 2.7 s | 2.1 s |
| **7** | **16** | **30,937** | **4.15** | **2.4 s** | **2.0 s** |
| 6 | 16 | 9,553 | 2.33 | 2.3 s | 2.0 s |
| 8 | 32 | 11,337 | 2.66 | 2.4 s | 2.0 s |

Render time hardly changes, because past a certain depth the cost is dominated by
shading and the walls, not by finding triangles. Memory changes a lot: deep trees with tiny
leaves store each triangle about 10 times. Leaves of 16 keep memory low. A fixed depth of 7
was fine up to the dense torus, but too shallow for the dragon (with cells of 4 units, each
leaf would hold hundreds of its 0.3-unit triangles). That is why the depth now scales
with the mesh, as described in the build steps. Try other values with `-octdepth 9`, or
other leaf sizes with `make clean && make CFLAGS="-O2 -DOCT_LEAF_SIZE=8"`.

**Scaling to the dragon.** With 871,420 primitives (the dragon plus the 6 room quads) the
tree has 860,729 nodes and 753,138 leaves, reaches depth 10, and stores each triangle 4.85
times on average (903,809 nodes and 5.15 references with `-rot -35`, since the turned model
lines up differently with the cells). It is built in under 1 s, after 0.5–0.7 s spent
reading the 34 MB PLY file. The renderer then traces 0.41–0.45 M
paths/s, not far from the 2,310-primitive torus (0.62 M paths/s). The cost of a ray grows
roughly with the depth of the tree, i.e. logarithmically, instead of with the number of
triangles. Brute force would need about 870,000 tests per ray: extrapolating from the
torus benchmark, the 600x600, 512 spp dragon image would take about a month. Peak memory is about 360 MB,
mostly the `Primitive` structs (~256 bytes each at the time, one struct for all three shape
types; mesh triangles now have their own 52-byte `Triangle`, section 3, and the peak is 129 MB).

**When it does not help.** The plain Cornell box has 18 primitives, and 5 of them are huge
walls that touch every cell along the sides of the box. Walking the tree there is pure
overhead: 12 s instead of 7 s. `-accel auto` therefore uses no tree at all for such small
scenes, and the BVH (next section) from 64 primitives up.

**Known simplifications** (each a possible next step):
- Primitives are assigned by *bounding box* overlap. An exact triangle-box test
  (separating axis theorem) would put long thin triangles in fewer cells.
- **Mailboxing** (done): a triangle stored in several cells could be tested several times
  by the same ray. Each traversal keeps a 64-slot direct-mapped table of the ids it tested
  and skips a repeat. Skipping is safe: the earlier test used a farther or equal `tmax`, so
  a repeat can only find the same hit again. Same images; 7–10% faster on the dragon and
  the dense torus (`make CFLAGS="-O2 -DOCT_NO_MAILBOX"` turns it off for comparison). The
  gain is small because a ray visits only a few leaves and the box tests dominate.
- An octree always splits at the *center*, wherever the geometry is. A BVH or a k-d tree
  built with the surface area heuristic (SAH) adapts to the geometry, and is what
  production ray tracers use. That is the next section.

### 5.12 The BVH (bounding volume hierarchy)

**The idea.** The octree cuts *space* into fixed cubes, so a triangle that crosses a cut
ends up in several cells. A BVH cuts the *list of primitives* instead. Split the
primitives into two groups, put a box around each group, and repeat inside each group
until the groups are tiny. Every primitive is in **exactly one** leaf. The two boxes of a
node may overlap, but they fit tightly around their own geometry, wherever it is.

```
          [box around everything]
           /                   \
  [box: dragon head]      [box: body + floor]
     /        \               /        \
  [horns]  [jaw]      [tail, 3 tris] [floor quad]
```

**Building** (`bvh_build`, `build`):

1. For every primitive, compute its bounding box and that box's **centroid** (center).
   Grouping is decided by centroids: a centroid is a single point, so it always lies on
   exactly one side of a plane.
2. A node computes the box of its primitives and the box of their centroids. It then
   splits along one axis with a plane, chosen by one of two strategies:
   - **`-bvhsplit middle`**: the longest axis of the centroid box, cut at its middle.
     Simple and fast to build.
   - **`-bvhsplit sah`** (default): the **surface area heuristic**. For a random ray that
     hits a box, the probability of also hitting a smaller box inside it is about the
     ratio of their surface areas. So the expected cost of a split is

$$C = C_{trav} + \frac{A_L}{A}\,N_L\,C_{isect} + \frac{A_R}{A}\,N_R\,C_{isect}$$

     (`A` = surface area, `N` = primitive count, with `C_trav = C_isect = 1`). Trying every
     possible plane is too slow, so the centroids are dropped into `-bvhbins` (16)
     equal **buckets** per axis. Two sweeps, one from the left and one from the right,
     give the area and count on each side of each of the 15 bucket boundaries.
     `sah_split` picks the cheapest of the 3 × 15 candidates (**binned SAH**).
3. **Leaf or split?** A leaf costs `N` intersection tests. SAH keeps the node as a leaf
   when the best split is not cheaper, unless the leaf would exceed `-bvhleaf` (4)
   primitives. The table below shows that the leaves average 1.8 primitives: SAH usually
   decides to split before the limit is reached.
4. The primitive indices are **partitioned in place** (like quicksort's partition):
   left group first, right group after. So every node, and in the end every leaf, owns a
   contiguous range `refs[start .. start + count)`, and no index is ever duplicated.
5. Fallback: if all centroids coincide (or every one lands on one side) and the node is
   too big for a leaf, the range is simply cut in half. That is valid, just not optimal.

**Traversal** (`bvh_intersect`) uses an explicit stack instead of recursion:

1. Pop a node. If its box is entered beyond the best hit so far, skip it.
2. Leaf: intersect its primitives, shrinking `best`.
3. Inner node: test **both** children's boxes, and push the farther one first so the
   **nearer one is popped next**. Hits found near the front shrink `best` early, and then
   whole far subtrees are skipped by step 1. The entry distance is stored on the stack,
   so step 1 costs nothing.
4. Shadow rays (`any_hit = 1`) return at the first hit.

**4-wide** (`-bvhwidth 4`, the default; `bvh_make_wide`). After the build, the binary tree
is collapsed into nodes of 4 children: a node takes its two children, then again and
again replaces the inner child with the largest box by that child's two children. The 4
child boxes are stored side by side (`BvhWideNode`: lo x of the 4, lo y of the 4, ...),
so one step of a ray tests 4 boxes in a loop the compiler turns into SIMD, and the tree
is half as deep. The hit children go on the stack farthest first, leaves included, so the
nearest is always next. The dragon (table below, 320x320, 8 threads): 1.87 → 3.38 M paths
per second, the same image. The GPU keeps the binary tree: in its kernel the 4-wide one
measured 5-10% slower (docs/PERFORMANCE.md).

**Octree vs. BVH on the dragon** (871,420 primitives, 300x300, 32 spp):

| | octree | BVH (SAH) | BVH (middle) |
|---|---|---|---|
| nodes / leaves | 903,809 / 790,833 | 962,993 / 481,497 | 591,957 / 295,979 |
| references per primitive | 5.15 (duplicates) | 1 (no duplicates) | 1 |
| depth | 10 | 26 | 31 |
| build time | **0.6 s** | 3.7 s | 0.8 s |
| render time | 4.9 s | **2.7 s** | 2.9 s |

The octree builds fastest, but stores each triangle about 5 times and visits many cells
that contain only slivers of triangles. The SAH BVH builds slowest, but each ray tests far
fewer boxes and triangles. On the dragon, the simple middle split is almost as good,
because the scan's triangles are very uniform. SAH pays off more on uneven scenes, like a
small detailed model next to huge walls. The BVH is the default for big scenes because
render time dominates. In the viewer you can switch between all three with key **B** and
watch the pass time.

**BVH tuning** (same dragon render):

| setting | build | render |
|---|---|---|
| default (`-bvhleaf 4 -bvhbins 16`) | 3.7 s | 2.7 s |
| `-bvhleaf 1` | 3.9 s | 3.0 s |
| `-bvhleaf 2` | 3.9 s | 2.9 s |
| `-bvhleaf 8` | 3.7 s | 2.6 s |
| `-bvhbins 8` | 2.8 s | 2.4 s |
| `-bvhbins 32` | 5.6 s | 2.5 s |

More buckets make the build slower and the tree barely better. 8 buckets were even a
little faster here, which is within the run-to-run noise (about ±0.2 s). Forcing tiny
leaves (`-bvhleaf 1`) adds nodes without helping.

**Parallel build.** The two children of a node own disjoint ranges of the primitive list,
so they can be built at the same time. `bvh_build` splits the top of the tree (nodes with
more than 32,768 primitives) with OpenMP tasks, builds each smaller subtree serially into
its own node array, and a final pass copies everything in preorder into one array. The
split decisions are the same functions as the serial build (`choose_split`), and the
preorder copy reproduces the serial layout exactly, so the tree, its statistics and the
images are the same for any number of threads. The bounds of all primitives are also
computed in parallel. Dragon, 4 cores / 8 threads: **3.5 s → 1.1 s**.

### 5.13 The real-time viewer (`view`)

`view` is a window you can fly through while the **same** engine renders. It uses the
same scene, acceleration structure and `trace_path()` as `render`. What it adds is a
camera you can steer, **progressive rendering**, and a second backend for the moments
when the path tracer is too slow to follow you: the rasterizer.

**Three modes** (`-backend`, key **M** cycles them):

| mode | while the camera moves | when it stops |
|---|---|---|
| `hybrid` (default) | the rasterizer draws every frame (local lighting, no shadows) | after 0.12 s the path tracer takes over and paints over the raster image |
| `rt` | the path tracer, 1 sample per pixel at 1/`preview` resolution | path tracer, full resolution |
| `raster` | the rasterizer | the rasterizer |

Hybrid mode is what game engines' editors do: a fast approximation while you navigate, the
real thing once you look. Both backends read the same `Scene`; nothing is converted or
copied when the mode changes.

**Window.** `platform/window.c` is the only file that knows about X11. `XCreateSimpleWindow`
opens a fixed-size window, an `XImage` is wrapped around a pixel array, and `XPutImage`
copies it to the screen: that is the whole "graphics API". `window_present` takes the
engine's `Framebuffer`, so the rasterizer draws straight into what is shown, and the path
tracer's average is converted into the same framebuffer. On a Wayland desktop (GNOME) this
runs through Xwayland, already part of the system. `XkbSetDetectableAutoRepeat` stops held
keys from sending fake "released" events, so holding **W** means "keep moving".

**Camera.** The view is a position plus two angles, *yaw* (left/right) and *pitch*
(up/down), turned into a direction:

```
forward = (cos(pitch)·sin(yaw), sin(pitch), cos(pitch)·cos(yaw))
right   = normalize(forward × up)
```

`W`/`S` move along `forward`, `A`/`D` along `right`, `E`/`Q` along y, all scaled by
`speed × dt` (the time since the last frame), so the speed does not depend on the frame rate.
Mouse drags and arrow keys change the angles (pitch stops just short of ±90° so the view
never flips). Every frame builds a fresh `Camera` with `camera_make` (path tracer) and the
matching matrices with `render_params_camera` (rasterizer).

**Progressive rendering.** Instead of an image, the viewer keeps a running **sum** and a
sample **count** per pixel. The window shows `sum / count`, converted to sRGB through a
4096-entry lookup table (`image_pack_srgb`; calling `pow` for 400,000 pixels 30 times a
second would be slow).
- **Still camera:** the image is refined one sample per pixel per *pass*. A pass is
  rendered in **bands** of rows. After each band the program checks the keyboard and
  mouse again, and the band height adapts so one band takes about 10–40 ms. The window
  therefore stays responsive even when a full pass takes a second (the dragon at
  640×640: about 250 ms).
- **Moving camera, `rt` mode:** every change throws the sums away. That frame is rendered
  at **1/`preview`** resolution (1/4 by default: 16× fewer pixels) with `-movespp` samples
  per pixel, and shown blown up. This is grainy but follows the mouse: about 16 ms per frame
  on the dragon. When the camera stops, rendering restarts at full resolution.
- **Moving camera, `hybrid` mode:** every change draws a raster frame instead (about
  10 ms for the box and 80 ms for the full 871k-triangle dragon at 640×640, with the
  rasterizer on all cores, see RASTERIZER.md).
- Pixels that have no sample yet keep their old color, so a new pass paints over the
  previous image (or the raster frame) from top to bottom instead of flashing black.
- After `-s` samples per pixel (4096) it stops and sleeps until something changes.

The random seed of every sample is computed from (pixel, sample number), so every sample is
different but the result is reproducible.

**Extras.**
- **C** prints the `render` command for the current view (`-cam`, `-look`, `-fov`
  plus your scene options), to render it offline in high quality.
- **P** saves what the window shows as a PNG: the path traced average at full precision,
  or the raster frame.
- **V** changes the raster shading (Phong, Gouraud, flat, wireframe).
- **B** rebuilds the scene with another acceleration structure.
- `[` / `]` change the number of bounces live: with 1–2 bounces you see direct light only.

**Tested** on this machine: a helper program sent synthetic key presses and mouse drags to
the window. Moving, looking around, zooming, **M**, **V**, **B**, **P**, **C**, `[`/`]` and
the preview keys worked in all three modes, and the sanitizer build reported no errors.

### 5.14 Caustics by light tracing (`-caustics lt`)

A caustic on the floor, lamp light focused by the glass sphere, is the path
`camera → floor → glass → glass → lamp`. The path tracer can only find it by chance: from
the floor it bounces at random and must happen to go through the glass and hit the small
lamp. Next event estimation cannot help, because the glass blocks the shadow ray. The
same holds for the light the mirror sphere throws on the walls, and for caustic light seen
after more bounces (`camera → wall → floor → glass → lamp`). These paths are the white
speckles of section 7.

Traced from the other end they are easy. `light_trace_caustics` starts paths at the lamp
(a uniform point, a cosine-distributed direction, power `β = Le · Area · π`), follows them
through mirrors and glass with exactly the path tracer's code, and at every diffuse surface
it reaches **connects to the camera**: a shadow ray to the eye, `camera_project` for the
pixel, and

$$\text{pixel} \mathrel{+}= \frac{\beta \cdot \frac{\text{albedo}}{\pi} \cdot \cos\theta_x}{d^2 \cdot A_{pixel} \cdot \cos^3\theta_c} \cdot \frac{1}{N}$$

`cos θ_x` is between the surface's geometric normal and the direction to the camera,
`d` the distance to the camera, `A_pixel` the area of one pixel on the virtual screen and
`cos³ θ_c / d²` turns area on the surface into area on that screen (`θ_c` is the angle to
the view axis); `N` is the number of light paths. On smooth meshes the light arriving is
also weighted by `|cos_shading| / |cos_geometric|` (the "adjoint BSDF" correction) so that
it matches what the path tracer's BRDF computes. The light path then keeps bouncing
(diffuse, mirror, glass, with Russian roulette), and every further diffuse vertex is
connected too. A medium collision ends it.

**Every path counted exactly once.** Light tracing covers precisely the paths
`lamp → S → (anything) → D → camera`: the first vertex after the lamp is a mirror or glass
(S), the last one before the camera diffuse (D), no medium vertex. With `skip_caustics` the
path tracer leaves out precisely the same set from its side: it saw a diffuse surface first
and reaches the lamp right after a mirror or glass, without a medium vertex. Everything
else, including everything the camera sees *through* glass or in a mirror, stays with the
path tracer. The two estimates are simply added.

**Same image for any number of threads.** Light paths land in random pixels, so threads
would write the same pixels. The pixels are summed as 32.32 fixed-point integers with
atomic adds; integer addition does not depend on the order, so the result is identical
for 1 or 7 threads (checked).

**Measured** (spheres scene, 200 × 200, 256 samples, error against a 4096-sample path
traced reference; the image means agree to 4 digits in all cases):

| estimator | RMSE | time |
|---|---|---|
| path tracing, no MIS (the old one) | 0.0189 | 1.8 s |
| + MIS | 0.0187 | 1.9 s |
| + light tracing (first diffuse vertex only) | 0.0160 | |
| + light tracing (every diffuse vertex, the final version) | **0.0104** | 2.3 s |

45% less error is what about 3.3× more samples would give, for 30% more time (400 × 400 at
128 spp: 3.3 s → 4.3 s). The wall speckles disappear; the remaining grain is inside the
glass sphere (paths the camera sees through glass, a job for full bidirectional path
tracing). The glass dragon: −22% RMSE. `-photons x` changes the number of light paths
(default: one per camera path). The viewer path traces caustics (light tracing would need
its own progressive buffer); `-caustics pt -mis off -sampler random` reproduces the old images bit for bit.
The GPU has the same light tracer (and, for real time, a photon cache): WEB.md 2.6.

### 5.15 Physically based materials (PBR1, `core/brdf.h`)

Every material is now one model, glTF 2.0's **metallic-roughness** with transmission
(docs/PBR.md): a base colour, `metallic` (0 a dielectric like plastic or stone, 1 a metal),
`roughness` (0 polished, 1 matte), an index of refraction and `transmission` (how much of
the light a dielectric does not reflect goes through it, refracted). The old materials are
presets of it: diffuse = a rough dielectric (roughness 1), mirror = a smooth metal of the
mirror's colour, glass = a smooth clear dielectric. `-metal`, `-rough`, `-ior`,
`-transmission` and `-base` change the model's material (USAGE.md).

**The BSDF** has four lobes:

- **Specular** reflection off microfacets: tiny mirrors whose normals `h` follow the GGX
  distribution `D(h)` (width `α = roughness²`), partly hidden from the viewer and from the
  light by each other (Smith's masking-shadowing `G2`, the height-correlated form), each
  reflecting a Fresnel share `F(v·h)` (Schlick): `F0 = ((η − 1)/(η + 1))² = 0.04` for a
  dielectric, the base colour for a metal:
  $$f_s = rac{F\,D\,G_2}{4\,|n·v|\,|n·l|}$$
- **Diffuse**: Lambert with the base colour, for the light the specular layer lets in and
  out again: `(1 − Es(v)) (1 − Es(l)) / (1 − Es_avg)`, where `Es` is how much the specular
  lobe reflects at that angle (tabulated, `core/brdf_tables.h`). glTF's reference writes
  `1 − F(v·h)`, which at grazing angles returns up to 30% more light than arrives; this
  form keeps exactly what arrives and is reciprocal.
- **Transmission**: refraction through the microfacets (Walter et al. 2007), tinted by the
  base colour. Roughness 0 is the old glass.
- **Multiple scattering** (metals): light that bounces between microfacets before leaving is
  missing from `f_s` (a white metal of roughness 1 keeps only 31% of the light seen straight
  on). Kulla & Conty's lobe gives it back: a white rough metal keeps all of it.

Roughness up to 0.02 is treated as perfectly smooth: the specular and transmission lobes
are then a mirror reflection and a refraction (delta lobes, as 5.6).

**Sampling.** A lobe is picked by about the light it reflects from this view. For the
microfacet lobes, a normal `h` is drawn from the **visible normals** (those facing the
viewer, weighted by how much of them it sees: Heitz 2018, in the spherical-cap form of
Dupuy & Benyoub 2023), and `l` is `v` reflected or refracted about it; the diffuse and
multiple-scattering lobes take a cosine-weighted direction. The density of the direction is
that of all the lobes together, so the weight `f cos / pdf` stays small whichever lobe
drew it (one-sample MIS between the lobes). Next event estimation evaluates the rough part
of the BSDF at the lamp's direction and weighs itself against that density (5.5.1).

**The partition for light tracing (5.14)** follows the lobes: a vertex where a delta lobe
was taken is an S vertex, any other a D vertex. Light paths start with an S, connect the
rough part of the BSDF to the camera, and keep bouncing by sampling the BSDF from the
light's side (through glass weighted by `(η_from/η_to)²`, the adjoint of the radiance's).

**Checked** by `bin/test_brdf` (make test): reciprocity (reflection `f(v,l) = f(l,v)`,
refraction `f(v,l)/η_l² = f(l,v)/η_v²`); sampling against evaluation (the mean weight of
the samples equals the integral of `f |cos|` by quadrature, and the density integrates to
the share of samples it gives, for eight materials at three angles); the white furnace
(white materials keep 99.9-100% of the light at every angle, rough glass 92-99.9%: its
light bouncing between microfacets is not given back). Images with light tracing and
without agree to 0.1 of 255. The GPU path tracer has the same BSDF in WGSL
(`gpu/brdf_gpu.c`): `bin/test_gpu_brdf` runs both on 65536 cases and finds the same
values, and GPU and CPU images agree within 0.25% (PBR.md, PBR2).

---

## 6. The scene

Dimensions and colors follow Peter Shirley's *Ray Tracing: The Next Week*, which is based
on the published Cornell measurements:

| element | geometry | material |
|---|---|---|
| left wall | quad at x = 555 | diffuse red (0.65, 0.05, 0.05) |
| right wall | quad at x = 0 | diffuse green (0.12, 0.45, 0.15) |
| floor, ceiling, back wall | quads | diffuse white (0.73, 0.73, 0.73) |
| lamp | 130×105 quad at y = 554, facing down | emission (15, 15, 15) |
| tall box | 165×330×165, rotated 15°, at (265, 0, 295) | white |
| short box | 165×165×165, rotated −18°, at (130, 0, 65) | white |
| mirror sphere (spheres scene) | r = 100 at (370, 100, 370) | mirror, albedo 0.95 |
| glass sphere (spheres scene) | r = 100 at (185, 100, 170) | glass, ior 1.5 |

The lamp sits 1 unit below the ceiling, so the ceiling quad does not cover it, and the
emission of 15 makes the room about as bright as the reference images. The front of the box
is open; rays that leave through it see black. That is the black area reflected in the
mirror sphere.

**Rotated boxes.** Instead of the usual "instance transform" (rotating every *ray* into
the object's frame), `add_box` rotates the 6 face quads themselves once, at build time:

```
rotate_y(p) = ( cos·x + sin·z,  y,  −sin·x + cos·z )
face corner -> rotate -> translate;   face edges u, v -> rotate only
```

This is simpler, and with a handful of quads there is no cost.

**One light.** Next event estimation only knows `scene.light`, the single quad lamp.
Supporting several lights would mean picking one at random per sample and dividing by the
probability of that choice.

**Mesh scene.** `-scene mesh` keeps the 5 walls and the lamp and puts the `.obj` model in
the middle of the floor, scaled to 330 units across, in warm orange (0.80, 0.55, 0.25), or
in mirror or glass with `-mat`. Any OBJ works; try
`bin/render -scene mesh -obj models/torus_dense.obj -mat glass`.

**The Stanford dragon.** It is the classic test model: scanned in 1996 at the Stanford
Computer Graphics Laboratory with a Cyberware 3030 MS scanner, then reconstructed into one
surface with the *vrip* program. `make dragon-download` fetches `dragon_recon.tar.gz` from
the Stanford 3D Scanning Repository, which contains the full model
(`dragon_vrip.ply`: 437,645 vertices, 871,414 triangles) and three decimated versions
(`_res2` 202k, `_res3` 48k, `_res4` 11k triangles), all ASCII PLY. The file has y pointing
up like our world, so it can be used as is. `-rot -35` turns it to a three-quarter view.
`make dragon` renders it in orange and in glass:

```sh
bin/render -scene mesh -obj models/dragon_recon/dragon_vrip.ply -rot -35 -n 600 -s 512
```

The repository allows the models to be used for research and non-commercial purposes, with
credit to the Stanford Computer Graphics Laboratory. Because the dragon is a symbol of
Chinese culture, it also asks people not to animate or morph it, apply Boolean operations
to it, or simulate it breaking, exploding or melting.

---

## 7. Noise and fireflies

The boxes scene converges smoothly: nearly all light reaches a surface either directly
(sampled explicitly) or after diffuse bounces.

The spheres scene is harder. Paths like *wall → mirror → lamp* or *floor → glass → lamp*
can only find the light by chance, because the mirror and glass fix the direction, so the
lamp cannot be aimed at through them. When such a path *does* hit the small, bright lamp, it
carries a huge value: a single white speckle, a **firefly**. These are real light
(caustics and reflected highlights), just very badly sampled. Your options are:

| remedy | effect |
|---|---|
| more samples | correct but slow: noise ∝ 1/√N |
| `-clamp 4` | cap each sample's brightest channel at 4. Speckles vanish instantly, but some real energy is thrown away (the caustic gets dimmer): **biased** |
| multiple importance sampling (on by default, `-mis`) | combines light sampling and BRDF / phase sampling (section 5.5.1): fewer fireflies near the lamp and in fog |
| light tracing (on by default in `render`, `-caustics lt`) | traces the caustic paths from the lamp (section 5.14): the floor caustic and all caustic light seen on diffuse surfaces become smooth, unbiased |
| full bidirectional path tracing / photon mapping | would also cover the paths the camera sees *through* glass or in a mirror (`camera → glass → floor → glass → lamp`), the remaining grain inside the glass sphere |

`make final` renders the spheres scene both ways, so the trade-off can be compared directly:
`out/cornell_spheres_final.png` (unbiased, slight grain) and
`out/cornell_spheres_final_clamped.png` (clean, dimmer caustic).

---

## 8. Limitations and next steps

Roughly in order of effort:

1. **Faster BVH traversal.** A more compact node layout (32 bytes, children stored next
   to each other), or 4-wide nodes tested with SIMD. The octree could get an exact
   triangle-box overlap test.
2. **Bigger models.** Triangles are now 52 bytes (float); the 7.2-million-triangle *XYZ
   RGB Asian dragon* would need about 0.4 GB for them plus the trees. Sharing vertex
   positions too (3 indices per triangle instead of 3 corners) would halve that again, at
   the cost of an indirection per intersection test; a binary PLY reader is missing.
   Texture coordinates (`vt`) are read past but not used yet.
3. **Glossy materials** (e.g. GGX microfacets) for brushed metal or plastic; MIS (section
   5.5.1) is already there to keep them noise-free.
3b. **Full bidirectional path tracing** (all combinations of camera and light subpaths,
   weighted by MIS) for the caustics seen through glass and in mirrors.
4. **Tone mapping** (Reinhard, ACES) instead of hard clamping, so bright areas roll off
   gently.
5. **Depth of field**: start camera rays from a random point on a lens disk instead of
   one pinhole.
6. **A smarter viewer**: shadow maps for the raster preview, a level of detail (a
   decimated mesh) so the hybrid preview stays fluid on the full dragon, reprojecting the
   old path traced image when the camera moves (temporal reuse).
7. **Denoising**: even a simple edge-aware blur guided by normals and albedo removes most
   of the remaining grain.

---

## 9. References

- Peter Shirley et al., *Ray Tracing in One Weekend* and *Ray Tracing: The Next Week*
  (free online): camera, spheres, quads, the Cornell box setup.
- Matt Pharr, Wenzel Jakob, Greg Humphreys, *Physically Based Rendering* (pbr-book.org,
  free online): rendering equation, Monte Carlo, importance sampling, MIS, Russian roulette.
- James Kajiya, *The Rendering Equation*, SIGGRAPH 1986.
- Christophe Schlick, *An Inexpensive BRDF Model for Physically-based Rendering*, 1994.
- Melissa O'Neill, *PCG: A Family of Simple Fast Space-Efficient Statistically Good
  Algorithms for Random Number Generation*, 2014.
- Tom Duff et al., *Building an Orthonormal Basis, Revisited*, JCGT 2017.
- The Cornell box: Cornell University Program of Computer Graphics,
  graphics.cornell.edu/online/box.
- The Stanford dragon: Stanford University Computer Graphics Laboratory, *The Stanford 3D
  Scanning Repository*, graphics.stanford.edu/data/3Dscanrep. Brian Curless and Marc Levoy,
  *A Volumetric Method for Building Complex Models from Range Images*, SIGGRAPH 1996 (vrip).
- Tomas Möller and Ben Trumbore, *Fast, Minimum Storage Ray/Triangle Intersection*,
  Journal of Graphics Tools, 1997.
- J. David MacDonald and Kellogg Booth, *Heuristics for Ray Tracing Using Space
  Subdivision*, The Visual Computer, 1990 (the surface area heuristic).
- Ingo Wald, *On fast Construction of SAH-based Bounding Volume Hierarchies*, IEEE
  Symposium on Interactive Ray Tracing, 2007 (binned SAH).
- PBRT, chapter "Primitives and Intersection Acceleration": BVH construction and traversal.
