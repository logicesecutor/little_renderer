# little_renderer: how to use everything

This is the reference for every program, `make` target, command-line option, key and
tunable constant in this folder. How the code works is explained elsewhere: `README.md`
covers the rasterizer and `raytracer/RAYTRACER.md` covers the ray tracer.

| program | folder | what it is |
|---|---|---|
| `raytracer` | `raytracer/` | offline path tracer: renders the Cornell box into a `.ppm` image |
| `rtview` | `raytracer/` | the same path tracer in a window you can fly through in real time |
| `main` | `.` | the rasterizer: one image per milestone into `out/*.ppm` |
| `viewer` | `.` | the rasterizer in a spinning real-time window |
| `tools/gen_models.py` | `tools/` | writes the test models (cube, sphere, torus, dense torus) |

Everything is written from scratch in C. The only libraries are `libm`, `libX11` (only for
the two windows) and, optionally, OpenMP, which spreads the work over the CPU cores.

---

## 1. Quick start

```sh
cd ~/Projects/little_renderer
make models                      # test models into models/ (once)

cd raytracer
make                             # builds raytracer and rtview
./rtview                         # fly around the Cornell box (window)
./raytracer                      # render it to out/cornell_boxes.ppm (~10 s)

make dragon-download             # Stanford dragon, 11 MB (once)
./rtview -scene mesh -obj ../models/dragon_recon/dragon_vrip.ply -rot -35
```

Every program prints its options with `-h`. PPM images open in GIMP, Eye of GNOME
(`eog file.ppm`) and most image viewers.

---

## 2. Building

### Ray tracer (`raytracer/Makefile`)

| command | what it does |
|---|---|
| `make` or `make all` | builds `raytracer` and `rtview` |
| `make raytracer` / `make rtview` | builds just one |
| `make run` | quick previews of the 3 scenes into `out/` |
| `make view` | opens `rtview`; pass options with `make view ARGS="-scene spheres"` |
| `make final` | slow, clean 600×600 / 1024 spp renders of every scene |
| `make bench` | none vs octree vs BVH on the torus; checks the 3 images are identical |
| `make bench-dragon` | octree vs BVH on the Stanford dragon; checks the images are identical |
| `make dragon-download` | downloads the Stanford dragon into `../models/dragon_recon/` |
| `make dragon` | renders the dragon (orange and glass) into `out/` |
| `make view-dragon` | opens `rtview` on the dragon (`ARGS=...` works here too) |
| `make clean` | deletes the binaries |

Build flags, combinable with any target:

| flag | effect |
|---|---|
| `make SINGLE=1` | no OpenMP: everything on one thread (easier to debug) |
| `make SAN=1` | address + undefined-behaviour sanitizers: catches bad memory access; ~10× slower |
| `make CFLAGS="-O2 -fopenmp -DOCT_LEAF_SIZE=8"` | your own flags, e.g. to change a compile-time constant (run `make clean` first) |

### Rasterizer (`Makefile` in the top folder)

| command | what it does |
|---|---|
| `make` / `make main` | builds the headless demo `main` |
| `make viewer` | builds the X11 spinning viewer |
| `make run` | builds and runs `main` |
| `make models` | runs `tools/gen_models.py` |
| `make SAN=1 main` | with sanitizers |
| `make clean` | deletes `main` and `viewer` |

---

## 3. `raytracer` and `rtview` options

Both programs accept the same options; the defaults differ where noted. Values are given
as `-option value`.

### Image and quality

| option | meaning | default | tips |
|---|---|---|---|
| `-n size` | width = height in pixels | 400 (rtview: 640) | time grows with the pixel count |
| `-width px`, `-height px` | set width and height separately | | the camera adapts to the aspect ratio |
| `-s samples` | raytracer: light paths per pixel. rtview: stop refining after this many | 128 (rtview: 4096) | noise falls like 1/√s: 4× the samples = half the grain |
| `-d bounces` | maximum bounces per path | 50 | 3–5 is fast but darker (less indirect light); Russian roulette ends most paths early anyway |
| `-clamp max` | caps the brightness of a single sample | 0 (off) | `-clamp 4` removes fireflies (glass, mirrors) but darkens caustics: biased |
| `-threads n` | CPU threads | 0 = all cores | same as the `OMP_NUM_THREADS` environment variable |
| `-o file` | raytracer: output PPM. rtview: prefix for screenshots (`<prefix>_001.ppm`, ...) | `out/cornell_<scene>.ppm` (rtview: `out/view`) | the folder must exist (`make` targets create `out/`) |

### Scene

| option | meaning | default |
|---|---|---|
| `-scene boxes` | the classic Cornell box: a tall box and a short box | ✓ |
| `-scene spheres` | a mirror sphere and a glass sphere | |
| `-scene mesh` | a model from a file, standing in the middle of the box | |
| `-obj file` | the model for `-scene mesh`: `.obj`, or ASCII `.ply` (binary PLY is not supported) | `../models/torus.obj` |
| `-mat diffuse\|mirror\|glass` | material of the model | diffuse (warm orange) |
| `-rot degrees` | turns the model around the vertical axis | 0 |

The model is always scaled so its largest side is 330 units (the box is 555), centered,
and stood on the floor. Files without normals get smooth normals computed automatically.

### Camera

| option | meaning | default |
|---|---|---|
| `-cam x,y,z` | camera position | `278,278,-800` (in front of the open side) |
| `-look x,y,z` | the point the camera looks at | `278,278,0` |
| `-fov degrees` | vertical field of view | 40 |

Coordinates: the box spans 0..555 on every axis, y is up, the open side faces −z, and
x = 555 (the red wall) appears on the **left**. The lamp is under the ceiling at
y = 554. The easiest way to get numbers is to fly there in `rtview` and press **C**.

### Acceleration structure

| option | meaning | default |
|---|---|---|
| `-accel auto` | BVH from 64 primitives up, otherwise none | ✓ |
| `-accel none` | test every primitive for every ray (fine for the 18-quad box) | |
| `-accel octree` | the octree (`octree.c`) | |
| `-accel bvh` | the BVH (`bvh.c`) | |
| `-octdepth n` | octree max depth; 0 = automatic (log₄(n/16) + 2, clamped to 6..14) | 0 |
| `-bvhleaf n` | BVH: max primitives per leaf (SAH may stop earlier) | 4 |
| `-bvhbins n` | BVH: buckets per axis when evaluating SAH splits (2..256) | 16 |
| `-bvhsplit sah\|middle` | BVH: surface area heuristic, or cut the longest axis in the middle | sah |

All three structures give exactly the same image; only the speed changes
(`make bench`, `make bench-dragon`). The program prints the build statistics (nodes,
leaves, depth, build time) at start-up.

### rtview only

| option | meaning | default |
|---|---|---|
| `-preview n` | while moving, render at 1/n resolution (1 = always full resolution) | 4 |
| `-movespp n` | samples per pixel of each low-resolution frame while moving | 1 |
| `-speed units` | movement speed in scene units per second | 300 |
| `-autoquit seconds` | quit after this time and save a screenshot (scripts, benchmarks) | 0 = never |

---

## 4. `rtview` controls

| key | action |
|---|---|
| `W` `A` `S` `D` | move forward / left / back / right |
| `E` / `Q` | move up / down |
| left mouse drag | look around |
| arrow keys | look around |
| mouse wheel | zoom (field of view ±2°) |
| `Shift` (hold) | move 4× faster |
| `+` / `-` | movement speed ×1.5 / ÷1.5 |
| `[` / `]` | one bounce less / more (restarts refining) |
| `B` | cycle the acceleration structure: none → octree → BVH (rebuilds it, prints stats) |
| `F` | low-resolution preview while moving on / off |
| `1` `2` `3` `4` | preview resolution 1/1, 1/2, 1/4, 1/8 |
| `R` | back to the starting camera |
| `P` | save the current image as PPM (at the resolution and sample count shown) |
| `C` | print a `raytracer` command that renders exactly this view |
| `H` | print the controls in the terminal |
| `Esc` | quit |

The title bar shows: acceleration structure, samples per pixel so far (/ the `-s` limit),
time per full pass, current resolution (1/1 when still), bounces and speed.

**How it works.** While you move, every frame is one sample per pixel at 1/4 resolution,
so it follows the mouse but looks grainy. When you stop, it restarts at full resolution
and keeps adding one sample per pixel per pass, painting top to bottom, so the image
gets cleaner the longer you wait. It stops at `-s` samples per pixel and then sits idle.

**Display.** It uses X11. On a GNOME/Wayland desktop it runs through Xwayland, which is
already there. Start it from a terminal in your desktop session. Over SSH or from a
script, set `DISPLAY=:0`.

---

## 5. Recipes

**Quick look vs final quality**
```sh
./raytracer -n 200 -s 16                    # < 1 s, very grainy, enough to check a camera
./raytracer -n 400 -s 128                   # ~7 s, the default
./raytracer -n 800 -s 1024 -o out/big.ppm   # ~4 min, clean
```
(times for the boxes scene on 8 cores; glass, mirrors and big meshes are slower)
```sh
# the loop to learn what a parameter does: change ONE thing, compare the images
./raytracer -s 64 -d 2  -o out/d2.ppm       # direct light + one bounce: dark, hard
./raytracer -s 64 -d 50 -o out/d50.ppm      # full global illumination
```

**Find a view in the window, render it offline**
1. `./rtview` (add your `-scene ...` options), fly to a nice spot, press **C**.
2. Copy the printed line, e.g.
   `./raytracer -scene boxes -cam 278.0,278.0,-497.0 -look 278.0,278.0,3.0 -fov 40 -n 640 -s 512 -o out/from_viewer.ppm`,
   and change `-n` / `-s` as you like.

**Your own model**
```sh
./rtview    -scene mesh -obj ~/Downloads/bunny.obj -rot 30
./raytracer -scene mesh -obj ~/Downloads/bunny.obj -rot 30 -mat glass -clamp 4 -s 512
```

**Compare acceleration structures**
```sh
./raytracer -scene mesh -obj ../models/torus_dense.obj -accel none   -n 100 -s 8
./raytracer -scene mesh -obj ../models/torus_dense.obj -accel octree -n 100 -s 8
./raytracer -scene mesh -obj ../models/torus_dense.obj -accel bvh    -n 100 -s 8
./raytracer -scene mesh -obj ../models/torus_dense.obj -accel bvh -bvhsplit middle -n 100 -s 8
```
Or press **B** in `rtview` and watch "ms/pass" in the title bar.

**Less noise from glass and mirrors**: more samples (`-s 2048`), or `-clamp 4`
(fast, slightly darker caustics).

**Use fewer cores** (to keep the computer responsive): `-threads 4`.

**Debug a crash**: `make clean && make SAN=1 SINGLE=1`, then run the same command with
a small `-n` and `-s`.

---

## 6. Constants you can tweak in the code

These have no command-line option. Change them and run `make` again.

| what | where | value |
|---|---|---|
| wall colors, light color/strength (15), camera default | `raytracer/scene.c`, `scene_cornell_box` | Cornell / Shirley values |
| box sizes and angles, sphere positions | `raytracer/scene.c` | see the comments |
| model size (330) and position, model color | `raytracer/scene.c`, `SCENE_MESH` branch | |
| ray offset against self-hits (`EPS`) | `raytracer/tracer.c` | 1e-4 |
| bounces before Russian roulette (`RR_START`) | `raytracer/tracer.c` | 3 |
| octree leaf size (`OCT_LEAF_SIZE`) | `raytracer/octree.h` (or `-DOCT_LEAF_SIZE=`) | 16 |
| octree depth limit / node limit | `raytracer/octree.h` | 14 / 8,000,000 |
| BVH depth limit (`BVH_MAX_DEPTH`) | `raytracer/bvh.h` | 100 |
| SAH costs (traversal 1, intersection 1) | `raytracer/bvh.c`, `sah_split` | 1 / 1 |
| `-accel auto` threshold (`AUTO_ACCEL_MIN_PRIMS`) | `raytracer/options.h` | 64 |
| rtview turn speed / mouse sensitivity | `raytracer/rtview.c` | 1.5 rad/s / 0.004 rad per pixel |
| rasterizer image size, light, ambient | `main.c`, `renderer.c` (`AMBIENT`, `render_default_params`) | 640×480, 0.12 |
| rasterizer viewer window size | `viewer.c` (`W`, `H`) | 800×600 |

---

## 7. The rasterizer programs

```sh
./main [model.obj] [texture.ppm]      # default: models/torus.obj + a checkerboard
./viewer [model.obj] [texture.ppm]    # real-time spinning view
```

`main` writes `out/0_gradient.ppm` ... `out/9_near_clipping.ppm`, one per milestone, and
prints the render time of each shading mode.

`viewer` keys:

| key | action |
|---|---|
| `1`–`5` | wireframe, flat, Gouraud, Phong, textured |
| `Space` | pause / resume the rotation |
| `C` | back-face culling on / off |
| `Esc` / `Q` | quit |

---

## 8. Troubleshooting

| problem | cause / fix |
|---|---|
| `cannot open X display` | run `rtview`/`viewer` from a terminal inside your desktop, or `export DISPLAY=:0` |
| `could not load ../models/torus.obj` | run `make models` in the top folder (or pass `-obj`) |
| `could not write out/...` | the `out/` folder is missing: `mkdir -p out` |
| `ply: only ASCII PLY files are supported` | convert the file to ASCII PLY or OBJ (e.g. with Blender or MeshLab) |
| white speckles | fireflies from glass/mirror paths: more `-s`, or `-clamp 4` |
| image too dark with small `-d` | fewer bounces = less indirect light; use the default 50 |
| rtview feels slow on big models | smaller window (`-n 400`), coarser preview (key `4`), fewer bounces (key `[`) |
| everything is black | the camera is outside the box looking away; press `R` in rtview, or check `-cam`/`-look` |
