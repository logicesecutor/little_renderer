# Roadmap: what comes next, in what order

The order of work agreed with the user, and where each step stands. **Keep it up to date:**
mark a step done (with its commit) when it lands, move steps when the plan changes, and
write down why. How each part works is in the other docs; the speed ideas, with their
measurements, are in [PERFORMANCE.md](PERFORMANCE.md) (the IDs P*, R* come from there);
the design of the physically based materials is in [PBR.md](PBR.md).

Status: `next` · `planned` · `later` · `stopped` · `done`.

## The order (decided 2026-11-14)

| # | step | status | what it is |
|---|---|---|---|
| 1 | **PNG, and the FPS counter on the image** | done (050baf2, 97091cd) | below |
| 2 | **P3: a cheaper hull–hull test** | done | below; PERFORMANCE.md P3 |
| 3 | **P8: a parallel prefix sum for the contact lists** | done | below; PERFORMANCE.md P8 |
| 4 | **R4b: PLOC, a GPU tree builder of SAH quality** | done | below; PERFORMANCE.md R4b |
| 5 | **PBR: one physically based material model for all four pipelines** | done: PBR1–PBR6 (2026-11-14/15) | [PBR.md](PBR.md), phases PBR1–PBR6 |
| 6 | **Our own Vulkan backend**: a WGSL → SPIR-V compiler and `webgpu.h` on Vulkan in place of wgpu-native, then native extensions, hardware ray tracing (R6) and remote rendering to browsers | done (merged into `main`, 2026-11-16; remote rendering not yet) | [VULKAN.md](VULKAN.md); PERFORMANCE.md X1, R6, P10 |

## The order (decided 2026-11-17, branch `next`)

| # | step | status | what it is |
|---|---|---|---|
| 7 | **Physics robustness**: continuous collision checks for very fast bodies, tall piles that come to rest, the CPU solver on all cores | done (f9ca27f … 8b3a18f) | PHYSICS.md 1.3b, 1.3c, 1.5; PERFORMANCE.md P6, P11 |
| 8 | **Bigger scenes** (20–50k instances): culling and simpler distant copies of meshes in the rasterizer, the top level built on the GPU, instances updated in place | done (4f0edd0 … 824333e) | PERFORMANCE.md R8 |
| 9 | **The browser build again**: the sky scene, the denoiser and the fast physics in a browser (without hardware ray tracing) | next: branch `web`, plan below (proposed 2026-11-17) | WEB.md |

`next` merged into `main` on 2026-11-17 (877237c, with the user's removal of the dragon
from the repository; the dragon stays on disk, untracked: `make dragon-download`).

## Step 9: the browser (branch `web`, proposed 2026-11-17)

**Where it stands** (Chrome 1280x800, RTX 3070, the page measured by a headless Chrome of
its own on the GPU, `out_next/cdp.mjs`): the web build is up to date and loads; one kernel
was refused by Chrome's WGSL compiler (the bodies' sweep: a barrier after a return that
depends on the thread; fixed, 24ed363), so until then the GPU bodies did not step in the
page. Measured since:

| page | browser | natively (wgpu, same GPU) |
|---|---|---|
| sky demo rasterized, GPU bodies | 125 frames/s (the display's rate) | 126+ |
| sky demo path traced, denoised | 30-33 frames/s, 22-24 ms a pass | 10 ms a pass |
| the first ~20 s of the path traced page | ~10 frames/s (light probe baked on the CPU, 3.2 s; pipelines compiled when first used) | the probe in 0.37 s |
| sky demo, bodies on the CPU (`?gpurigid=off`) | 97-127 ms a physics step at 800-900 bodies (1-2 frames/s) | 45 ms on one core, 30 on 16 |
| blast: air and cloud on the GPU | 125 frames/s | |

The steps, in order (each its own commits; measured before and after with W2's tool):

| # | step | what |
|---|---|---|
| W2 | **Measure and check in a real browser, automatically** | `tools/webbench.mjs` (Node 22+, no packages): the page in a headless Chrome of its own (a temporary profile) on this machine's GPU: console, frame rate, the viewer's line, `Module._lr_prof()`, in a log; `make web-bench`. `make web-check`: every pipeline the page makes compiled by Chrome, failing on any GPU error (the check that caught the sweep kernel) |
| W3 | **Every pipeline right in the browser** | the image set rendered in Chrome (GPU raster, GPU path traced, denoised) against the native wgpu images; `test_gpu_rigid` and `test_gpu_fluid` as pages; the wavefront kernels (their first compile in Chrome took over a minute, PERFORMANCE.md W1): one module per kernel, or kept off in browsers |
| W4 | **The path tracer as fast as natively** | find the 2.3x: Chrome's own timestamps (`timestamp-query`), the kernel's code from Chrome's compiler (the traversal stack in private memory, bounds checks), workgroup sizes. The start: pipelines made asynchronously (`createComputePipelineAsync`) instead of on first use; the light probe baked on the GPU (it already path traces) or made once and kept |
| W5 | **The denoiser in the page** | on by default with the path tracer: its cost per frame there, its look while the camera and the bodies move, its options in the page |
| W6 | **The bodies on the CPU in a browser** (no WebGPU, or `?gpurigid=off`) | 2.5x slower than natively on one core: WASM SIMD (`-msimd128`) for the contacts' math; threads (Emscripten pthreads, which need a cross-origin isolated page: `make serve` can send the headers, a static host cannot, so optional); fewer substeps when a step falls behind; P4 (CPU or GPU chosen by the number of bodies) |
| W7 | **The page** | the sky scene a link with its sky (today 20 MB of data to load: a 1k sky for the web, or fetched when that link is opened), a bigger-scene link (`props`), loading progress, the frame-rate HUD |
| W8 | **Docs and publishing** | WEB.md, README, PERFORMANCE.md W1; optionally the page on GitHub Pages (with the GitHub Actions item below) |

## Everything else to do (backlog, gathered 2026-11-17)

Not scheduled; the IDs point to PERFORMANCE.md (ideas with measurements) or to the sections
of this file.

Agreed for later (2026-11-17):

- **A Linux window for our Vulkan layer.** Today it shows a window only on Windows (Win32
  surface); an X11 or Wayland surface would let the Linux laptop's integrated GPU run the
  live viewer through our own layer.
- **Automated builds and tests**: GitHub Actions building and running `make test` on Linux
  and Windows (float and double, wgpu and our layer where a runner can) for every push.

Physics:

- **P5, AVBD solver** ("very good to have"; started on the branch `physics-avbd`, which the user rebases).
- **P11, the rest**: tall piles on the GPU fall fully asleep in 2 of 3 runs; the rest needs a
  Gauss-Seidel GPU solver by colours (P5's colouring).
- **P6, the rest**: SIMD in the CPU solver (AVX2 natively, WASM SIMD: W6).
- **P4**: CPU or GPU chosen by the number of bodies. **P9**: fewer speculative contacts on
  the CPU. **P3b**: GJK/MPR for hulls if the full test costs again. **P7**: a broad-phase tree
  past ~10k bodies (the CPU's is sort and sweep now).
- **P10, the rest**: a world partly asleep still pays all the GPU dispatches.

Rendering:

- **R7, anti-aliasing**: MSAA in the GPU rasterizer, FXAA/SMAA, coverage MSAA in the CPU
  rasterizer, TAA in the viewer, the path tracer's denoiser guide jittered and a better pixel
  filter.
- **R2**: upscaler / temporal SVGF.
- **R1 in browsers**: the wavefront kernels (W3).
- **R8, the rest**: the wgpu path tracer's GPU top level traces 10-25% slower than the CPU's
  SAH tree (PLOC or a refit could close it); simpler copies of the unit sphere for the
  physics' balls; levels of detail for the path tracer's preview.
- **M1**: surface depth from textures (height maps and parallax, detail maps; below).
- Mixed materials for the falling meshes (`-meshmat mirror,diffuse`).

Air and cloud: **A1** (sparse bricks for the cloud's advection), **A2** (the air at a lower
rate when nothing violent happens).

Our Vulkan layer (VULKAN.md "Order"): **remote rendering** to browsers (native rendering
streamed where the browser lacks a technology, e.g. hardware ray tracing); the browsers'
new features (subgroups, ray tracing) as they ship.

Tests and tools: the web checks of W2 in the GitHub Actions; the GPU physics tests are
timing-sensitive on piles (fixed with tolerances; a seeded, waited mode would make them
exact).

## 1. PNG, and the FPS counter on the image

PNG replaces PPM completely (the user's choice: the engine writes PNG itself, in C, and PPM
goes away), reading and writing, so that every image the engine makes or reads is one any
program opens.

- `formats/png.{c,h}`, from scratch, no library: CRC-32 and Adler-32; a deflate encoder (LZ77
  with hash chains and fixed Huffman codes: a good part of zlib's ratio in a few hundred
  lines); the PNG filters chosen per row (the smallest sum of absolute differences, as libpng
  does). A decoder too, needed by the PBR textures (PBR5, glTF): inflate (stored, fixed and
  dynamic Huffman blocks), all five filters, colour types grey, grey + alpha, RGB, RGBA and
  palette, bit depths 1–16 (to 8 bits), Adam7 interlacing.
- Everything that wrote `.ppm` writes `.png`: `render` and `sim` (`frame_NNNN.png`), the
  viewer's screenshots, the examples and `make` targets; `tools/frames_to_gif.py` reads PNG;
  `texture_load_ppm` becomes `texture_load_png`; the tests and `make bench` compare PNGs.
  The default output names change (`out/cornell_boxes.png` ...), the docs with them.
- A test: write and read back random images of every colour type and size (round trip
  exact), and decode reference files written by another encoder (Pillow, via uv, once, kept
  in the repo).
- **FPS on the image**: a small readout (frames per second and ms per frame) in a corner of
  the frame, drawn with a built-in 5×7 pixel font, on the CPU framebuffer and in the GPU's
  compositing pass; on by default, a key toggles it. The title keeps it too.

## 2. P3: a cheaper hull–hull test

The hulls' contacts are the biggest cost of a pile (2.6 of 4.7 ms per step with 1500 hulls
on the GPU). The separating-axis test tries every pair of edges (48 corners per hull: about
19k pairs per pair of hulls).

- Fewer corners where they do not show: measure the hull's error against the mesh for
  24, 32 and 48 corners.
- GJK for the distance (and EPA or the SAT on the touching faces only for the depth) instead
  of all edge pairs; the contact points from clipping the two touching faces, as now.
- Kept contacts (PhysX's persistent contact manifold): each pair remembers its separating
  axis or touching faces from the last step; a pair that barely moved starts there and most
  of the time needs one test.
- CPU and GPU together; measured with `mesh_rain` and the tall piles of P11.

Done (2026-11-14): the remembered pairs (the third point), CPU and GPU; the corners measured
and kept at 48 (32 would be 1.6x faster but leave up to 7% of a round shape out); GJK not
needed for now (PERFORMANCE.md P3b). Results in PERFORMANCE.md's Done.

## 3. P8: a parallel prefix sum for the contact lists

"Contact lists 1" is one GPU thread adding up the contacts of every body (0.1–0.4 ms).
A work-efficient scan (Blelloch; or decoupled look-back, Merrill & Garland 2016) over
workgroups, with a fallback when subgroup operations are missing.

Done (2026-11-14): a single workgroup of 256 threads is enough for the bodies there are
(each sums a run of them, the 256 sums are scanned together): 0.10–0.14 ms → 0.004 ms.

## 4. R4b: PLOC

Meister & Bittner, "Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy
Construction" (2018): a GPU builder close to SAH quality, for when the CPU has no top-level
tree and for meshes that change. Compared with today's LBVH and the CPU's SAH tree on build
time, tree cost (bvh_cost) and ray speed.

Done (2026-11-14): meshes' trees 8-9% cheaper than the LBVH's and rays 3-4% faster, built
1.3-1.8x slower (the 871k-triangle dragon: 10.5 ms instead of 8.4); the top level as good
either way. So PLOC builds meshes' trees and the LBVH the top level (`LR_GPU_BVH`); the
CPU's SAH trees are still the fastest to trace, and are used whenever they are current.

## 5. PBR

The design and its six phases are in [PBR.md](PBR.md): the glTF metallic-roughness model
with transmission, one shading code for the CPU and GPU rasterizers and path tracers, an
environment for the rasterizer (light probes from the scene and HDR skies), textures, and
glTF files with PNG and JPEG textures. All six phases are done (`-gltf`, `make
gltf-download` for Khronos's sample models).

## Later: M1, surface depth from textures (an idea, agreed 2026-11-15)

Make surfaces look deep and detailed without more triangles: the model stays simple and
textures carry the relief. Normal maps (PBR5) already bend the light; what they cannot do is
move what is seen (a brick's edge does not hide the mortar behind it), cast the relief's own
shadows or change the silhouette. The textures to add, each with a fallback when a model
lacks it:

- **height maps and parallax**: a height (depth) texture; the rasterizers march the view ray
  through it (parallax occlusion mapping, Tatarchuk 2006; relief mapping) so the texture
  coordinates shift where the surface would really be, with the relief's self-shadowing
  towards the lamp from the same march. The path tracers the same at each hit (the ray
  marched in the height field, then the shading point and normal there), so all four
  pipelines still agree. The procedural bricks get their height map (it is already their
  normal map's source);
- **detail maps**: a second, finer normal and colour texture repeated many times over the
  first, so close-ups stay sharp without huge textures;
- the textures glTF already brings in, used further: the occlusion texture also on the
  lamp's light where the relief hides it (only the light from around today); and a
  displacement map as the step after, where the triangles are subdivided only near the camera
  (tessellation) for real silhouettes, if parallax is not enough.

glTF has no height map in its core material: an option to give one (`-heightmap`, as
`-normalmap`), and the extensions that carry one read when models use them.
