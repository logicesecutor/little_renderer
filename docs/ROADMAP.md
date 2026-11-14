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
| 4 | **R4b: PLOC, a GPU tree builder of SAH quality** | next | below; PERFORMANCE.md R4b |
| 5 | **PBR: one physically based material model for all four pipelines** | planned | [PBR.md](PBR.md), phases PBR1–PBR6 |

Later (agreed, not scheduled): P5 (AVBD solver) and P10 (the GPU's fixed cost per step),
both "very good to have"; P9, P11 (P3 may settle P11), P4, P6, P7; R2, R6; mixed materials
for the falling meshes (`-meshmat mirror,diffuse`).

**Stopped: the web.** No browser checks and no web work until the user resumes it
(PERFORMANCE.md W1). Meanwhile `make web` is not run; when the web resumes, the web build is
brought up to date with everything done since (the GPU kernels of P1, P2 and later steps
must then be checked under Tint).

## 1. PNG, and the FPS counter on the image

PNG replaces PPM completely (the user's choice: the engine writes PNG itself, in C, and PPM
goes away), reading and writing, so that every image the engine makes or reads is one any
program opens.

- `core/png.{c,h}`, from scratch, no library: CRC-32 and Adler-32; a deflate encoder (LZ77
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

## 5. PBR

The design and its six phases are in [PBR.md](PBR.md): the glTF metallic-roughness model
with transmission, one shading code for the CPU and GPU rasterizers and path tracers, an
environment for the rasterizer (light probes from the scene and HDR skies), textures, and
glTF files with PNG and JPEG textures.
