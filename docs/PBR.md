# PBR: one physically based material model for every pipeline

The plan for physically based materials (step 5 of [ROADMAP.md](ROADMAP.md)), and what was
done. Status: all six phases done (PBR1-PBR4 2026-11-14, PBR5 and PBR6 2026-11-15, below).

## Goals (decided 2026-11-14)

- **One material model**: glTF 2.0's metallic-roughness, the web's standard: base colour,
  metallic, roughness, emission, plus transmission and index of refraction
  (KHR_materials_transmission, KHR_materials_ior) for glass.
- **One shading code** for all four pipelines: the CPU rasterizer, the CPU path tracer, the
  GPU rasterizer and the GPU path tracer. The BRDF is written once in C (`core/brdf.h`,
  inline functions on `real`) and once in WGSL (one string in `gpu/`, used by both GPU
  pipelines), and a test evaluates both on the same inputs and compares.
- **Today's materials become presets** of it: diffuse = a rough dielectric, mirror = a smooth
  metal, glass = a smooth transmissive dielectric. The existing images may change a little
  (Fresnel at grazing angles, the diffuse lobe weighted by what the specular one leaves): the
  reference images are made again and the change is noted in the docs.
- **The rasterizer gets real lighting**: the material lit by the area lamp, and by an
  environment, for reflections and ambient: **light probes** made from the scene (a closed
  room like the Cornell box) and **HDR skies** from files (open scenes); the path tracer
  gets the sky as a light too.
- **Textures** for every parameter, and **glTF 2.0 files** (`.gltf` and `.glb`) with **PNG
  and JPEG** textures, all decoded by the engine's own code.

## The model

A material: `base_color` (RGB), `metallic`, `roughness`, `ior` (1.5), `transmission` (0..1),
`emission` (RGB, lights), and optional textures: base colour (sRGB, alpha), metallic-roughness
(B = metallic, G = roughness, as glTF), normal map (tangent space), emissive, occlusion (the
rasterizer only; the path tracer finds occlusion itself).

The BRDF (glTF's appendix B, which every glTF viewer follows):

- **Specular**: microfacets with the GGX / Trowbridge-Reitz distribution
  (`α = roughness²`), the height-correlated Smith masking-shadowing, Schlick's Fresnel.
  Dielectrics: F0 = ((ior − 1) / (ior + 1))² (0.04 at 1.5); metals: F0 = base colour.
- **Diffuse**: Lambert with the base colour, weighted by (1 − F) (what the specular layer
  does not reflect), zero for metals.
- **Transmission**: the rough dielectric BTDF (Walter et al. 2007, the same GGX); roughness 0
  is today's glass exactly.
- **Energy lost by rough microfacets** (single scattering darkens rough metals): the
  Kulla–Conty compensation, measured on the white furnace test; added if the loss shows.

The path tracer samples it: the lobe by its Fresnel-weighted share; GGX by its **visible
normals** (Heitz 2018; the spherical-cap form of Dupuy & Benyoub 2023, shorter); the lamp
by next event estimation, combined with the BRDF's sample by MIS as today.

## The rasterizer's lighting

- **The lamp**: the same BRDF, lit by the quad lamp as by its "most representative point"
  (Karis 2013: the point of the quad nearest the reflected ray for the specular, its centre
  for the diffuse, with the energy normalized). Linearly transformed cosines (Heitz et al.
  2016, exact for polygonal lights) are the step after, if the highlights look wrong: they
  need fitted tables, which would be fitted by our own tool.
- **The environment**, split-sum (Karis 2013): a cube map prefiltered by roughness (each
  mip level convolved with GGX by importance sampling), a 2D table of the BRDF's scale and
  bias over (n·v, roughness) computed at start, and the irradiance as 9 spherical harmonics
  for the diffuse. Where the environment comes from:
  - **light probes**: the scene path traced into a cube map from a point (the room's centre
    by default, or chosen), made again when asked (the bodies move, the probe does not
    follow every frame);
  - **HDR skies**: Radiance `.hdr` files (RGBE, run-length encoded), as an equirectangular
    map turned into the cube map; the path tracer uses the same map as a light at infinity,
    sampled by its luminance (a 2D CDF).
- **Glass** in the rasterizer: the environment along the refracted ray, mixed with the
  reflection by Fresnel, drawn after the opaque surfaces.
- No shadows (as today); shadow maps are a later step if wanted.

## Textures

Texture coordinates from OBJ (`vt`, already read) and glTF; tangents computed per vertex for
normal maps (by the triangles' UV directions, as MikkTSpace does, so that glTF's normal maps
look right). Mip-maps made at load; the rasterizers filter trilinearly, the path tracers
bilinearly (ray differentials later). On the GPU the textures go into a few atlases (one per
format), each texture a rectangle with its own mips and filtering in the shader; the choice
against texture arrays (sizes must match per layer) is made in PBR5 after a measurement.

## glTF

`core/gltf.{c,h}`: `.gltf` (JSON + `.bin` + image files, or data URIs in base64) and `.glb`
(one binary file). A small JSON parser of our own; meshes (positions, normals, texture
coordinates 0, tangents, indices; triangles only), the node tree with its transforms (each
node with a mesh becomes an instance), the materials above with their textures. Images: PNG
(the decoder of ROADMAP step 1) and JPEG: baseline and progressive JPEG (Huffman, IDCT,
4:2:0 / 4:2:2 / 4:4:4 chroma, restart markers) in `core/jpeg.{c,h}`. Programs take
`-gltf file` to load one into the room (or, with an HDR sky, into an open scene).

## Phases

| phase | what | done when |
|---|---|---|
| PBR1 | The material model, the presets, the C BRDF; the CPU path tracer with VNDF sampling and rough glass; options `-base`, `-metal`, `-rough`, `-ior`, `-transmission` (and `-mat` presets) | white furnace test (a white rough metal and a white dielectric in a uniform environment keep their energy, up to the lost multiple scattering), BRDF reciprocity and sampling-vs-evaluation tests; the reference images remade. **Done** (below) |
| PBR2 | The GPU path tracer with the WGSL copy | the C/WGSL comparison test; CPU and GPU images agree (mean radiance within 1%, as today). **Done** (below) |
| PBR3 | Both rasterizers with the same BRDF, lit by the lamp | raster vs path traced direct light on a metal and a plastic sphere: same highlights. **Done** (below) |
| PBR4 | The environment: light probes, `.hdr` skies, prefiltering, the split-sum table, SH irradiance; the sky as a light in the path tracer; an open scene to use it | raster reflections of the room on the mirror dragon; a sky-lit scene path traced and rasterized. **Done** (below) |
| PBR5 | Textures and normal maps in all four pipelines | a textured model the same in the four. **Done** (below) |
| PBR6 | glTF + JPEG; `-gltf`; the showcase with glTF models | the Khronos sample models (DamagedHelmet, MetalRoughSpheres, ...) load and look as in other viewers. **Done** (below) |

Every phase: CPU and GPU together, Windows and Linux (float and double) builds and
`make test`; the measurements in [PERFORMANCE.md](PERFORMANCE.md); one commit per phase.

### PBR1, done (2026-11-14)

- `core/scene.h`: `Material` gains `metallic`, `roughness`, `transmission`; `MAT_PBR` takes
  them as given, `MAT_DIFFUSE` / `MAT_MIRROR` / `MAT_GLASS` / `MAT_LIGHT` are presets
  (`material_resolved`), so old code (`{MAT_GLASS, tint, emission, ior}`) keeps working;
  constructors `material_diffuse`, `material_mirror`, `material_glass`, `material_light`,
  `material_pbr`.
- `core/brdf.h`: the BSDF in C (`bsdf_make`, `bsdf_eval`, `bsdf_sample`); its tables in
  `core/brdf_tables.h`, made by `tools/gen_brdf_tables.py`. RAYTRACER.md 5.15 explains it.
- `rt/tracer.c`: the path tracer and the light tracer shade with it (NEE and MIS over the
  BSDF's rough part, perfect lobes as S vertices of the caustics partition).
- `-base`, `-metal`, `-rough`, `-ior`, `-transmission` change the model's material (`bin/render`,
  `view`, `sim`; `SceneOptions.mesh_set`, `scene_mesh_material`).
- `bin/test_brdf` (in `make test`): reciprocity, sampling against evaluation by quadrature,
  the white furnace.
- The README's images remade with the CPU path tracer.

Decided on the way (the open questions):

- **Multiple scattering: compensated.** The furnace test showed the loss: a white metal of
  roughness 1 kept 31% of the light seen straight on, 76% at grazing angles; roughness 0.5
  86-92%. Kulla & Conty's lobe (two 32 x 32 tables) brings every white metal to 99.9-100%.
- **The diffuse layer: albedo-scaled, not glTF's `1 - F(v.h)`.** glTF's form returned up to
  28% more light than arrived (white plastic, roughness 0.3, seen at 84°); scaling the
  diffuse by what the specular lobe leaves both ways, `(1 - Es(v)) (1 - Es(l)) / (1 - Es_avg)`,
  keeps exactly what arrives and stays reciprocal. A first try with `F(n.v)` in place of the
  lobe's albedo kept only 45% of it at grazing angles on a rough white dielectric (its
  rough specular lobe reflects far less than F there).
- **Perfectly smooth below roughness 0.02** (delta lobes).
- **Through glass the BSDF is the radiance one** (`f(v, l) / eta_l^2` reciprocal); the
  light tracer weighs refractions by `(eta_from / eta_to)^2`.

### PBR2, done (2026-11-14)

- `gpu/brdf_gpu.c`: `GPU_WGSL_BSDF`, the C of `core/brdf.h` function by function in WGSL
  (`smooth` is a WGSL keyword: the flag is `perfect` there). A module that includes it
  defines `bsdf_tab(k)`; the path tracer keeps the tables after the blue-noise mask in its
  binding 10 (no new binding: its 8 storage buffers are all used).
- `gpu/trace_gpu.c`: the materials go up resolved (base colour, metallic, roughness, ior,
  transmission); the path tracer, the wavefront tracer, the light tracer (and the photon
  cache it fills) and the denoiser's guide (which walks through perfect mirrors and glass
  to the first surface with a rough part) shade with it, as `rt/tracer.c` does.
- The GGX distribution now takes `|n x h|^2` from the cross product instead of `1 - (n.h)^2`
  (C and WGSL): at the peak of a sharp lobe the subtraction lost the digits that the
  C/WGSL comparison then saw as up to 12% different densities.
- `bin/test_gpu_brdf` (in `make test`): 65536 cases (materials of every kind, views and
  lights in every direction, the sampling's numbers): the same evaluation, density,
  decision, direction and weight in WGSL and C, all of them (worst relative difference
  2e-4 for the evaluation, 1e-3 for the weights).
- GPU against CPU images (256×256, 4096 against 512 samples, linear mean): the boxes,
  the spheres, gold, a rough white metal, frosted glass, blue plastic, smooth red plastic,
  a half-metal half-clear material, a copper dragon, a metal torus in fog: all within
  0.25% (the CPU's noise).
- Cost on the GPU: PERFORMANCE.md (+2% to +28%).

### PBR3, done (2026-11-14)

- `raster/render.c`: `render_shade`, the BSDF lit by the quad lamp (RASTERIZER.md,
  "Lighting"): the diffuse-like lobes by the lamp's exact form factor; the microfacet lobe
  at the lamp's centre, over its 4 quarters or at its point nearest the reflected ray, by
  how the lobe's solid angle compares with the lamp's; an ambient light through the BSDF's
  albedo for a uniform environment. Flat and Gouraud shade with it at the triangle's point
  and at the vertices (the varying is now a colour).
- `core/brdf.h`: `bsdf_spec_lobe`, `bsdf_diffuse_lobes`, `bsdf_albedo_uniform` (the lobes
  one at a time; the last one PBR4's environment uses too), and their WGSL copies in
  `GPU_WGSL_BSDF`.
- `gpu/raster_gpu.c`: the same in WGSL (the lamp and the resolved materials in its
  uniforms, the BSDF's tables in a second uniform buffer).
- Against the path traced direct light (`LR_RASTER_AMBIENT=0` against `-d 1 -caustics pt`,
  a model sphere 300 px, linear radiance at the sphere's top, where the highlight is): the
  walls the same; metal +10%, plastic +11%, rough metal +2%, mirror -10%, half-rough metal
  -6%; the highlights in the same places, of the same shapes and sizes. A first version
  with the representative point alone gave rough lobes 28% too much (it lights the whole
  lobe from the lamp's best point): the quarters fixed it.
- With the ambient light: the spheres' room has the path tracer's mean brightness (0.0949
  against 0.0947, linear); the boxes' floor is brighter (no shadows), the ceiling and the
  back wall within 10%.
- CPU against GPU rasterizer: within 1/255 but for a few pixels on triangle edges, float
  and double. Cost: PERFORMANCE.md.

### PBR4, done (2026-11-14)

- `core/hdr.{c,h}`: Radiance `.hdr` files read (flat, old and new run-length encoding) and
  written. `core/env.{c,h}`: the environment map: lookups, sampling by luminance, the
  levels prefiltered with GGX (filtered importance sampling, 128 samples per texel), the
  irradiance's 9 harmonics, a procedural sky, a probe's box (parallax).
- The path tracers (CPU, GPU single kernel and wavefront) see the sky where rays leave the
  scene and sample it as a light (RAYTRACER.md 5.16). On the GPU the sky is a texture
  (its 8 storage buffers are all used) with its sampling distributions in a second one;
  the kernels are compiled twice, with the sky's code and without it (`SKY_ON`), the second
  made only when a scene has a sky: without one the sky's code cost 11% on the boxes.
- `rt/probe.{c,h}`: light probes, path traced (the lamp seen directly left out; the
  rasterizers light with it themselves). Each scene says where its probe goes: the room's
  centre and box; the open scene above its spheres.
- The rasterizers (CPU and GPU) light with the probe or the sky, split-sum
  (RASTERIZER.md, "Lighting"), and draw the sky behind. The CPU rasterizer clips against a
  guard band now (a floor to the horizon left holes).
- `-scene sky`, `-sky`, `-skyscale`, `-skyrot`, `-probe`, `-proberes`, `-probespp`; the
  viewer's `L` makes the probe again.
- Test skies: Poly Haven's `kloofendal_48d_partly_cloudy` and `studio_small_09` (2k, CC0),
  downloaded into `models/skies/` (not in git).
- `bin/test_env` (in `make test`): .hdr round trip; the sampling's density (integrates to
  1, is the one `env_pdf` states, the sky's integral by it equals the quadrature's), on the
  procedural sky turned and not and on the downloaded skies; a constant environment stays
  constant through the levels and the harmonics; the harmonics' irradiance and the levels
  against quadrature for an environment of degree 2 (0.003% and 0.5%); the path tracer's
  white furnace under a white sky.
- GPU against CPU path tracing (200 px, 1024 against 256 samples, linear mean): the open
  scene under the procedural sky, both downloaded skies (one turned), the spheres' room
  with a sky through its front, the open scene in fog: all within 0.4% (the CPU's noise),
  the wavefront the same. GPU against CPU rasterizer: a mean difference of 0.05-0.2 of
  255.
- Cost: PERFORMANCE.md.

Decided on the way:

- **Equirectangular maps, not cube maps.** One kind of lookup for every pipeline (C and
  WGSL), the same map the path tracer samples. The prefiltered levels halve down to 64 x
  32 and then stay (a 16 x 8 level made the roughest lookups 4% wrong near the poles), so
  they are not one texture's mip levels: on the GPU they are one buffer, read with the C's
  bilinear lookups.
- **The probe holds the indirect light only**: the lamp seen directly is left out (the
  rasterizers light with the lamp exactly), its light reflected by anything is in.
- **The open scene gets a probe too**: lit by the sky map alone, its spheres reflected the
  photo's own landscape instead of the scene's floor.
- **No split-sum table of our own**: the BSDF's tables (`BRDF_E`, `BRDF_B`, the directional
  albedo and its Fresnel-free part) are that table.

### PBR5, done (2026-11-15)

- `core/texture.{c,h}`: `TexMap`, an RGBA8 image with its mip levels (2 x 2 averages, in
  linear light for colour: a black and white checker averages to half the light, sRGB 188),
  read from PNG files or made here (`checker`, `bricks` with its `bricks_normal` and
  `bricks_mr`, `plate_mr`); bilinear and trilinear lookups, repeating, and the level for a
  pixel's footprint. Texture coordinates as glTF's: v down the image (OBJ's turned over at
  load).
- Materials take glTF's three textures: base colour (times the factor), metallic and
  roughness (blue and green, times the factors) and a normal map (tangent space, green up
  the image), and `uv_scale`. `material_at` and `material_normal` (core/scene.h) are the one
  place that reads them; every pipeline calls them (the GPU's copy is WGSL).
- Meshes keep their `vt` and get a tangent per texture coordinate, from the triangles'
  texture directions (MikkTSpace's idea: summed per vertex, the handedness from dP/dv, so
  mirrored coordinates work); spheres, quads and the boxes' faces have their own (a sphere
  by longitude and latitude). Instances turn their tangents with their transforms.
- The CPU rasterizer interpolates the coordinates and the tangent, and works out how they
  change to the next pixel exactly (from the barycentrics' gradients): trilinear filtering
  at the right level. Phong reads the textures per pixel, Gouraud and flat per vertex.
- The path tracers (CPU, GPU single kernel and wavefront, light tracing) read level 0
  bilinearly (no ray differentials yet: at a distance the noise averages the texels).
- On the GPU every texture's levels go into one RGBA8 atlas (rectangles packed in rows,
  wrapping inside each, filtered by hand as the C does) with a table of where they are,
  bound the same way in the rasterizer and the path tracer.
- `-tex`, `-mrmap`, `-normalmap`, `-uvscale` texture the model; the open scene's floor is
  brick paving.
- **The irradiance grid.** One probe gave the whole room its light: a ceiling far from the
  lamp's bright floor got the same as one close to it, and the mesh scene's probe ended up
  inside the model. Now each scene with a probe gets a grid too (`IrrGrid`, core/env.h):
  5 x 5 x 5 probes over the room (5 x 3 x 5 in the open), each 9 harmonics of the light
  arriving there, path traced (2048 paths), blended trilinearly by the rasterizers for the
  diffuse light. Probes inside something (a back face of a closed shape seen first, or no
  ray escaping where the room's centre's do) are left out of the blend. The probe itself
  (reflections) is placed at the most open of 27 points around where the scene asks. A
  single path brighter than 4 (the lamp seen through glass) is clamped, but the sky seen
  directly is not in the paths: it is sampled by its luminance on its own (a sun would
  have been clamped away: the open floor came out at a third of its light).
- A normal map can tilt a normal towards a lamp the surface itself faces away from; the
  path tracer finds no light there, so the rasterizers take only the part of the lamp
  above the surface's own plane (`lamp_hz`): bricks on the underside of a sphere no longer
  show lit edges.
- `bin/test_tex` (in `make test`): the mip chain (odd sides, the last level the image's
  mean in linear light, sRGB averaging), bilinear exactness and wrapping, the level for a
  footprint, the tangents and their handedness (mirrored too), a normal map bending the
  normal up the image, the bricks' map leaning the right way, `material_at`'s factors and
  `uv_scale`.
- Checked: GPU against CPU rasterizer on textured models (sphere with bricks, checker cube,
  a metal torus with `plate_mr`, Gouraud, the rooms, the open scene): a mean difference of
  0.07-0.12 of 255. GPU, wavefront and CPU path tracing of textured models, instances
  included: within 0.3%. The CPU rasterizer against the path tracer: the rooms within about
  5%, the textured floor under the sky alike.
- Cost: PERFORMANCE.md.

Decided on the way:

- **One atlas, not texture arrays.** An array's layers must all be one size, and the path
  tracer has no binding to spare for one texture per map: one RGBA8 texture holds every
  level of every map, and the shaders filter by hand. The rasterizer and the path tracer
  read it the same way.
- **The grid for the diffuse light, the probe for reflections.** Reflections need detail
  (a map), the diffuse light needs where it is: a grid of 9 harmonics each is small and
  fast to make, and the GPU blends it the same way.
- **Filtering only where it pays**: trilinear in the rasterizers, level 0 in the path
  tracers (many samples per pixel average it anyway).

### PBR6, done (2026-11-15)

- `core/jpeg.{c,h}`: JPEG read by our own code: baseline, extended (SOF1, 16-bit tables) and
  progressive (spectral selection, successive approximation), Huffman, restart markers, grey
  and 3 components (YCbCr, or RGB by Adobe's marker), any sampling factors. The IDCT is
  libjpeg's accurate integer one, the chroma upsampling its "fancy" triangle filter, the
  colour conversion its fixed point: the pixels are **exactly libjpeg's** on all 14 test
  files. A 2048 x 2048 progressive JPEG decodes in 120 ms (the Linux VM; libjpeg-turbo with
  SIMD: 73).
- `core/json.{c,h}`: a small JSON reader into a tree (UTF-8, \uXXXX escapes and surrogate
  pairs; errors with their line).
- `core/gltf.{c,h}`: `.gltf` (buffers and images in files beside it or in data: URIs) and
  `.glb`; accessors of every component type (normalized or not, strided), triangles, strips
  and fans, indexed or not; the default scene's node tree (matrix or translation / rotation /
  scale): each node with a mesh becomes an instance of a scene mesh (one per glTF mesh, all
  its primitives, each triangle with its primitive's material); the model placed as an .obj
  is (`scene_load_gltf(scene, path, size, base, rot)`). Its tangents are used (computed as
  ours when missing: the same convention, bitangent up the image).
- Materials gain glTF's last pieces: **emission** (factor x texture x
  KHR_materials_emissive_strength) in all four pipelines, the paths that hit a glowing surface
  count it (it is not sampled as a lamp); the **occlusion** texture, used by the rasterizers on
  the light from around (the path tracers find occlusion themselves); the normal map's
  **scale**; the textures' **wrap modes** (repeat, clamp, mirror; CPU and GPU).
- `-gltf file` (USAGE.md): the model in the room (the default with it), or under the sky in
  place of the spheres (`-scene sky`); turned to face the camera. The showcase puts the
  damaged helmet in the cube's place when `make gltf-download` has fetched the samples.
- Test models (CC BY 4.0, Khronos glTF-Sample-Assets): DamagedHelmet, MetalRoughSpheres,
  NormalTangentMirrorTest, OrientationTest, BoxTextured (.glb, and as .gltf with its .bin and
  .png), in `models/gltf/` (not in git).
- `bin/test_jpeg` and `bin/test_gltf` (in `make test`): JPEG files written by libjpeg
  (`tools/gen_jpeg_tests.py`) decoded to its exact pixels, broken files refused; glTF models
  made by the test as .gltf with data: URIs, with files, and as .glb: instances, placement,
  the nodes' transforms, normals, texture coordinates, tangents, every material field, the
  texture's pixels and sampler; JSON values and errors; broken files refused. Both run clean
  under AddressSanitizer and UBSan.
- Checked on the samples (in the room, and the helmet under a sky): GPU, wavefront and CPU
  path tracing agree within 0.04%; the GPU and CPU rasterizers within 0.07-0.13 of 255; the
  rooms' rasterized light against the path tracer's as in PBR5 (boxes +7.8%, spheres -2.6%,
  the open scene +4.2%). They
  look as in other viewers: the helmet's visor, glow and worn metal, the spheres' metallic /
  roughness sweep, the orientation test's arrows, the mirrored normal maps (rasterizer and
  path tracer alike).
- Cost: PERFORMANCE.md.

Decided and found on the way:

- **Atlas packing**: the GPU atlas is now as narrow as fits under 8192 rows (1024 to 8192
  wide; the helmet's five 2048 x 2048 maps did not fit in one 2048-wide column); if nothing
  fits, the largest texture's top level is left out on the GPU (a message says so).
- **Probe normalization** (the rasterizers): a single reflection probe made somewhere else
  showed the room's light in corners that see none of it (the orientation test's cube front,
  facing the room's open side, shone blue). The specular from the probe is now scaled by how
  much light arrives around the reflected direction at the point (the irradiance grid)
  against at the probe (its own harmonics), never brightened; the luminances from the first
  4 harmonics (the full 9 cost 12 ms more per frame on the boxes).
- **Inside or behind**: a grid probe counted as inside something when any of its rays hit a
  back face; thin open panels (glTF's double-sided surfaces) then left whole layers of probes
  unused (a lighter rectangle on the wall behind the normal-map test). Now it takes most rays.
- **Light tracing with roughness textures**: the CPU and GPU decided whether a scene can have
  caustics from the untextured materials, so a roughness map with smooth texels lost those
  caustics in both (the path tracer leaves them to light tracing): textured materials count
  now. One difference remains, the same on CPU and GPU: on the normal-map mirror test, light
  tracing gives 0.4% less than path tracing alone (normal maps are not reciprocal: the bent
  normal is tilted towards the light in one and towards the camera in the other).
- **At most 16 materials** (`MAX_MATS`): the GPU packs a triangle's material in 4 bits. Every
  sample model fits; a model with more shares the last one (with a message). More materials
  need a wider packing: later, with the Vulkan backend's scene layout.
- Not read (gltf.h): animations and skins, cameras, lights, morph targets, sparse accessors,
  alpha masking and blending (drawn opaque), texture coordinates 1, KHR_texture_transform;
  Draco and meshopt compression are refused, KTX2 and WebP images skipped.

## Open questions (decided during the phases)

- The lamp in the rasterizer: representative point first; linearly transformed cosines if
  the highlights differ visibly from the path tracer. (Decided in PBR3: no LTC; the
  representative point for narrow lobes, quadrature over the lamp for wide ones, the
  exact form factor for the diffuse: within about 10% of the path tracer.)
- Multiple-scattering compensation: only if the furnace test shows the loss. (Decided in
  PBR1: it showed, compensated.)
- Textures on the GPU: atlases or arrays. (Decided in PBR5: one RGBA8 atlas, below.)
- Test files (Khronos samples, HDR skies) come from outside: downloaded only with the user's
  permission, as usual.

## References

- Khronos, glTF 2.0 specification, appendix B (BRDF implementation), and the
  KHR_materials_transmission / KHR_materials_ior extensions.
- B. Walter, S. Marschner, H. Li, K. Torrance, "Microfacet Models for Refraction through
  Rough Surfaces", EGSR 2007.
- E. Heitz, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs", JCGT
  2014; "Sampling the GGX Distribution of Visible Normals", JCGT 2018.
- J. Dupuy, A. Benyoub, "Sampling Visible GGX Normals with Spherical Caps", HPG 2023.
- B. Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013 course (split sum,
  representative point).
- E. Heitz, J. Dupuy, S. Hill, D. Neubelt, "Real-Time Polygonal-Light Shading with
  Linearly Transformed Cosines", SIGGRAPH 2016.
- C. Kulla, A. Conty, "Revisiting Physically Based Shading at Imageworks", SIGGRAPH 2017
  course (energy compensation).
- R. Ramamoorthi, P. Hanrahan, "An Efficient Representation for Irradiance Environment
  Maps", SIGGRAPH 2001 (9 spherical harmonics).
- G. Ward, the Radiance picture format (RGBE); ITU T.81 (JPEG); W3C PNG specification,
  RFC 1950/1951 (zlib, deflate).
