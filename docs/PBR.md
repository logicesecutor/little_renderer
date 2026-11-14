# PBR: one physically based material model for every pipeline (plan)

The plan for physically based materials (step 5 of [ROADMAP.md](ROADMAP.md)). Status: a plan,
nothing implemented yet; each phase is marked here as it lands.

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
| PBR1 | The material model, the presets, the C BRDF; the CPU path tracer with VNDF sampling and rough glass; options `-base`, `-metal`, `-rough`, `-ior`, `-transmission` (and `-mat` presets) | white furnace test (a white rough metal and a white dielectric in a uniform environment keep their energy, up to the lost multiple scattering), BRDF reciprocity and sampling-vs-evaluation tests; the reference images remade |
| PBR2 | The GPU path tracer with the WGSL copy | the C/WGSL comparison test; CPU and GPU images agree (mean radiance within 1%, as today) |
| PBR3 | Both rasterizers with the same BRDF, lit by the lamp | raster vs path traced direct light on a metal and a plastic sphere: same highlights |
| PBR4 | The environment: light probes, `.hdr` skies, prefiltering, the split-sum table, SH irradiance; the sky as a light in the path tracer; an open scene to use it | raster reflections of the room on the mirror dragon; a sky-lit scene path traced and rasterized |
| PBR5 | Textures and normal maps in all four pipelines | a textured model the same in the four |
| PBR6 | glTF + JPEG; `-gltf`; the showcase with glTF models | the Khronos sample models (DamagedHelmet, MetalRoughSpheres, ...) load and look as in other viewers |

Every phase: CPU and GPU together, Windows and Linux (float and double) builds and
`make test`; the measurements in [PERFORMANCE.md](PERFORMANCE.md); one commit per phase.

## Open questions (decided during the phases)

- The lamp in the rasterizer: representative point first; linearly transformed cosines if
  the highlights differ visibly from the path tracer.
- Multiple-scattering compensation: only if the furnace test shows the loss.
- Textures on the GPU: atlases or arrays.
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
