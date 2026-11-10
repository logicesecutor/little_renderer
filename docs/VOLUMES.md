# Volumetric clouds (participating media)

A cloud is not a surface. It is a region full of tiny water droplets, and light travelling
through it is scattered, again and again, until it leaves the cloud or is absorbed. The
engine models this as a **participating medium**: a voxel grid of density in the shared
core (`core/volume.h`), filled with a procedural cumulus made from noise (`core/noise.c`),
rendered physically by the path tracer (`rt/medium.c`, `rt/tracer.c`), and in real time
by the rasterizer's raymarcher (`raster/volume_march.c`, section 7.1).

```sh
bin/render -scene cloud                         # the cloud alone in the box -> out/cornell_cloud.ppm
bin/render -scene boxes -cloud on -density 0.05 # a cloud added to any scene
bin/view -scene cloud                           # fly around it: raymarched while moving, path traced when still
bin/render -scene cloud -backend raster         # the raymarched version, in milliseconds
make cloud                                      # 600x600, 1024 samples per pixel
make cloud-frames                               # 48 animation frames (-time 0 .. 23.5 s) into out/cloud_frames/
make test                                       # checks the sampling against exact answers
```

Every option is in USAGE.md; this file explains how it works and where it can go next.

---

## 1. Code structure

| file | layer | job |
|---|---|---|
| `core/noise.h/c` | core | Perlin gradient noise, Worley cellular noise, their fractal sums (fBm, billows). Seeded, thread-safe. |
| `core/volume.h/c` | core | `Volume`: the density grid, its bounds and optical properties (`sigma_t`, `albedo`, `g`); trilinear `volume_density`; `volume_update` (majorant, active box, majorant grid of 8³-voxel bricks, `version`); the Henyey-Greenstein phase function; the procedural cumulus (`Cloud`, `cloud_density`, `volume_make_cloud`, `volume_new_cloud`, `volume_add_cloud`). |
| `core/scene.h/c` | core | `Scene.volumes` (up to 8 media, owned, freed by `scene_free`), `scene_add_volume`, `SceneOptions.cloud`, `SceneOptions.fog`, `SCENE_CLOUD`. |
| `rt/medium.h/c` | rt | delta tracking (where a ray scatters), ratio tracking (how much light gets through), both walking the majorant grid; the same for all media of a scene at once; Henyey-Greenstein sampling. |
| `rt/tracer.c` | rt | the path loop with a medium: scattering events, light sampling inside the cloud, shadows of the cloud on surfaces. |
| `raster/volume_march.h/c` | raster | real time: raymarching over the rasterized frame, with a light grid (section 7.1). |
| `app/options.c` | app | `-scene cloud`, `-cloud on`, `-density`, `-albedo`, `-g`, `-detail`, `-seed`, `-time`, `-cloudres`, `-fog`. |
| `examples/test_medium.c` | | unit test of the sampling against exact answers (`make test`). |

The grid lives in `core/` on purpose. The path tracer is only its first user. A real-time
raymarcher (the viewer), an animation and a fluid simulation (forces) all read or write the
same `Volume`. Section 7 shows how each of these plugs in.

Without a cloud (`scene.volume == NULL`) the path tracer takes exactly the same steps as
before, random numbers included: the reference images are bit-for-bit unchanged.

---

## 2. The physics in four quantities

Along a short step `ds` through density `ρ(x)`, a photon interacts with probability
`σ_t ρ(x) ds`. `σ_t` (`-density`, the **extinction coefficient**, per scene unit) is the
sum of two parts:

- **absorption** `σ_a`: the photon is gone (turned into heat);
- **scattering** `σ_s`: the photon continues in a new direction.

The **single-scattering albedo** `α = σ_s / σ_t` (`-albedo`) says which one wins. Cloud
droplets barely absorb visible light: `α ≈ 0.99` or more, which is why clouds are white,
and why thick clouds are grey underneath: the light is not absorbed, it is sent back up.

**Transmittance.** The fraction of light that crosses a segment without interacting is the
Beer-Lambert law

$$T(a, b) = \exp\Big(-\int_a^b \sigma_t\,\rho(x(t))\,dt\Big)$$

The exponent is the **optical depth** `τ`. `τ = 1` lets 37% through, `τ = 5` less than 1%.
`1/σ_t` is the **mean free path** at full density: with the default `σ_t = 0.08` it is
12.5 units, so the 300-unit-tall cloud is dozens of mean free paths thick in its core.

**Phase function.** When a photon scatters, its new direction is random, with the
probability density `p(cos θ)` per steradian, where `θ` is the angle between the old and
new directions. The engine uses **Henyey-Greenstein**:

$$p_{HG}(\cos\theta) = \frac{1}{4\pi}\,\frac{1 - g^2}{(1 + g^2 - 2g\cos\theta)^{3/2}}$$

`g` (`-g`) is the average cosine. `g = 0` scatters the same in every direction, `g > 0`
forward (the photon tends to keep going), `g < 0` backward. Real water droplets are
strongly forward-scattering, about `g = 0.85`. That gives the bright "silver lining" when
you look towards the light through a cloud's edge.

**The rendering equation in a medium** (the radiative transfer equation, integral form):
the light reaching the camera along a ray is the light from the next surface, dimmed by
`T`, plus light scattered *into* the ray at every point along the way:

$$L(x, \omega) = T(x, x_s)\,L_s(x_s, \omega) + \int_0^{s} T(x, x_t)\,\sigma_s\rho(x_t) \int_{S^2} p(\omega, \omega')\,L(x_t, \omega')\,d\omega'\,dt$$

The inner `L` is again the same equation, so in a dense cloud light scatters many times
(**multiple scattering**). That is what makes clouds look soft and white rather than dark
and smoky, and why a path tracer, which follows the bounces one by one, handles it
naturally.

---

## 3. The algorithms (`rt/medium.c`)

### 3.1 Where does the ray scatter? Delta tracking

In a *uniform* medium of extinction `σ`, the distance to the next interaction is
exponential, `t = −ln(1 − ξ)/σ` with `ξ` uniform in `[0, 1)`. Our cloud is not uniform.
**Delta tracking** (Woodcock et al., 1965) makes it uniform with a trick: add imaginary
"null" particles everywhere, just enough to fill every point up to the densest one,
`σ_max = σ_t · max_density` (the **majorant**). Then:

```
t = entry point of the ray into the cloud's active box
loop:
    t += −ln(1 − ξ₁) / σ_max              tentative collision in the uniform medium
    if t ≥ end (surface or box exit):  no interaction, the ray gets through
    if ξ₂ < σ_t ρ(x(t)) / σ_max:        real collision at t      (else: null, keep going)
```

A null collision changes nothing, so the walk produces real collisions with exactly the
right distribution, without ever integrating the density. The cost is one density lookup
per tentative collision: proportional to `σ_max` times the length of the walk.

**The active box.** `volume_update` computes the smallest box around the non-empty voxels,
and every walk is clipped to it, so rays that miss the cloud cost one box test.

**The majorant grid.** A single `σ_max` for the whole cloud is wasteful where the cloud is
thin: every tentative step there is short, and most are null. `volume_update` therefore
also cuts the grid into **bricks** of 8³ voxels and stores, per brick, the largest density
that the trilinear interpolation can return inside it (the max over the brick's voxels and
the ring of voxels around it that the interpolation reads). The walk crosses the bricks one
by one with a 3D DDA (Amanatides & Woo 1987: per axis the distance between brick borders is
constant, so the next border is the nearest of three precomputed values), and in each brick
delta tracking uses that brick's own majorant. At a border the exponential step simply
starts again with the next brick's majorant (it has no memory); empty bricks are skipped
for free. If even the global majorant gives an optical depth below 4 over the ray's whole
segment, the walk has only a few steps anyway, and one segment with the global majorant is
cheaper than walking the bricks: `rt/medium.c` picks per ray.

Measured (200 × 200, 64 spp): the 256-resolution cloud 3.1 s → 2.2 s, cloud + fog
3.0 → 2.7 s, a thin cloud (`-density 0.01`) 0.9 → 0.8 s, the default cloud unchanged
(it is dense almost everywhere inside its box, so the global majorant was already tight).
Brick sizes 4 and 16 were not better (`-DVOLUME_BRICK=n`), and `-DMEDIUM_NO_BRICKS`
turns the grid off for comparison.

**Several media.** Where media overlap their extinctions add. Each medium's collisions are
an independent random process, and the first collision of the sum is the earliest of each
medium's own first collision (the minimum of independent exponential clocks). So
`scene_medium_collision` tracks every medium on its own, each only up to the best collision
found so far, and the winner decides how the light scatters (its albedo, its phase
function); `scene_medium_transmittance` multiplies the transmittances. `-fog sigma` uses
this: a uniform haze filling the room (albedo 0.95, `g` 0.4) together with the cloud.

### 3.2 How much light gets through? Ratio tracking

Shadow rays (from a surface or a point in the cloud towards the lamp) need `T`, not a
single yes/no. Delta tracking would give 1 or 0 at random: correct on average but very
noisy. **Ratio tracking** (Novák et al., 2014) takes the same tentative collisions and
multiplies, instead of deciding:

```
T = 1
at every tentative collision:  T *= 1 − σ_t ρ(x) / σ_max
```

The expected value is exactly `exp(−τ)`, with much less noise. When `T` falls below 0.1,
Russian roulette (survive with probability 1/2, then double `T`) stops long walks without
bias.

### 3.3 In which direction? Sampling Henyey-Greenstein

Inverting the cumulative distribution of `p_HG` in `cos θ` gives a closed form:

$$\cos\theta = \frac{1}{2g}\left(1 + g^2 - \left(\frac{1 - g^2}{1 + g - 2g\xi}\right)^2\right)$$

(and `cos θ = 1 − 2ξ` when `g ≈ 0`). The azimuth is uniform, and `onb_from_normal` turns
`(θ, φ)` around the ray's direction into a world direction.

### 3.4 The path loop (`trace_path` in `rt/tracer.c`)

Each bounce first finds the next surface as before, then lets delta tracking decide
whether the ray scatters in the medium on the way there:

```
hit = closest surface (may be none)
if the medium has a real collision at t < hit.t:
    throughput *= albedo                         scattered, not absorbed
    radiance   += throughput * light_in_medium(p)   next event estimation, below
    direction   = sample Henyey-Greenstein around the old direction
    emit = MIS                                   the lamp was just sampled directly (RAYTRACER.md 5.5.1)
    Russian roulette; continue
else: the surface, exactly as before
```

Why the weights are so simple: delta tracking produces a real collision with exactly the
probability density of the physical process, so the Monte Carlo weight of the distance is
1. Of the real collisions a fraction `α` scatters, hence `throughput *= albedo`. And the
new direction is drawn with the phase function's own density, so `phase / pdf = 1`.

**Direct light inside the cloud** (`sample_light_in_medium`), the volume version of next
event estimation (RAYTRACER.md 5.5): pick a point on the lamp, check that no surface is in
the way, and dim it by the cloud between the two points (ratio tracking). That is the
cloud's **self-shadowing**: the bottom is darker because the light reaching it has already
crossed the cloud.

$$L_{direct} = p_{HG}(\omega \cdot \omega_l)\; L_e\; T(p \leftrightarrow x_l)\;\frac{\cos\theta_l \cdot \text{Area}}{d^2}$$

With MIS (the default, RAYTRACER.md 5.5.1) this sample is weighted against the chance
that the phase-sampled direction finds the same lamp point, and a phase-sampled ray that
hits the lamp counts with the complementary weight. That matters with strong forward
scattering and in fog near the lamp (fog + spheres: −28% error).

There is no `cos θ` at the scattering point (no surface), and the phase function takes the
place of the BRDF. The light arrives travelling along `−ω_l` and leaves along `−ω`, so the
scattering angle is `cos θ = ω · ω_l`.

**Shadows of the cloud on surfaces.** The surfaces' own light sampling now multiplies by the
transmittance of the shadow ray, so the floor under the cloud is darker, with a soft edge.

---

## 4. Modelling the cloud (`core/volume.c`)

A cumulus forms when warm air rises, cools, and its water condenses. It condenses at about
the same height everywhere (the base is flat), and the rising bubbles pile up into round
lobes on top: the "cauliflower". `cloud_density` builds exactly that, in the cloud's own
coordinates (the box centred at 0, its longest side spanning [−1, 1], so puffs stay round):

1. **Puffs**: `-seed` places `puffs` (12) spheres, a big central tower and smaller ones
   around it, sitting on the base. Shape = the strongest puff at the point, `1 − dist/r`.
2. **Flat base**: multiply by a `smoothstep` just above the base height, and fade near the
   box walls so the grid never cuts the cloud.
3. **Domain warping**: before evaluating the shape, push the point around with smooth
   Perlin fBm, so the puffs get irregular outlines instead of perfect spheres.
4. **Erosion**: subtract `detail × (1 − noise)`, where the noise is 75% *billows*
   (inverted Worley noise, round bumps) and 25% fine Perlin detail. The thick core
   survives; the thin edges break up into wisps. `-detail 0` gives smooth blobs, `1` very
   eroded ones.
5. **Contrast**: multiply by 5 and clamp to [0, 1]: a dense core with a short soft edge.

**Animation** (`-time`): the erosion noise is sampled at `point + wind × time`, so the
billows drift and boil while the overall shape stays put. Re-baking the 128 × 89 × 105
grid (1.2 million voxels) takes about 0.2 s on 8 threads; `make cloud-frames` renders a
sequence.

**Why a grid, and not `cloud_density` called directly by the renderer?** One noise
evaluation costs about 1 µs (12 Perlin lookups and 81 Worley cells); a trilinear grid lookup is
about 100 times cheaper, and the path tracer makes billions of them. The grid is also what
a simulation needs (section 7). `cloud_density` stays public for whoever wants to evaluate
the cloud without a grid.

**Resolution** (`-cloudres`, 128 along the longest side): the voxels are about 3 units
wide in the Cornell box. Finer detail needs more voxels (memory grows with the cube:
`-cloudres 256` = about 10 million voxels, 38 MB).

---

## 5. The parameters, and what they look like

| option | default | try | what happens |
|---|---|---|---|
| `-density` | 0.08 | 0.02 / 0.3 | thin, translucent haze / a solid-looking cumulus with a dark base. Render time grows with density (more scattering events per path). |
| `-albedo` | 0.99 | 0.8 / `0.9,0.6,0.3` | smoky, darker / tinted (each channel absorbed differently, like dust or a sunset-lit cloud) |
| `-g` | 0.3 | 0 / 0.85 / −0.3 | uniform / physically correct droplets: very forward, the cloud looks greyer from the side and glows when backlit / backscattering |
| `-detail` | 0.6 | 0 / 0.95 | smooth blobs / ragged, wispy edges |
| `-seed` | 1 | 2, 3, ... | another cloud |
| `-time` | 0 | 0 .. 30 | the billows drift with the wind |
| `-cloudres` | 128 | 64 / 256 | faster to build, blurrier / sharper detail, 8× the memory |

**Why `g = 0.3` by default and not the physical 0.85.** In the Cornell box the lamp is
above the cloud and the camera looks at it from the side. With strong forward scattering
most of the light keeps going down through the cloud instead of turning towards the camera,
and the cloud looks dim and grey. Real clouds still look white at 0.85 because they are
hundreds of mean free paths thick and lit by the whole sky; in a small box a softer `g`
gives the familiar look. Use `-g 0.85` (with more `-density`) for the physical value.

**Noise.** A cloud is harder to render than walls: paths bounce many times inside it, and
with forward scattering a few paths that head straight to the lamp carry most of the light.
Expect visible grain below ~512 samples per pixel. `-clamp 4` helps against the brightest
speckles (with a small bias).

**Speed** (8 threads, i5-8250U): `-scene cloud`, 300 × 300, 256 spp: about 15 s (1.5 M
paths/s; the boxes scene without a cloud runs at about 2.4 M). The viewer in `rt` mode at 320 × 320: about
70 ms per pass.

---

## 6. How it was checked

- **Exact answers** (`make test`, `examples/test_medium.c`): in a uniform medium of optical
  depth 1 and in a medium whose density rises along the ray (exact `τ` by numerical
  integration), ratio tracking and delta tracking both converge to `exp(−τ)` (within 1
  standard error with 400,000 samples); Henyey-Greenstein sampling gives mean cosine `g`
  for `g = −0.5, 0, 0.3, 0.85`; `p_HG` integrates to 1. The same along three rays through a
  blob in a 64 × 48 × 40 grid (8 × 6 × 5 bricks: the majorant-grid walk), and for two
  overlapping media (transmittance = product, a collision belongs to medium 1 with
  probability σ₁ / (σ₁ + σ₂)). Passes in float and in double.
- **Nothing else changed**: 13 reference images without a cloud (double path tracer,
  float path tracer, rasterizer) are bit-for-bit identical.
- **Sanitizers**: the address/undefined-behaviour build renders clouds, a camera inside
  the cloud, tinted albedo, negative `g` and a 4-voxel grid without errors.

---

## 7. The road to the next steps

Everything below builds on what exists: `Volume` (the grid), `volume_density` (lookup),
`volume_update` (keep the majorant right), `cloud_density` (the procedural shape).

### 7.1 Real-time clouds in the viewer (raymarching): done

`raster/volume_march.c` draws every medium of a scene over a frame the rasterizer has
drawn, in milliseconds. `bin/view` (raster and hybrid modes), `bin/render -backend raster`
and `bin/sim -backend raster` use it; `-rastercloud off` turns it off.

```
for each pixel:                                   (or each 2x2 block, -marchscale 2)
    ray from the eye; stop it at the distance the z-buffer gives (depth -> view depth)
    for each step (about one voxel of the finest medium the ray is in, <= -march steps):
        sigma = sigma_t * density(x)
        S = albedo * ( E_lamp(x) * sum_i a^i exp(-b^i tau(x)) phase_hg(cos, g c^i) * brightness
                      + L_ambient * (0.35 + 0.65 exp(-tau(x) / 4)) )
        L += T * (1 - exp(-sigma ds)) * S             energy-conserving step (Hillaire 2015)
        T *= exp(-sigma ds)                           stop when T < 0.003
    pixel = L + T * pixel                             in linear color, back to sRGB
```

- **The light grid** `tau(x)`: the optical depth from x to the lamp through *all* media,
  on a grid of at most 48 cells along each medium's longest side (`-lightres`), filled by
  marching from every cell centre to the lamp, only inside the medium's active box. It is
  rebuilt only when a medium changes (`Volume.version`, incremented by `volume_update`) or
  the lamp moves: free for a static cloud, about 15–40 ms per frame for a simulated one.
- **The lamp**: irradiance `L A cos / (d² + A/π)` (exact on the axis of a disk of the
  lamp's area, finite close to it, the inverse square law far away).
- **Multiple scattering** by octaves (Wrenninge, "Art-Directable Multiple Volumetric
  Scattering", 2015): the lamp term is added again with weight 0.7^i, optical depth 0.3^i
  and anisotropy 0.5^i (3 octaves). Light that bounced many times inside got through less
  cloud and forgot its direction. The constants were tuned against the path traced cloud.
- **Ambient**: the walls' light, estimated from the lamp's power spread over the walls with
  their mean albedo (interreflections included), tinted by their mean color.
- **Brightness 2**: the rasterizer's lamp has no distance falloff and no shadows, so its
  rooms are about twice as bright as the path tracer's; the media's lamp light is doubled
  to sit right in them.
- **Speed**: one step per voxel of the finest medium the ray is currently in (a 16³ fog
  takes 38-unit steps, a 128³ cloud 4-unit steps), 4× longer through empty bricks of the
  majorant grid, never stepping over the start of a finer medium. `-marchscale 2` (the
  viewer's default) marches one ray per 2×2 block and upsamples with depth-aware weights,
  so silhouettes in front of the cloud stay sharp: 3–4× faster. 640×640 on 4 cores: the
  cloud 63 ms full / 21 ms half resolution, cloud + fog 220 / 64 ms.
- **Not modelled**: shadows of the media on the surfaces, light the media bounce onto the
  walls, shadows of the bodies on the cloud. The path tracer has all of these.
- **Checked** (`bin/test_medium`): an absorbing block in an unlit scene leaves exactly
  `exp(−σ d)` of the background, with and without a surface inside the block (the z-buffer
  stops the ray), at full and half resolution; and every scene without media rasterizes
  byte-identical images to before. The result is the same for any number of threads.

### 7.2 Animation

Already possible offline with `-time`. In the viewer: re-bake every frame (0.2 s at 128³,
so roughly 5 fps), or bake two time steps and blend them, or, better, let a simulation move
the density (next).

### 7.3 Forces: a fluid simulation on the same grid

**Done**: `phys/fluid.c` does this (MAC grid, multigrid projection, MacCormack advection
of the cloud), coupled with rigid bodies in `phys/world.c`. See [PHYSICS.md](PHYSICS.md),
`bin/sim` and `bin/view -sim on`. What follows is the plan it was built from.

To make the cloud react to wind, a moving object or buoyancy, store a **velocity field**
next to the density (three more grids, or one staggered MAC grid) and advance it each time
step with **Stable Fluids** (Stam 1999):

1. **forces**: `u += dt · f`, e.g. wind, a user "push", buoyancy `∝ (temperature − ambient)`
   upwards;
2. **advect**: semi-Lagrangian, `q_new(x) = q(x − dt · u(x))`, for the velocity itself and
   for the density (trilinear lookups, exactly `volume_density`'s interpolation);
3. **project**: solve a Poisson equation for the pressure (Jacobi or conjugate gradient
   iterations) and subtract its gradient, so the flow is incompressible and swirls;
4. optionally **vorticity confinement** (Fedkiw, Stam & Jensen 2001) to bring back the
   small curls that numerical diffusion smooths away;
5. `volume_update(v)`, so the renderers see the new majorant and active box.

A sketch of the API: `FluidSim *fluid_create(Volume *v)`, `fluid_add_force(...)`,
`fluid_step(sim, dt)`. The path tracer and a raymarcher need no change.

### 7.4 Outdoor sky

Clouds belong in the sky: a **sun** (a directional light: next event estimation picks its
direction, no area), a **sky** (light arriving from every direction that escapes the
scene, e.g. a simple gradient or a physical model such as Preetham's), a ground plane, and
a much bigger cloud layer. Path tracing then needs no other change; `g = 0.85` and a dual
Henyey-Greenstein lobe (forward + a little backward) give the real look, and a tone
mapping operator handles the sun's dynamic range.

### 7.5 Faster

- Done: the **majorant grid** with a 3D DDA (section 3.1), **several media**, and
  **multiple importance sampling** between phase and light sampling (section 3.4).
- **Residual ratio tracking** (Novák et al. 2014) for even less noisy shadows.
- **Equi-angular sampling** (Kulla & Fajardo 2012) of the distance along a ray near the
  lamp: the remaining grain of fog right around the lamp.

---

## 8. References

- E. Woodcock, T. Murphy, P. Hemmings, S. Longworth, *Techniques used in the GEM code for
  Monte Carlo neutronics calculations in reactors and other systems of complex geometry*,
  1965 (delta tracking).
- J. Novák, A. Selle, W. Jarosz, *Residual Ratio Tracking for Estimating Attenuation in
  Participating Media*, SIGGRAPH Asia 2014.
- J. Novák, I. Georgiev, J. Hanika, W. Jarosz, *Monte Carlo Methods for Volumetric Light
  Transport Simulation*, Eurographics 2018 state of the art report.
- L. Henyey, J. Greenstein, *Diffuse radiation in the galaxy*, Astrophysical Journal, 1941.
- K. Perlin, *Improving Noise*, SIGGRAPH 2002. S. Worley, *A Cellular Texture Basis
  Function*, SIGGRAPH 1996.
- A. Schneider, N. Vos, *The Real-time Volumetric Cloudscapes of Horizon Zero Dawn*,
  SIGGRAPH 2015 Advances in Real-Time Rendering.
- S. Hillaire, *Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite*,
  SIGGRAPH 2016 course.
- J. Stam, *Stable Fluids*, SIGGRAPH 1999. R. Fedkiw, J. Stam, H. W. Jensen, *Visual
  Simulation of Smoke*, SIGGRAPH 2001.
- M. Pharr, W. Jakob, G. Humphreys, *Physically Based Rendering*, chapters on volume
  scattering and light transport in participating media (pbr-book.org).
