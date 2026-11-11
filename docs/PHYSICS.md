# The physics engine: rigid bodies, air, and a cloud that reacts

`engine/phys/` is a small physics engine written from scratch on the engine's own math
(`core/lrmath.h`). It moves two kinds of things and couples them both ways:

- **rigid bodies**: spheres and boxes that fall, bounce, slide, spin and stack;
- **the air** of the room, an incompressible fluid on a grid, which **carries the cloud**:
  the cloud's density is the same `Volume` the path tracer renders (`core/volume.h`).

A ball falling through the cloud pushes the air aside, the air drags the cloud along, and a
hole and a wake open behind the ball. A balloon, nearly as light as the air, is carried by
it. An explosion makes the air rush outwards, bursts the cloud, and throws the bodies.

```sh
make && bin/sim                       # the "drop" demo: 120 frames into out/sim/ + out/sim.gif
bin/sim -demo blast -frames 90        # an explosion in the cloud
bin/view -sim on                      # live, in the window (Space pause, K boom, N drop a ball)
bin/test_physics                      # the checks against known answers (also `make test`)
```

| file | what it does |
|---|---|
| `core/lrmath.h` | (added) `Mat3`, `Quat` and their operations: the rotation math of the bodies |
| `phys/rigid.h/.c` | rigid bodies: integration, collision detection, the contact solver |
| `phys/fluid.h/.c` | the air: Stable Fluids on a MAC grid, the cloud's advection |
| `phys/poisson.h/.c` | the pressure equation, solved by multigrid |
| `phys/world.h/.c` | both together: the coupling, explosions, and keeping a `Scene` in sync |
| `app/simdemo.h/.c` | the demos of `bin/sim` and `bin/view -sim on` (apps layer) |
| `app/frame.h/.c` | one path traced frame (apps layer) |
| `apps/sim.c` | the offline simulation + rendering program |
| `examples/test_physics.c` | the tests |
| `tools/frames_to_gif.py` | frames → animated GIF (Pillow; `uv run` installs it by itself) |

`phys/` only includes `core/` (`make check-layers` checks it): the physics knows nothing
about the renderers. The renderers know nothing about the physics either; the only meeting
points are the `Volume` (the fluid writes it, the path tracer reads it) and the `Scene`'s
primitives (`phys_sync_scene` moves them).

**Units.** The scene's units are treated as centimetres: the Cornell box is 5.55 m, gravity
is 981 cm/s², densities are g/cm³ (water 1, steel 7.8, air 0.0012), masses are grams.

---

## 1. Rigid bodies (`phys/rigid.c`)

### 1.1 State and integration

A `Body` has a shape (sphere: radius; box: half sizes), a mass `m`, an inertia tensor, and
its state: position `x` (center of mass), velocity `v`, orientation `q` (a unit
quaternion) and angular velocity `ω` (world frame).

The inertia tensor in the body's own frame is diagonal:

| shape | I (body frame) |
|---|---|
| sphere, radius r | `I = 2/5 m r²` on every axis |
| box, half sizes (a, b, c) | `Ix = m/3 (b² + c²)`, `Iy = m/3 (a² + c²)`, `Iz = m/3 (a² + b²)` |

The solver needs the inverse in world coordinates: `I⁻¹_world = R · I⁻¹_body · Rᵀ`, with
`R` the rotation matrix of `q` (`body_inv_inertia_world`).

Each step of `dt` is **semi-implicit (symplectic) Euler**:

```
v += dt · (g + F/m)          ω += dt · I⁻¹_world τ          (forces first)
... contacts change v and ω ...
x += dt · v                  q += dt · ½ (0, ω) q,  then normalize q
```

Updating the velocity before the position (instead of after, as explicit Euler does) keeps
the energy of an orbit or a bounce from creeping up, and it is what the contact solver
needs: it corrects the velocities that are about to be used. The quaternion derivative
`q̇ = ½ ω q` is integrated the same way and renormalized so it stays a rotation
(`quat_integrate`).

Why quaternions and not angles: four numbers, no gimbal lock, and composing / normalizing
them is cheap. `Quat` (`w + xi + yj + zk`) has `quat_axis_angle`, `quat_mul`,
`quat_rotate`, `quat_to_mat3`, `quat_norm` in `lrmath.h`.

### 1.2 Collision detection

Every step, every pair of bodies whose bounding spheres overlap is tested, and every body
against the six walls of the room. Each touching pair gives one or more **contacts**: a
point `p`, a normal `n` (from body b towards body a) and a penetration depth.

| pair | method |
|---|---|
| sphere – wall | distance of the center to the plane |
| box – wall | each of the 8 corners against the plane |
| sphere – sphere | distance between the centers |
| sphere – box | the sphere's center in the box's frame, clamped to the box = the closest point; if the center is inside, leave by the nearest face |
| box – box | separating axis test + face clipping (below) |

**Box against box.** Two convex shapes do not overlap if and only if there is an axis on
which their projections do not overlap. For two boxes it is enough to try 15 axes: the 3
face normals of each, and the 9 cross products of an edge direction of one with an edge
direction of the other. On an axis `L` the box projects to an interval of half-length
`Σ hᵢ |eᵢ · L|`, so the overlap is `ra + rb − |(xa − xb) · L|`. If one is negative, there
is no contact; otherwise the axis with the smallest overlap is the contact normal.
Face axes are preferred (an edge axis must be clearly better), and a's faces over b's, so
that the choice does not flicker between steps when a box rests on another.

The contact points come from **clipping** (as in Box2D / ODE):

- face axis: the box owning that face is the *reference*, the other the *incident* box.
  Its face most opposed to the normal is taken (4 corners) and clipped against the 4 side
  planes of the reference face (Sutherland–Hodgman). The clipped points below the
  reference face are the contacts: a box resting flat on another gets 4 contacts, at the
  corners of their overlap, which is what makes it stable.
- edge axis: one contact between the closest points of the two edges.

### 1.3 The contact solver: sequential impulses

At a contact the bodies must stop approaching. With the velocity of the two contact
points `v_rel = (va + ωa × ra) − (vb + ωb × rb)` (`ra`, `rb` = contact point minus the
centers), the condition is `v_rel · n ≥ 0`. An impulse `j n` (applied +j to a, −j to b)
changes `v_rel · n` by `j / m_eff`, where

```
1 / m_eff = 1/ma + 1/mb + n · ((I⁻¹a (ra × n)) × ra) + n · ((I⁻¹b (rb × n)) × rb)
```

(the last two terms: part of the impulse goes into spinning the body instead of moving it).
So the impulse that exactly stops the approach is `j = m_eff · (−v_rel · n)`.

With many contacts, fixing one disturbs the others. **Sequential impulses** (Catto,
"Iterative Dynamics with Temporal Coherence", GDC 2005) visits every contact many times
(16 iterations per step) and each time applies the change of impulse needed, while
clamping the **accumulated** impulse of each contact to `≥ 0` (contacts push, never
pull). This is projected Gauss–Seidel on the contact constraints: it converges to the
impulses that satisfy all of them at once.

Three additions make it behave:

| term | what it does | value |
|---|---|---|
| restitution | if the bodies approach faster than 60 cm/s, the target is `v_rel · n = −e · v_before` (they bounce back with e of their speed); `e` = the larger of the two bodies' | balls 0.45–0.6, walls 0.4 |
| Baumgarte bias | overlap left over (it always is: steps are discrete) is removed gradually: the target separating speed is `0.2 · (depth − slop) / dt` | slop 0.5 cm |
| friction | two tangent directions `t1`, `t2`; the same kind of impulse, but clamped to `±μ jn` (Coulomb's law: friction at most μ times the normal force), `μ = √(μa μb)` | 0.5 (balloons 0.3) |

**Warm starting.** The contacts of the previous step are kept. A new contact between the
same two bodies, within 1 cm of an old one and with nearly the same normal, starts with
the old contact's accumulated impulses (applied right away). A resting stack needs the
same impulses step after step, so the iterations only have to correct small changes. In
`test_physics` a stack of 3 boxes and a ball goes from sinking and drifting 1.4 cm in 3 s
(without warm starting, with the old corner-based contacts) to 0.0003 cm.

**Substeps.** `PhysWorld` runs the rigid bodies 4 times per physics step (`substeps`), at
1/240 s for the default 1/60 s step: stiffer contacts and no tunnelling for a steel ball
falling at 10 m/s (4 cm per substep, against a radius of 25–40 cm).

---

## 2. The air (`phys/fluid.c`)

### 2.1 Stable Fluids

The air is modelled as an **incompressible, inviscid fluid**: the Euler equations

```
∂u/∂t + (u · ∇) u = −∇p/ρ + f          (momentum: carried by itself, pushed by pressure and forces)
∇ · u = 0                              (incompressible: as much air leaves any region as enters)
```

are split into steps (Stam, "Stable Fluids", SIGGRAPH 1999), each easy on its own:

```
fluid_step(dt):
  1. forces         wind (u relaxes towards the wind velocity), buoyancy of the cloud,
                    vorticity confinement
  2. advection      u(x) ← u(x − dt u(x))                          semi-Lagrangian
  3. projection     solve ∇²p = ∇ · u, then u ← u − ∇p             incompressibility
  4. the cloud      ρ(x) ← ρ(x − dt u(x)), MacCormack-corrected     (section 3)
```

Between steps the **boundary conditions** are enforced: no air crosses a closed wall
(normal velocity 0), and the air in cells inside a body moves with the body.

**Semi-Lagrangian advection.** To know what arrives at a point `x`, follow the flow
backwards: the value now at `x` is the old value at `x − dt u(x)`, read with trilinear
interpolation. Whatever `dt`, the result is an interpolation of existing values, so it
can never blow up: this is the "stable" in Stable Fluids. The price is some numerical
blur (interpolation averages), which vorticity confinement and MacCormack counter.

### 2.2 The MAC grid

Velocities are stored **staggered** (Harlow & Welch 1965, "marker and cell"): the pressure
at the cell centers, and each velocity component on the faces it crosses: `u` on the x
faces, `v` on the y faces, `w` on the z faces. A grid of `nx · ny · nz` cells has
`(nx+1) ny nz` values of `u`, and so on (`fluid_iu`, `fluid_iv`, `fluid_iw`).

Why: with everything at the centers, the divergence `(u[x+1] − u[x−1]) / 2h` skips the
cell itself, so a checkerboard pattern of velocities has zero divergence on paper and the
projection cannot see it. The first version of this file did that, and after the
projection the flow still had a divergence of a third of its speed per cell. On the MAC
grid the divergence of a cell is exactly the flow through its six faces,

```
div = (u[x+1] − u[x] + v[y+1] − v[y] + w[z+1] − w[z]) / h
```

and the pressure gradient across a face is `(p[i] − p[j]) / h`: the discrete divergence of
the discrete gradient is the discrete Laplacian, exactly. After the projection the flow
is divergence-free up to the solver's tolerance (`test_physics`: < 10⁻⁴ of the speed).

`fluid_velocity(p)` interpolates each component from its own grid (offset by half a cell
on the two other axes).

### 2.3 Forces

- **Wind**: `u += (wind − u) · min(1, rate dt)`, the air drifts towards the wind velocity
  (`wind_rate` 0.3 /s). In the Cornell box, closed except for its front, a uniform wind
  cannot get through, so the projection turns it into swirls; it is meant for open
  domains (`fluid->open`, below).
- **Buoyancy**: `v += buoyancy · ρ(x) · dt` on the y faces: the cloud's air is warm and
  moist, lighter than the air around it, and rises (`-buoyancy`, 0 by default: the cloud
  stays where it floats).
- **Vorticity confinement** (Fedkiw, Stam & Jensen, "Visual Simulation of Smoke", 2001):
  the numerical blur kills small swirls. Compute the vorticity `ω = ∇ × u` (at the cell
  centers), the direction `N = ∇|ω| / |∇|ω||` towards stronger rotation, and push the air
  with `f = ε h (N × ω)`: it spins the swirls back up, at the scale of the grid.
  `ε = vorticity` (0.25).

### 2.4 Projection and boundary conditions

Subtracting the gradient of a pressure `p` from `u` must leave `∇ · (u − ∇p) = 0`, so
`∇²p = ∇ · u`: a **Poisson equation**, one unknown per cell. Per cell, with `h` the cell
size:

```
(Σ p_neighbors − k · p) / h² = div
```

What counts as a neighbor is the boundary condition:

| next to the cell | in the equation | meaning |
|---|---|---|
| a fluid cell | its `p` in the sum, counted in `k` | the pressure difference pushes the air |
| a closed wall, or a cell inside a body | left out of the sum and of `k` | **Neumann**: the face's velocity is given (0, or the body's), no pressure acts across it |
| an open face of the box | `p = 0` beyond it: counted in `k`, nothing in the sum | **Dirichlet**: open air at rest pressure; air can leave and enter |

`fluid->open` is a bit mask of open faces (`FLUID_OPEN_XLO` ... `FLUID_OPEN_ZHI`,
`FLUID_OPEN_SKY` = all but the floor). The Cornell box is open at the front
(`FLUID_OPEN_ZLO`, the camera side): air pushed by a falling ball, or by an explosion,
leaves there, and the cloud with it.

Then `u −= (p[i] − p[j]) / h` on every face between two fluid cells (or a fluid cell and
an open face); faces fixed by a wall or a body keep their value.

The pressure `p` here already includes `dt / ρ` (it is the impulse that makes the flow
divergence-free), so no `dt` appears.

### 2.5 Multigrid (`phys/poisson.c`)

Gauss–Seidel (replace each `p` by the value that satisfies its own equation, given the
neighbors) removes the error quickly at the scale of a cell, but the error at the scale of
the room shrinks by only about `1 − π²/n²` per sweep: a 48³ grid needs thousands of
sweeps. The first version used 60 sweeps of over-relaxed Gauss–Seidel; an explosion's
pressure then never spread through the room, and the air barely moved.

**Multigrid** fixes it. After a few sweeps the remaining error is smooth; a smooth error
is well represented on a grid with half as many cells per axis, where it is rough again
and a few sweeps reduce it again; recursively:

```
V-cycle(level):
    if coarsest (fewer than 8 cells on a side): 40 sweeps, done
    3 sweeps                               (pre-smoothing)
    r = rhs − L p                          (residual)
    rhs_coarse = mean of r over the 8 children of each coarse cell
    p_coarse = 0; V-cycle(coarser)
    p += p_coarse of the parent cell       (piecewise constant prolongation)
    3 sweeps                               (post-smoothing)
```

A coarse cell is solid only if all its children are; open faces stay open at every
level. One V-cycle costs about as much as 7 sweeps on the fine grid and cuts the error by
about 3× whatever the grid size (measured: max divergence 27 → 9.7 → 2.5 → 0.19 → 0.002
for 1, 2, 4, 8, 16 cycles on 48³, with a source of rate 30). Each step does 5 V-cycles,
starting from the previous step's pressure (a good guess, the flow changes little per
step). The 3+3 sweeps matter: with 2+2 the reduction was only about 1.7× per cycle.

All sweeps are **red-black**: cells are colored like a 3D checkerboard, and each half-sweep
updates one color from the other only. Cells of one color do not depend on each other, so
they are updated in parallel, and the result is the same for any number of threads.

### 2.6 Explosions: air that appears

A plain outward kick does nothing in an incompressible fluid: an outward flow from a point
has positive divergence, and the projection removes exactly that. An explosion is gas
that appears (it is not incompressible at all). `fluid_add_expansion(center, R, s, t)`
models it: for `t` seconds the projection solves for

```
∇ · u = s (1 − (r/R)²)      inside the sphere, 0 elsewhere
```

instead of 0. The flow leaving the sphere is `Q = ∫ div dV = s 4π R³ (2/15)`, so at a
distance `d` outside it the air moves outwards at `Q / (4π d²)` (in open air).
`test_physics` measures 1.05× this value at d = 180 on a 48³ grid with all faces open (the
5% is the open boundary being 278 cm away instead of at infinity). `phys_blast` uses a
core of half the blast radius, `s` chosen so that the air leaves the core at the given
speed, for 0.15 s.

---

## 3. The cloud in the air

### 3.1 MacCormack advection

The cloud's density lives in the `Volume` (128³ voxels over the whole room in the demos,
finer than the 48³ air cells: the air's motion is smooth, the cloud's edges are not).
Each step every voxel is advected through the air's velocity, interpolated at the voxel.

Plain semi-Lagrangian advection blurs a little every step, and after a hundred steps the
cloud's billows would be mush. **MacCormack** (Selle et al., "An Unconditionally Stable
MacCormack Method", 2008) estimates that error and removes it:

```
φ̂ = A(φ)                   one step backwards (semi-Lagrangian)
φ̃ = A⁻¹(φ̂)                 and forwards again: if A were exact, φ̃ = φ
φ_new = φ̂ + (φ − φ̃) / 2    remove half the round-trip error: second order accurate
clamp φ_new to [min, max] of the 8 voxels φ̂ was interpolated from
```

The clamp keeps it stable and never creates new extremes (no negative density, no
overshoot at sharp edges).

### 3.2 Compression

Density is a conserved quantity: `∂ρ/∂t + ∇ · (ρu) = 0`, which along the flow is
`dρ/dt = −ρ ∇ · u`. Where the air is incompressible (everywhere, normally) this term is 0
and advection alone is right. Inside an explosion's source it is not: the same cloud fills
more space, so after advection the density is multiplied by `exp(−div · dt)`. Without
this, an explosion inside the cloud inflated it into a dark ball filling the room (the
mass grew with the volume); with it, the cloud bursts into a thin shell, and its total
mass is kept.

### 3.3 Details

- Only the voxels the cloud can reach are processed: the box around the cloud
  (`Volume.active`) grown by the largest distance the air moves in `dt` (the backward pass
  over a box twice as far, since the correction reads it). For the demos this is 20–50% of
  the grid.
- Densities below 10⁻⁴ are set to 0, so that the active box does not slowly grow to the
  whole room with invisible traces.
- No cloud inside a body: voxels whose air cell is solid are emptied.
- `volume_update` at the end: the majorant grid of the path tracer (VOLUMES.md) follows the
  cloud.
- `dissipation` (0 by default) makes the cloud fade by that fraction per second.

---

## 4. Coupling (`phys/world.c`)

`phys_step(dt)` does, in this order:

```
1. the air's force on each body, from the air as it is now:
       drag      F = ½ ρ_air C_d A |Δu| Δu,   Δu = air velocity around the body − body velocity
       buoyancy  F = −ρ_air V g             (Archimedes: the weight of the air displaced)
2. rigid bodies: 4 substeps with these forces + gravity + contacts
3. bodies → air: cells whose center is inside a body become solid, with the velocity of
   the body's material there (v + ω × r)
4. the air: fluid_step(dt)
```

**The air around a body** is the mean of the air velocity at 14 points (6 axes, 8
diagonals) on a sphere just outside the cells the body covers, since inside it the air
is the body.

**Area** `A` is what the body shows to the flow: `πr²` for a sphere; for a box, each pair
of faces contributes its area times `|cos|` of its normal with the flow direction (the
exact area of a box's shadow). `C_d` = 0.47 for a sphere, 1.05 for a box.

**Stability of the drag.** The drag is explicit (computed once per step). For a very light
body in a fast flow it could overshoot (reverse the relative velocity within one step), so
its impulse is capped at what brings the body to the air's speed.

**What it gives.** A steel ball of radius 35 cm (1.4 tonnes) falling at 10 m/s barely
feels the air: drag / weight ≈ 10⁻³. A
balloon of density 0.00105 (helium plus a skin, slightly lighter than air) gets an upward
net force of `(ρ_air − ρ) V g`, rises at the speed where the drag balances it, about 1 m/s,
and follows every gust. In between, a wooden box (0.5–0.7) is pushed by an explosion's
wind but not by a ball's wake.

**Explosions** (`phys_blast(center, radius, speed, impulse)`): the air expands for 0.15 s
(section 2.6), and every body inside the radius gets the shock wave's push, an impulse of
`impulse · A · (1 − d/radius)` along the direction away from the center (capped at the
air's speed), plus a little spin.

**One-way vs two-way.** Bodies push the air through the solid cells (their exact
velocity); the air pushes bodies through drag and buoyancy. The momentum the drag takes
from the air is not removed from the air explicitly: the solid-cell boundary already makes
the air flow around the body. This "weak" coupling is the usual choice in graphics
(Bridson, *Fluid Simulation for Computer Graphics*, ch. 5): stable and cheap, at the cost
of not conserving momentum exactly between the two.

### 4.1 The scene

`phys_add_to_scene(pw, body, scene, material)` adds the body's primitives (a sphere, or a
box as 6 quads with `u × v` pointing outwards) and remembers the first one in `Body.prim`.
`phys_sync_scene` rewrites them at the bodies' current poses (`make_sphere`, `make_quad`),
in place. After that, the path tracer needs its acceleration structure rebuilt
(`scene_set_accel`: 0.1 ms for the ~100 primitives of a demo); the rasterizer reads the
primitives directly.

### 4.2 Using it in your own program

`examples/hello_physics.c` (`make bin/hello_physics`) is the whole thing in 50 lines: a
steel ball falling through a cloud, 30 path traced frames. The core of it:

```c
static Scene scene;
SceneOptions so = {.kind = SCENE_EMPTY};
scene_cornell_box(&scene, &so);                       // the room and the lamp

// the cloud: one grid over the whole room, so the air can carry it anywhere
Aabb room = {vec3(0, 0, 0), vec3(555, 555, 555)};
Volume *cloud = volume_create(96, 96, 96, room);
CloudParams cp = cloud_default_params();
volume_add_cloud(cloud, (Aabb){vec3(80, 220, 110), vec3(475, 440, 450)}, &cp);
scene_add_volume(&scene, cloud);

PhysWorld pw;
phys_init(&pw);
rigid_set_room(&pw.rigid, room);
pw.fluid = fluid_create(room, 555.0 / 40, cloud);     // 40^3 cells of air
int ball = rigid_add_sphere(&pw.rigid, vec3(278, 500, 278), 40, 7.8);   // steel
Material steel = {MAT_MIRROR, vec3(0.95, 0.93, 0.9), vec3(0, 0, 0), 1};
phys_add_to_scene(&pw, ball, &scene, scene_add_material(&scene, steel));

for (int f = 0; f < 30; f++){
    for (int k = 0; k < 2; k++) phys_step(&pw, 1.0 / 60);   // 30 frames per second
    phys_sync_scene(&pw, &scene);
    scene_set_accel(&scene, ACCEL_NONE, NULL);
    frame_render_rt(&scene, &cam, &img, 16, &tp, 0, 0, f);  // app/frame.h
    image_write_ppm(&img, path);
}
phys_free(&pw);          // the air, not the cloud: the scene owns it
scene_free(&scene);
```

Rigid bodies without air: leave `pw.fluid` NULL, or use `RigidWorld` alone
(`rigid_init`, `rigid_add_*`, `rigid_step`). Air without bodies: `fluid_create` and
`fluid_step` alone, with `fluid_add_impulse` (a gust) or `fluid_add_expansion` (a blast).

---

## 5. The demos (`app/simdemo.c`)

| | `-demo drop` (default) | `-demo blast` |
|---|---|---|
| the cloud | across the upper half of the room | lower, over the floor |
| bodies | `-balls` (5) and `-boxes` (3) appear under the ceiling one every 0.4 s, interleaved, and fall through the cloud: glass (2.5 g/cm³), steel (7.8, a mirror), plastic (1.0, orange/blue/yellow), wooden boxes (0.7, tumbling); `-balloons` (3) start on the floor at the back and rise | all of them at rest on the floor; balloons slightly heavier than air (0.0016) |
| event | | at t = 1 s an explosion in the lower part of the cloud (radius 400, air 1100 cm/s, impulse 50000) |
| what to watch | holes and wakes in the cloud behind each ball, the cloud dragged down; balloons drifting up through it | the cloud bursting into a shell, balloons and boxes thrown, steel balls rolling |

`bin/sim` writes each frame to `out/sim/frame_NNNN.ppm` (`-o folder`) and then
`out/sim.gif` (`-gif off` to skip). Each frame: physics for 1/fps seconds (steps of at most
1/60 s), sync, rebuild, path trace. With `-backend raster` the frames are rasterized and
the cloud raymarched (VOLUMES.md 7.1): the whole 120-frame animation takes about 40 s
instead of 20 minutes, mostly physics. Options: `bin/sim -h` and USAGE.md.

`bin/view -sim on` runs the same demo live, on a coarser grid (air 32³, cloud 96³, one
physics step of up to 1/30 s per frame). Every frame the simulation advances by the time
that passed (at most 1/20 s: a slow machine shows slow motion). In the default hybrid
mode the rasterizer and the raymarched cloud draw every frame (about 12 frames per second
on a 4-core laptop, physics included); Space pauses, and the path tracer then refines the
frozen moment. In `rt` mode (key M) the path tracer renders one pass per frame and the
window shows a running average of the last passes (each new one weighs 1/3). K sets off
an explosion, N drops a steel ball, T restarts.

---

## 6. Tests (`examples/test_physics.c`, `make test`)

Each test compares with an answer known in advance:

| test | expected | measured (float) |
|---|---|---|
| free fall, 1 s at 1/240 s | v = −981, y = y₀ − 490.5 (Euler's known offset g·dt·t/2 allowed) | −981.002, 507.5 (offset 2.0) |
| bounce, restitution 0.5, dropped from 400 cm | comes back to e² · 400 = 100 cm | 100.03 |
| momentum: 2 spheres + 2 rotated boxes colliding, no gravity | conserved | change < 10⁻⁶ of the total |
| stack: 3 boxes + a ball, 3 s | stays put, at rest | 0.0003 cm drift, fastest point 0.015 cm/s |
| a tilted box dropped | settles on a face (center at one of its half sizes) | 0.01 cm off |
| projection, closed box, strong radial kick | divergence-free | max \|div\|·h/\|u\| < 10⁻⁴ |
| steady wind 100 cm/s for 1 s, open at both ends | the cloud moves 100 cm, keeps its mass | 91 cm, mass ×1.005 |
| multigrid, expansion source, 10 V-cycles | divergence left / rate < 1% | 0.04% |
| expansion in open air | outward speed `s R³ (2/15) / d²` | 1.05× (up and sideways) |

The cloud moves 91 and not 100 cm in the wind test because the air needs a moment to reach
the wind's speed through the projection at the walls; the 10% tolerance covers it.

Also checked: the float and double builds, address + undefined-behaviour sanitizers on the
tests and on a whole blast simulation, and that every scene without physics renders
byte-identical images to before (the physics only adds code).

---

## 7. Performance

8 threads, demo settings (air 48³, cloud 128³):

| part | time per physics step (1/60 s) |
|---|---|
| rigid bodies, 4 substeps, 11 bodies | 0.1–0.5 ms |
| air: forces + vorticity | 3–5 ms |
| air: velocity advection | 8–12 ms |
| air: projection (5 V-cycles) | 15–25 ms |
| cloud: MacCormack on the active voxels | 25–35 ms |
| total | 55–75 ms (up to ~150 ms while an explosion spreads the cloud through the room) |

Rendering dominates offline: a 320×320 frame at 64 samples per pixel takes 9–14 s with
the cloud (more as the cloud spreads); the 120 frames of `make sim` took 20 minutes, of
which 75 s were physics. The rasterizer with the raymarched cloud takes about 70 ms per
320×320 frame (the light grid rebuilt every frame included). In the live viewer the
physics is the limit, which is why `view -sim on` defaults to `-fluidres 32 -cloudres 96`
(about 30 ms per step).

---

## 8. Limits and next steps

- **Shapes**: spheres and boxes. Convex meshes would need GJK/EPA (or SAT over the mesh's
  faces and edges) for detection; the solver is shape-agnostic already.
- **Broad phase**: all pairs (n² / 2), fine for tens of bodies. A uniform grid or
  sweep-and-prune for thousands.
- **Rolling**: a ball rolls on forever on a flat floor (no rolling resistance), as in most
  game engines; angular damping (0.1 /s) slowly stops it.
- **Two-way coupling** is weak (section 4); a monolithic solve (Batty, Bertails & Bridson
  2007, "A Fast Variational Framework for Accurate Solid-Fluid Coupling") would conserve
  momentum between bodies and air.
- **Solid cells** are marked at cell resolution: thin or small bodies (< 1 cell) do not
  block the air. Cut cells (fractional face areas) would fix it.
- **The cloud** is passive: it does not condense or evaporate. A temperature and a water
  vapour field with a condensation rule (Harris et al. 2003, "Simulation of Cloud Dynamics
  on Graphics Hardware") would make it grow and dissolve on its own.
- **Outdoor sky**: `fluid->open = FLUID_OPEN_SKY` already gives an open domain where a wind
  blows through; with a sky light and a larger domain it is the next demo.
- **Real-time view**: done, the rasterizer raymarches the cloud (VOLUMES.md 7.1). The
  physics step is now the slower half of a live frame; the MacCormack pass over the
  cloud's voxels is the place to optimize next (a sparse brick list instead of a box).

## References

- J. Stam, "Stable Fluids", SIGGRAPH 1999; "Real-Time Fluid Dynamics for Games", GDC 2003.
- R. Fedkiw, J. Stam, H. W. Jensen, "Visual Simulation of Smoke", SIGGRAPH 2001.
- A. Selle, R. Fedkiw, B. Kim, Y. Liu, J. Rossignac, "An Unconditionally Stable MacCormack
  Method", J. Sci. Comput. 2008.
- F. Harlow, J. E. Welch, "Numerical Calculation of Time-Dependent Viscous Incompressible
  Flow of Fluid with Free Surface", Phys. Fluids 1965 (the MAC grid).
- R. Bridson, *Fluid Simulation for Computer Graphics*, 2nd ed., CRC Press 2015.
- W. L. Briggs, V. E. Henson, S. F. McCormick, *A Multigrid Tutorial*, 2nd ed., SIAM 2000.
- E. Catto, "Iterative Dynamics with Temporal Coherence", GDC 2005; Box2D (contact
  clipping, warm starting).
- D. Baraff, "Physically Based Modeling: Rigid Body Simulation", SIGGRAPH course notes 2001.
- C. Ericson, *Real-Time Collision Detection*, Morgan Kaufmann 2004 (separating axes,
  closest points of segments).
