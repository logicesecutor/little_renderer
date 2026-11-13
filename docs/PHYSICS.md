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

Once per physics step (not per substep: 1.3), every pair of bodies whose bounding spheres
come within a margin of each other is tested, and every body against the six walls of the
room. Each pair that touches, or is closer than the margin, gives one or more **contacts**:
a point `p`, a normal `n` (from body b towards body a) and a depth (how far they overlap;
negative: the gap between them, a *speculative* contact, 1.3).

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

### 1.2b Mesh bodies (`phys/convex.c`)

Two more shapes come from a triangle mesh (the scene's `SceneMesh`, in its own space):

- **`BODY_HULL`**, a moving body: the mesh's **convex hull**. `rigid_geom_hull` takes the
  vertices furthest along 288 directions spread over the sphere (a Fibonacci spiral), keeps
  at most 48 of them (each next one the furthest from those kept), and builds their hull
  incrementally (add a point, remove the faces it sees, close the hole from their horizon
  to it). Coplanar triangles become one polygon face. Its **mass** comes from the hull
  split into tetrahedra from the centroid: volume, centre of mass, and the inertia tensor
  from each tetrahedron's covariance (`det/120 · (aaᵀ + bbᵀ + ccᵀ + (a+b+c)(a+b+c)ᵀ)`). The
  tensor's eigenvectors (Jacobi rotations) are the body's axes, so its inverse inertia
  stays diagonal, like a box's: the body frame has the centre of mass at the origin and
  the principal axes as axes, and `body_mesh_xform` gives the mesh's place for drawing.
- **`BODY_MESH`**, fixed: the mesh's **triangles** themselves (`rigid_geom_mesh`), in a
  BVH (median splits), a static collider of any shape: balls roll in its hollows, hulls
  land on it.

| pair | method |
|---|---|
| hull – wall | the hull's corners below the wall, at most 4 kept |
| sphere – hull | the face the centre is furthest outside of; inside: out by it; outside: the closest point of the faces facing the centre |
| hull – hull, hull – box | separating axis test over both hulls' faces and the pairs of edges that make a face of the Minkowski difference (Gregorius, GDC 2013: the two edges' arcs on the Gauss map cross), then face clipping as for boxes (the incident face against the reference face's sides; at most 4 points: the deepest, the furthest from it, the one spanning the most area, the one adding the most) |
| sphere – mesh | each triangle near the sphere: its closest point |
| box / hull – mesh | each triangle near the body (the BVH, with the body's box in the mesh's frame): the same separating axis test, the triangle two-sided (a mesh need not be closed); its edges against only the hull edges on the hull's outline seen along them; at most 4 contacts for the pair |

The air sees both as obstacles through an **occupancy grid**, 48³ bits over the shape's
box in its own frame: a cell centre inside the hull's planes, or, for a mesh, inside by
the parity of the crossings of three rays (along x, y and z: the majority decides, so a
mesh with small holes still has an inside). `body_contains` reads it, so do the GPU's
obstacle kernel (the bits uploaded once per shape) and the CPU's `mark_solids`.

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
| restitution | if the bodies approached faster than 60 cm/s when the step began, once they touch the target is `v_rel · n = −e · v_before` (they bounce back with e of that speed; once per contact); `e` = the larger of the two bodies' | balls 0.45–0.6, walls 0.4 |
| Baumgarte bias | overlap left over (it always is: steps are discrete) is removed gradually: the target separating speed is `0.2 · (depth − slop) / dt`, at most 300 cm/s | slop 0.5 cm |
| friction | two tangent directions `t1`, `t2`; the same kind of impulse, but clamped to `±μ jn` (Coulomb's law: friction at most μ times the normal force), `μ = √(μa μb)` | 0.5 (balloons 0.3) |

**Warm starting.** The contacts of the previous step are kept. A new contact between the
same two bodies, within 1 cm of an old one and with nearly the same normal, starts with
the old contact's accumulated impulses (applied right away). A resting stack needs the
same impulses step after step, so the iterations only have to correct small changes. In
`test_physics` a stack of 3 boxes and a ball goes from sinking and drifting 1.4 cm in 3 s
(without warm starting, with the old corner-based contacts) to 0.0003 cm.

**Substeps.** `PhysWorld` runs the solver 4 times per physics step (`substeps`, in
`rigid_steps`), at 1/240 s for the default 1/60 s step: stiffer contacts and no tunnelling
for a steel ball falling at 10 m/s (4 cm per substep, against a radius of 25–40 cm).

**Contacts found once per step** (Catto's "soft step", Box2D v3). Looking for contacts is
the costly part (the hulls' separating axes, the meshes' triangles), so it is done once per
step, and the 4 substeps follow the contacts instead of finding them again. Each contact
keeps its point in the frame of each body (`la`, `lb`; a wall's: the world point) and its
depth when found (`depth0`). In each substep both points are moved with their bodies, and

```
depth = depth0 − ((xa + Ra la) − (xb + Rb lb)) · n
```

the depth when found minus how far the two points have moved apart along the normal (the
normal stays as found). For the bodies' motion within one step (millimetres, a few degrees)
this is what a new search would find.

**Speculative contacts.** A pair that does not touch at the start of the step may touch
before its end: its contacts must be there from the start. So the pairs are tested with a
margin, how far they can close in the step:

```
margin = 0.1 + dt (closing speed + |ωa| ra + |ωb| rb),   at most 0.1 + (ra + rb) / 2
```

(the closing speed along the line between their centres, the spin of their outermost
points; `ra`, `rb` their bounding radii; against a wall the whole speed). A contact with a
gap (depth < 0) lets the bodies approach by at most the gap in the substep: its target is
`v_rel · n ≥ depth / h`, which is a pull of nothing until the gap would close, and exactly
what stops them at the surface when it would. The iterations skip the gaps that cannot
close in the substep even at twice the speed (most of a pile's), and a gap does not bounce
until it has closed: the bounce comes in the next substep (or the next step: a contact
stopped in the last substep remembers its speed, `vn0`), from the speed when the step
began. Bouncing a gap that is about to close, or bouncing from the speed after the solver
has stopped the bodies, made piles jump and blow up, as did an uncapped push out (a light
balloon squeezed into a wall by heavy balls was shot out ever faster).

### 1.4 The bodies on the GPU (`gpu/rigid_gpu.c`)

With a GPU (`view -sim on`, `sim`; `-gpurigid off` to keep them on the CPU) the bodies are
stepped there, all of them: balls, boxes, hulls, fixed meshes and the walls. The state stays
on the GPU; `phys_step` calls it through the hook `PhysWorld.rigid_step`, and the bodies
come back to the CPU a step late, asynchronously (the scene is drawn from that copy; offline
and in the tests every step waits for it, `gpu_rigid_set_wait`). A body changed on the CPU
(a new one, one pushed by `phys_blast`, one the program moved) has `Body.dirty` set and is
sent again.

Each step is a chain of compute passes, the contacts once (with margins, 1.3) and the rest
in each substep:

| pass | when | what it does |
|---|---|---|
| forces | every substep | gravity, the air's drag and buoyancy, damping |
| pairs | once per step | one thread per pair of bodies (n²/2) tests the bounding spheres (with the margin) and sorts the pair into three lists: simple (balls and boxes), hulls, body–mesh |
| contacts | once per step | the simple pairs one thread each (`rigid.c`'s tests); hull pairs one workgroup each, 64 threads sharing the separating-axis search (`convex.c`'s SAT and face clipping); the walls |
| meshes | once per step | for each body near a fixed mesh one workgroup walks the mesh's tree breadth first and lists the triangles near the body; then one thread per triangle runs the test, and the best 4 contacts of each pair are kept |
| contact lists | once per step | each body's contacts: count, prefix sum, fill |
| prepare | every substep | each contact's points where its bodies are now, its depth, masses and target (as the CPU's) |
| solve | every substep | the contacts' impulses, 16 iterations, warm started |
| move | every substep | integrate |
| wake, sleep | once per step | sleeping (1.5): which sleeping bodies wake (one thread per pair), and, at the end, how long each has been still and who falls asleep |

The lists' sizes are only known on the GPU, so their dispatches are indirect (a small
kernel writes the workgroup counts).

**Mass splitting.** The CPU's solver (1.3) visits the contacts one after another, each
seeing the impulses of the ones before: that is what makes it converge, and it is
sequential. A GPU solves all the contacts at once (Jacobi), and a plain Jacobi solve
overshoots: a box resting on 4 contacts gets 4 full corrections. Tonge, Benevolenski &
Voroshilov (2012, "Mass splitting for jitter-free parallel rigid body simulation") split
each body into as many copies as it has contacts, each with mass `m / k` (inertia `I / k`):
every contact pushes its own copy, then the body takes the average of its copies'
velocities. That is a sequential-impulse solve of a different but equivalent system, it
converges like 1.3 and does not depend on the order. Warm starting is the same as on the
CPU: each new contact looks for the old one of the same pair, within 1 cm, and starts from
its impulses.

### 1.5 Sleeping

A pile at rest still costs a full step: its contacts are found and solved every step, only
to push nothing anywhere. So a body at rest falls asleep: when it has been **still** (its
centre and its outermost point slower than 5 cm/s) for **0.5 s**, its velocity is set to 0
and it is no longer moved (`Body.sleeping`). To the bodies still awake it is a fixed body,
so they rest on it; pairs of two sleeping bodies, and their walls, are not tested at all.

Each body decides alone, as in ODE's auto-disable. Box2D and PhysX put a whole island (all
the bodies that touch, through each other) to sleep at once, when all of them are still;
in a pile of hulls some body always jitters a little, and that kept whole islands awake
(a test with the island rule, and with "a body and all it touches": 250 hulls, about 10%
asleep after 40 s). Alone, the still ones fall asleep, hold the others still as fixed
bodies, and the pile falls asleep from the bottom up (all 250 after 35 s).

A sleeping body **wakes** when:

- a body faster than 20 cm/s (four times the "still" speed, so that a jittering neighbour
  does not wake it) comes within the contact margin of it (1.3), before the contacts are
  looked for, so it takes part in the same step. The bodies it then moves wake their
  neighbours in turn, a step later each;
- the program changes it: `Body.dirty` (a new position or velocity), `body_apply_impulse`
  (explosions), or any velocity it was given;
- the force from outside on it (the air's) changes by more than a quarter of its weight.

A slower body that reaches a sleeping one stops against it, as against a wall.
`RigidWorld.sleep = 0` turns sleeping off (the tests of the solver do; `test_sleep` tests
it); so does `mesh_rain -sleep off`. On the GPU the same rules run in two passes (1.4): the
sleeping body keeps its mass aside and shows 1 / mass = 0, so every other pass treats it
as fixed without knowing about sleep.

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

Every body is drawn as an **instance** (`Body.inst`): `phys_add_to_scene(pw, body, scene,
material)` adds a ball as an instance of the unit sphere (`INST_SPHERE`) and a box as one of
the unit box (`INST_BOX`), placed by a transform that scales, turns and moves it
(`xform_axes`); a mesh body is an instance of its scene mesh (`phys_add_mesh_to_scene(pw,
body, scene, mesh, material)`). `phys_sync_scene` only moves the instances
(`scene_move_instance`): no triangle or primitive changes, on the GPU neither (SDK.md
section 3). The path tracer hits the unit shapes exactly (no tessellation), and its
acceleration structure is two levels: the scene's fixed primitives and each mesh keep
their trees, and only the small tree over the instances is rebuilt, by
`scene_update_instances` (0.02 ms for 600 bodies). The rasterizer draws the unit sphere
and box once uploaded, one draw per body.

`phys_add_to_scene_prims` is the old way: the body's primitives (a sphere, or a box as 6
quads) are added to the scene and `phys_sync_scene` rewrites them in place (`Body.prim`);
the scene's tree then has to be rebuilt every frame (`scene_set_accel`).

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
    scene_update_instances(&scene, NULL);              // the bodies are instances (4.1)
    frame_render_rt(&scene, &cam, &img, 16, &tp, 0, 0, f);  // app/frame.h
    image_write_ppm(&img, path);
}
phys_free(&pw);          // the air, not the cloud: the scene owns it
scene_free(&scene);
```

Mesh bodies (a mesh loaded in its own frame, as for instances):

```c
int mesh = scene_load_mesh_object(&scene, "models/torus.obj", mat, 120);   // the scene's copy
const SceneMesh *sm = &scene.meshes[mesh];
int hull = rigid_geom_hull(&pw.rigid, sm->tris, sm->ntris, 0);           // its convex hull
int b = rigid_add_hull_body(&pw.rigid, hull, xform_rot_y(vec3(278, 300, 278), 30, 1), 0.8);
phys_add_mesh_to_scene(&pw, b, &scene, mesh, -1);                          // drawn as an instance

int big = scene_load_mesh_object(&scene, "models/dragon_recon/dragon_vrip_res2.ply", mat, 260);
int tris = rigid_geom_mesh(&pw.rigid, scene.meshes[big].tris, scene.meshes[big].ntris);
int c = rigid_add_mesh_collider(&pw.rigid, tris, xform_rot_y(vec3(278, 0, 250), 0, 1));
phys_add_mesh_to_scene(&pw, c, &scene, big, -1);                           // fixed, never moves
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

**Meshes** in either demo: `-collider on` stands the model (`-obj`) in the middle of the room,
260 units, fixed; `-meshes n` adds n copies of it, 120 units, as convex hulls (density
0.8): in the drop demo they fall one every 0.4 s, turned at random; in the blast they rest
on the floor until it throws them. For example
`bin/sim -obj models/dragon_recon/dragon_vrip_res2.ply -collider on -meshes 4`: dragons
falling through the cloud and landing around a big one, the air flowing around all of them.
Rigid bodies take 0.2 ms per 1/60 s step on average for that (2 ms at most, when hulls
land on the 200k-triangle dragon), the GPU air 0.6 ms.

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
| the hull of a box mesh, turned and moved in its own frame | the box's volume, centre and principal moments | exact (< 10⁻⁴) |
| that hull dropped tilted, then a second one on it | lands on a face; the second rests on it | 0.01 cm off; 0.4 cm drift |
| a torus's hull dropped tilted | lands flat (centre at the tube radius 20) | 20.5 (the 48-corner hull) |
| a ball, a hull and a box on a fixed mesh (a block, top at 100) | rest on it | 0.03 cm off each |
| the inside of a torus mesh as the air sees it (48³ cells) | 2π²Rr² | 0.988× |
| two hulls colliding, no gravity | momentum conserved | change < 10⁻⁵ |

The cloud moves 91 and not 100 cm in the wind test because the air needs a moment to reach
the wind's speed through the projection at the walls; the 10% tolerance covers it.

`examples/test_gpu_rigid.c` (in `make test`; skipped by a build without the GPU) runs the
same kind of checks on the GPU's bodies: free fall, the bounce, momentum, the stack of 3
boxes and a ball, the tilted box, a hull landing on a face and a second one stacked on it,
the torus puck, a ball, a hull and a box resting on a fixed mesh, and a pile of 60 bodies
(the same on the CPU and the GPU: nothing leaves the room, nothing overlaps, everything at
rest). Then it times 64 to 2048 bodies on both (section 7).

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

**The bodies on the GPU** (1.4; RTX 3070, a 1/60 s step of 4 substeps, a pile of balls,
boxes and hulls; `test_gpu_rigid`):

| bodies | CPU (one thread) | GPU |
|---|---|---|
| 64 | 0.67 ms | 0.67 ms |
| 256 | 2.9 ms | 0.80 ms |
| 1024 | 20.3 ms | 1.07 ms |
| 2048 | | 1.60 ms |

(Since the contacts are found once per step, 1.3: the GPU 1.6–1.8 times faster than when
they were found in every substep, 1.73 → 1.07 ms at 1024 bodies; the CPU about 8% slower,
18.8 → 20.3 ms: its balls' and boxes' tests are cheap, and the speculative contacts give
the solver more contacts. With hulls the gain is larger: 1500 hulls in `mesh_rain`, 8.9 →
4.7 ms per step on the GPU, their contacts 7.1 → 2.6 ms.)

**Sleeping** (1.5) takes a settled pile off the CPU almost entirely: 1024 balls and boxes at
rest, 22.7 → 3.5 ms per step (907 asleep). On the GPU it removes the hulls' contacts (600
hulls in `mesh_rain`: 2.5 → 0.9 ms per step on average), but not the fixed cost of the
passes, which are launched for the room for contacts: the settled 1024 balls and boxes take
1.07 ms either way (PERFORMANCE.md P10).

Below about 64 bodies the GPU's fixed cost (some 10 dispatches per substep) is larger than
the work; it still takes the bodies off the CPU, which the live viewer needs: with 600
bodies `view -sim on` draws 164 frames per second (the screen's limit) with the bodies on
the GPU, 142 with them on the CPU (19 ms of physics per frame). The dragons
(`-collider on -meshes 3`) take 1.3 ms per step on the GPU, most of it the hulls' contacts
(0.35 ms) and the triangles near the big dragon (0.23 ms). Set `LR_GPU_TIMING=1` to print
the time of every pass at the end (natively; WEB.md 3).

---

## 8. Limits and next steps

- **Shapes**: spheres, boxes, convex hulls of meshes, and fixed triangle meshes. A moving
  concave body (a mug that holds a ball) would need a convex decomposition (V-HACD) into
  several hulls of one body; two fixed meshes never collide with each other.
- **Mesh contacts**: a body sliding over a mesh can catch on the triangles' inner edges
  (each triangle is tested on its own); "internal edge" filtering would smooth it.
- **Broad phase**: all pairs (n² / 2), fine for tens of bodies on the CPU; on the GPU one
  thread per pair is still cheap at 2000 bodies (section 7). A uniform grid or
  sweep-and-prune for tens of thousands.
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
  clipping, warm starting); "Solver2D", 2024, and Box2D v3 (the soft step: contacts found
  once per step, followed through the substeps; speculative contacts; the push-out cap).
- D. Baraff, "Physically Based Modeling: Rigid Body Simulation", SIGGRAPH course notes 2001.
- R. Tonge, F. Benevolenski, A. Voroshilov, "Mass Splitting for Jitter-Free Parallel Rigid
  Body Simulation", SIGGRAPH 2012.
- C. Ericson, *Real-Time Collision Detection*, Morgan Kaufmann 2004 (separating axes,
  closest points of segments).
