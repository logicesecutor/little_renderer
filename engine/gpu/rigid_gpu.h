// rigid_gpu.h - the rigid bodies of phys/rigid.h, stepped on the GPU.
//
// Balls, boxes, convex hulls, fixed meshes and the room's walls, entirely in compute shaders:
// the air's forces, contacts (every pair of bodies by their bounding spheres, then the same
// tests as rigid.c and convex.c: separating axes and face clipping, a fixed mesh's tree of
// triangles, the walls), and the contact solver. The CPU's solver visits the contacts one after another (sequential
// impulses); a GPU solves them all at once, so each body's mass is split among its contacts
// (Tonge, Benevolenski & Voroshilov 2012, "Mass splitting for jitter-free parallel rigid body
// simulation"): every contact pushes its share of the body, and the body takes the sum, which
// converges like the sequential solver, independent of the order. Impulses are warm started
// from the step before, matched contact by contact, as on the CPU.
//
// The state stays on the GPU; the bodies come back to the CPU a step late (asynchronously),
// which is what phys_sync_scene and the CPU's drag use. A body changed on the CPU (new, pushed
// by phys_blast, moved by the program) has Body.dirty set and is sent again.
//
// Usage: once gpu_state(g) == GPU_READY, gr = gpu_rigid_create(g, &pw): from then on
// phys_step(&pw, dt) steps the bodies on the GPU (through PhysWorld.rigid_step).
#ifndef LR_RIGID_GPU_H
#define LR_RIGID_GPU_H

#include "gpu/gpu.h"
#include "phys/world.h"

typedef struct GpuRigid GpuRigid;

GpuRigid *gpu_rigid_create(Gpu *g, PhysWorld *pw);
// Offline (bin/sim, tests): each step waits for the bodies to come back, so the CPU's copy
// is never behind (slower, but the same result whatever the GPU's speed)
void gpu_rigid_set_wait(GpuRigid *gr, int wait);
// the bodies back to the CPU now (waits for the GPU)
void gpu_rigid_sync(GpuRigid *gr);
// gives the bodies back to the CPU and frees the GPU's copy
void gpu_rigid_destroy(GpuRigid *gr);

#endif
