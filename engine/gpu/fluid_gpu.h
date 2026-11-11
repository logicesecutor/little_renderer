// fluid_gpu.h - the air of phys/fluid.h (and the cloud it carries), stepped on the GPU.
//
// The same Stable Fluids step as fluid_step, kernel for kernel, as WebGPU compute shaders:
// obstacles from the rigid bodies, walls, wind, buoyancy, vorticity confinement,
// semi-Lagrangian advection, the pressure projection (the same multigrid V-cycles: red-black
// Gauss-Seidel, restriction, prolongation) and the MacCormack advection of the cloud. The
// state stays on the GPU: the GPU view (gpu/view_gpu.h) draws the cloud straight from there,
// and the only thing read back every step is the air velocity around each body (14 samples,
// the drag of phys/world.c), one step late. The CPU's copy of the cloud is refreshed only when
// asked (gpu_fluid_sync_cloud), for the CPU renderers.
//
// Usage: once gpu_state(g) == GPU_READY, gf = gpu_fluid_create(g, &pw): from then on
// phys_step(&pw, dt) steps the air on the GPU (through PhysWorld.air_step). Builds without a
// GPU backend return NULL, and the air stays on the CPU.
#ifndef LR_FLUID_GPU_H
#define LR_FLUID_GPU_H

#include "gpu/gpu.h"
#include "gpu/view_gpu.h"
#include "phys/world.h"

typedef struct GpuFluid GpuFluid;

// takes over pw->fluid, starting from its current state; NULL if there is no fluid or GPU
GpuFluid *gpu_fluid_create(Gpu *g, PhysWorld *pw);
// v draws the cloud from the GPU's densities (no copy through the CPU)
void gpu_fluid_attach_view(GpuFluid *gf, GpuView *v);
// asks for the cloud's densities back in the CPU's Volume (asynchronously: they arrive a
// frame or so later, then volume_update runs), for the CPU renderers
void gpu_fluid_sync_cloud(GpuFluid *gf);
// Offline (bin/sim): when the GPU falls behind, wait for it instead of folding the step's time
// into the next one, so the result does not depend on how fast the GPU is. Not in a browser.
void gpu_fluid_set_wait(GpuFluid *gf, int wait);
// steps submitted to the GPU and not finished yet (beyond 2, phys_step waits for the GPU:
// the step's time is added to the next one)
int gpu_fluid_pending(const GpuFluid *gf);
// gives the air back to the CPU (pw->fluid keeps the velocities it had when the GPU took
// over, and the cloud as of the last sync) and frees the GPU's copy
void gpu_fluid_destroy(GpuFluid *gf);

#endif
