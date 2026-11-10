// lr.h - little_renderer: a rendering engine written from scratch in C.
// Include this one header and link with liblr.a (see SDK.md):
//
//   cc -Iengine my_app.c build/float/liblr.a -lm -lX11 -fopenmp
//
// Layers (each one only uses the layers above it):
//   core/      math, random numbers, images, camera, meshes, scenes,
//              noise and volumes (clouds)                             <- shared by everything
//   raster/    the rasterizer: triangles -> pixels with a z-buffer, clouds by raymarching
//   rt/        the path tracer: rays, intersections, octree / BVH, light transport,
//              participating media
//   phys/      the physics engine: rigid bodies, the air (stable fluids) carrying the
//              cloud, and the coupling between them
//   platform/  the window (X11)
//   app/       command-line options of the bundled programs
//
// Precision: `real` is float by default (fast); build with `make DOUBLE=1` (-DLR_DOUBLE)
// for double. An app must be compiled with the same setting as the library it links.
#ifndef LR_H
#define LR_H

#include "core/lrmath.h"
#include "core/rng.h"
#include "core/image.h"
#include "core/texture.h"
#include "core/camera.h"
#include "core/mesh.h"
#include "core/scene.h"
#include "core/noise.h"
#include "core/volume.h"

#include "raster/raster.h"
#include "raster/render.h"
#include "raster/volume_march.h"

#include "rt/intersect.h"
#include "rt/octree.h"
#include "rt/bvh.h"
#include "rt/accel.h"
#include "rt/medium.h"
#include "rt/tracer.h"

#include "phys/rigid.h"
#include "phys/fluid.h"
#include "phys/world.h"

#include "platform/window.h"

#endif
