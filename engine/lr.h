// lr.h - little_renderer: a rendering engine written from scratch in C.
// Include this one header and link with liblr.a (see SDK.md):
//
//   cc -Iengine my_app.c build/float/liblr.a -lm -lX11 -fopenmp      (Linux)
//   cc -Iengine my_app.c build/float/liblr.a -lm -lgdi32 -fopenmp    (Windows, MSYS2)
//
// Layers (each one only uses the layers above it):
//   base/      math (`real`, vectors), random numbers, the clock, noise, the sampler
//   formats/   PNG, JPEG, Radiance .hdr and JSON, read and written by our own code
//   scene/     images, textures, camera, meshes, materials and the BRDF, the environment,
//              volumes (clouds), glTF models, the Scene         <- shared by every backend
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

#include "base/lrmath.h"
#include "base/clock.h"
#include "base/rng.h"
#include "base/noise.h"

#include "formats/png.h"
#include "formats/jpeg.h"
#include "formats/json.h"

#include "scene/image.h"
#include "scene/texture.h"
#include "scene/camera.h"
#include "scene/mesh.h"
#include "scene/scene.h"
#include "scene/gltf.h"
#include "scene/brdf.h"
#include "scene/volume.h"

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
