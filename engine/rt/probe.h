// probe.h - light probes: what light arrives at one point of the scene from every direction,
// path traced into an environment map (scene/env.h), for the rasterizers (docs/PBR.md, PBR4).
//
// A rasterizer knows only the lamp; the probe brings it the rest: the room's walls and what
// stands in it, lit, for reflections and for the light that bounces around. The lamp seen
// directly is left out of the map (the rasterizers light with the lamp themselves), its light
// reflected by anything is in. A probe of a room keeps the room's box, so lookups from other
// points are corrected for where they are (env_box_dir). It is made when asked: things that
// move later are seen where they were then.
#ifndef LR_PROBE_H
#define LR_PROBE_H

#include "scene/env.h"
#include "scene/scene.h"

#define PROBE_RES 256        // default map width (128 high) in a room
#define PROBE_RES_OPEN 512   // ... under a sky (it shows the sky in mirrors)
#define PROBE_SPP 32         // default paths per texel
#define PROBE_DEPTH 8        // bounces per path

// The probe at `center`, w x w / 2 texels, spp paths each (random directions inside the
// texel); box != NULL: lookups corrected by this box. The scene needs its acceleration
// structure (scene_set_accel). Runs in parallel (OpenMP). NULL if out of memory.
Env *probe_bake(const Scene *s, Vec3 center, const Aabb *box, int w, int spp);

// The scene's probe: where the scene says (Scene.probe_at, probe_box), else, with a lamp, at
// the centre of the box around its quads but the lamp - or, if that point is inside
// something (a model standing in the middle of the room), above it; NULL for a scene with neither a lamp
// nor a sky, or neither a place nor quads. w = 0: PROBE_RES, or PROBE_RES_OPEN without a lamp.
Env *probe_bake_scene(const Scene *s, int w, int spp);

// The scene's irradiance grid (scene/env.h IrrGrid): over the probe's box (a room), or
// around the probe (an open scene: 1400 x 400 x 1400 units), 5 x 5 x 5 (5 x 3 x 5 open)
// probes of `paths` paths each, those inside something marked unused. NULL where
// probe_bake_scene gives none.
IrrGrid *probe_grid_bake_scene(const Scene *s, int paths);
#define PROBE_GRID_PATHS 2048
#define PROBE_GRID_CLAMP 4.0f   // the largest radiance a grid probe's sample counts

#endif
