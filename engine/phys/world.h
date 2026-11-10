// world.h - physics: rigid bodies and the air (with its cloud) in one world, coupled both ways.
//
//   bodies -> air    every step the cells inside a body are marked solid and take the
//                    body's velocity, so the pressure projection makes the air flow around
//                    it: a falling ball pushes the cloud aside and leaves a wake behind it.
//   air -> bodies    drag, 1/2 rho_air C_d A |du| du, with du = air velocity around the body
//                    minus the body's velocity (A = the area the body shows to the flow), and
//                    buoyancy, rho_air V g upwards (Archimedes). Heavy bodies barely notice;
//                    a balloon (density close to the air's) is carried by the wind.
//
// The world also keeps a Scene in sync: phys_add_to_scene creates a sphere or 6 quads for a
// body, phys_sync_scene moves them to the body's current pose. After syncing, the path
// tracer needs its acceleration structure rebuilt (scene_set_accel); the rasterizer reads
// the primitives directly.
#ifndef LR_PHYS_WORLD_H
#define LR_PHYS_WORLD_H

#include "core/scene.h"
#include "phys/fluid.h"
#include "phys/rigid.h"

typedef struct {
    RigidWorld rigid;
    Fluid *fluid;            // the air, or NULL (bodies only). Owned: phys_free frees it.
    real air_density;        // g / cm^3 (0.0012, air at 20 C)
    real drag_sphere;        // drag coefficient of a sphere (0.47)
    real drag_box;           // of a box (1.05)
    int substeps;            // rigid-body steps per phys_step (stiffer contacts)
    real time;               // seconds simulated
    double ms_rigid, ms_fluid;   // timings of the last step
} PhysWorld;

// gravity, defaults, no fluid; then set pw->fluid = fluid_create(...) if wanted
void phys_init(PhysWorld *pw);
void phys_free(PhysWorld *pw);

// one step of dt seconds: drag and buoyancy -> rigid bodies (substeps) -> obstacles -> air
void phys_step(PhysWorld *pw, real dt);

// air velocity around a body: the mean over 14 points on a shell just outside it
Vec3 phys_air_velocity_around(const PhysWorld *pw, int body);
// the force the air puts on a body now (drag + buoyancy)
Vec3 phys_air_force(const PhysWorld *pw, int body);

// An explosion. For 0.15 s the air expands from the inner half of the sphere and rushes
// out of it at `speed` (fluid_add_expansion), carrying the cloud with it; and every
// body inside the sphere gets the shock wave's push: an impulse of
// `impulse` (g / (cm s), per cm^2 of the area it shows to the blast) times the same fade,
// so a heavy box slides while a balloon flies (never faster than the air itself).
void phys_blast(PhysWorld *pw, Vec3 center, real radius, real speed, real impulse);

// adds the primitives for body i to the scene (material = scene material index), remembers
// where they are (Body.prim) and returns the index of the first one
int phys_add_to_scene(PhysWorld *pw, int body, Scene *s, int material);
// moves every body's primitives to its current pose
void phys_sync_scene(const PhysWorld *pw, Scene *s);

#endif
