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
// The world also keeps a Scene in sync: phys_add_to_scene makes a body an instance of a unit
// sphere or box (mesh bodies: of their mesh), phys_sync_scene moves the instances to the
// bodies' current poses. After syncing, the path tracer needs its top level over the
// instances rebuilt (scene_update_instances, microseconds); the rasterizer reads them directly.
#ifndef LR_PHYS_WORLD_H
#define LR_PHYS_WORLD_H

#include "core/scene.h"
#include "phys/fluid.h"
#include "phys/rigid.h"

typedef struct PhysWorld {
    RigidWorld rigid;
    Fluid *fluid;            // the air, or NULL (bodies only). Owned: phys_free frees it.
    real air_density;        // g / cm^3 (0.0012, air at 20 C)
    real drag_sphere;        // drag coefficient of a sphere (0.47)
    real drag_box;           // of a box (1.05)
    int substeps;            // rigid-body steps per phys_step (stiffer contacts)
    real time;               // seconds simulated
    double ms_rigid, ms_fluid;   // timings of the last step
    // The air can be stepped elsewhere, e.g. on the GPU (gpu/fluid_gpu.h). When air_step is
    // set, phys_step calls it instead of marking the obstacles and stepping pw->fluid on the
    // CPU, and the drag reads air_around: the air velocity around each body (what
    // phys_air_velocity_around computes on the CPU), kept up to date by whoever steps the air.
    void (*air_step)(void *user, struct PhysWorld *pw, real dt);
    void *air_user;
    // The rigid bodies can be stepped elsewhere too (gpu/rigid_gpu.h): phys_step then calls
    // rigid_step with the air's force on each body (held for the substeps); it returns 0 when
    // it cannot take this world (the CPU steps it). The bodies' state comes back from it
    // later (a step or so); a body changed on the CPU must get Body.dirty = 1.
    int (*rigid_step)(void *user, struct PhysWorld *pw, real dt, int substeps, const Vec3 *forces);
    void *rigid_user;
    Vec3 *air_around;
    int nair_around;
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

// adds body i (a ball or a box) to the scene as an instance of the unit sphere or box
// (material = scene material index), remembers it (Body.inst) and returns it (-1 for a mesh
// body). Moving it then changes only its transform: the world's structure stays as it is
// (scene_update_instances), on the GPU only a record per body moves.
int phys_add_to_scene(PhysWorld *pw, int body, Scene *s, int material);
// the old way: a sphere primitive or 6 quads (Body.prim), rewritten by phys_sync_scene
// (then the world's structure must be built again: scene_set_accel)
int phys_add_to_scene_prims(PhysWorld *pw, int body, Scene *s, int material);
// a mesh body (BODY_HULL, BODY_MESH) is drawn as an instance of the scene's mesh `mesh`
// (the one its shape was made from; material -1: the mesh's own). Returns the instance.
int phys_add_mesh_to_scene(PhysWorld *pw, int body, Scene *s, int mesh, int material);
// moves every body's primitives (and instances) to its current pose
void phys_sync_scene(const PhysWorld *pw, Scene *s);

#endif
