// simdemo.h - the physics demos of the bundled programs (bin/sim and bin/view -sim on):
// the Cornell box, a cloud the air can carry anywhere in the room, bodies that appear on a
// schedule, and the explosion of the blast demo. Apps only: the engine's physics API is
// phys/world.h (see docs/PHYSICS.md for using it directly).
#ifndef LR_SIMDEMO_H
#define LR_SIMDEMO_H

#include "app/options.h"
#include "core/rng.h"
#include "core/scene.h"
#include "phys/world.h"

#define SIM_MAX_SPAWNS 64

// a body that enters the simulation at time t
typedef struct {
    double t;
    BodyShape shape;
    Vec3 pos, half, vel;
    Quat rot;
    real density, restitution, friction;
    int material;
} Spawn;

typedef struct {
    PhysWorld pw;
    Volume *cloud;           // the room-wide grid the air carries (owned by the scene)
    Spawn spawns[SIM_MAX_SPAWNS];
    int nspawns, next_spawn;
    double blast_t;          // time of the scheduled explosion, < 0 = none
    Vec3 blast_at;
    real blast_radius, blast_speed, blast_impulse;
    int mat_drop;            // material of balls dropped with sim_drop_ball
    Rng rng;
    int steps;               // physics steps so far
    double max_dt;           // longest physics step (1/60 s; the viewer uses 1/30)
} Sim;

// Builds the room (o->scene.fog is honoured), the cloud, the air and the schedule of
// bodies for o->sim.demo; applies the camera overrides and the thread count. The path
// tracer's acceleration structure is NOT built (sim_step changes the scene every step).
int sim_setup(Sim *sim, const Options *o, Scene *s);
// advance dt seconds (in physics steps of at most max_dt), add the bodies that are due,
// fire the explosion when due, and move the scene's primitives to the bodies
void sim_step(Sim *sim, Scene *s, double dt);
// interactive extras (the viewer's keys)
void sim_drop_ball(Sim *sim, Scene *s);
void sim_blast_now(Sim *sim);
void sim_free(Sim *sim);

// one line about the state, e.g. "t 1.23 s | 9 bodies | air 12 ms"
void sim_status(const Sim *sim, char *buf, int len);

#endif
