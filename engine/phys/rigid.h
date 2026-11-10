// rigid.h - physics: rigid bodies (spheres and boxes) falling, colliding and resting.
//
// Units are the scene's: the Cornell box is 555 units, think of them as centimetres, so
// gravity is 981 units / s^2 and densities are in g / cm^3 (water = 1).
//
// One step (rigid_step):
//   1. velocities += (gravity + external forces / mass) * dt   (semi-implicit Euler)
//   2. find the contacts (body-body and body-wall)
//   3. sequential impulses: push the bodies apart at every contact, again and again
//      (iterations), so that they stop approaching; friction and bounce included
//   4. positions += velocity * dt, orientations += angular velocity * dt
#ifndef LR_RIGID_H
#define LR_RIGID_H

#include "core/lrmath.h"
#include "core/scene.h"   // Aabb

typedef enum { BODY_SPHERE, BODY_BOX } BodyShape;

typedef struct {
    BodyShape shape;
    Vec3 half;               // box: half the size per axis; sphere: radius in all three
    real mass, inv_mass;     // inv_mass = 0: a fixed body that nothing moves
    real volume;             // for buoyancy in air
    Vec3 inv_inertia;        // inverse inertia tensor in the body's own frame (diagonal)
    Vec3 pos, vel;           // center of mass and its velocity (world)
    Quat rot;                // orientation (body frame -> world)
    Vec3 angvel;             // angular velocity (world), radians / s
    real restitution;        // 0 = no bounce .. 1 = perfectly elastic
    real friction;           // Coulomb coefficient
    Vec3 force, torque;      // external forces for the next step (cleared by rigid_step)
    int material;            // scene material used to draw it
    int prim;                // first scene primitive that shows it (see phys/world.h), -1 = none
} Body;

typedef struct {
    int a, b;                // bodies; b = -1 for a wall of the room
    Vec3 p, n;               // contact point; normal, pointing from b towards a
    real depth;              // how far they overlap
    // solver state
    Vec3 ra, rb, t1, t2;
    real mass_n, mass_t1, mass_t2, bias, jn, jt1, jt2;
} Contact;

typedef struct {
    Body *bodies;
    int nbodies, cap;
    Vec3 gravity;
    Aabb room;               // all bodies stay inside this box (6 walls)
    int has_room;
    real wall_restitution, wall_friction;
    int iterations;          // impulse iterations per step (more = stiffer stacks)
    real linear_damping, angular_damping;   // fraction of velocity lost per second
    Contact *contacts;       // of the last step
    int ncontacts, cap_contacts;
    Contact *prev;           // of the step before: their impulses "warm start" the solver
    int nprev, cap_prev;
    int warm_start;          // 1 (default): start each contact from last step's impulse
} RigidWorld;

void rigid_init(RigidWorld *w);   // gravity 981 down, no room, 16 iterations
void rigid_free(RigidWorld *w);
void rigid_set_room(RigidWorld *w, Aabb room);

// density in g / cm^3 (scene units: cm). Return the body's index.
int rigid_add_sphere(RigidWorld *w, Vec3 center, real radius, real density);
int rigid_add_box(RigidWorld *w, Vec3 center, Vec3 half, Quat rot, real density);

// world-frame inverse inertia R * I^-1 * R^T
Mat3 body_inv_inertia_world(const Body *b);
// velocity of the body's material at world point p (v + omega x r)
Vec3 body_point_velocity(const Body *b, Vec3 p);
// is world point p inside the body?
int body_contains(const Body *b, Vec3 p);
// apply an impulse (momentum change) at world point p
void body_apply_impulse(Body *b, Vec3 impulse, Vec3 p);

void rigid_step(RigidWorld *w, real dt);

// diagnostics: total kinetic energy and momentum
real rigid_kinetic_energy(const RigidWorld *w);
Vec3 rigid_momentum(const RigidWorld *w);

#endif
