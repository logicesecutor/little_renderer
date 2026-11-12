// medium.h - path tracer: light travelling through a participating medium (core/volume.h).
//
// Inside a cloud a ray does not go straight to the next surface: at every point it may be
// scattered (sent in a new direction) or absorbed, with a probability that depends on the
// local density. These functions answer the three questions a path tracer asks:
// All of them walk the volume's majorant grid brick by brick (3D DDA), so thin regions are
// crossed in long steps and empty bricks are skipped.
//   where does the ray scatter?           medium_sample_distance  (delta tracking)
//   how much light gets through a segment? medium_transmittance    (ratio tracking)
//   in which direction does it scatter?    medium_sample_phase     (Henyey-Greenstein)
// All three are unbiased: on average they give the exact answer, whatever the density.
#ifndef LR_MEDIUM_H
#define LR_MEDIUM_H

#include "core/camera.h"
#include "core/rng.h"
#include "core/scene.h"
#include "core/volume.h"

// Follows the ray (its direction must have unit length) from tmin to tmax. Returns 1 and
// the distance t of a scattering-or-absorption event ("real collision") if one happens
// before tmax, 0 if the ray gets through.
int medium_sample_distance(const Volume *v, const Ray *ray, real tmin, real tmax, Rng *rng, real *t);

// Fraction of light that crosses the medium from tmin to tmax along the ray (unit
// direction): exp(-integral of sigma_t * density), estimated without bias.
real medium_transmittance(const Volume *v, const Ray *ray, real tmin, real tmax, Rng *rng);

// All media of a scene at once (they may overlap: extinctions add). The first real
// collision of any medium before tmax: returns 1, its distance and which medium.
int scene_medium_collision(const Scene *s, const Ray *ray, real tmin, real tmax, Rng *rng, real *t, int *which);
// product of the transmittances of all media of the scene
real scene_medium_transmittance(const Scene *s, const Ray *ray, real tmin, real tmax, Rng *rng);

// A new direction after scattering, for a ray that was travelling along d (unit length),
// drawn with probability density phase_hg(dot(d, new), g).
Vec3 medium_sample_phase(Vec3 d, real g, Rng *rng);
Vec3 medium_sample_phase_u(Vec3 d, real g, real xi, real u_phi);   // the same from 2 given numbers

#endif
