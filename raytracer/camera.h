// camera.h - a pinhole camera: one origin, a virtual screen 1 unit in front of it.
#ifndef CAMERA_H
#define CAMERA_H

#include "geometry.h"
#include "rng.h"

typedef struct {
    Vec3 origin;
    Vec3 upper_left;   // top-left corner of the virtual screen
    Vec3 horizontal;   // full screen width, pointing right
    Vec3 vertical;     // full screen height, pointing up
} Camera;

static inline Camera camera_make(Vec3 from, Vec3 at, Vec3 up, double vfov_deg, double aspect){
    double h = tan(vfov_deg * PI / 180.0 / 2.0);
    double view_h = 2.0 * h, view_w = aspect * view_h;

    Vec3 w = v_norm(v_sub(from, at));     // points BACKWARDS (camera looks down -w)
    Vec3 u = v_norm(v_cross(up, w));      // right
    Vec3 v = v_cross(w, u);               // up

    Camera c;
    c.origin = from;
    c.horizontal = v_scale(u, view_w);
    c.vertical = v_scale(v, view_h);
    c.upper_left = v_sub(v_add(v_sub(from, v_scale(c.horizontal, 0.5)), v_scale(c.vertical, 0.5)), w);
    return c;
}

// s in [0,1] left -> right, t in [0,1] top -> bottom
static inline Ray camera_ray(const Camera *c, double s, double t){
    Vec3 target = v_sub(v_add(c->upper_left, v_scale(c->horizontal, s)), v_scale(c->vertical, t));
    Ray r = {c->origin, v_norm(v_sub(target, c->origin))};
    return r;
}

#endif
