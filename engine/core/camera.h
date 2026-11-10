// camera.h - rays and a pinhole camera. Header only.
// The path tracer shoots camera_ray()s; the rasterizer builds the matching view and
// projection matrices from the same from/at/up/fov (see render_params_camera).
#ifndef LR_CAMERA_H
#define LR_CAMERA_H

#include "core/lrmath.h"

typedef struct {
    Vec3 o;   // origin
    Vec3 d;   // direction (not necessarily unit length)
} Ray;

typedef struct {
    Vec3 origin;
    Vec3 upper_left;   // top-left corner of the virtual screen, 1 unit in front of the eye
    Vec3 horizontal;   // full screen width, pointing right
    Vec3 vertical;     // full screen height, pointing up
} Camera;

static inline Camera camera_make(Vec3 from, Vec3 at, Vec3 up, real vfov_deg, real aspect){
    real h = r_tan(vfov_deg * PI / 180.0 / 2.0);
    real view_h = 2.0 * h, view_w = aspect * view_h;

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
static inline Ray camera_ray(const Camera *c, real s, real t){
    Vec3 target = v_sub(v_add(c->upper_left, v_scale(c->horizontal, s)), v_scale(c->vertical, t));
    Ray r = {c->origin, v_norm(v_sub(target, c->origin))};
    return r;
}

// The reverse of camera_ray: where on the screen is the world point p? Returns 0 if p is
// behind the camera, else 1 with (s, t) (inside the image when both are in [0, 1)) and the
// cosine between the view axis and the direction to p. Used by light tracing to find the
// pixel a light path lands in.
static inline int camera_project(const Camera *c, Vec3 p, real *s, real *t, real *cos_axis){
    Vec3 center = v_add(c->upper_left, v_sub(v_scale(c->horizontal, 0.5), v_scale(c->vertical, 0.5)));
    Vec3 fw = v_sub(center, c->origin);                     // unit: the screen is 1 unit away
    Vec3 q = v_sub(p, c->origin);
    real z = v_dot(q, fw);
    if (z <= 0) return 0;
    Vec3 e = v_sub(v_scale(q, 1.0 / z), fw);                // offset from the screen center
    *s = 0.5 + v_dot(e, c->horizontal) / v_dot(c->horizontal, c->horizontal);
    *t = 0.5 - v_dot(e, c->vertical) / v_dot(c->vertical, c->vertical);
    *cos_axis = z / v_len(q);
    return 1;
}

// area of the virtual screen (1 unit in front of the eye)
static inline real camera_screen_area(const Camera *c){
    return v_len(c->horizontal) * v_len(c->vertical);
}

#endif
