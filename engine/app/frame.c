#include "app/frame.h"

#include "base/rng.h"

void frame_render_rt(const Scene *s, const Camera *cam, Image *img, int spp, const TraceParams *tp,
                     real clamp, double caustics_photons, uint64_t seed){
    const int W = img->w, H = img->h;
    trace_image(s, cam, tp, W, H, spp, seed, clamp, img->px, NULL, NULL);
    if (caustics_photons > 0)
        light_trace_caustics(s, cam, W, H, (long long)((double)W * H * spp * caustics_photons), tp,
                             0x5EED0F11A57ULL ^ (seed * 0x9E3779B97F4A7C15ULL), img->px);
}
