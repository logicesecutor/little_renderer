#include "app/frame.h"

#include "core/rng.h"

void frame_render_rt(const Scene *s, const Camera *cam, Image *img, int spp, const TraceParams *tp,
                     real clamp, double caustics_photons, uint64_t seed){
    const int W = img->w, H = img->h;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int y = 0; y < H; y++){
        for (int x = 0; x < W; x++){
            uint64_t pixel = (uint64_t)y * W + x;
            Rng rng;
            rng_seed(&rng, 0x853c49e6748fea9bULL ^ (pixel * 0x9E3779B97F4A7C15ULL) ^ (seed * 0xD1B54A32D192ED03ULL), pixel);
            Vec3 sum = vec3(0, 0, 0);
            for (int k = 0; k < spp; k++){
                real u = (x + rng_real(&rng)) / W, v = (y + rng_real(&rng)) / H;
                Vec3 c = trace_path_ex(s, camera_ray(cam, u, v), &rng, tp);
                if (!(c.x == c.x && c.y == c.y && c.z == c.z)) continue;   // skip NaNs
                if (clamp > 0 && v_max(c) > clamp) c = v_scale(c, clamp / v_max(c));
                sum = v_add(sum, c);
            }
            img->px[pixel] = v_scale(sum, 1.0f / spp);
        }
    }
    if (caustics_photons > 0)
        light_trace_caustics(s, cam, W, H, (long long)((double)W * H * spp * caustics_photons), tp,
                             0x5EED0F11A57ULL ^ (seed * 0x9E3779B97F4A7C15ULL), img->px);
}
