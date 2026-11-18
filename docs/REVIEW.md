# Review of `main` (37b805c, 2026-11-18): findings and proposed changes

A proposal only: **no code was changed**. Every item has an id, so the ones to apply can be
picked by id. The line numbers are those of `main` at 37b805c.

**Since then** (in `development`, from main 30ab6ee, 2026-11-18): wgpu-native was removed (the
native GPU is our Vulkan layer alone: docs/STATUS.md 1.8; the "wgpu" and `VK=1` builds of
section 0 are now one, `make`) and the performance review's speedups went in
(docs/PERF_REVIEW.md: `rigid.c`, `convex.c`, `env.c`, `mesh.c`, `bvh.c`, `lrmath.h`,
`denoise.wgsl`, `gpu.c`, `vk_surface.c`, `wgsl.c` changed), so some line numbers have moved; each
item is checked against the code as it is before it is applied. The fixes are made in
`development`. Fixed so far: **B4** (every kind of build in folders of its own: STATUS.md
step 15).

- Categories: **bug**, **dead** (dead code), **useless**, **dup** (duplication), **smell**, **design**.
- Status of a bug: **CONFIRMED** (reproduced by a run, or proven by reading every path / by
  `git`, `grep`, `make -q`) or **SUSPECTED** (the reasoning holds, but it was not reproduced).
- Risk: what applying the change could break. Effort: S (under an hour), M (an afternoon), L (more).

## 0. What was done, and the state found

| check | result |
|---|---|
| `make -j8` (wgpu), `make VK=1 -j8`, `make DOUBLE=1 -j8`, `make web` | all clean: no warning, no error |
| `make check-layers` | layers ok |
| `make test`, `make VK=1 test` | all 14 tests pass on both (RTX 3070) |
| `gcc -fsyntax-only` on every C file with `-Wshadow -Wstrict-prototypes -Wmissing-prototypes -Wundef -Wcast-qual -Wformat=2 -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wnull-dereference -Wunused-macros -Wvla -Wimplicit-fallthrough` | 16 warnings in all: 9 `-Wcast-qual` (callbacks, `gpu_state`), 6 `-Wshadow` (S10), 1 `-Wduplicated-branches` (spirv.c:981, harmless) |
| `clang --analyze` on engine (CPU build), apps, examples, tests | 20 reports, all read: false positives (gltf.c `g_free(&gg)`, texture.c:154, rigid.c:595/1134) or harmless (dead stores jpeg.c:536, bench.c:274) |
| dead code: every external symbol of `liblr.a` against all sources and docs; every WGSL function, shader and header macro | section 2 |
| duplicates: every `static` function name defined in more than one file, then compared | section 4 |
| read in full | options.c, view.c, render.c, sim.c, gpu.c, gpu_internal.h, view_gpu.c, png.c, hdr.c, json.c, mesh.c, image.c, omp_web.c, hud.c, gpu_frame.c, frame.c, window*.c, the Makefile, tools/*.mjs, wgsl_embed.c, vk_check.py, frames_to_gif.py, shell.html |
| read in part (the risky paths) | gltf.c, jpeg.c, texture.c, scene.c, accel.c, rigid.c, simdemo.c, bench.c, raster_gpu.c, trace_gpu.c, denoise_gpu.c, fluid_gpu.c, vk_core.c, the shaders (by search) |
| not reviewed line by line | webgpu/wgsl/spirv.c and front.c, convex.c, fluid.c, poisson.c, render.c (raster), tracer.c, bvh.c, env.c, volume*.c, most WGSL bodies, tools/gen_*.py |

The general state is good: the tree builds without warnings in four configurations, the
untrusted-input code (PNG, JPEG, HDR, JSON, glTF) checks its sizes carefully, and the tools
find almost nothing. The real bugs below are few and mostly at the edges (options, tools, the
web build, failure paths); the larger gains are in duplication and in a few oversized files.

## 1. Bugs, by severity

| id | sev. | status | where | what is wrong | proposed change | risk | effort |
|---|---|---|---|---|---|---|---|
| **B1** | medium | CONFIRMED (reading, `git log -S`) | options.c:209-259 (no default), :310; USAGE.md:331; WEB.md 2.6 | `-photoncache` is documented as **on** by default (USAGE's table; WEB.md 2.6 describes the cache as what the viewer shows while moving), but `options_parse` starts from `memset 0` and never sets it: it is **off** unless asked for. It is also the only option missing from `view -h`. | Decide which is meant. If the docs are: `o->photon_cache = prog == PROG_VIEW;` and a line in `options_usage`. If the code is: correct USAGE.md and WEB.md. | The first changes what the viewer shows and costs while moving with caustics (suite group I may move). | S |
| **B2** | medium | CONFIRMED (reading; needs a failing pipeline to trigger) | gpu_frame.c:57-70; bench.c:200-205; view_gpu.c:948-952; view.c:529 | When the path tracer's pipelines fail (`gpu_trace_ready` < 0), `gpu_view_trace_more` sets `trace_failed` and returns without a result. `gpu_frame_trace` (render, sim) and bench's `trace_until` loop `while (passes < spp)` and so **spin for ever**; the viewer, when still, stays on the rasterized image at 0 spp and never falls back to the CPU path tracer (it only does at the next camera move). | `gpu_view_trace_more` returns 0 on failure (or add `gpu_view_trace_failed`); the two loops return 1 (render and sim then fall back to the CPU as they already do for other failures); view.c sets `o.gpu_trace = 0`. | low | S |
| **B3** | low | CONFIRMED | options.c:109, :111 against volume.c:121, :123 | `-h` prints wrong defaults: `-density (0.05)` is 0.08, `-g (0.6)` is 0.3 (USAGE.md has the right ones). The help text carries every default as a hand-typed copy. | Print the defaults from `cloud_default_params()`, `march_default_params()`, `accel_default_params()` and the `Options` just filled (`%g`), or at least correct the two. See S1 for the general cure. | none | S |
| **B4** | low | CONFIRMED (`make -q SAN=1` and `make -q SINGLE=1` say up to date after `make`) | Makefile:20-25, :72 | `SAN=1` and `SINGLE=1` build into the same `build/float-gpu` and `bin` as a plain build: after `make`, `make SAN=1` does nothing, and after a partial rebuild the library mixes objects of both kinds. | Give them their own folders as `DOUBLE` and `VK` have (`BUILD`/`BIN` suffix `-san`, `-single`), or a stamp file of the flags that every object depends on. | new output folders (USAGE.md to follow) | S |
| **B5** | low | CONFIRMED (reading) | view.c:1134 | `if (o.sim.on && o.preview == 4) preview = 2;` uses the default value as "not given": an explicit `-sim on -preview 4` is silently turned into 2. | A `preview_set` flag in `options_parse` (as `march_scale_set`, `fluidres_set`), or 0 = automatic. | none | S |
| **B6** | low | CONFIRMED (both reproduced with `curl` against the server) | tools/serve.mjs:21-23 | (a) `file.startsWith(root)` also accepts a sibling folder whose name begins the same: `GET /../build2/secret.txt` was served from `web/build2`. (b) `decodeURIComponent` throws on a malformed escape (`GET /%`): the uncaught `URIError` ends the server. | (a) `file !== root && !file.startsWith(root + sep)` gives 403; (b) `try`/`catch` gives 400. | none (it listens on 127.0.0.1 only, hence low) | S |
| **B7** | low | CONFIRMED (`render -fov 0`, `-fov 180`, `-fov -40`, `-probespp 0`, `-instances -5` all exit 0) | options.c:400-427 | Several numeric options are not checked while their neighbours are: `-fov` (0, 180 and negative give a degenerate camera), `-probespp 0` (a probe of no paths), `-proberes`, `-instances`, `-history`, `-simhz`, `-speed`, `-uvscale`, `-skyscale`, `-threads` (negative), and `-bodies` only with `-sim on`. `atoi`/`atof` also take `12x` as 12. | Range checks beside the existing ones; a small `parse_int`/`parse_real` on `strtol`/`strtod` that rejects trailing text. | a script that passed an out-of-range value now gets an error | S |
| **B8** | low | CONFIRMED (reading) | gltf.c:223-227, :394 | The `indices` accessor's `componentType` and `type` are not checked. `acc_index` reads 4 bytes for anything that is not 5121 or 5123, so a (malformed) file with indices of type byte, short or float reads up to 3 bytes past the range `acc_get` checked: garbage indices, and a heap over-read when the buffer ends there. | After `acc_get`: refuse the primitive unless `idx.ctype` is 5121, 5123 or 5125 and `idx.comps == 1`. | none | S |
| **B9** | low, web only | CONFIRMED (arithmetic, reading) | png.c:681, :728; hdr.c:104, :108; jpeg.c:591, :605-611 | The decoders bound each side (PNG 2^24, HDR and JPEG 65535/65536) but not the product: with a 32-bit `size_t` (the WebAssembly build) `(size_t)w * h * 4` wraps, a short buffer is allocated and the loops write past it. On 64 bits the allocation simply fails. The page only reads its own preloaded files, hence low. | One shared limit on pixels (for example `(uint64_t)w * h <= 1 << 28`) checked in the three decoders before allocating. | none | S |
| **B10** | low | SUSPECTED (not run: needs a very large cloud or an adapter with small limits) | view_gpu.c:437-439; gpu.c:465-472 | The media's density buffer is never compared with `g->max_storage` (the meshes are: raster_gpu.c:327, trace_gpu.c:891), so a cloud too large for the GPU gives validation errors instead of the CPU's march. And `gpu_buffer_reserve` asks for `size + size / 4 + 1024`: a buffer that fits the limit exactly is asked 25% over it. | Clamp `want` to `max_storage` when `size` itself fits; in `upload_volumes` report and fall back when `nd * 4` does not fit. | low | S-M |
| **B11** | low | SUSPECTED (not reproduced: the window was not driven) | view_gpu.c:659-668 | When `wgpuSurfaceGetCurrentTexture` does not succeed (Outdated, Lost, Timeout: a minimized window, a display change) the code prints once and draws nothing; the surface is never configured again, so the window may stay frozen, and a texture returned with a non-success status is not released. | On Outdated/Lost configure the surface again (`configure_surface`, exported from gpu.c) and skip the frame; release `st.texture` when it is not NULL. | low | S-M |
| **B12** | low | SUSPECTED (reading) | omp_web.c:176-182 | `make_team`: if a `pthread_create` fails and a later one succeeds, `T.made = i + 1` counts the missing thread; a region then waits at its barrier for a thread that does not exist. | Stop at the first failure. | none | S |
| **B13** | low | SUSPECTED (Linux, not run) | window.c:56-58 (no `FocusChangeMask`) | X11 has no equivalent of Win32's `WM_KILLFOCUS` and the page's `blur`: a movement key held while the window loses focus stays down (its release goes to another window). | `FocusChangeMask`, and on `FocusOut` the same list of releases (shared with D10). | none | S |
| **B14** | low | SUSPECTED (reading) | gpu.c:345-362, :366-374 | `gpu_wait` hands the addresses of its stack variables `done`/`done2` to the queue's callbacks, and leaves the loop early when the device is lost: a callback delivered later writes into a dead stack slot. | Flags inside `Gpu` (or a generation counter the callback checks). | none | S |
| **B15** | low, latent | CONFIRMED (reading) | vk_core.c:851-861 | `submit_ex` silently drops command buffers beyond 16 in one `wgpuQueueSubmit` (not submitted, not marked, never freed). The engine submits one at a time, so nothing hits it today. | `vk_error(... Validation ...)` when `n` exceeds the room, or grow `s.cbs`. | none | S |
| **B16** | doc | CONFIRMED (counted) | README.md:46, :190, :211; Makefile:6 | "76 WGSL shaders": there are 79. `apps/` is described as `render`, `view`, `sim` in the README (table and diagram) and in the Makefile's header: `bench` is the fourth. `LR_VK_DUMP` (vk_pipe.c) is in no doc. | Correct the three places; one line for `LR_VK_DUMP` in USAGE.md. | none | S |

## 2. Dead code (each proven by search)

| id | where | proof | proposed change | risk | effort |
|---|---|---|---|---|---|
| **DC1** | shaders/rigid/rigid_convex.wgsl:144-220 (`fn collide_convex`, 77 lines) | Its name occurs once in all 79 shaders and in no C string: nothing calls it. The full test now lives in `rigid_heavy.wgsl` and `feature_contacts`. | Remove it (and anything only it uses: check `closest_seg`, `face_loop`, `incident_face` keep other callers; they do today). | none at run time; `make test` and `make VK=1 test` compile every rigid kernel | S |
| **DC2** | options.c:260, :381, :423 (`backend_set`) | Set at :381, then only `(void)backend_set;`. | Remove the variable and the line. | none | S |
| **DC3** | omp_web.c:372, :374 (`if0_was`) | A thread-local that is only named in `(void)if0_was;`. | Remove. | none | S |
| **DC4** | gpu.c:30, :71 and gpu.h:58 (`gpu_timing`); gpu.c:13, :67 (`gpu_rt_offered`) | No caller in engine, apps, examples or tests (only their two definitions each). | Remove `gpu_timing`; either print `gpu_rt_offered` in the viewer's start-up line or remove it too (WEB.md 2.10 mentions it). | none | S |
| **DC5** | tools/vk_check.py:24 (`first`), :40 (`prefix`) | Assigned / passed, never read. | Remove. | none | S |
| **DC6** | .gitignore:64 (`/out/bench_*.ppm`), :1-55 | Nothing writes `.ppm` any more and `out/*` is ignored two lines below; the first 55 lines are a generic C template (kernel modules, `.ilk`, `.pdb`, ...). | Drop the `.ppm` line; optionally trim the template to what this tree makes. | none | S |
| **DC7** | .vscode/tasks.json:19-39 | "C/C++: gcc build active file" compiles one file alone with `/usr/bin/gcc`: no file of this tree builds that way. | Remove the task (keep the `make` one). | none | S |
| **DC8** | scene.c:296; sim.c:151-152; Makefile:~283 | An orphan comment ("rotate around the vertical (y) axis" above `mesh_material`); the same comment line twice; the comment of the `bench` target ("all acceleration structures on the torus") now sits above the perf suite's block, 17 lines before its target. | Remove / move. | none | S |
| **DC9** | sampler.h (`sampler_hash`, `sampler_owen`, `sampler_sobol1`), env.h (`env_prepare`), scene.h (`scene_mesh_material`), vk_impl.h (`vk_barrier`, `vk_buffer_address`, `vk_device_lost`, `vk_texture_ready`) | Each is used only inside the file that defines it and is not in SDK.md. | Make them `static` (and drop the declarations). | none within the tree; an outside program using them would break | S |

Not dead, but worth knowing: these documented API functions (SDK.md) have **no caller at all**
in the tree, not even a test: `convex_collide`, `fluid_kinetic_energy`, `rigid_kinetic_energy`,
`poisson_residual`, `irr_grid_irradiance`, `jpeg_read`, `phys_add_to_scene_prims`,
`phys_air_force`, `render_scene_light`, `scene_add_triangle`, `texmap_load_png`. Keep them as
API, but each deserves one line in a test (or to go, if it is not wanted).

## 3. Useless work

| id | where | what | proposed change | risk | effort |
|---|---|---|---|---|---|
| **U1** | render.c:152 (after options.c:477) | `render` on the GPU builds the scene's BVH twice: `options_setup_scene` builds the chosen structure, then `scene_set_accel(scene, ACCEL_BVH, ...)` frees it and builds it again (`scene_set_accel` has no early return, accel.c:175-188). On the dragon that is the whole build once more. | `if (scene_accel_kind(scene) != ACCEL_BVH)` as sim.c:118 does. | none | S |
| **U2** | Makefile:407 | `$(PYTHON) python -m zipfile ...` is `python3 python -m zipfile` when `PYTHON` is `python3`: it always fails and only the next alternative saves it. | `$(if $(findstring uv,$(PYTHON)),$(PYTHON) python,python3) -m zipfile`. | none | S |
| **U3** | Makefile:342 | `-sDEFAULT_TO_CXX` sets a setting to the value it already has (SUSPECTED: Emscripten's default is 1). | Remove after one `make web` to confirm. | none | S |
| **U4** | gpu.c:224 | `static WGPUFeatureName feats[2]`: `static` is not needed (the descriptor is consumed inside `wgpuAdapterRequestDevice`) and shares the array between GPUs. | A local array. | none | S |

## 4. Duplicated code

| id | where | what repeats | proposed change | risk | effort |
|---|---|---|---|---|---|
| **D1** | render.c:65-98; app/frame.c:5-33; view.c:128-155; bench.c:~258-272, ~950-962 | The CPU path tracer's image loop (seed per pixel, `sampler_make`, `sampler_2d`, `camera_ray`, `trace_path_sampled`, NaN skipped, clamp, sum) five times. render.c's and frame.c's differ only by a seed and a progress line. | One function in `rt/` (rows `y0..y1`, sample index range, into sums and counts) used by all five; render keeps its progress through a callback or by calling it band by band. | the images must stay bit-identical: the 64 reference images check it | M |
| **D2** | gltf.c:38-54; texture.c:190-203; jpeg.c:629-645; png.c:~805-812; hdr.c:125-140 | "The whole file into a growing buffer" five times (four the same loop, one with `ftell`). | `file_read_all(path, &n)` in `base/`. | none | S |
| **D3** | octree.c:9 and scene.c:96 (`grow`, identical); mesh.c:9 (`PUSH`); bvh.c:44, :350; scene.c:86; rigid.c:153, :413; render.c:316 and :218 (`xmalloc`); gltf.c (`vec_push`) | Growing arrays by hand, about ten times, each with its own `perror`/`exit`. | One `base/array.h` (`lr_grow`), with the out-of-memory policy of S5 in one place. | low | M |
| **D4** | image.c:94 and texture.c:80 (`to_srgb8`, identical); texture.c:70 and volume_march.c:280-287 (the 256-entry sRGB to linear table, twice); raster_shader.wgsl:369, view_composite.wgsl:33-38, tex.wgsl:5 | The sRGB transfer function, three times in C and three in WGSL. | texture.c calls `image_srgb8`; volume_march.c uses `texmap_srgb_table()`; the WGSL through D14's common file. | none (same formulas) | S |
| **D5** | png.c:15-31, :441-457; image.c:120-133; texture.c:~60-76, ~90-104; sampler.c:~130-140 | Six copies of "make the table on first use": `if (!ready)` / `#pragma omp critical` / `if (!ready)` / fill / `#pragma omp flush` / `ready = 1`. The first, unlocked read of `ready` is a data race by the letter of C11 (harmless on x86 and wasm). | Either constant tables generated at build time (as `brdf_tables.h`; the CRC table is 256 words), or one `LR_ONCE(flag, name) { ... }` macro in `base/`. | none | S-M |
| **D6** | gpu.c:91-96, :619-624; view_gpu.c:~178-184; fluid_gpu.c and rigid_gpu.c (`is_live`, `live_slot`, and `on_done`, `storage` as well) | Five "who is still alive" lists for late callbacks, each a fixed array with its own search; fluid_gpu.c and rigid_gpu.c share four helpers by copy. | `gpu_live_add/remove/has(kind, ptr)` in gpu.c; `storage` and `on_done` into gpu_internal.h. | low | S |
| **D7** | view_gpu.c:245-259; raster_gpu.c:188-190; brdf_gpu.c:36-41; fluid_gpu.c:228-262; trace_gpu.c:338-365 | Shader sources joined with `malloc(strlen + ...)`, `strcpy`, `strcat` and a hand-counted size, five times. trace_gpu.c also patches the text (`strstr` for `var stack...` at :351 and `fn traverse(` at :356) and silently does nothing if the shader's text has changed. | `char *gpu_wgsl_join(const char *const *parts, int n)`; the two patches fail loudly when their text is not found (or become `#ifdef`-like markers in the WGSL). | low | S |
| **D8** | gpu.c:808-818 and :880-889; :507-524 and :565-586; :79, :87, :545 with view_gpu.c:~531, ~1040; :724, :734, :747, :764, :772; view_gpu.c:192 | Inside the GPU helpers: the entry loop of `gpu_bind` and `gpu_bind_cached_buffers`; the pipeline descriptor of `gpu_compute` and `gpu_compute_async`; the `WGPUStringView` length expression for `%.*s` five times; `for (k < 400 && t->pending == 2) gpu_wait` five times; `make_layout` in view_gpu.c is an older `gpu_layout`. | Small static helpers (`fill_entries`, `compute_desc`, `gpu_sv_len`, `timer_settle`); view_gpu.c uses `gpu_layout`. | none | S |
| **D9** | raster_gpu.c:403-417; trace_gpu.c:1384-1393 (and `hw_records`) | "Send only the runs of instance records changed since version v, joining gaps under REC_GAP" in each backend. | An iterator over changed runs in `scene/` (`scene_changed_runs(s, since, gap, &first, &last)`). | low | S-M |
| **D10** | window_win32.c:25-31 and window_web.c:25-31 (`push`, the ring of 512 events); the list of keys released on `WM_KILLFOCUS` / `on_blur` | Identical code in the two platform files (and missing in the third: B13). | A private `platform/window_common.h` with the queue and `release_all_keys`. | none | S |
| **D11** | view.c:475 and :511, gpu_frame.c:52 (`GpuTraceParams` built field by field); view.c:483, :523, :822 (`last_rp` and its mode); :654-660 and :1023-1028 ("this view" command); :1014-1043 (the autoquit report, twice); sim.c:~165-172 and bench.c:~636-644 (quoting a command for `cmd.exe`) | Small repeats in the programs. | Four local helpers (`trace_params()`, `raster_params()`, `print_view_command()`, `autoquit_report()`), and `app_run_command()` in `engine/app`. | none | S |
| **D12** | tests: `check()` in 12 files (three signatures); `torus_tris`, `test_sleep`, `max_speed` in test_physics.c and test_gpu_rigid.c; `read_back`, `on_map` in test_gpu_basic.c and test_gpu_rt.c; `world_free` | Test helpers copied from test to test. | `tests/test_util.h` (static inline). | none | S |
| **D13** | examples/mesh_rain.c and app/simdemo.c:20, :45 (`rnd`, `random_rot`) | Copied; examples are meant to stand alone, so this one is a matter of taste. | Leave, or `quat_random(rng)` in `phys/`. | none | S |
| **D14** | WGSL: `hash32` x3 (denoise, trace, trace_resolve), `luma` x3, `tid` x4, `cidx` x2, `pc_key`/`pc_axis` x2, `onb` x2, `normal_to_world` x2, the sRGB functions x3 | Helpers repeated across shader files. | `common/util.wgsl`, prepended as `types.wgsl` already is (with D7's join). | each kernel's source grows a little; the web's compile time should be measured | S-M |

## 5. Smells

| id | where | what | proposed change | effort |
|---|---|---|---|---|
| **S1** | options.c:269-397, :17-166 | One chain of ~130 `strcmp` branches, and a help text that repeats every name and default by hand (B1 and B3 come from that). `-collider` (:359-363) spells on/off itself although `on_off` exists; `-aa`, `-sampler`, `-caustics`, `-gpu` could use `pick`. | A table `{name, kind, offset, default, programs, help}`: the parser and `-h` both read it. At least: `on_off` and `pick` everywhere. | M (table), S (helpers) |
| **S2** | options.c:498-545 | `options_print_command`: `-s 512` is fixed (:539), `mats[]` has a fourth entry repeating "diffuse" (:499), `-accel`, `-caustics`, `-photons`, `-bvh*` are not printed, paths with spaces are not quoted. | Print what differs from the defaults (falls out of S1's table); quote paths. | S |
| **S3** | view.c:233-349, :557-1045 | About 55 file-scope variables and one `frame()` of 490 lines doing input, camera, physics, four render paths, presentation, title and autoquit. | A `Viewer` struct and five functions (`viewer_input`, `viewer_update`, `viewer_render`, `viewer_present`, `viewer_report`). | M |
| **S4** | sim.c:101-140 | `err` is three things (-1 not drawn, 0 done, 1 failed) across two chains with empty `if (err >= 0){ }` branches. | `frame_on_gpu()` / `frame_on_cpu()` returning a small enum. | S |
| **S5** | 36 `exit(1)` in the library (scene.c 8, bvh.c 4, rigid.c 4, fluid.c 4, octree.c 3, render.c 3, ...) | Out of memory ends the process from inside library code in some files and returns NULL or -1 in others (formats, gltf, gpu). | One policy: a single `lr_oom()` (a hook the program can replace) used by D3's array, or error returns throughout. | M |
| **S6** | 22 `LR_*` variables read by `getenv` in 9 files | Each with its own caching idiom (`static int x = -1`, `-2`, or none: render.c:1046 reads `LR_RASTER_AMBIENT` on every frame). | `lr_env_int(name, def)` / `lr_env_on(name)` in `base/`, read once; one table of them in USAGE.md. | S |
| **S7** | gpu.c:420-424 | `gpu_state(const Gpu *)` polls the device (and casts the `const` away to do it). | Take `Gpu *`, or leave polling to `gpu_poll`. | S |
| **S8** | gpu.c:479-482, :809-812; view_gpu.c:192-194 | `gpu_layout` and `gpu_bind` stop silently at 24 and 16 entries, `make_layout` at 8. | An error message (or assert) when a layout is longer. | S |
| **S9** | gpu_internal.h:210-237 | Five `gpu_trace_*` functions take the same four buffers (`frame_u, dens, bricks, vstat`) one by one. | A `GpuMedia` struct passed once. | S |
| **S10** | rigid.c:497, :524, :537, ~:584; :502 | A `Placed` (two in `collide_convex`) is `malloc`ed and freed for every pair on every step; the parameter `m` is shadowed at :502 (hence the extra `mg`). Other `-Wshadow` hits: trace_gpu.c:1111, tracer.c:430, view.c:474 and :503 (the global `max_depth`), test_gpu_rigid.c:178. | A scratch `Placed` per thread (also a little speed: to be measured by the perf suite, F group); rename the shadows and add `-Wshadow` to CFLAGS. | S |
| **S11** | omp_web.c:334 | Every named `critical` uses the same mutex: two different critical sections nested would deadlock (none nest today). | A comment, or a small table of mutexes by name. | S |
| **S12** | tools/webbench.mjs:63 | The debugging port is fixed (9334): two runs at once (a perf run and `make web-check`) collide. | A random port as `-serve` already takes (:51), or `--remote-debugging-port=0` and read `DevToolsActivePort`. | S |
| **S13** | web/shell.html:53 | The page's log grows without end with `textContent +=` (each line copies the whole text). | Keep the last N lines (append a text node, drop the oldest). | S |
| **S14** | bench.c:~650 with view.c:551 | The viewer's stats file is read with `%63s %lf` in a loop that stops at the first value that is not a number (`mode rt`): any key written after it would be lost without a word. | Read line by line; or write `mode` last by rule and say so. | S |
| **S15** | Makefile:73, :107-113, :366-372 | Every object depends on every header (any header edit rebuilds everything); the web targets do not depend on the preloaded models nor on the flags (`DOUBLE`, `NOSIMD`, `THREADS`): switching them leaves a stale page. | `-MMD -MP` and `-include $(OBJ:.o=.d)`; a flags stamp for the web targets (with B4's). | S-M |
| **S16** | hud.c:49-63 | If `hud_draw`'s `malloc` fails the text is drawn anyway and can never be taken off the framebuffer. | Do not draw when the copy cannot be kept. | S |
| **S17** | window.c (`window_refresh_hz`) | X11 always answers 60: the path tracer's budget per frame is wrong on a faster display (known, noted in window.h). | XRandR when it is there (a later step). | M |
| **S18** | mesh.c:105-108, :70-90 | A face with more than 64 corners is cut short silently; a failed `realloc` in `read_line` ends the file silently. | A message for each. | S |
| **S19** | json.c:150-155 | Numbers go through `strtod`, which also takes `0x10`, `-inf`, `-nan`: not JSON. | Check the characters of the number first. | S |

## 6. Design observations (larger than a line)

| id | observation | proposal |
|---|---|---|
| **A1** | **Oversized files.** spirv.c 2259 lines, trace_gpu.c 2032, rigid.c 1548, view_gpu.c 1178, render.c 1155, view.c 1148, bench.c 1142. trace_gpu.c holds the scene's upload, two tree builders (LBVH, PLOC), the hardware ray tracing path, the wavefront kernels, the caustics and the photon cache. view_gpu.c holds the media, the composite, the HUD, FXAA, the capture and the path tracer's scheduler; `struct GpuView` has about 75 fields, 25 of them the scheduler's `t_*`. | trace_gpu.c into `trace_scene.c` (upload, sync), `trace_build.c` (LBVH, PLOC), `trace_hw.c` (ray queries, `LR_VK` only), `trace_wave.c`, `trace_light.c` (caustics, photons), with a `trace_internal.h`. view_gpu.c: the scheduler into `view_trace.c` with its own `TraceSched` struct; HUD + FXAA + capture into `view_post.c`. view.c: S3. No behaviour changes; one file at a time, each checked by the tests and the reference images. |
| **A2** | **Layouts shared by C and WGSL are kept in step by hand.** `GFrame`/`GVol` (view_gpu.c:84-97) mirror `struct Frame` in types.wgsl; the record sizes appear as bare numbers (112 and 28 floats in raster_gpu.c:405-427, 192, 96, 48, 36 and 32 in trace_gpu.c:1353-1437). | Named constants beside each struct, `_Static_assert(sizeof(GFrame) == ...)`, and the same constant written into the WGSL by the join of D7 where the shader needs it. |
| **A3** | **Three programs and the bench each decide "GPU or CPU" on their own.** `engine/app/gpu_frame.c` serves render and sim, view.c has its own state machine, bench.c its own loops (B2 had to be found in three places). | One small facade in `engine/app` (open, trace n samples, raster a frame, read back, fall back), used by all four. |
| **A4** | **Errors and messages.** Library code prints with `fprintf(stderr, ...)` and `printf` directly (also `options_setup_scene`), and ends the process on out of memory in places (S5). | An optional log hook (`lr_log(level, fmt, ...)`, default: stderr) and S5's policy; an embedding program can then stay quiet. |
| **A5** | **The public surface is everything.** `lr.h` includes every header; helpers used by one file are public (DC9); eleven documented functions have no caller and no test (section 2). | Mark what is internal `static` or move it to `*_internal.h`; one test line per documented function. |
| **A6** | **Tests cover the formats, the BSDF, the physics and the GPU well, but not:** the OBJ and PLY readers, the options and `-h` (B1, B3, B5, B7 would have been caught), the tools (B6), the WGSL compiler on its own (only through `VK=1` runs), the web OpenMP runtime. The decoders (PNG, JPEG, HDR, JSON, glTF, OBJ) are self-contained and would take a fuzzer (`make SAN=1` + a small driver) with little work. | `tests/test_options.c` (defaults against the docs' table, every option parsed, bad values refused), `tests/test_mesh.c`, a `make fuzz-png` style target later. |
| **A7** | **The CPU and GPU twins are deliberate** (README) and are not counted as duplication here. What is worth sharing between them is only data: constants, table layouts, record formats (A2). | none beyond A2 |

## 7. A suggested order

1. The bugs that change what a user sees: **B1** (decide), **B2**, **B3**, **B5**, **B7**; then **B4**, **B6**, **B8**, **B9**, **B16**.
2. The removals, all free: **DC1-DC9**, **U1-U4**.
3. The cheap sharing: **D2**, **D4**, **D6**, **D7**, **D8**, **D10**, **D11**, **D12**; **S4**, **S6**, **S7**, **S8**, **S12**, **S13**, **S16**.
4. The ones that touch images or speed, each with the reference images and the perf suite: **D1**, **D3** with **S5**, **D5**, **D9**, **D14**, **S10**.
5. The suspected bugs, each after a reproduction: **B10**, **B11**, **B12**, **B13**, **B14**, **B15**.
6. The larger work, one step at a time: **S1** (the options table, which also closes the class of B1/B3), **S3**, **A1**, **A3**, **A6**.
