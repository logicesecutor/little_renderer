# little_renderer - one Makefile for the whole engine. Run everything from this folder.
# Every target, option and key is documented in docs/USAGE.md; the library API in docs/SDK.md.
#
#   engine/{base,formats,scene,raster,rt,phys,platform,gpu,app}/*.c -> build/float-vk/liblr.a (the engine)
#   webgpu/{vk,wgsl}/*.c                      -> into the same library (our WebGPU, on Vulkan)
#   apps/*.c                                  -> bin/render, bin/view, bin/sim
#   examples/*.c                              -> bin/hello_raster, bin/hello_rt, ...
#   tests/*.c                                 -> bin/test_png, bin/test_physics, ... (make test runs them)
#
#   make            float build (fast)       make DOUBLE=1   double build into bin/double/
#   make SAN=1      sanitizers               make SINGLE=1   no OpenMP (one thread)
#
# Linux: gcc + make + libX11. Windows: the same Makefile in an MSYS2 UCRT64 (or MINGW64)
# shell, with MSYS2's gcc and make (docs/USAGE.md, "Windows"); programs get a .exe.
#
#   make NOGPU=1    without the GPU (CPU only). With it (the default) a machine without
#                   Vulkan 1.3 still runs everything, on the CPU.
CC = gcc
CFLAGS = -Wall -Wextra -O2 -g
# "make SAN=1": address + undefined-behaviour sanitizers (catches bad memory access)
SANFLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer
CFLAGS += $(if $(SAN),$(SANFLAGS))
# OpenMP spreads the path tracer over the CPU cores; "make SINGLE=1" builds without it
OMP = $(if $(SINGLE),,-fopenmp)
# "make NATIVE=1": -O3 and the instructions of x86-64 CPUs from 2013 on (AVX2:
# -march=x86-64-v3; other CPUs: -march=native), into folders of its own (build/*-native,
# bin/native). Faster on the CPU (docs/PERF_REVIEW.md, N8), and the same images to the bit as
# the default build's: multiplies and adds are not fused (-ffp-contract=off). Fused (FMA) it
# was faster still but rounded otherwise, and with gcc 13 two hulls colliding in test_physics
# blew up (its momentum check: 31164 for 0.001). Not the default: it needs such a CPU.
ARCHFLAGS = $(if $(findstring x86_64,$(shell $(CC) -dumpmachine)),-march=x86-64-v3,-march=native)
CFLAGS += $(if $(NATIVE),-O3 $(ARCHFLAGS) -ffp-contract=off)

# Precision of `real` (engine/base/lrmath.h). float is the default: half the memory and
# faster math. In float mode, constants like 0.5 are made float too
# (-fsingle-precision-constant), otherwise every `x * 0.5` would silently compute in double.
# Library and apps must agree, so the two builds live in separate folders.
ifdef DOUBLE
PREC = double
PRECFLAGS = -DLR_DOUBLE
BIN = bin/double
else
PREC = float
PRECFLAGS = -fsingle-precision-constant
BIN = bin
endif

# ---- the GPU, natively: our own WebGPU on Vulkan -------------------------------------------
# The engine's GPU code (engine/gpu) is written against the standard webgpu.h (our copy:
# webgpu/include/webgpu/webgpu.h). Natively our own layer implements it on Vulkan 1.3
# (webgpu/vk, docs/VULKAN.md), its WGSL compiled to SPIR-V by our compiler (webgpu/wgsl), with
# hardware ray tracing where the GPU has it. The Vulkan loader, which the GPU's driver
# installs, is opened at run time: no SDK and no library to build with. A machine without
# Vulkan 1.3 runs everything on the CPU. NOGPU=1 builds without the GPU. (In the browser the
# same engine runs on the browser's WebGPU: make web.)
ifeq ($(NOGPU),)
GPUFLAGS = -DLR_GPU -DLR_VK -Iwebgpu/include
GPULIBS = $(if $(filter Windows_NT,$(OS)),,-ldl)
GPUDIR = -vk
GPUSRC = $(wildcard webgpu/vk/*.c webgpu/wgsl/*.c)
endif

BUILD = build/$(PREC)$(GPUDIR)$(if $(NATIVE),-native)
BIN := $(BIN)$(if $(NATIVE),/native)
INC = -Iengine -Iwebgpu
HDR = $(wildcard engine/*.h engine/*/*.h webgpu/*/*.h webgpu/include/webgpu/*.h)
SRC = $(wildcard engine/*/*.c)
OBJ = $(patsubst webgpu/%.c,$(BUILD)/webgpu/%.o,$(patsubst engine/%.c,$(BUILD)/%.o,$(SRC) $(GPUSRC))) $(BUILD)/wgsl_shaders.o
LIB = $(BUILD)/liblr.a

# the shaders: engine/gpu/shaders/<folder>/*.wgsl, carried inside the library as C strings
# (lr_wgsl_<name>), made by tools/wgsl_embed.c (built with the same compiler: HOSTCC for
# another one)
HOSTCC ?= $(CC)
WGSL_FILES = $(sort $(wildcard engine/gpu/shaders/*/*.wgsl))
WGSL_EMBED = build/tools/wgsl_embed$(EXE)
WGSL_C = build/gen/wgsl_shaders.c

# the window: X11 on Linux, Win32 (gdi32) on Windows
ifeq ($(OS),Windows_NT)
LIBS = -lm -lgdi32
EXE = .exe
else
LIBS = -lm -lX11
EXE =
endif

APPS = $(addsuffix $(EXE),$(BIN)/render $(BIN)/view $(BIN)/sim $(BIN)/bench)
EXAMPLES = $(addsuffix $(EXE),$(BIN)/hello_raster $(BIN)/hello_rt $(BIN)/custom_scene $(BIN)/raster_milestones \
                              $(BIN)/hello_physics $(BIN)/mesh_rain)
# the unit tests, in the order make test runs them
TEST_NAMES = test_medium test_png test_jpeg test_brdf test_env test_tex test_gltf test_gpu_basic test_gpu_brdf \
             test_physics test_gpu_rigid test_gpu_fluid test_gpu_rt test_gpu_trace
TESTS = $(addsuffix $(EXE),$(addprefix $(BIN)/,$(TEST_NAMES)))

all: lib apps examples tests
lib: $(LIB)
apps: $(APPS)
examples: $(EXAMPLES)
tests: $(TESTS)

# ---- the library ------------------------------------------------------------------------

$(BUILD)/%.o: engine/%.c $(HDR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(PRECFLAGS) $(OMP) $(INC) $(GPUFLAGS) -c $< -o $@

$(BUILD)/webgpu/%.o: webgpu/%.c $(HDR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(PRECFLAGS) $(OMP) $(INC) $(GPUFLAGS) -c $< -o $@

$(WGSL_EMBED): tools/wgsl_embed.c
	@mkdir -p $(dir $@)
	$(HOSTCC) -O2 -Wall -Wextra $< -o $@

$(WGSL_C): $(WGSL_EMBED) $(WGSL_FILES)
	@mkdir -p $(dir $@)
	$(WGSL_EMBED) $@ $(WGSL_FILES)

$(BUILD)/wgsl_shaders.o: $(WGSL_C)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(GPUFLAGS) -c $< -o $@

$(LIB): $(OBJ)
	rm -f $@
	ar rcs $@ $^

# ---- apps, examples and tests (one .c file each, linked with the library) ----------------

$(BIN)/%$(EXE): apps/%.c $(LIB) $(HDR)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(PRECFLAGS) $(OMP) $(INC) $(GPUFLAGS) $< $(LIB) -o $@ $(GPULIBS) $(LIBS)

$(BIN)/%$(EXE): examples/%.c $(LIB) $(HDR)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(PRECFLAGS) $(OMP) $(INC) $(GPUFLAGS) $< $(LIB) -o $@ $(GPULIBS) $(LIBS)

$(BIN)/%$(EXE): tests/%.c $(LIB) $(HDR) tests/png_tests.h tests/jpeg_tests.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(PRECFLAGS) $(OMP) $(INC) $(GPUFLAGS) $< $(LIB) -o $@ $(GPULIBS) $(LIBS)

# Layering rule: base uses nothing else; formats only base; scene only base and formats; raster,
# rt, phys and platform only those three (and themselves); gpu (the GPU backend) uses them and the
# data types of raster, rt and phys (and our
# Vulkan layer's ray tracing header, vk/webgpu_rt.h); app may use everything. webgpu/ (vk, wgsl)
# includes nothing of the engine. Fails if a file includes a layer it should not.
check-layers:
	@ok=1; \
	for f in engine/base/*; do grep -Hn '#include "' $$f | grep -v '"base/' && ok=0; done; \
	for f in engine/formats/*; do grep -Hn '#include "' $$f | grep -v -e '"base/' -e '"formats/' && ok=0; done; \
	for f in engine/scene/*; do grep -Hn '#include "' $$f | grep -v -e '"base/' -e '"formats/' -e '"scene/' && ok=0; done; \
	for l in raster rt phys platform; do for f in engine/$$l/*; do \
	  grep -Hn '#include "' $$f | grep -v -e '"base/' -e '"formats/' -e '"scene/' -e "\"$$l/" && ok=0; done; done; \
	for f in engine/gpu/*.[ch]; do grep -Hn '#include "' $$f | grep -v -e '"base/' -e '"formats/' -e '"scene/' -e '"raster/' -e '"rt/' -e '"phys/' -e '"gpu/' -e '"vk/webgpu_rt.h"' && ok=0; done; \
	for f in webgpu/*/*; do grep -Hn '#include "' $$f | grep -v -e '"vk/' -e '"wgsl/' && ok=0; done; \
	if [ $$ok = 1 ]; then echo "layers ok"; else echo "layer violations above"; exit 1; fi

# ---- models -----------------------------------------------------------------------------

DRAGON = models/dragon_recon/dragon_vrip.ply

# Python for the tools in tools/: uv if it is installed (it reads each script's header and
# brings what it needs, e.g. Pillow), otherwise python3
# (on Windows uv installs into %USERPROFILE%\.local\bin, outside the MSYS2 shell's PATH)
ifeq ($(OS),Windows_NT)
UV_WIN = $(shell cygpath -u "$$USERPROFILE" 2>/dev/null)/.local/bin/uv.exe
endif
PYTHON = $(if $(shell command -v uv 2>/dev/null),uv run,$(if $(wildcard $(UV_WIN)),$(UV_WIN) run,python3))

# cube, sphere, torus, dense torus into models/ (made here), then the HDR skies, the glTF
# sample models and the Stanford dragon (downloaded, about 41 MB, only those not there yet;
# without a network they are skipped with a message and the rest still works)
models:
	$(PYTHON) tools/gen_models.py
	$(MAKE) --no-print-directory skies-download gltf-download dragon-download

# (re)make a GIF from frames: make gif FRAMES_DIR=out/sim FPS=30
FRAMES_DIR = out/sim
FPS = 30
gif:
	$(PYTHON) tools/frames_to_gif.py $(FRAMES_DIR) $(FRAMES_DIR).gif $(FPS)

# The Stanford dragon (871,414 triangles, plus three decimated versions), once, into
# models/dragon_recon/ (11 MB download, 45 MB unpacked); skipped if it is already there.
# Credit: Stanford University Computer Graphics Laboratory, 3D Scanning Repository.
DRAGON_URL = http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz
dragon-download:
	@if [ -s $(DRAGON) ]; then echo "have $(DRAGON)"; else echo "downloading $(DRAGON)"; \
	  mkdir -p models && curl -fsSL -o models/dragon_recon.tar.gz $(DRAGON_URL) \
	  && tar xzf models/dragon_recon.tar.gz -C models \
	  || echo "could not download $(DRAGON_URL) (no network?)"; \
	  rm -f models/dragon_recon.tar.gz; fi

# a file downloaded unless it is already there: $(call fetch,file,url)
fetch = if [ -s $(1) ]; then echo "have $(1)"; else echo "downloading $(1)"; \
	curl -fsSL --create-dirs -o $(1) $(2) || { rm -f $(1); echo "could not download $(2) (no network?)"; }; fi

# Khronos glTF sample models (docs/PBR.md, PBR6), once, into models/gltf/ (17 MB): what the
# showcase (the damaged helmet) and the glTF checks use. Credit: The Khronos Group,
# glTF-Sample-Assets (each model's README names its authors); CC BY 4.0, the helmet's earlier
# version CC BY-NC 4.0.
GLTF_URL = https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models
GLTF_MODELS = DamagedHelmet MetalRoughSpheres NormalTangentMirrorTest OrientationTest BoxTextured
gltf-download:
	@$(foreach m,$(GLTF_MODELS),$(call fetch,models/gltf/$(m).glb,$(GLTF_URL)/$(m)/glTF-Binary/$(m).glb);)
	@$(foreach f,BoxTextured.gltf BoxTextured0.bin CesiumLogoFlat.png,$(call fetch,models/gltf/BoxTextured/$(f),$(GLTF_URL)/BoxTextured/glTF/$(f));)

# HDR skies (docs/PBR.md, PBR4), once, into models/skies/ (13 MB): Poly Haven's, CC0
SKY_URL = https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k
SKIES = kloofendal_48d_partly_cloudy studio_small_09
skies-download:
	@$(foreach k,$(SKIES),$(call fetch,models/skies/$(k)_2k.hdr,$(SKY_URL)/$(k)_2k.hdr);)

# ---- running ----------------------------------------------------------------------------

# rasterizer step by step: one image per milestone into out/
demo: $(BIN)/raster_milestones$(EXE)
	mkdir -p out
	$(BIN)/raster_milestones

# quick previews of every scene, path traced and rasterized
run: $(BIN)/render$(EXE)
	mkdir -p out
	$(BIN)/render -scene boxes
	$(BIN)/render -scene spheres
	$(BIN)/render -scene mesh
	$(BIN)/render -scene boxes -backend raster
	$(BIN)/render -scene spheres -backend raster
	$(BIN)/render -scene mesh -backend raster
	$(BIN)/render -scene cloud

# real-time window (options: make view ARGS="-scene spheres -backend rt")
view: $(BIN)/view$(EXE)
	mkdir -p out
	$(BIN)/view $(ARGS)

view-dragon: $(BIN)/view$(EXE)
	mkdir -p out
	$(BIN)/view -scene mesh -obj $(DRAGON) -rot -35 $(ARGS)

# the unit tests: the medium sampling, PNG and JPEG files, the BSDF (and its WGSL), the environment, the textures, glTF models, the physics (CPU and GPU)
test: $(TESTS)
	mkdir -p out
	@for t in $(TEST_NAMES); do echo "$(BIN)/$$t"; $(BIN)/$$t || exit 1; done

# volumetric cloud in the box (docs/VOLUMES.md): a clean image, and an animation as frames
cloud: $(BIN)/render$(EXE)
	mkdir -p out
	$(BIN)/render -scene cloud -n 600 -s 1024 -o out/cornell_cloud_final.png

FRAMES = 48
cloud-frames: $(BIN)/render$(EXE)
	mkdir -p out/cloud_frames
	for i in $$(seq 0 $$(($(FRAMES) - 1))); do \
	  $(BIN)/render -scene cloud -n 300 -s 64 -time $$(echo "$$i * 0.5" | bc -l) \
	                -o out/cloud_frames/frame_$$(printf %03d $$i).png || exit 1; done
	@echo "frames in out/cloud_frames/ (e.g. ffmpeg -framerate 12 -i out/cloud_frames/frame_%03d.png out/cloud.mp4)"

# physics (docs/PHYSICS.md): a simulation rendered frame by frame, then a GIF
sim: $(BIN)/sim$(EXE)
	$(BIN)/sim -demo drop -frames 120 -s 64 -n 320 -o out/sim

sim-blast: $(BIN)/sim$(EXE)
	$(BIN)/sim -demo blast -frames 90 -s 64 -n 320 -o out/sim_blast

# slow, clean versions
final: $(BIN)/render$(EXE)
	mkdir -p out
	$(BIN)/render -scene boxes -n 600 -s 1024 -o out/cornell_boxes_final.png
	$(BIN)/render -scene spheres -n 600 -s 1024 -o out/cornell_spheres_final.png
	$(BIN)/render -scene spheres -n 600 -s 1024 -clamp 4 -o out/cornell_spheres_final_clamped.png
	$(BIN)/render -scene mesh -n 600 -s 1024 -o out/cornell_mesh_final.png

dragon: $(BIN)/render$(EXE)
	mkdir -p out
	$(BIN)/render -scene mesh -obj $(DRAGON) -rot -35 -n 600 -s 512 -o out/cornell_dragon.png
	$(BIN)/render -scene mesh -obj $(DRAGON) -rot -35 -mat glass -clamp 4 -n 600 -s 512 -o out/cornell_dragon_glass.png

# Compares two images: identical, or how many bytes differ. In the double build the three
# acceleration structures give identical images. In float a few pixels may differ: where a
# ray hits the shared edge of two triangles at exactly the same float distance, the one
# tested first wins, and that depends on the structure. The path then continues differently.
SAME = n=$$(cmp -l $(1) $(2) | wc -l); if [ $$n = 0 ]; then echo "$(1) = $(2): identical"; \
       else echo "$(1) vs $(2): $$n bytes differ (ties, see Makefile)"; fi

# all acceleration structures on the torus (2304 triangles)
# the performance suite (apps/bench.c, docs/STATUS.md 1.3): every case timed by the program,
# the results appended to out/perf/<commit>_<backend>.csv (vk, or cpu with NOGPU=1).
# Compare two runs: $(BIN)/bench -compare out/perf/a.csv out/perf/b.csv
# (the commit, + when the tree has changes; without git, the date: PERF_TAG=... to choose)
PERF_TAG ?= $(or $(shell git rev-parse --short HEAD 2>/dev/null),$(shell date +%Y%m%d-%H%M))$(if $(shell git status --porcelain --untracked-files=no 2>/dev/null),+)
PERF_BACKEND = $(if $(NOGPU),cpu,vk)
perf-quick: $(BIN)/bench$(EXE)
	mkdir -p out/perf
	$(BIN)/bench -quick -tag $(PERF_TAG) -o out/perf/$(PERF_TAG)_$(PERF_BACKEND).csv $(ARGS)
perf: $(BIN)/bench$(EXE)
	mkdir -p out/perf
	$(BIN)/bench -full -tag $(PERF_TAG) -o out/perf/$(PERF_TAG)_$(PERF_BACKEND).csv $(ARGS)

bench: $(BIN)/render$(EXE)
	mkdir -p out
	$(BIN)/render -scene mesh -n 200 -s 16 -accel none   -o out/bench_none.png
	$(BIN)/render -scene mesh -n 200 -s 16 -accel octree -o out/bench_octree.png
	$(BIN)/render -scene mesh -n 200 -s 16 -accel bvh    -o out/bench_bvh.png
	@$(call SAME,out/bench_none.png,out/bench_octree.png)
	@$(call SAME,out/bench_none.png,out/bench_bvh.png)

# octree vs BVH on the Stanford dragon (needs "make dragon-download")
bench-dragon: $(BIN)/render$(EXE)
	mkdir -p out
	$(BIN)/render -scene mesh -obj $(DRAGON) -rot -35 -n 300 -s 32 -accel octree -o out/bench_dragon_octree.png
	$(BIN)/render -scene mesh -obj $(DRAGON) -rot -35 -n 300 -s 32 -accel bvh    -o out/bench_dragon_bvh.png
	@$(call SAME,out/bench_dragon_octree.png,out/bench_dragon_bvh.png)

# ---- the web (Emscripten + WebAssembly) ----------------------------------------------------
#
# bin/view in a web page: web/build/index.html (+ .js, .wasm, .data). Needs Emscripten:
#   source <emsdk>/emsdk_env.sh   (Linux; Windows: F:/Tools/emsdk, see docs/WEB.md)
#   make web && make serve        then open http://localhost:8000
# The page passes its URL options to the program: index.html?scene=cloud&sim=on
EMCC ?= emcc
WEB = web/build
WEB_MODELS = models/cube.obj models/sphere.obj models/torus.obj models/torus_dense.obj \
             models/dragon_recon/dragon_vrip_res2.ply models/dragon_recon/dragon_vrip_res3.ply \
             models/dragon_recon/dragon_vrip_res4.ply $(if $(WEB_DRAGON),$(DRAGON)) \
             models/skies/kloofendal_48d_partly_cloudy_2k.hdr
WEB_SRC = $(SRC) $(WGSL_C) apps/view.c
# the GPU backend: WebGPU through Dawn's bindings for Emscripten (fetched by emcc once)
# (WGPU_REFERENCE=1: for a webgpu.h from before wgpu...Reference became wgpu...AddRef)
WEBGPU = -DLR_GPU --use-port=emdawnwebgpu $(if $(WGPU_REFERENCE),-DLR_WGPU_REFERENCE)
# WebAssembly SIMD (128-bit; every browser with WebGPU has it): the compiler vectorizes the
# CPU's loops where it can; NOSIMD=1 leaves it out
WEBSIMD = $(if $(NOSIMD),,-msimd128)
WEBBASE = -O3 -Iengine $(if $(DOUBLE),-DLR_DOUBLE) -Wall -Wextra -Wno-unknown-pragmas $(WEBGPU) $(WEBSIMD) -sDEFAULT_TO_CXX \
          -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=256MB -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=4MB \
          -sEXIT_RUNTIME=0 -sENVIRONMENT=web
WEBFLAGS = $(WEBBASE) --shell-file web/shell.html $(foreach m,$(wildcard $(WEB_MODELS)),--preload-file $(m)@$(m))

# the performance suite in a page (apps/bench.c): built with JSPI so it can wait for the GPU
# (Chrome 137+); tools/webbench.mjs runs it in a headless Chrome and keeps its results
# (dragon_vrip_res4: the sky demo's props, so its cases are the scene measured natively)
WEB_BENCH_MODELS = models/torus.obj models/dragon_recon/dragon_vrip_res2.ply models/dragon_recon/dragon_vrip_res4.ply \
                   models/skies/kloofendal_48d_partly_cloudy_2k.hdr models/gltf/DamagedHelmet.glb
web-bench: $(WEB)/bench.html $(if $(THREADS),$(WEB)/bench_mt.html)
$(WEB)/bench.html: $(SRC) $(WGSL_C) apps/bench.c $(HDR)
	@mkdir -p $(WEB)
	$(EMCC) $(WEBBASE) -DLR_WEB_WAIT -sJSPI $(SRC) $(WGSL_C) apps/bench.c -o $@ \
	    $(foreach m,$(wildcard $(WEB_BENCH_MODELS)),--preload-file $(m)@$(m))
$(WEB)/bench_mt.html: $(SRC) $(WGSL_C) apps/bench.c $(HDR) web/include/omp.h
	@mkdir -p $(WEB)
	$(EMCC) $(WEBBASE) $(WEBMT) -DLR_WEB_WAIT -sJSPI $(SRC) $(WGSL_C) apps/bench.c -o $@ \
	    $(foreach m,$(wildcard $(WEB_BENCH_MODELS)),--preload-file $(m)@$(m))

# THREADS=1: also the threaded page, index_mt.js beside index.js: the CPU's loops (the
# engine's OpenMP pragmas) on WebAssembly threads (Emscripten's pthreads; the part of an
# OpenMP runtime they need is engine/base/omp_web.c, its header web/include/omp.h), one per
# hardware thread. Threads need SharedArrayBuffer, so a cross-origin isolated page (make
# serve sends the headers); the page (web/shell.html) loads index_mt.js when it is isolated
# and index.js otherwise (a static host that cannot send them; ?threads=1 forces it).
WEBMT = -pthread -Xclang -fopenmp -Iweb/include -sENVIRONMENT=web,worker \
        -sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency -sDEFAULT_PTHREAD_STACK_SIZE=2MB
web: $(WEB)/index.html $(if $(THREADS),$(WEB)/index_mt.js)
$(WEB)/index.html: $(WEB_SRC) $(HDR) web/shell.html
	@mkdir -p $(WEB)
	$(EMCC) $(WEBFLAGS) $(WEB_SRC) -o $@
$(WEB)/index_mt.js: $(WEB_SRC) $(HDR) web/include/omp.h
	@mkdir -p $(WEB)
	$(EMCC) $(WEBBASE) $(WEBMT) $(foreach m,$(wildcard $(WEB_MODELS)),--preload-file $(m)@$(m)) $(WEB_SRC) -o $@

# a local web server for the page (the browser will not run WebAssembly from file://)
# the GPU air against the CPU air in a browser (open web/build/test_gpu_fluid.html, or see
# docs/WEB.md; natively it is part of make test)
webtest: $(WEB)/test_gpu_fluid.html
$(WEB)/test_gpu_fluid.html: $(SRC) $(WGSL_C) $(HDR) tests/test_gpu_fluid.c
	@mkdir -p $(WEB)
	$(EMCC) $(WEBBASE) $(SRC) $(WGSL_C) tests/test_gpu_fluid.c -o $@

# (Node: tools/serve.mjs, which also sends the headers of a cross-origin isolated page, which
# the threaded build needs; without Node, Python's server, without them)
serve:
	node tools/serve.mjs $(WEB) -port 8000 || (cd $(WEB) && (python3 -m http.server 8000 || uv run python -m http.server 8000))

# every pipeline of the page compiled and run once in a headless Chrome on this GPU (the bench
# page's cases that make them), failing on any GPU error (tools/webbench.mjs check)
web-check: $(WEB)/bench.html $(if $(THREADS),$(WEB)/bench_mt.html)
	node tools/webbench.mjs check -serve
	$(if $(THREADS),node tools/webbench.mjs check -serve -page bench_mt.html)

clean:
	rm -rf build bin $(WEB)

.PHONY: all lib apps examples tests check-layers models dragon-download gltf-download skies-download demo run view view-dragon final dragon bench bench-dragon perf perf-quick web-bench web-check test cloud cloud-frames sim sim-blast gif web webtest serve clean
