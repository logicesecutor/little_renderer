# little_renderer - one Makefile for the whole engine. Run everything from this folder.
# Every target, option and key is documented in docs/USAGE.md; the library API in docs/SDK.md.
#
#   engine/{core,raster,rt,phys,platform,app}/*.c  -> build/float/liblr.a   (the engine)
#   apps/*.c                                  -> bin/render, bin/view, bin/sim
#   examples/*.c                              -> bin/hello_raster, bin/hello_rt, ...
#
#   make            float build (fast)       make DOUBLE=1   double build into bin/double/
#   make SAN=1      sanitizers               make SINGLE=1   no OpenMP (one thread)
#
# Linux: gcc + make + libX11. Windows: the same Makefile in an MSYS2 UCRT64 (or MINGW64)
# shell, with MSYS2's gcc and make (docs/USAGE.md, "Windows"); programs get a .exe.
#
#   make deps       the GPU: fetches wgpu-native (a WebGPU library) into third_party/wgpu;
#                   from then on every build uses the GPU (docs/WEB.md, "native GPU").
#                   Without it everything builds and runs on the CPU, as before.
CC = gcc
CFLAGS = -Wall -Wextra -O2 -g
# "make SAN=1": address + undefined-behaviour sanitizers (catches bad memory access)
SANFLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer
CFLAGS += $(if $(SAN),$(SANFLAGS))
# OpenMP spreads the path tracer over the CPU cores; "make SINGLE=1" builds without it
OMP = $(if $(SINGLE),,-fopenmp)

# Precision of `real` (engine/core/lrmath.h). float is the default: half the memory and
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

# ---- the GPU, natively: WebGPU through wgpu-native, if make deps fetched it ------------------
# The engine's GPU code (engine/gpu) is written against the standard webgpu.h, so any WebGPU
# implementation can stand here; wgpu-native is linked statically (no DLL to carry along),
# and reaches the GPU through Vulkan (Linux) or Direct3D 12 / Vulkan (Windows), which the
# GPU's driver provides. NOGPU=1 builds without it even when it is there.
WGPU_VERSION = v29.0.1.1
WGPU_DIR = third_party/wgpu
ifeq ($(OS),Windows_NT)
WGPU_ZIP = wgpu-windows-x86_64-gnu-release.zip
WGPU_SYSLIBS = -lws2_32 -luserenv -lbcrypt -lntdll -lopengl32 -ld3dcompiler -lole32 -loleaut32 -luuid \
               -lpropsys -lruntimeobject -ladvapi32 -lshell32 -luser32 -lkernel32 -lsynchronization -ldwmapi
else
WGPU_ZIP = wgpu-linux-x86_64-release.zip
WGPU_SYSLIBS = -lpthread -ldl -lm
endif
ifeq ($(NOGPU)$(wildcard $(WGPU_DIR)/lib/libwgpu_native.a),$(WGPU_DIR)/lib/libwgpu_native.a)
GPUFLAGS = -DLR_GPU -DLR_WGPU_NATIVE -I$(WGPU_DIR)/include
GPULIBS = $(WGPU_DIR)/lib/libwgpu_native.a $(WGPU_SYSLIBS)
GPUDIR = -gpu
endif

BUILD = build/$(PREC)$(GPUDIR)
INC = -Iengine
HDR = $(wildcard engine/*.h engine/*/*.h)
SRC = $(wildcard engine/*/*.c)
OBJ = $(patsubst engine/%.c,$(BUILD)/%.o,$(SRC))
LIB = $(BUILD)/liblr.a

# the window: X11 on Linux, Win32 (gdi32) on Windows
ifeq ($(OS),Windows_NT)
LIBS = -lm -lgdi32
EXE = .exe
else
LIBS = -lm -lX11
EXE =
endif

APPS = $(addsuffix $(EXE),$(BIN)/render $(BIN)/view $(BIN)/sim)
EXAMPLES = $(addsuffix $(EXE),$(BIN)/hello_raster $(BIN)/hello_rt $(BIN)/custom_scene $(BIN)/raster_milestones \
                              $(BIN)/test_medium $(BIN)/test_physics $(BIN)/test_gpu_rigid $(BIN)/test_png $(BIN)/test_brdf $(BIN)/hello_physics \
                              $(BIN)/mesh_rain)

all: lib apps examples
lib: $(LIB)
apps: $(APPS)
examples: $(EXAMPLES)

# ---- the library ------------------------------------------------------------------------

$(BUILD)/%.o: engine/%.c $(HDR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(PRECFLAGS) $(OMP) $(INC) $(GPUFLAGS) -c $< -o $@

$(LIB): $(OBJ)
	ar rcs $@ $^

# ---- apps and examples (one .c file each, linked with the library) -----------------------

$(BIN)/%$(EXE): apps/%.c $(LIB) $(HDR)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(PRECFLAGS) $(OMP) $(INC) $(GPUFLAGS) $< $(LIB) -o $@ $(GPULIBS) $(LIBS)

$(BIN)/%$(EXE): examples/%.c $(LIB) $(HDR)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(PRECFLAGS) $(OMP) $(INC) $(GPUFLAGS) $< $(LIB) -o $@ $(GPULIBS) $(LIBS)

# Layering rule: core uses nothing else; raster, rt, phys and platform use only core (and
# themselves); gpu (the GPU backend) uses core and the data types of raster, rt and phys; app may
# use everything. Fails if a file includes a layer it should not.
check-layers:
	@ok=1; \
	for f in engine/core/*; do grep -Hn '#include "' $$f | grep -v '"core/' && ok=0; done; \
	for l in raster rt phys platform; do for f in engine/$$l/*; do \
	  grep -Hn '#include "' $$f | grep -v -e '"core/' -e "\"$$l/" && ok=0; done; done; \
	for f in engine/gpu/*; do grep -Hn '#include "' $$f | grep -v -e '"core/' -e '"raster/' -e '"rt/' -e '"phys/' -e '"gpu/' && ok=0; done; \
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

# cube, sphere, torus, dense torus into models/
models:
	$(PYTHON) tools/gen_models.py

# (re)make a GIF from frames: make gif FRAMES_DIR=out/sim FPS=30
FRAMES_DIR = out/sim
FPS = 30
gif:
	$(PYTHON) tools/frames_to_gif.py $(FRAMES_DIR) $(FRAMES_DIR).gif $(FPS)

# The Stanford dragon (871,414 triangles), once.
# Credit: Stanford University Computer Graphics Laboratory, 3D Scanning Repository.
dragon-download:
	mkdir -p models
	curl -L -o models/dragon_recon.tar.gz http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz
	tar xzf models/dragon_recon.tar.gz -C models
	rm models/dragon_recon.tar.gz

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

# the unit tests: the medium sampling, PNG files, the BSDF, the physics (CPU and GPU)
test: $(BIN)/test_medium$(EXE) $(BIN)/test_png$(EXE) $(BIN)/test_brdf$(EXE) $(BIN)/test_physics$(EXE) $(BIN)/test_gpu_rigid$(EXE)
	$(BIN)/test_medium
	$(BIN)/test_png
	$(BIN)/test_brdf
	$(BIN)/test_physics
	$(BIN)/test_gpu_rigid

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
             models/dragon_recon/dragon_vrip_res4.ply $(if $(WEB_DRAGON),$(DRAGON))
WEB_SRC = $(SRC) apps/view.c
# the GPU backend: WebGPU through Dawn's bindings for Emscripten (fetched by emcc once)
WEBGPU = -DLR_GPU --use-port=emdawnwebgpu
WEBBASE = -O3 -Iengine $(if $(DOUBLE),-DLR_DOUBLE) -Wall -Wextra -Wno-unknown-pragmas $(WEBGPU) -sDEFAULT_TO_CXX \
          -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=256MB -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=4MB \
          -sEXIT_RUNTIME=0 -sENVIRONMENT=web
WEBFLAGS = $(WEBBASE) --shell-file web/shell.html $(foreach m,$(wildcard $(WEB_MODELS)),--preload-file $(m)@$(m))

web: $(WEB)/index.html
$(WEB)/index.html: $(WEB_SRC) $(HDR) web/shell.html
	@mkdir -p $(WEB)
	$(EMCC) $(WEBFLAGS) $(WEB_SRC) -o $@

# a local web server for the page (the browser will not run WebAssembly from file://)
# the GPU air against the CPU air (open web/build/test_gpu_fluid.html, or see docs/WEB.md)
webtest: $(WEB)/test_gpu_fluid.html
$(WEB)/test_gpu_fluid.html: $(SRC) $(HDR) examples/test_gpu_fluid.c
	@mkdir -p $(WEB)
	$(EMCC) $(WEBBASE) $(SRC) examples/test_gpu_fluid.c -o $@

serve:
	cd $(WEB) && (python3 -m http.server 8000 || uv run python -m http.server 8000)

# ---- the native GPU's library ------------------------------------------------------------
# make deps: wgpu-native $(WGPU_VERSION), the build for this system, into third_party/wgpu
# (kept out of git). make deps-clean removes it: the builds are CPU-only again.
WGPU_URL = https://github.com/gfx-rs/wgpu-native/releases/download/$(WGPU_VERSION)/$(WGPU_ZIP)
WINTAR = /c/Windows/System32/tar.exe
deps:
	@mkdir -p $(WGPU_DIR)
	@echo "downloading $(WGPU_URL)"
	curl -fL --retry 3 -o $(WGPU_DIR)/wgpu.zip $(WGPU_URL)
	cd $(WGPU_DIR) && (unzip -oq wgpu.zip 2>/dev/null || ([ -x $(WINTAR) ] && $(WINTAR) -xf wgpu.zip) || \
	    $(PYTHON) python -m zipfile -e wgpu.zip . 2>/dev/null || python3 -m zipfile -e wgpu.zip .)
	rm -f $(WGPU_DIR)/wgpu.zip
	echo $(WGPU_VERSION) > $(WGPU_DIR)/VERSION
	@echo "wgpu-native $(WGPU_VERSION) is in $(WGPU_DIR): make now builds with the GPU"

deps-clean:
	rm -rf $(WGPU_DIR)

clean:
	rm -rf build bin $(WEB)

.PHONY: all lib apps examples check-layers models dragon-download demo run view view-dragon final dragon bench bench-dragon test cloud cloud-frames sim sim-blast gif web webtest serve deps deps-clean clean
