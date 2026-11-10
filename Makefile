# little_renderer - one Makefile for the whole engine. Run everything from this folder.
# Every target, option and key is documented in docs/USAGE.md; the library API in docs/SDK.md.
#
#   engine/{core,raster,rt,phys,platform,app}/*.c  -> build/float/liblr.a   (the engine)
#   apps/*.c                                  -> bin/render, bin/view, bin/sim
#   examples/*.c                              -> bin/hello_raster, bin/hello_rt, ...
#
#   make            float build (fast)       make DOUBLE=1   double build into bin/double/
#   make SAN=1      sanitizers               make SINGLE=1   no OpenMP (one thread)
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

BUILD = build/$(PREC)
INC = -Iengine
HDR = $(wildcard engine/*.h engine/*/*.h)
SRC = $(wildcard engine/*/*.c)
OBJ = $(patsubst engine/%.c,$(BUILD)/%.o,$(SRC))
LIB = $(BUILD)/liblr.a
LIBS = -lm -lX11

APPS = $(BIN)/render $(BIN)/view $(BIN)/sim
EXAMPLES = $(BIN)/hello_raster $(BIN)/hello_rt $(BIN)/custom_scene $(BIN)/raster_milestones $(BIN)/test_medium $(BIN)/test_physics $(BIN)/hello_physics

all: lib apps examples
lib: $(LIB)
apps: $(APPS)
examples: $(EXAMPLES)

# ---- the library ------------------------------------------------------------------------

$(BUILD)/%.o: engine/%.c $(HDR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(PRECFLAGS) $(OMP) $(INC) -c $< -o $@

$(LIB): $(OBJ)
	ar rcs $@ $^

# ---- apps and examples (one .c file each, linked with the library) -----------------------

$(BIN)/%: apps/%.c $(LIB) $(HDR)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(PRECFLAGS) $(OMP) $(INC) $< $(LIB) -o $@ $(LIBS)

$(BIN)/%: examples/%.c $(LIB) $(HDR)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(PRECFLAGS) $(OMP) $(INC) $< $(LIB) -o $@ $(LIBS)

# Layering rule: core uses nothing else; raster, rt, phys and platform use only core (and
# themselves); app may use everything. Fails if a file includes a layer it should not.
check-layers:
	@ok=1; \
	for f in engine/core/*; do grep -Hn '#include "' $$f | grep -v '"core/' && ok=0; done; \
	for l in raster rt phys platform; do for f in engine/$$l/*; do \
	  grep -Hn '#include "' $$f | grep -v -e '"core/' -e "\"$$l/" && ok=0; done; done; \
	if [ $$ok = 1 ]; then echo "layers ok"; else echo "layer violations above"; exit 1; fi

# ---- models -----------------------------------------------------------------------------

DRAGON = models/dragon_recon/dragon_vrip.ply

# cube, sphere, torus, dense torus into models/
models:
	python3 tools/gen_models.py

# The Stanford dragon (871,414 triangles), once.
# Credit: Stanford University Computer Graphics Laboratory, 3D Scanning Repository.
dragon-download:
	mkdir -p models
	curl -L -o models/dragon_recon.tar.gz http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz
	tar xzf models/dragon_recon.tar.gz -C models
	rm models/dragon_recon.tar.gz

# ---- running ----------------------------------------------------------------------------

# rasterizer step by step: one image per milestone into out/
demo: $(BIN)/raster_milestones
	mkdir -p out
	$(BIN)/raster_milestones

# quick previews of every scene, path traced and rasterized
run: $(BIN)/render
	mkdir -p out
	$(BIN)/render -scene boxes
	$(BIN)/render -scene spheres
	$(BIN)/render -scene mesh
	$(BIN)/render -scene boxes -backend raster
	$(BIN)/render -scene spheres -backend raster
	$(BIN)/render -scene mesh -backend raster
	$(BIN)/render -scene cloud

# real-time window (options: make view ARGS="-scene spheres -backend rt")
view: $(BIN)/view
	mkdir -p out
	$(BIN)/view $(ARGS)

view-dragon: $(BIN)/view
	mkdir -p out
	$(BIN)/view -scene mesh -obj $(DRAGON) -rot -35 $(ARGS)

# unit test of the participating-medium sampling (exact answers, see examples/test_medium.c)
test: $(BIN)/test_medium $(BIN)/test_physics
	$(BIN)/test_medium
	$(BIN)/test_physics

# volumetric cloud in the box (docs/VOLUMES.md): a clean image, and an animation as frames
cloud: $(BIN)/render
	mkdir -p out
	$(BIN)/render -scene cloud -n 600 -s 1024 -o out/cornell_cloud_final.ppm

FRAMES = 48
cloud-frames: $(BIN)/render
	mkdir -p out/cloud_frames
	for i in $$(seq 0 $$(($(FRAMES) - 1))); do \
	  $(BIN)/render -scene cloud -n 300 -s 64 -time $$(echo "$$i * 0.5" | bc -l) \
	                -o out/cloud_frames/frame_$$(printf %03d $$i).ppm || exit 1; done
	@echo "frames in out/cloud_frames/ (e.g. ffmpeg -framerate 12 -i out/cloud_frames/frame_%03d.ppm out/cloud.mp4)"

# physics (docs/PHYSICS.md): a simulation rendered frame by frame, then a GIF
sim: $(BIN)/sim
	$(BIN)/sim -demo drop -frames 120 -s 64 -n 320 -o out/sim

sim-blast: $(BIN)/sim
	$(BIN)/sim -demo blast -frames 90 -s 64 -n 320 -o out/sim_blast

# slow, clean versions
final: $(BIN)/render
	mkdir -p out
	$(BIN)/render -scene boxes -n 600 -s 1024 -o out/cornell_boxes_final.ppm
	$(BIN)/render -scene spheres -n 600 -s 1024 -o out/cornell_spheres_final.ppm
	$(BIN)/render -scene spheres -n 600 -s 1024 -clamp 4 -o out/cornell_spheres_final_clamped.ppm
	$(BIN)/render -scene mesh -n 600 -s 1024 -o out/cornell_mesh_final.ppm

dragon: $(BIN)/render
	mkdir -p out
	$(BIN)/render -scene mesh -obj $(DRAGON) -rot -35 -n 600 -s 512 -o out/cornell_dragon.ppm
	$(BIN)/render -scene mesh -obj $(DRAGON) -rot -35 -mat glass -clamp 4 -n 600 -s 512 -o out/cornell_dragon_glass.ppm

# Compares two images: identical, or how many bytes differ. In the double build the three
# acceleration structures give identical images. In float a few pixels may differ: where a
# ray hits the shared edge of two triangles at exactly the same float distance, the one
# tested first wins, and that depends on the structure. The path then continues differently.
SAME = n=$$(cmp -l $(1) $(2) | wc -l); if [ $$n = 0 ]; then echo "$(1) = $(2): identical"; \
       else echo "$(1) vs $(2): $$n bytes differ (ties, see Makefile)"; fi

# all acceleration structures on the torus (2304 triangles)
bench: $(BIN)/render
	mkdir -p out
	$(BIN)/render -scene mesh -n 200 -s 16 -accel none   -o out/bench_none.ppm
	$(BIN)/render -scene mesh -n 200 -s 16 -accel octree -o out/bench_octree.ppm
	$(BIN)/render -scene mesh -n 200 -s 16 -accel bvh    -o out/bench_bvh.ppm
	@$(call SAME,out/bench_none.ppm,out/bench_octree.ppm)
	@$(call SAME,out/bench_none.ppm,out/bench_bvh.ppm)

# octree vs BVH on the Stanford dragon (needs "make dragon-download")
bench-dragon: $(BIN)/render
	mkdir -p out
	$(BIN)/render -scene mesh -obj $(DRAGON) -rot -35 -n 300 -s 32 -accel octree -o out/bench_dragon_octree.ppm
	$(BIN)/render -scene mesh -obj $(DRAGON) -rot -35 -n 300 -s 32 -accel bvh    -o out/bench_dragon_bvh.ppm
	@$(call SAME,out/bench_dragon_octree.ppm,out/bench_dragon_bvh.ppm)

clean:
	rm -rf build bin

.PHONY: all lib apps examples check-layers models dragon-download demo run view view-dragon final dragon bench bench-dragon test cloud cloud-frames sim sim-blast clean
