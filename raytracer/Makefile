# Ray tracer build. Every command and option is documented in ../USAGE.md.
CC = gcc
CFLAGS = -Wall -Wextra -O2 -g
# OpenMP only spreads the rows over the CPU cores. "make SINGLE=1" builds without it.
CFLAGS += $(if $(SINGLE),,-fopenmp)
# "make SAN=1" adds the address + undefined-behaviour sanitizers
SANFLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer
CFLAGS += $(if $(SAN),$(SANFLAGS))

# the engine, shared by both programs
ENGINE = geometry.c octree.c bvh.c scene.c meshfile.c tracer.c image.c options.c
HDR = $(wildcard *.h)

all: raytracer rtview

# offline renderer: writes a PPM
raytracer: main.c $(ENGINE) $(HDR)
	$(CC) $(CFLAGS) main.c $(ENGINE) -o $@ -lm

# real-time viewer: an X11 window you can fly through (needs libx11-dev)
rtview: rtview.c $(ENGINE) $(HDR)
	$(CC) $(CFLAGS) rtview.c $(ENGINE) -o $@ -lm -lX11

# quick previews of every scene
run: raytracer
	mkdir -p out
	./raytracer -scene boxes
	./raytracer -scene spheres
	./raytracer -scene mesh

# open the real-time viewer on the Cornell box (try: make view ARGS="-scene spheres")
view: rtview
	mkdir -p out
	./rtview $(ARGS)

# slow, clean versions
final: raytracer
	mkdir -p out
	./raytracer -scene boxes -n 600 -s 1024 -o out/cornell_boxes_final.ppm
	./raytracer -scene spheres -n 600 -s 1024 -o out/cornell_spheres_final.ppm
	./raytracer -scene spheres -n 600 -s 1024 -clamp 4 -o out/cornell_spheres_final_clamped.ppm
	./raytracer -scene mesh -n 600 -s 1024 -o out/cornell_mesh_final.ppm

# all acceleration structures on the torus (2304 triangles); the images must be identical
bench: raytracer
	mkdir -p out
	./raytracer -scene mesh -n 200 -s 16 -accel none   -o out/bench_none.ppm
	./raytracer -scene mesh -n 200 -s 16 -accel octree -o out/bench_octree.ppm
	./raytracer -scene mesh -n 200 -s 16 -accel bvh    -o out/bench_bvh.ppm
	cmp out/bench_none.ppm out/bench_octree.ppm && cmp out/bench_none.ppm out/bench_bvh.ppm && echo "images identical"

# octree vs BVH on the Stanford dragon (needs "make dragon-download")
bench-dragon: raytracer
	mkdir -p out
	./raytracer -scene mesh -obj $(DRAGON) -rot -35 -n 300 -s 32 -accel octree -o out/bench_dragon_octree.ppm
	./raytracer -scene mesh -obj $(DRAGON) -rot -35 -n 300 -s 32 -accel bvh    -o out/bench_dragon_bvh.ppm
	cmp out/bench_dragon_octree.ppm out/bench_dragon_bvh.ppm && echo "images identical"

# The Stanford dragon (871,414 triangles). Download it once with "make dragon-download".
# Credit: Stanford University Computer Graphics Laboratory, 3D Scanning Repository.
DRAGON = ../models/dragon_recon/dragon_vrip.ply
dragon: raytracer
	mkdir -p out
	./raytracer -scene mesh -obj $(DRAGON) -rot -35 -n 600 -s 512 -o out/cornell_dragon.ppm
	./raytracer -scene mesh -obj $(DRAGON) -rot -35 -mat glass -clamp 4 -n 600 -s 512 -o out/cornell_dragon_glass.ppm

# fly around the dragon in real time
view-dragon: rtview
	mkdir -p out
	./rtview -scene mesh -obj $(DRAGON) -rot -35 $(ARGS)

dragon-download:
	mkdir -p ../models
	curl -L -o ../models/dragon_recon.tar.gz http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz
	tar xzf ../models/dragon_recon.tar.gz -C ../models
	rm ../models/dragon_recon.tar.gz

clean:
	rm -f raytracer rtview

.PHONY: all run view final bench bench-dragon dragon view-dragon dragon-download clean
