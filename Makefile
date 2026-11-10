CC = gcc
CFLAGS = -Wall -Wextra -g -O2
# make SAN=1 -> build with the address + undefined-behaviour sanitizers (catches bad array writes)
SANFLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer
CFLAGS += $(if $(SAN),$(SANFLAGS))

LIB_SRC = framebuffer.c raster.c mesh.c texture.c renderer.c
HDR = $(wildcard *.h)

main: main.c $(LIB_SRC) $(HDR)
	$(CC) $(CFLAGS) main.c $(LIB_SRC) -o $@ -lm

# real-time window through plain Xlib (needs libx11-dev, no other library)
viewer: viewer.c $(LIB_SRC) $(HDR)
	$(CC) $(CFLAGS) viewer.c $(LIB_SRC) -o $@ -lm -lX11

models:
	python3 tools/gen_models.py

run: main
	./main

clean:
	rm -f main viewer

.PHONY: run clean models
