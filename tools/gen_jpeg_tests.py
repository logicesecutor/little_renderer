#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["pillow"]
# ///
"""gen_jpeg_tests.py - writes examples/jpeg_tests.h: JPEG files written by libjpeg (through
Pillow) and the pixels libjpeg decodes from them, for examples/test_jpeg.c (core/jpeg.c must
decode them to within a step of libjpeg's own pixels).

    uv run tools/gen_jpeg_tests.py

Baseline and progressive; 4:4:4, 4:2:2, 4:2:0 and grey; optimized Huffman tables; restart
markers; 16-bit quantization tables (SOF1); RGB without the colour transform (Adobe); a size
that is not a whole number of MCUs.
"""
import io

from PIL import Image

W, H = 29, 17


def pattern(mode):
    img = Image.new("RGB", (W, H))
    px = img.load()
    for y in range(H):
        for x in range(W):
            r = (x * 7 + y * 3) % 256
            g = (128 + 100 * ((x // 4 + y // 3) % 2) + x * y) % 256
            b = (255 - x * 5 + (x * y * 13) % 37) % 256
            px[x, y] = (r, g, b)
    return img.convert(mode)


CASES = [
    ("baseline 4:4:4", "RGB", dict(quality=90, subsampling=0)),
    ("baseline 4:2:2", "RGB", dict(quality=85, subsampling=1)),
    ("baseline 4:2:0", "RGB", dict(quality=75, subsampling=2)),
    ("baseline grey", "L", dict(quality=80)),
    ("baseline optimized tables", "RGB", dict(quality=70, subsampling=2, optimize=True)),
    ("baseline quality 100", "RGB", dict(quality=100, subsampling=0)),
    ("progressive 4:2:0", "RGB", dict(quality=80, subsampling=2, progressive=True)),
    ("progressive 4:4:4", "RGB", dict(quality=95, subsampling=0, progressive=True)),
    ("progressive 4:2:2", "RGB", dict(quality=60, subsampling=1, progressive=True)),
    ("progressive grey", "L", dict(quality=85, progressive=True)),
    ("restart markers, baseline", "RGB", dict(quality=80, subsampling=2, restart_marker_blocks=3)),
    ("restart markers, progressive", "RGB", dict(quality=80, subsampling=2, progressive=True, restart_marker_rows=1)),
    ("16-bit tables (SOF1)", "RGB", dict(subsampling=0, qtables=[[300] + [2] * 63, [400] + [3] * 63])),
    ("RGB, no transform (Adobe)", "RGB", dict(quality=90, subsampling=0, keep_rgb=True)),
]


def main():
    out = ["// jpeg_tests.h - written by tools/gen_jpeg_tests.py (do not edit): JPEG files written by",
           "// libjpeg (Pillow) and the RGB pixels libjpeg decodes from them, for test_jpeg.c",
           f"#define JPEG_TEST_W {W}", f"#define JPEG_TEST_H {H}",
           "typedef struct { const char *name; const unsigned char *jpg; unsigned size; const unsigned char *rgb; } JpegTest;"]
    names = []
    for i, (name, mode, opts) in enumerate(CASES):
        buf = io.BytesIO()
        try:
            pattern(mode).save(buf, "JPEG", **opts)
        except TypeError as e:          # an option this Pillow does not know
            print("skipped", name, e)
            continue
        data = buf.getvalue()
        rgb = Image.open(io.BytesIO(data)).convert("RGB").tobytes()
        out.append(f"static const unsigned char jpg_{i}[{len(data)}] = {{{','.join(map(str, data))}}};")
        out.append(f"static const unsigned char rgb_{i}[{len(rgb)}] = {{{','.join(map(str, rgb))}}};")
        names.append((name, i, len(data)))
        print(f"{name}: {len(data)} bytes")
    out.append("static const JpegTest jpeg_tests[] = {")
    for name, i, n in names:
        out.append(f'    {{"{name}", jpg_{i}, {n}, rgb_{i}}},')
    out.append("};")
    open("examples/jpeg_tests.h", "w", newline="\n").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
