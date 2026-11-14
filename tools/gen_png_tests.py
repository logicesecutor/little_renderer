#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["pillow"]
# ///
"""gen_png_tests.py - writes examples/png_tests.h: PNG files made by other encoders, and the
pixels they hold, for examples/test_png.c (core/png.c must decode them exactly).

    uv run tools/gen_png_tests.py

Two kinds: files written by Pillow (libpng-like output, the common colour types), and files
written here with Python's zlib (its dynamic Huffman codes, and stored blocks at level 0)
covering what Pillow does not write: interlacing, 16 bits, 2- and 4-bit samples, every
filter type, transparency keys. The expected pixels follow core/png.h's rules: RGBA 8 bits,
16-bit samples keep their high byte, fewer bits are scaled to 0..255.
"""
import io
import struct
import zlib

from PIL import Image

W, H = 13, 9


def val(x, y, c, maxv):
    """the test pattern: every channel a different mix, wrapping in 0..maxv"""
    return (x * (37 + 11 * c) + y * (53 + 7 * c) + 17 * c + x * y * (c + 1)) % (maxv + 1)


def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def filt(row, prior, bpp, f):
    out = bytearray(len(row))
    for i in range(len(row)):
        a = row[i - bpp] if i >= bpp else 0
        b = prior[i]
        c = prior[i - bpp] if i >= bpp else 0
        p = [0, a, b, (a + b) // 2, paeth(a, b, c)][f]
        out[i] = (row[i] - p) & 255
    return bytes(out)


def pack_row(samples, depth):
    if depth == 8:
        return bytes(samples)
    if depth == 16:
        return b"".join(struct.pack(">H", s) for s in samples)
    out, acc, n = bytearray(), 0, 0
    for s in samples:
        acc = (acc << depth) | s
        n += depth
        if n == 8:
            out.append(acc)
            acc, n = 0, 0
    if n:
        out.append(acc << (8 - n))
    return bytes(out)


def own_png(ctype, depth, spp, pixels, interlace, level, plte=None, trns=None):
    """pixels[y][x] = tuple of raw samples; rows filtered with every filter type in turn"""
    bits = spp * depth
    bpp = max(1, bits // 8)
    passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)] \
        if interlace else [(0, 0, 1, 1)]
    raw, k = bytearray(), 0
    for xs, ys, dx, dy in passes:
        xsel = list(range(xs, W, dx))
        ysel = list(range(ys, H, dy))
        if not xsel or not ysel:
            continue
        prior = bytes(len(pack_row([0] * (len(xsel) * spp), depth)))
        for y in ysel:
            row = pack_row([s for x in xsel for s in pixels[y][x]], depth)
            f = k % 5
            k += 1
            raw += bytes([f]) + filt(row, prior, bpp, f)
            prior = row
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, depth, ctype, 0, 0, interlace))
    if plte:
        out += chunk(b"PLTE", plte)
    if trns:
        out += chunk(b"tRNS", trns)
    z = zlib.compress(bytes(raw), level)
    out += chunk(b"IDAT", z[:len(z) // 2]) + chunk(b"IDAT", z[len(z) // 2:])    # split across chunks
    return out + chunk(b"IEND", b"")


def scale(v, depth):
    return v >> 8 if depth == 16 else (v if depth == 8 else v * 255 // ((1 << depth) - 1))


cases = []   # (name, png bytes, file channels, expected RGBA bytes)

# --- Pillow
for mode, ch in (("RGB", 3), ("RGBA", 4), ("L", 1), ("LA", 2)):
    im = Image.new(mode, (W, H))
    px = [[tuple(val(x, y, c, 255) for c in range(ch)) for x in range(W)] for y in range(H)]
    im.putdata([p if ch > 1 else p[0] for row in px for p in row])
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    exp = bytearray()
    for row in px:
        for p in row:
            if ch == 1:
                exp += bytes((p[0], p[0], p[0], 255))
            elif ch == 2:
                exp += bytes((p[0], p[0], p[0], p[1]))
            elif ch == 3:
                exp += bytes(p + (255,))
            else:
                exp += bytes(p)
    cases.append(("pillow " + mode, b.getvalue(), ch, bytes(exp)))

im = Image.new("P", (W, H))
pal = [(i * 13 % 256, i * 29 % 256, i * 71 % 256) for i in range(16)]
im.putpalette([c for p in pal for c in p])
idx = [[val(x, y, 0, 15) for x in range(W)] for y in range(H)]
im.putdata([i for row in idx for i in row])
b = io.BytesIO()
im.save(b, "PNG", transparency=bytes([255 - 16 * i for i in range(16)]))
exp = bytearray()
for row in idx:
    for i in row:
        exp += bytes(pal[i] + (255 - 16 * i,))
cases.append(("pillow P + tRNS", b.getvalue(), 4, bytes(exp)))

im = Image.new("1", (W, H))
bw = [[val(x, y, 0, 1) for x in range(W)] for y in range(H)]
im.putdata([255 * v for row in bw for v in row])
b = io.BytesIO()
im.save(b, "PNG")
exp = b"".join(bytes((255 * v,) * 3 + (255,)) for row in bw for v in row)
cases.append(("pillow 1-bit", b.getvalue(), 1, exp))

# --- Python zlib


def own_case(name, ctype, depth, spp, interlace, level, file_ch, plte=None, trns=None, key=None):
    maxv = (1 << depth) - 1
    px = [[tuple(val(x, y, c, maxv) for c in range(spp)) for x in range(W)] for y in range(H)]
    if key is not None:                    # make some pixels match the transparency key
        for y in range(H):
            for x in range(0, W, 3):
                px[y][x] = key
    data = own_png(ctype, depth, spp, px, interlace, level, plte, trns)
    exp = bytearray()
    for row in px:
        for p in row:
            if ctype == 3:
                i = p[0]
                exp += bytes((plte[3 * i], plte[3 * i + 1], plte[3 * i + 2], trns[i] if trns and i < len(trns) else 255))
                continue
            s = [scale(v, depth) for v in p]
            a = 0 if key is not None and p == key else 255
            if ctype == 0:
                exp += bytes((s[0], s[0], s[0], a))
            elif ctype == 4:
                exp += bytes((s[0], s[0], s[0], s[1]))
            elif ctype == 2:
                exp += bytes((s[0], s[1], s[2], a))
            else:
                exp += bytes(s)
    cases.append((name, data, file_ch, bytes(exp)))


own_case("grey 2-bit interlaced", 0, 2, 1, 1, 9, 1)
own_case("RGB 16-bit interlaced, stored blocks", 2, 16, 3, 1, 0, 3)
own_case("RGB 8-bit, transparent key", 2, 8, 3, 0, 1, 4, trns=struct.pack(">HHH", 10, 20, 30), key=(10, 20, 30))
own_case("grey + alpha 16-bit", 4, 16, 2, 0, 6, 2)
own_case("RGBA 16-bit interlaced", 6, 16, 4, 1, 9, 4)
own_case("grey 4-bit", 0, 4, 1, 0, 9, 1)
plte = bytes(c for i in range(16) for c in (i * 15, 255 - i * 15, (i * 97) % 256))
own_case("palette 4-bit interlaced", 3, 4, 1, 1, 9, 3, plte=plte)

with open("examples/png_tests.h", "w") as f:
    f.write("// png_tests.h - written by tools/gen_png_tests.py (do not edit): PNG files from other\n"
            "// encoders and the RGBA pixels they hold, for test_png.c\n")
    f.write("#define PNG_TEST_W %d\n#define PNG_TEST_H %d\n" % (W, H))
    f.write("typedef struct { const char *name; const unsigned char *png; unsigned size; int channels; "
            "const unsigned char *rgba; } PngTest;\n")
    for i, (name, data, ch, exp) in enumerate(cases):
        for tag, d in (("png", data), ("rgba", exp)):
            f.write("static const unsigned char %s_%d[%d] = {" % (tag, i, len(d)))
            f.write(",".join(str(c) for c in d))
            f.write("};\n")
    f.write("static const PngTest png_tests[] = {\n")
    for i, (name, data, ch, exp) in enumerate(cases):
        f.write('    {"%s", png_%d, %d, %d, rgba_%d},\n' % (name, i, len(data), ch, i))
    f.write("};\n")
print("wrote examples/png_tests.h: %d files" % len(cases))
