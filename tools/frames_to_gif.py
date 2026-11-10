#!/usr/bin/env python3
"""frames_to_gif.py - the frames of bin/sim (or any folder of frame_NNNN.ppm) into a GIF.

    python3 tools/frames_to_gif.py out/sim out/sim.gif 30

Needs Pillow (pip install pillow). Every frame gets its own 256-color palette (adaptive,
with dithering), which suits the smooth gradients of the path traced images.
"""
import glob
import os
import sys

from PIL import Image


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    folder, out = sys.argv[1], sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    files = sorted(glob.glob(os.path.join(folder, "frame_*.ppm")))
    if not files:
        print("no frame_*.ppm in", folder)
        return 1
    frames = [Image.open(f).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                                    dither=Image.Dither.FLOYDSTEINBERG) for f in files]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(round(1000 / fps)),
                   loop=0, optimize=False, disposal=1)
    print("wrote %s: %d frames, %dx%d, %.0f fps" % (out, len(frames), frames[0].width, frames[0].height, fps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
