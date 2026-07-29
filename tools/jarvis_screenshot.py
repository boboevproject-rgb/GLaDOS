"""Capture the screen so the assistant can look at it.

The assistant runs this and then reads the resulting PNG, which its vision
model sees directly. The image is downscaled because a full 4K frame is slow
to transfer and adds no detail the model can use.

Usage:
    python tools/screenshot.py            # all monitors, saved to tools/screen.png
    python tools/screenshot.py out.png    # custom destination
"""

from pathlib import Path
import sys

from PIL import ImageGrab

MAX_WIDTH = 1600
DEFAULT_OUT = Path(__file__).parent / "screen.png"


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)

    image = ImageGrab.grab(all_screens=True)
    if image.width > MAX_WIDTH:
        height = round(image.height * MAX_WIDTH / image.width)
        image = image.resize((MAX_WIDTH, height))
    image.save(out, optimize=True)
    print(f"{out} ({image.width}x{image.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
