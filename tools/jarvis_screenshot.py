"""Capture the screen so the assistant can look at it.

The assistant runs this and then reads the resulting PNG, which its vision
model sees directly. The image is downscaled because a full-resolution frame
is slow to transfer and adds no detail the model can use.

Because it is downscaled, coordinates read off the image are NOT screen
coordinates. The scale is recorded in screen.json so tools/click.py can
convert them; never pass image coordinates straight to pyautogui.

Usage:
    python tools/screenshot.py            # all monitors -> tools/screen.png
    python tools/screenshot.py out.png    # custom destination
"""

import json
from pathlib import Path
import sys

from PIL import ImageGrab

MAX_WIDTH = 1600
HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "screen.png"
META = HERE / "screen.json"


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)

    image = ImageGrab.grab(all_screens=True)
    real_width, real_height = image.size
    if image.width > MAX_WIDTH:
        height = round(image.height * MAX_WIDTH / image.width)
        image = image.resize((MAX_WIDTH, height))
    image.save(out, optimize=True)

    scale = real_width / image.width
    META.write_text(
        json.dumps(
            {
                "image": str(out),
                "image_size": list(image.size),
                "screen_size": [real_width, real_height],
                "scale": scale,
            }
        ),
        encoding="utf-8",
    )
    print(
        f"{out} — снимок {image.width}x{image.height}, экран {real_width}x{real_height}. "
        f"Координаты со снимка передавай в tools/click.py, он пересчитает (масштаб {scale:.3g})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
