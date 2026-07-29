"""Click a point the assistant located on the latest screenshot.

Takes coordinates as they appear in tools/screen.png and converts them to real
screen coordinates using the scale recorded by tools/screenshot.py, so the
assistant never has to do the arithmetic (getting it wrong means clicking the
wrong thing entirely).

pyautogui's corner failsafe is left enabled on purpose: slamming the mouse
into a screen corner aborts whatever is running.

Usage:
    python tools/click.py 820 460             # left click
    python tools/click.py 820 460 --double
    python tools/click.py 820 460 --right
    python tools/click.py 820 460 --move      # only move the cursor there
"""

import json
from pathlib import Path
import sys

import pyautogui

META = Path(__file__).parent / "screen.json"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) < 2:
        print("нужны координаты со снимка: python tools/click.py X Y [--double|--right|--move]")
        return 2

    if not META.exists():
        print("нет tools/screen.json — сначала сделай снимок: python tools/screenshot.py")
        return 2

    meta = json.loads(META.read_text(encoding="utf-8"))
    scale = meta["scale"]
    x, y = round(int(args[0]) * scale), round(int(args[1]) * scale)

    width, height = meta["screen_size"]
    if not (0 <= x < width and 0 <= y < height):
        print(f"точка {x},{y} вне экрана {width}x{height} — проверь координаты по снимку")
        return 1

    pyautogui.moveTo(x, y, duration=0.2)
    if "--move" in flags:
        action = "курсор наведён"
    elif "--double" in flags:
        pyautogui.doubleClick()
        action = "двойной клик"
    elif "--right" in flags:
        pyautogui.rightClick()
        action = "правый клик"
    else:
        pyautogui.click()
        action = "клик"
    print(f"{action} в {x},{y} (со снимка {args[0]},{args[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
