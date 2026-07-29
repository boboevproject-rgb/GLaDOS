"""Type text into whatever window currently has focus.

Text goes through the clipboard and Ctrl+V rather than simulated keystrokes:
pyautogui's typewrite cannot produce Cyrillic on Windows and would silently
emit garbage. The previous clipboard contents are restored afterwards so the
user does not lose what they had copied.

Usage:
    python tools/type.py "текст для ввода"
    python tools/type.py "логин" --enter      # press Enter afterwards
    python tools/type.py --key enter          # just press a key
    python tools/type.py --hotkey ctrl s      # key combination
"""

import sys
import time

import pyautogui
import pyperclip


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print('нужен текст: python tools/type.py "текст" [--enter]')
        return 2

    if args[0] == "--key":
        if len(args) < 2:
            print("нужно имя клавиши, например: --key enter")
            return 2
        pyautogui.press(args[1])
        print(f"нажата клавиша {args[1]}")
        return 0

    if args[0] == "--hotkey":
        keys = args[1:]
        if not keys:
            print("нужны клавиши, например: --hotkey ctrl s")
            return 2
        pyautogui.hotkey(*keys)
        print(f"нажато сочетание {'+'.join(keys)}")
        return 0

    text = args[0]
    saved = ""
    try:
        saved = pyperclip.paste()
    except Exception:
        pass  # empty or non-text clipboard is fine, nothing to restore

    pyperclip.copy(text)
    time.sleep(0.05)  # give the clipboard a moment to settle
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.1)

    if "--enter" in args:
        pyautogui.press("enter")

    try:
        pyperclip.copy(saved)
    except Exception:
        pass

    print(f"введено {len(text)} символов" + (" и нажат Enter" if "--enter" in args else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
