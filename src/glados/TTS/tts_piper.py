"""Piper-based speech synthesizer with multilingual (e.g. Russian) voice support.

Uses the `piper-tts` package (ONNX runtime + bundled espeak-ng phonemizer),
so no English-only phonemizer model is involved. Voice models are stored in
``models/TTS/piper/<voice_name>.onnx`` with a ``.onnx.json`` config next to them.

Russian voices can be downloaded from https://huggingface.co/rhasspy/piper-voices
(e.g. ``ru_RU-dmitri-medium``, ``ru_RU-irina-medium``).
"""

import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from numpy.typing import NDArray
import piper as _piper_pkg
from piper import PiperVoice

from ..utils.resources import resource_path

PIPER_VOICES_DIR = resource_path("models/TTS/piper")


def _safe_espeak_data_dir() -> Path | None:
    """Return an espeak-ng data dir at an ASCII-only path.

    espeak-ng opens files with ANSI APIs on Windows and fails on non-ASCII
    paths (e.g. a venv under a Cyrillic user folder), so the bundled data is
    copied once to LOCALAPPDATA if needed.
    """
    src = Path(_piper_pkg.__file__).parent / "espeak-ng-data"
    if not src.is_dir():
        return None
    try:
        str(src).encode("ascii")
        return src
    except UnicodeEncodeError:
        base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
        dst = Path(base) / "piper" / "espeak-ng-data"
        if not dst.is_dir():
            shutil.copytree(src, dst)
        return dst


def get_piper_voices(path: Path = PIPER_VOICES_DIR) -> list[str]:
    """List locally available Piper voice names (onnx files in the piper models dir)."""
    if not path.is_dir():
        return []
    return sorted(p.stem for p in path.glob("*.onnx"))


class SpeechSynthesizer:
    """Piper text-to-speech synthesizer conforming to SpeechSynthesizerProtocol."""

    sample_rate: int

    def __init__(self, voice: str = "ru_RU-dmitri-medium", use_cuda: bool = False) -> None:
        model_path = PIPER_VOICES_DIR / f"{voice}.onnx"
        if not model_path.is_file():
            available = get_piper_voices()
            raise FileNotFoundError(
                f"Piper voice model not found: {model_path}. "
                f"Locally available piper voices: {available or 'none'}. "
                "Download voices from https://huggingface.co/rhasspy/piper-voices"
            )
        espeak_dir = _safe_espeak_data_dir()
        if espeak_dir is not None:
            self.voice = PiperVoice.load(model_path, use_cuda=use_cuda, espeak_data_dir=espeak_dir)
        else:
            self.voice = PiperVoice.load(model_path, use_cuda=use_cuda)
        self.sample_rate = self.voice.config.sample_rate

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        chunks = [chunk.audio_int16_array for chunk in self.voice.synthesize(text)]
        if not chunks:
            return np.array([], dtype=np.float32)
        audio_int16 = np.concatenate(chunks)
        return (audio_int16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
