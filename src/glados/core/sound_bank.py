"""Bank of pre-recorded voice lines for instant, zero-cost reactions.

Loads WAV files (e.g. the Russian J.A.R.V.I.S. movie-dub sound pack) grouped
into categories like "startup", "wake_ack" or "processing". The engine plays
them directly through the audio queue, so frequent reactions skip the TTS
engine entirely — no latency and no cloud credits spent.

The audio files themselves are user-provided local assets and are not part of
the repository.
"""

from pathlib import Path
import random

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import soundfile as sf


class SoundBank:
    """Pre-loaded, resampled voice lines grouped by category."""

    def __init__(self, sounds_dir: str | Path, sounds: dict[str, list[str]], sample_rate: int) -> None:
        """Eagerly load all configured WAVs, downmixed to mono at `sample_rate`.

        Args:
            sounds_dir: Base directory with the WAV files.
            sounds: Mapping of category name to WAV filenames inside `sounds_dir`.
            sample_rate: Target sample rate (must match the TTS engine's rate,
                as both share one audio player).
        """
        self.sample_rate = sample_rate
        self._bank: dict[str, list[tuple[NDArray[np.float32], str]]] = {}
        base = Path(sounds_dir)
        for category, filenames in sounds.items():
            clips: list[tuple[NDArray[np.float32], str]] = []
            for name in filenames:
                path = base / name
                try:
                    audio, rate = sf.read(path, dtype="float32")
                except Exception as e:
                    logger.warning(f"SoundBank: cannot load '{path}': {e}")
                    continue
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if rate != sample_rate:
                    n = int(len(audio) * sample_rate / rate)
                    audio = np.interp(
                        np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
                    ).astype(np.float32)
                clips.append((audio, path.stem.strip()))
            if clips:
                self._bank[category] = clips
        loaded = {cat: len(clips) for cat, clips in self._bank.items()}
        logger.info(f"SoundBank loaded: {loaded}")

    def has(self, category: str) -> bool:
        return category in self._bank

    def random(self, category: str) -> tuple[NDArray[np.float32], str] | None:
        """Return a random (audio, phrase) clip of the category, or None."""
        clips = self._bank.get(category)
        if not clips:
            return None
        return random.choice(clips)
