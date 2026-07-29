"""GigaAM v2 ASR — a Russian-first speech recognizer (Sber, MIT).

Parakeet is multilingual but only mediocre on Russian; GigaAM is trained on
~700k hours of Russian speech and is markedly more accurate on it (single-digit
WER on CPU), which matters for wake-word and command recognition.

Audio is fed straight from memory as float32 @ 16 kHz — the same format the
audio pipeline already produces — so no temp WAV files are involved.

Requires the optional `gigaam` extra: ``uv sync --extra cpu --extra gigaam``.
"""

from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import torch

SAMPLE_RATE = 16000
DEFAULT_MODEL = "v2_rnnt"  # RNN-T: best accuracy/latency balance for short commands


class AudioTranscriber:
    """GigaAM-based transcriber conforming to TranscriberProtocol."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        import gigaam  # imported lazily: heavy (torch) and optional

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 is a GPU-only win; on CPU it is slower and less accurate
        logger.info(f"Loading GigaAM ASR '{model_name}' on {self.device}...")
        self.model = gigaam.load_model(
            model_name,
            device=self.device,
            fp16_encoder=self.device != "cpu",
        )

    def transcribe(self, audio_source: NDArray[Any]) -> str:
        """Transcribe float32 mono audio sampled at 16 kHz."""
        audio = np.asarray(audio_source, dtype=np.float32).squeeze()
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if audio.size == 0:
            return ""
        wav = torch.from_numpy(audio).to(self.model._device).to(self.model._dtype).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=self.model._device)
        with torch.inference_mode():
            encoded, encoded_len = self.model.forward(wav, length)
            text: str = self.model.decoding.decode(self.model.head, encoded, encoded_len)[0]
        return text.strip()

    def transcribe_file(self, audio_path: Path) -> str:
        """Transcribe an audio file (resampled by GigaAM's own loader)."""
        result: str = self.model.transcribe(str(audio_path))
        return result.strip()
