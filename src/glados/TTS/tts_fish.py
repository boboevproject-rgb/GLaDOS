"""Fish Audio cloud TTS with community voice models (e.g. Russian J.A.R.V.I.S. clones).

Uses the official Fish Audio HTTP API (https://docs.fish.audio) with a
``reference_id`` pointing to a public voice model from the fish.audio catalog.
Requires an API key in the ``FISH_API_KEY`` environment variable (create one at
https://fish.audio -> API keys). Voices are addressed from the config as
``fish:<reference_id>``.

This is a cloud engine: synthesis needs internet and spends Fish Audio credits.
Keep a local engine (Piper/Silero) configured as a fallback for offline use.
"""

from io import BytesIO
import os

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import requests
import soundfile as sf

API_URL = "https://api.fish.audio/v1/tts"
DEFAULT_MODEL = "s1"
REQUEST_TIMEOUT_S = 30


class SpeechSynthesizer:
    """Fish Audio cloud synthesizer conforming to SpeechSynthesizerProtocol."""

    sample_rate: int
    # The S1 model normalizes numbers/dates in the text's own language, so the
    # English-only SpokenTextConverter must be skipped for this engine.
    handles_text_normalization: bool = True

    def __init__(
        self,
        reference_id: str,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        sample_rate: int = 44100,
    ) -> None:
        self.api_key = api_key or os.environ.get("FISH_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Fish Audio API key not found. Set the FISH_API_KEY environment "
                "variable (create a key at https://fish.audio)."
            )
        self.reference_id = reference_id
        self.model = model
        self.sample_rate = sample_rate
        self._session = requests.Session()

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        text = text.strip()
        if not text:
            return np.array([], dtype=np.float32)
        try:
            response = self._session.post(
                API_URL,
                json={
                    "text": text,
                    "reference_id": self.reference_id,
                    "format": "wav",
                    "sample_rate": self.sample_rate,
                    "latency": "balanced",
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Model": self.model,
                },
                timeout=REQUEST_TIMEOUT_S,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Fish Audio TTS request failed: {e}")
            return np.array([], dtype=np.float32)

        audio, wav_rate = sf.read(BytesIO(response.content), dtype="float32")
        if audio.ndim > 1:  # downmix, the player expects mono
            audio = audio.mean(axis=1)
        if wav_rate != self.sample_rate:
            n = int(len(audio) * self.sample_rate / wav_rate)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
            ).astype(np.float32)
        return audio
