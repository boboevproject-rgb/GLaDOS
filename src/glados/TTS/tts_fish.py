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
KITTA_API_URL = "https://fishaudio.org/api/open/v1/speech/tts"
DEFAULT_MODEL = "s1"
KITTA_DEFAULT_MODEL = "fishaudio-s21pro-flash"
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
        model: str | None = None,
        sample_rate: int = 44100,
        backend: str = "official",
        fallback_voice: str | None = "ru_RU-dmitri-medium",
    ) -> None:
        self.api_key = api_key or os.environ.get("FISH_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Fish Audio API key not found. Set the FISH_API_KEY environment "
                "variable (create a key at https://fish.audio or https://fishaudio.org)."
            )
        if backend not in ("official", "kitta"):
            raise ValueError(f"Unknown Fish Audio backend: {backend}")
        self.backend = backend
        self.reference_id = reference_id
        self.model = model or (KITTA_DEFAULT_MODEL if backend == "kitta" else DEFAULT_MODEL)
        self.sample_rate = sample_rate
        self._session = requests.Session()
        # Cloud synthesis fails on expired credits, rate limits or a dropped
        # connection. Speaking in a local voice beats going mute, so a Piper
        # voice is loaded on first failure and used from then on.
        self._fallback_voice = fallback_voice
        self._fallback: object | None = None
        self._warned_fallback = False

    def _request_audio(self, text: str) -> bytes:
        if self.backend == "kitta":
            response = self._session.post(
                KITTA_API_URL,
                json={
                    "text": text,
                    "voiceId": self.reference_id,
                    "modelId": self.model,
                    "format": "mp3",
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=REQUEST_TIMEOUT_S,
            )
        else:
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
        return response.content

    def _speak_locally(self, text: str) -> NDArray[np.float32]:
        """Synthesize with the local Piper voice, resampled to our rate.

        The audio player is configured with this instance's `sample_rate`, so
        the fallback output must match it regardless of the voice's own rate.
        """
        if self._fallback_voice is None:
            return np.array([], dtype=np.float32)
        if self._fallback is None:
            from ..TTS import tts_piper

            self._fallback = tts_piper.SpeechSynthesizer(voice=self._fallback_voice)
        audio = self._fallback.generate_speech_audio(text)
        rate = self._fallback.sample_rate
        if rate != self.sample_rate and audio.size:
            n = int(len(audio) * self.sample_rate / rate)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
            ).astype(np.float32)
        return audio

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        text = text.strip()
        if not text:
            return np.array([], dtype=np.float32)
        try:
            content = self._request_audio(text)
        except requests.RequestException as e:
            if not self._warned_fallback:
                logger.error(f"Fish Audio TTS unavailable ({e}); switching to the local voice")
                self._warned_fallback = True
            return self._speak_locally(text)

        audio, wav_rate = sf.read(BytesIO(content), dtype="float32")
        if audio.ndim > 1:  # downmix, the player expects mono
            audio = audio.mean(axis=1)
        if wav_rate != self.sample_rate:
            n = int(len(audio) * self.sample_rate / wav_rate)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
            ).astype(np.float32)
        return audio
