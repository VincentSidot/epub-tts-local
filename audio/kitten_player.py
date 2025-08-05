import torch
from torch import Tensor
from typing import Optional
from kittentts import KittenTTS
from .base import AudioPlayerBase


class KittenAudioPlayer(AudioPlayerBase):
    def __init__(
        self,
        volume: float = 0.5,
        target_punctuation_stop=None,
        target_token_length: int = 64,
        playback_blocksize: int = 640,
        queue_maxsize: int = 2,  # Few items in queue for Kitten as it's blazingly fast 🔥
        kitten_voice: Optional[str] = None,
        kitten_speed: float = 1.0,
        crossfade_ms: int = 30,  # Kitten player has less crossfade time than base player
    ):
        model = KittenTTS("KittenML/kitten-tts-nano-0.1")
        self._model = model
        self._kitten_voice = kitten_voice
        self._kitten_speed = kitten_speed

        super().__init__(
            volume=volume,
            orig_sr=24000,  # Kitten fixed SR
            target_punctuation_stop=target_punctuation_stop,
            target_token_length=target_token_length,
            playback_blocksize=playback_blocksize,
            queue_maxsize=queue_maxsize,
        )

    def _generate(self, chunk_text: str) -> Tensor:
        audio_np = self._model.generate(
            chunk_text,
            voice=self._kitten_voice,
            speed=self._kitten_speed,
        )
        return torch.from_numpy(audio_np).unsqueeze(0)  # (1, N)
