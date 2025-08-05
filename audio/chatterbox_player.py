import torch
import logging
from torch import Tensor
from typing import Generator, Optional
from chatterbox.tts import ChatterboxTTS
from .base import AudioPlayerBase


class ChatterboxAudioPlayer(AudioPlayerBase):
    def __init__(
        self,
        volume: float = 0.5,
        target_punctuation_stop=None,
        target_token_length: int = 64,
        playback_blocksize: int = 640,
        queue_maxsize: int = 5,  # Higher queue size for Chatterbox has it's takes longer to generate audio
        device: Optional[str] = None,
        voice_clone: Optional[str] = None,
        crossfade_ms: int = 100,  # Default crossfade time
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        logging.info(f"Using device: {device}")
        model = ChatterboxTTS.from_pretrained(device=device)
        self._model = model
        self._voice_clone = voice_clone

        super().__init__(
            volume=volume,
            orig_sr=model.sr,
            target_punctuation_stop=target_punctuation_stop,
            target_token_length=target_token_length,
            playback_blocksize=playback_blocksize,
            queue_maxsize=queue_maxsize,
        )

    def _generate(self, chunk_text: str) -> Tensor:
        output = self._model.generate(chunk_text, audio_prompt_path=self._voice_clone)
        if isinstance(output, Generator):
            return torch.cat(list(output), dim=-1)
        elif isinstance(output, Tensor):
            return output
        else:
            raise ValueError("Unexpected Chatterbox output type.")
