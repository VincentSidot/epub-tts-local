import queue
import threading
import sounddevice as sd
import soundfile as sf
from typing import Generator, List, Optional
import numpy as np
from torch import Tensor
import torch


class AudioPlayerBase:
    def __init__(
        self,
        volume: float,
        orig_sr: int,
        target_punctuation_stop: Optional[List[str]],
        target_token_length: int,
        playback_blocksize: int,
        queue_maxsize: int = 3,  # Default queue size
        crossfade_ms: int = 100,
    ):
        self._queue = queue.Queue(queue_maxsize)
        self._stop_signal = object()
        self._TARGET_TOKEN_LENGTH = target_token_length
        self._PLAYBACK_BLOCKSIZE = playback_blocksize
        self._volume = volume
        self._previous_audio: Optional[Tensor] = None
        self._orig_sr = orig_sr

        self._crossfade_ms = crossfade_ms
        if target_punctuation_stop is None:
            self._TARGET_PUNCTUATION_STOP = [".", "!", "?", ";"]
        else:
            self._TARGET_PUNCTUATION_STOP = target_punctuation_stop

    def _cleanup(self, text: str) -> str:
        return " ".join(text.replace("\n", " ").strip().split())

    def _chunk_text(
        self, text_generator: Generator[str, None, None]
    ) -> Generator[str, None, None]:
        def split_text_iterator(text: Generator[str, None, None]):
            for block in text:
                yield from self._cleanup(block).split()

        current_chunk = []
        accumulated_length = 0
        ignore_next_token = False

        for word in split_text_iterator(text_generator):
            word_length = len(word)
            if accumulated_length + word_length >= self._TARGET_TOKEN_LENGTH:
                for punctuation in self._TARGET_PUNCTUATION_STOP:
                    if punctuation in word:
                        current_chunk.append(word)
                        yield " ".join(current_chunk)
                        current_chunk = []
                        accumulated_length = 0
                        ignore_next_token = True
                        break

            if ignore_next_token:
                ignore_next_token = False
            else:
                current_chunk.append(word)
                accumulated_length += word_length + 1

        if current_chunk:
            yield " ".join(current_chunk)

    def _crossfade(self, audio: Tensor) -> Tensor:
        overlap = int(self._orig_sr * self._crossfade_ms / 1000)
        if overlap == 0:
            return audio
        if self._previous_audio is None:
            self._previous_audio = audio[..., -overlap:]
            return audio[..., :-overlap]
        else:
            fade_out = torch.linspace(1, 0, self._previous_audio.size(-1)).to(
                audio.device
            )
            fade_in = torch.linspace(0, 1, overlap).to(audio.device)
            crossfaded = (
                self._previous_audio * fade_out + audio[..., :overlap] * fade_in
            )
            self._previous_audio = audio[..., -overlap:]
            return torch.cat([crossfaded, audio[..., overlap:-overlap]], dim=-1)

    def _data_stream(self, chunk_size) -> Generator[np.ndarray, None, None]:
        buffer = np.zeros((chunk_size, 1))
        buffer_size = 0
        while True:
            audio_data_tensor = self._queue.get()
            if audio_data_tensor is self._stop_signal:
                if buffer_size > 0:
                    buffer[buffer_size:] = 0
                    yield buffer
                yield self._stop_signal
                break

            scaled_audio = audio_data_tensor * self._volume
            crossfade_audio = self._crossfade(scaled_audio)
            audio_data = crossfade_audio.T.numpy().reshape(-1, 1)

            audio_data_size = audio_data.size
            fill_size_needed = chunk_size - buffer_size
            if audio_data_size < fill_size_needed:
                buffer[buffer_size : buffer_size + audio_data_size] = audio_data
                buffer_size += audio_data_size
            else:
                end_size = audio_data_size - fill_size_needed
                yield buffer
                idx = 0
                while end_size > chunk_size:
                    yield audio_data[idx : idx + chunk_size]
                    idx += chunk_size
                    end_size -= chunk_size
                buffer[:end_size] = audio_data[idx : idx + end_size]
                buffer_size = end_size

    def _generate(self, chunk_text: str) -> Tensor:
        """Implemented by subclasses"""
        raise NotImplementedError

    def stream_to_file(
        self, text_generator: Generator[str, None, None], file_path: str
    ) -> None:
        def writer_worker():
            with sf.SoundFile(
                file_path,
                mode="w",
                samplerate=self._orig_sr,
                channels=1,
                subtype="PCM_16",
            ) as file:
                for audio_data in self._data_stream(self._PLAYBACK_BLOCKSIZE):
                    if audio_data is self._stop_signal:
                        break
                    file.write(audio_data)

        writer_thread = threading.Thread(target=writer_worker)
        writer_thread.start()
        for chunk in self._chunk_text(text_generator):
            self._queue.put(self._generate(chunk))
        self._queue.put(self._stop_signal)
        writer_thread.join()

    def stream(self, text_generator: Generator[str, None, None]) -> None:
        data_generator = self._data_stream(self._PLAYBACK_BLOCKSIZE)
        event = threading.Event()

        def callback(outdata, frames, time, status):
            try:
                audio_data = next(data_generator)
            except StopIteration:
                raise sd.CallbackStop
            if audio_data is self._stop_signal:
                raise sd.CallbackStop
            outdata[:] = audio_data

        with sd.OutputStream(
            samplerate=self._orig_sr,
            channels=1,
            callback=callback,
            finished_callback=event.set,
            blocksize=self._PLAYBACK_BLOCKSIZE,
        ):
            for chunk in self._chunk_text(text_generator):
                self._queue.put(self._generate(chunk))
            self._queue.put(self._stop_signal)
            event.wait()

    def __call__(self, text: str) -> None:
        self.stream((t for t in [text]))
