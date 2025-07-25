import queue
import threading
import sounddevice as sd
import soundfile as sf
import logging
from typing import Generator
import torchaudio.transforms as ta
import numpy as np
from typing import Generator, List, Optional
from torch import Tensor
import torch

# Use of https://github.com/rsxdalv/chatterbox/tree/fast
from chatterbox.tts import ChatterboxTTS


class AudioPlayer:
    def __init__(
        self,
        volume: float = 0.5,
        sampling_rate_ratio: float = 1.0,  # This is used to adjust the sampling rate of the audio playback
        target_punctuation_stop: Optional[List[str]] = None,
        target_token_length: int = 64,
        playback_blocksize: int = 640,
        queue_maxsize: int = 10,
        device: Optional[str] = None,  # Force the device
        voice_clone: Optional[str] = None,  # Optional voice clone for text generation
    ) -> None:
        """
        Initialize the AudioPlayer with a pre-trained model.
        Args:
            volume (float): Volume level for audio playback (0.0 to 1.0).
            sampling_rate_ratio (float): Ratio to adjust the sampling rate of the audio playback.
            target_punctuation_stop (Optional[List[str]]): List of punctuation marks to stop at when chunking text.
            target_token_length (int): Target length for each text chunk in tokens.
            playback_blocksize (int): Size of the audio buffer for playback.
            queue_size (int): Maximum size of the queue before generation pause
        """
        # Automatically detect the best available device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        logging.info(f"Using device: {device}")

        self.__device = device
        self.__model = ChatterboxTTS.from_pretrained(device=device)
        self.__queue = queue.Queue(queue_maxsize)
        self.__stop_signal = object()
        self.__TARGET_TOKEN_LENGTH = target_token_length
        self.__PLAYBACK_BLOCKSIZE = playback_blocksize
        self.__volume = volume
        self.__previous_audio: Optional[Tensor] = None
        self.__voice_clone = voice_clone

        self.__orig_sr = self.__model.sr
        self.__target_sr = int(self.__orig_sr * sampling_rate_ratio)

        if self.__target_sr != self.__orig_sr:
            self.__resampler = ta.Resample(
                orig_freq=self.__orig_sr, new_freq=self.__target_sr
            )
        else:
            self.__resampler = None

        if target_punctuation_stop is None:
            self.__TARGET_PUNCTUATION_STOP = [".", "!", "?", ";"]
        else:
            self.__TARGET_PUNCTUATION_STOP = target_punctuation_stop

    @property
    def device(self) -> str:
        return self.__device

    def torch_compile(self) -> None:
        """
        Compile the model using torch.compile for performance optimization.
        This is optional and can be skipped if not needed.
        """
        try:

            if self.__device == "cuda":
                backend = "cudagraphs"
            else:
                backend = "inductor"

            self.__model.t3._step_compilation_target = torch.compile(
                self.__model.t3._step_compilation_target,
                fullgraph=True,
                backend=backend,
            )
            self.__model.t3.init_patched_model()  # Initialize the patched model
            logging.info("Model compiled successfully.")
        except ImportError:
            logging.warning("torch.compile is not available. Skipping compilation.")

    def change_model_encoding(self, dtype):
        logging.info(f"")
        self.__model.t3.to(dtype=dtype)
        self.__model.conds.t3.to(dtype=dtype)

    def __cleanup(self, text: str) -> str:
        """
        Clean up the text by removing unwanted characters.
        Cleaning is done by removing extra spaces and normalizing the text to a
        single space. Also, it removes newlines.
        Args:
            text (str): The text to be cleaned.
        Returns:
            str: The cleaned text.
        """
        text = text.replace("\n", " ")
        text = text.strip()
        text = " ".join(text.split())
        return text

    def __chunk_text(
        self, text_generator: Generator[str, None, None]
    ) -> Generator[str, None, None]:
        """
        Split the text into chunks of near MAX_TOKEN_LENGTH.
        It will target a punctuation mark to split the text, so the token size
        may vary slightly.
        Args:
            text_generator (Generator[str, None, None]): A generator that yields text blocks.
        Returns:
            list[str]: A list of text chunks.
        """

        def split_text_iterator(
            text: Generator[str, None, None],
        ) -> Generator[str, None, None]:
            for block in text:
                block = self.__cleanup(block)
                yield from block.split()

        current_chunk = []
        accumulated_length = 0

        ignore_next_token = False
        for word in split_text_iterator(text_generator):
            word_length = len(word)

            if accumulated_length + word_length >= self.__TARGET_TOKEN_LENGTH:
                # Check for punctuation to split the text
                for punctuation in self.__TARGET_PUNCTUATION_STOP:
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
                accumulated_length += word_length + 1  # +1 for the space

        if current_chunk:
            yield " ".join(current_chunk)

    def __crossfade(self, audio: Tensor, duration_ms=100) -> Tensor:

        sr = self.__target_sr  # Sample rate of the audio
        overlap = int(sr * duration_ms / 1000)

        if overlap == 0:
            return audio  # No crossfade if overlap is zero

        if self.__previous_audio is None:
            self.__previous_audio = audio[..., -overlap:]
            return audio[..., :-overlap]  # Return the audio without crossfade
        else:
            next_overlap = audio[..., :overlap]

            fade_out = torch.linspace(1, 0, self.__previous_audio.size(-1)).to(
                audio.device
            )
            fade_in = torch.linspace(0, 1, overlap).to(audio.device)

            crossfaded = self.__previous_audio * fade_out + next_overlap * fade_in

            self.__previous_audio = audio[..., -overlap:]  # Update previous audio
            return torch.cat([crossfaded, audio[..., overlap:-overlap]], dim=-1)

    def __data_stream(self, chunk_size) -> Generator[np.ndarray, None, None]:
        """
        Generate audio data from the model in real-time. With fixed chunk size.
        """
        buffer = np.zeros((chunk_size, 1))  # Buffer to hold audio data mono
        buffer_size = 0
        while True:
            audio_data_tensor = self.__queue.get()
            if audio_data_tensor is self.__stop_signal:
                if buffer_size > 0:
                    buffer[buffer_size:] = np.zeros(
                        (chunk_size - buffer_size, 1)
                    )  # Fill the rest of the buffer with silence
                    yield buffer
                yield self.__stop_signal
                break

            # Resample if needed
            if self.__resampler is not None:
                audio_data_tensor = self.__resampler(audio_data_tensor)

            scaled_audio = audio_data_tensor * self.__volume  # 🔈 scale volume here
            crossfade_audio = self.__crossfade(scaled_audio)

            audio_data = crossfade_audio.T.numpy()
            audio_data = audio_data.reshape(-1, 1)  # ensure shape is (N, 1)

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

    def __generate(self, chunck_text: str) -> Tensor:
        output = self.__model.generate(
            chunck_text, audio_prompt_path=self.__voice_clone
        )

        # If output is generator consume it
        if isinstance(output, Generator):
            data = []
            for chunk in output:
                if not isinstance(chunk, Tensor):
                    raise ValueError("Output must be a generator or tensor")
                data.append(chunk)

            return torch.cat(data, dim=-1)

        # Else if output is tensor
        elif isinstance(output, Tensor):
            return output
        else:
            raise ValueError(
                "Output must be a generator or tensor or generator of tensors"
            )

    def stream_to_file(
        self, text_generator: Generator[str, None, None], file_path: str
    ) -> None:

        def writer_worker():
            with sf.SoundFile(
                file_path,
                mode="w",
                samplerate=self.__target_sr,
                channels=1,
                subtype="PCM_16",  # 16-bit PCM WAV
            ) as file:
                for audio_data in self.__data_stream(self.__PLAYBACK_BLOCKSIZE):
                    if audio_data is self.__stop_signal:
                        break  # stop signal received

                    assert audio_data.shape[1] == 1, "Audio data must be mono"
                    file.write(audio_data)

        writer_thread = threading.Thread(target=writer_worker)
        writer_thread.start()

        for chunk in self.__chunk_text(text_generator):
            audio_chunk = self.__generate(chunk)
            self.__queue.put(audio_chunk)
            logging.info(f"Processed text chunk: {chunk}")

        self.__queue.put(self.__stop_signal)
        writer_thread.join()

        logging.info("Writing complete.")

    def stream(self, text_generator: Generator[str, None, None]) -> None:

        data_generator = self.__data_stream(self.__PLAYBACK_BLOCKSIZE)
        event = threading.Event()

        def callback(outdata, frames, time, status):
            assert frames == self.__PLAYBACK_BLOCKSIZE
            if status.output_underflow:
                logging.warning("Output underflow: increase blocksize?")
                return
            assert not status
            try:
                audio_data = next(data_generator)
            except StopIteration:
                raise sd.CallbackStop

            if audio_data is self.__stop_signal:
                raise sd.CallbackStop

            assert outdata.shape == audio_data.shape

            outdata[:] = audio_data

        logging.info("Starting playback...")

        # Start the audio stream
        with sd.OutputStream(
            samplerate=self.__target_sr,
            channels=1,  # Mono audio
            callback=callback,
            finished_callback=event.set,
            blocksize=self.__PLAYBACK_BLOCKSIZE,
        ):
            for chunk in self.__chunk_text(text_generator):
                audio_chunk = self.__generate(chunk)
                self.__queue.put(audio_chunk)
                logging.info(f"Processed text chunk: {chunk}")
            self.__queue.put(self.__stop_signal)
            event.wait()  # Wait for the stream to finish

        logging.info("Finished streaming audio.")

    def __call__(self, text: str) -> None:

        def text_generator() -> Generator[str, None, None]:
            yield text

        self.stream(text_generator())
