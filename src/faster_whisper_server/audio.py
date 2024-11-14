from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, BinaryIO, AsyncGenerator

import numpy as np
import soundfile as sf

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

def audio_samples_from_file(file: BinaryIO) -> NDArray[np.float32]:
    """
    Read audio samples from a file.

    :param file: BinaryIO object of the audio file
    :return: Audio samples as a numpy array
    """
    try:
        audio_and_sample_rate = sf.read(
            file,
            format="RAW",
            channels=1,
            samplerate=SAMPLES_PER_SECOND,
            subtype="PCM_16",
            dtype="float32",
            endian="LITTLE",
        )
        audio = audio_and_sample_rate
        return audio
    except Exception as e:
        logger.error(f"Error reading audio file: {e}")
        return np.array([], dtype=np.float32)

class Audio:
    def __init__(
        self,
        data: NDArray[np.float32] = np.array([], dtype=np.float32),
        start: float = 0.0,
    ) -> None:
        """
        Initialize the Audio class.

        :param data: Audio data as a numpy array
        :param start: Start time of the audio
        """
        self.data = data
        self.start = start

    def __repr__(self) -> str:
        return f"Audio(start={self.start:.2f}, end={self.end:.2f})"

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def duration(self) -> float:
        return len(self.data) / SAMPLES_PER_SECOND

    def after(self, ts: float) -> 'Audio':
        """
        Get the audio data after a specified time.

        :param ts: Time from the start of the audio
        :return: New Audio object with data after the specified time
        """
        assert ts <= self.duration
        return Audio(self.data[int(ts * SAMPLES_PER_SECOND) :], start=ts)

    def extend(self, data: NDArray[np.float32]) -> None:
        """
        Extend the audio data.

        :param data: New audio data to append
        """
        # logger.debug(f"Extending audio by {len(data) / SAMPLES_PER_SECOND:.2f}s")
        self.data = np.append(self.data, data)
        # logger.debug(f"Audio duration: {self.duration:.2f}s")

class AudioStream(Audio):
    def __init__(
        self,
        data: NDArray[np.float32] = np.array([], dtype=np.float32),
        start: float = 0.0,
    ) -> None:
        """
        Initialize the AudioStream class.

        :param data: Initial audio data
        :param start: Start time of the audio
        """
        super().__init__(data, start)
        self.closed = False
        self.modify_event = asyncio.Event()

    def extend(self, data: NDArray[np.float32]) -> None:
        """
        Extend the audio data and notify any waiting tasks.

        :param data: New audio data to append
        """
        assert not self.closed
        super().extend(data)
        self.modify_event.set()

    def close(self) -> None:
        """
        Close the audio stream and notify any waiting tasks.
        """
        assert not self.closed
        self.closed = True
        self.modify_event.set()
        logger.info("AudioStream closed")

    async def chunks(self, min_duration: float, max_duration: float = None) -> AsyncGenerator[NDArray[np.float32], None]:
        """
        Asynchronously yield chunks of audio data.

        :param min_duration: Minimum duration of each chunk
        :param max_duration: Maximum duration of each chunk (optional)
        :yield: Chunks of audio data
        """
        i = 0.0  # end time of last chunk
        while True:
            await self.modify_event.wait()
            self.modify_event.clear()

            if self.closed:
                if self.duration > i:
                    yield self.after(i).data
                return
            if max_duration and self.duration - i > max_duration:
                # Trim data if it exceeds max_duration
                yield self.after(i).data[:int(max_duration * SAMPLES_PER_SECOND)]
                i += max_duration
            elif self.duration - i >= min_duration:
                # If `i` shouldn't be set to `duration` after the yield
                # because by the time assignment would happen more data might have been added
                i_ = i
                i = self.duration
                # NOTE: probably better to just do a slice
                yield self.after(i_).data

# Example usage
async def main():
    # Open an audio file
    with open("path/to/audio/file.wav", "rb") as file:
        audio_data = audio_samples_from_file(file)

    # Create an AudioStream object
    audio_stream = AudioStream(audio_data)

    # Extend the audio stream with new data (example)
    new_data = np.random.rand(1000)  # Example new data
    audio_stream.extend(new_data)

    # Close the audio stream (example)
    # audio_stream.close()

    # Yield chunks of audio data
    async for chunk in audio_stream.chunks(min_duration=1.0, max_duration=5.0):
        logger.info(f"Received chunk of {len(chunk) / SAMPLES_PER_SECOND:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
