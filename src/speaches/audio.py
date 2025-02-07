from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING, BinaryIO

import numpy as np
import soundfile as sf
import ffmpeg

from speaches.config import SAMPLES_PER_SECOND

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from numpy.typing import NDArray

    from speaches.routers.speech import ResponseFormat


logger = logging.getLogger(__name__)


# aip 'Write a function `resample_audio` which would take in RAW PCM 16-bit signed, little-endian audio data represented as bytes (`audio_bytes`) and resample it (either downsample or upsample) from `sample_rate` to `target_sample_rate` using numpy'  # noqa: E501
def resample_audio(audio_bytes: bytes, sample_rate: int, target_sample_rate: int) -> bytes:
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    duration = len(audio_data) / sample_rate
    target_length = int(duration * target_sample_rate)
    resampled_data = np.interp(
        np.linspace(0, len(audio_data), target_length, endpoint=False), np.arange(len(audio_data)), audio_data
    )
    return resampled_data.astype(np.int16).tobytes()


def convert_audio_format(
    audio_bytes: bytes,
    sample_rate: int,
    audio_format: ResponseFormat,
    format: str = "RAW",  # noqa: A002
    channels: int = 1,
    subtype: str = "PCM_16",
    endian: str = "LITTLE",
) -> bytes:
    # NOTE: the default dtype is float64. Should something else be used? Would that improve performance?
    data, _ = sf.read(
        io.BytesIO(audio_bytes),
        samplerate=sample_rate,
        format=format,
        channels=channels,
        subtype=subtype,
        endian=endian,
    )

    if audio_format == "aac":
        try:
            # Write to WAV in memory
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, data, samplerate=sample_rate, format='WAV')
            wav_bytes = wav_buffer.getvalue()
            
            # Convert WAV to AAC using ffmpeg
            input_stream = ffmpeg.input('pipe:', format='wav')
            output_stream = ffmpeg.output(
                input_stream,
                'pipe:', 
                acodec='aac',
                ab='192k',
                f='adts'  # AAC container format
            )
            
            out_bytes, _ = ffmpeg.run(
                output_stream, 
                input=wav_bytes,  # Use the WAV bytes
                capture_stdout=True,
                capture_stderr=True
            )
            
            return out_bytes
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
            raise
    else:
        converted_audio_bytes_buffer = io.BytesIO()
        sf.write(converted_audio_bytes_buffer, data, samplerate=sample_rate, format=audio_format)
        return converted_audio_bytes_buffer.getvalue()


def audio_samples_from_file(file: BinaryIO) -> NDArray[np.float32]:
    audio_and_sample_rate = sf.read(
        file,
        format="RAW",
        channels=1,
        samplerate=SAMPLES_PER_SECOND,
        subtype="PCM_16",
        dtype="float32",
        endian="LITTLE",
    )
    audio = audio_and_sample_rate[0]
    return audio  # pyright: ignore[reportReturnType]


class Audio:
    def __init__(
        self,
        data: NDArray[np.float32] = np.array([], dtype=np.float32),
        start: float = 0.0,
    ) -> None:
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

    def after(self, ts: float) -> Audio:
        assert ts <= self.duration
        return Audio(self.data[int(ts * SAMPLES_PER_SECOND) :], start=ts)

    def extend(self, data: NDArray[np.float32]) -> None:
        # logger.debug(f"Extending audio by {len(data) / SAMPLES_PER_SECOND:.2f}s")
        self.data = np.append(self.data, data)
        # logger.debug(f"Audio duration: {self.duration:.2f}s")


# TODO: trim data longer than x
class AudioStream(Audio):
    def __init__(
        self,
        data: NDArray[np.float32] = np.array([], dtype=np.float32),
        start: float = 0.0,
    ) -> None:
        super().__init__(data, start)
        self.closed = False

        self.modify_event = asyncio.Event()

    def extend(self, data: NDArray[np.float32]) -> None:
        assert not self.closed
        super().extend(data)
        self.modify_event.set()

    def close(self) -> None:
        assert not self.closed
        self.closed = True
        self.modify_event.set()
        logger.info("AudioStream closed")

    async def chunks(self, min_duration: float) -> AsyncGenerator[NDArray[np.float32], None]:
        i = 0.0  # end time of last chunk
        while True:
            await self.modify_event.wait()
            self.modify_event.clear()

            if self.closed:
                if self.duration > i:
                    yield self.after(i).data
                return
            if self.duration - i >= min_duration:
                # If `i` shouldn't be set to `duration` after the yield
                # because by the time assignment would happen more data might have been added
                i_ = i
                i = self.duration
                # NOTE: probably better to just to a slice
                yield self.after(i_).data
