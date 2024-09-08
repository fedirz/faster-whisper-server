from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, BinaryIO

import numpy as np
import soundfile as sf

from faster_whisper_server.config import SAMPLES_PER_SECOND
from faster_whisper_server.logger import logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from numpy.typing import NDArray


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
