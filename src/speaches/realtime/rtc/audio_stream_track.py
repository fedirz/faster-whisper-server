import asyncio
import base64
import io
import logging

from aiortc import MediaStreamTrack
from av.audio.frame import AudioFrame
import numpy as np
from openai.types.beta.realtime import ResponseAudioDeltaEvent

from speaches.audio import audio_samples_from_file
from speaches.realtime.context import SessionContext
from speaches.realtime.input_audio_buffer_event_router import resample_audio_data

logger = logging.getLogger(__name__)


class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, ctx: SessionContext) -> None:
        super().__init__()
        self.ctx = ctx
        # self.q = ctx.pubsub.subscribe()
        self.frame_queue = asyncio.Queue()  # Queue for AudioFrames
        self._timestamp = 0
        self._sample_rate = 48000
        self._frame_duration = 0.01  # in seconds
        self._samples_per_frame = int(self._sample_rate * self._frame_duration)
        self._running = True

        # Start the frame processing task
        self._process_task = asyncio.create_task(self._audio_frame_generator())

    async def recv(self) -> AudioFrame:
        """Receive the next audio frame."""
        if not self._running:
            raise MediaStreamError("Track has ended")  # noqa: EM101

        try:
            frame = await self.frame_queue.get()
            await asyncio.sleep(
                0.005
            )  # NOTE: I believe some delay is neccessary to prevent buffers from being dropped.
        except asyncio.CancelledError as e:
            raise MediaStreamError("Track has ended") from e  # noqa: EM101
        else:
            return frame

    async def _audio_frame_generator(self) -> None:
        """Process incoming numpy arrays and split them into AudioFrames."""
        try:
            async for event in self.ctx.pubsub.subscribe_to("response.audio.delta"):
                assert isinstance(event, ResponseAudioDeltaEvent)

                if not self._running:
                    return

                # copied from `input_audio_buffer.append` handler
                audio_array = audio_samples_from_file(io.BytesIO(base64.b64decode(event.delta)))
                audio_array = resample_audio_data(audio_array, 24000, 48000)

                # Convert to int16 if not already
                if audio_array.dtype != np.int16:
                    audio_array = (audio_array * 32767).astype(np.int16)

                # Split the array into frame-sized chunks
                frames = self._split_into_frames(audio_array)

                # Create AudioFrames and add them to the frame queue
                logger.info(f"Received audio: {len(audio_array)} samples")
                logger.info(f"Split into {len(frames)} frames")
                for frame_data in frames:
                    frame = self._create_frame(frame_data)
                    self.frame_queue.put_nowait(frame)

        except asyncio.CancelledError:
            logger.warning("Audio frame generator task cancelled")

    def _split_into_frames(self, audio_array: np.ndarray) -> list[np.ndarray]:
        # Ensure the array is 1D
        if len(audio_array.shape) > 1:
            audio_array = audio_array.flatten()

        # Calculate number of complete frames
        n_frames = len(audio_array) // self._samples_per_frame

        frames = []
        for i in range(n_frames):
            start = i * self._samples_per_frame
            end = start + self._samples_per_frame
            frame = audio_array[start:end]
            frames.append(frame)

        remaining = len(audio_array) % self._samples_per_frame
        if remaining > 0:
            logger.info(f"Processing remaining {remaining} samples")
            last_frame = audio_array[-remaining:]
            padded_frame = np.pad(last_frame, (0, self._samples_per_frame - remaining), "constant", constant_values=0)
            logger.info(f"Padded frame range: {padded_frame.min()}, {padded_frame.max()}")
            frames.append(padded_frame)

        return frames

    def _create_frame(self, frame_data: np.ndarray) -> AudioFrame:
        """Create an AudioFrame from numpy array data.

        Args:
            frame_data: Numpy array containing exactly samples_per_frame samples

        Returns:
            AudioFrame object

        """
        frame = AudioFrame(
            format="s16",
            layout="mono",
            samples=self._samples_per_frame,
        )
        frame.sample_rate = self._sample_rate

        # Convert numpy array to bytes and update frame
        frame.planes[0].update(frame_data.tobytes())

        # Set timestamp
        frame.pts = self._timestamp
        self._timestamp += self._samples_per_frame

        return frame

    def stop(self) -> None:
        """Stop the audio track and cleanup."""
        self._running = False
        if hasattr(self, "_process_task"):
            self._process_task.cancel()
        super().stop()


class ToneAudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._timestamp = 0
        self._sample_rate = 24000
        self._samples_per_frame = self._sample_rate // 100  # 10ms
        self._running = True
        self._frequency = 440  # 440 Hz tone

    async def recv(self) -> AudioFrame:
        if not self._running:
            raise MediaStreamError("Track has ended")  # noqa: EM101

        # Generate sine wave for this frame
        t = np.linspace(0, self._samples_per_frame / self._sample_rate, self._samples_per_frame)
        samples = np.sin(2 * np.pi * self._frequency * t)
        samples = (samples * 32767).astype(np.int16)

        # Create frame
        frame = AudioFrame(
            format="s16",
            layout="mono",
            samples=self._samples_per_frame,
        )
        frame.sample_rate = self._sample_rate
        frame.pts = self._timestamp
        frame.planes[0].update(samples.tobytes())

        self._timestamp += self._samples_per_frame

        # Sleep for frame duration
        await asyncio.sleep(0.01)  # 10ms

        return frame

    def stop(self) -> None:
        self._running = False
        super().stop()


class MediaStreamError(Exception):
    pass
