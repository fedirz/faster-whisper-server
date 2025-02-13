import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from speaches.realtime.utils import generate_item_id

SAMPLE_RATE = 16000
MS_SAMPLE_RATE = 16
MAX_VAD_WINDOW_SIZE_SAMPLES = 3000 * MS_SAMPLE_RATE


# NOTE not in `src/speaches/realtime/input_audio_buffer_event_router.py` due to circular import
class VadState(BaseModel):
    audio_start_ms: int | None = None
    audio_end_ms: int | None = None
    # TODO: consider keeping track of what was the last audio timestamp that was processed. This value could be used to control how often the VAD is run.


# TODO: use `np.int16` instead of `np.float32` for audio data
class InputAudioBuffer:
    def __init__(self) -> None:
        self.id = generate_item_id()
        self.data: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.vad_state = VadState()

    @property
    def size(self) -> int:
        """Number of samples in the buffer."""
        return len(self.data)

    @property
    def duration(self) -> float:
        """Duration of the audio in seconds."""
        return len(self.data) / SAMPLE_RATE

    @property
    def duration_ms(self) -> int:
        """Duration of the audio in milliseconds."""
        return len(self.data) // MS_SAMPLE_RATE

    def append(self, audio_chunk: NDArray[np.float32]) -> None:
        """Append an audio chunk to the buffer."""
        self.data = np.append(self.data, audio_chunk)
