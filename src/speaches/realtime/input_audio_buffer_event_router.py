import base64
from io import BytesIO
import logging
from typing import Literal

from faster_whisper.transcribe import get_speech_timestamps
from faster_whisper.vad import VadOptions
import numpy as np
from numpy.typing import NDArray
from openai.types.beta.realtime.error_event import Error

from speaches.audio import audio_samples_from_file
from speaches.realtime.context import SessionContext
from speaches.realtime.event_router import EventRouter
from speaches.realtime.input_audio_buffer import (
    MAX_VAD_WINDOW_SIZE_SAMPLES,
    MS_SAMPLE_RATE,
    InputAudioBuffer,
    InputAudioBufferTranscriber,
)
from speaches.types.realtime import (
    ErrorEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearedEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferCommittedEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    TurnDetection,
)

logger = logging.getLogger(__name__)

event_router = EventRouter()

empty_input_audio_buffer_commit_error = Error(
    type="invalid_request_error",
    message="Error committing input audio buffer: the buffer is empty.",
)

type SpeechTimestamp = dict[Literal["start", "end"], int]


# NOTE: `signal.resample_poly` **might** be a better option for resampling audio data
# TODO: also found in src/speaches/audio.py. Remove duplication
def resample_audio_data(data: NDArray[np.float32], sample_rate: int, target_sample_rate: int) -> NDArray[np.float32]:
    ratio = target_sample_rate / sample_rate
    target_length = int(len(data) * ratio)
    return np.interp(np.linspace(0, len(data), target_length), np.arange(len(data)), data).astype(np.float32)


# TODO: also found in src/speaches/routers/vad.py. Remove duplication
def to_ms_speech_timestamps(speech_timestamps: list[SpeechTimestamp]) -> list[SpeechTimestamp]:
    for i in range(len(speech_timestamps)):
        speech_timestamps[i]["start"] = speech_timestamps[i]["start"] // MS_SAMPLE_RATE
        speech_timestamps[i]["end"] = speech_timestamps[i]["end"] // MS_SAMPLE_RATE
    return speech_timestamps


def vad_detection_flow(
    input_audio_buffer: InputAudioBuffer, turn_detection: TurnDetection
) -> InputAudioBufferSpeechStartedEvent | InputAudioBufferSpeechStoppedEvent | None:
    audio_window = input_audio_buffer.data[-MAX_VAD_WINDOW_SIZE_SAMPLES:]

    speech_timestamps = to_ms_speech_timestamps(
        get_speech_timestamps(
            audio_window,
            vad_options=VadOptions(
                threshold=turn_detection.threshold,
                min_silence_duration_ms=turn_detection.silence_duration_ms,
                speech_pad_ms=turn_detection.prefix_padding_ms,
            ),
        )
    )
    if len(speech_timestamps) > 1:
        logger.warning(f"More than one speech timestamp: {speech_timestamps}")

    speech_timestamp = speech_timestamps[-1] if len(speech_timestamps) > 0 else None

    # logger.debug(f"Speech timestamps: {speech_timestamps}")
    if input_audio_buffer.vad_state.audio_start_ms is None:
        if speech_timestamp is None:
            return None
        input_audio_buffer.vad_state.audio_start_ms = (
            input_audio_buffer.duration_ms - len(audio_window) // MS_SAMPLE_RATE + speech_timestamp["start"]
        )
        return InputAudioBufferSpeechStartedEvent(
            item_id=input_audio_buffer.id,
            audio_start_ms=input_audio_buffer.vad_state.audio_start_ms,
        )

    else:  # noqa: PLR5501
        if speech_timestamp is None:
            # TODO: not quite correct. dependent on window size
            input_audio_buffer.vad_state.audio_end_ms = (
                input_audio_buffer.duration_ms - turn_detection.prefix_padding_ms
            )
            return InputAudioBufferSpeechStoppedEvent(
                item_id=input_audio_buffer.id,
                audio_end_ms=input_audio_buffer.vad_state.audio_end_ms,
            )

        elif speech_timestamp["end"] < 3000 and input_audio_buffer.duration_ms > 3000:  # FIX: magic number
            input_audio_buffer.vad_state.audio_end_ms = (
                input_audio_buffer.duration_ms - turn_detection.prefix_padding_ms
            )

            return InputAudioBufferSpeechStoppedEvent(
                item_id=input_audio_buffer.id,
                audio_end_ms=input_audio_buffer.vad_state.audio_end_ms,
            )

    return None


# Client Events


@event_router.register("input_audio_buffer.append")
def handle_input_audio_buffer_append(ctx: SessionContext, event: InputAudioBufferAppendEvent) -> None:
    audio_chunk = audio_samples_from_file(BytesIO(base64.b64decode(event.audio)))
    # convert the audio data from 24kHz (sample rate defined in the API spec) to 16kHz (sample rate used by the VAD and for transcription)
    audio_chunk = resample_audio_data(audio_chunk, 24000, 16000)
    input_audio_buffer_id = next(reversed(ctx.input_audio_buffers))
    input_audio_buffer = ctx.input_audio_buffers[input_audio_buffer_id]
    input_audio_buffer.append(audio_chunk)
    if ctx.session.turn_detection is not None:
        vad_event = vad_detection_flow(input_audio_buffer, ctx.session.turn_detection)
        if vad_event is not None:
            ctx.pubsub.publish_nowait(vad_event)


@event_router.register("input_audio_buffer.commit")
def handle_input_audio_buffer_commit(ctx: SessionContext, _event: InputAudioBufferCommitEvent) -> None:
    input_audio_buffer_id = next(reversed(ctx.input_audio_buffers))
    input_audio_buffer = ctx.input_audio_buffers[input_audio_buffer_id]
    if input_audio_buffer.size == 0:
        ctx.pubsub.publish_nowait(ErrorEvent(error=empty_input_audio_buffer_commit_error))
    else:
        ctx.pubsub.publish_nowait(
            InputAudioBufferCommittedEvent(
                previous_item_id=next(reversed(ctx.conversation.items), None),  # FIXME
                item_id=input_audio_buffer_id,
            )
        )
        input_audio_buffer = InputAudioBuffer(ctx.pubsub)
        ctx.input_audio_buffers[input_audio_buffer.id] = input_audio_buffer


@event_router.register("input_audio_buffer.clear")
def handle_input_audio_buffer_clear(ctx: SessionContext, _event: InputAudioBufferClearEvent) -> None:
    ctx.input_audio_buffers.popitem()
    # OpenAI's doesn't send an error if the buffer is already empty.
    ctx.pubsub.publish_nowait(InputAudioBufferClearedEvent())
    input_audio_buffer = InputAudioBuffer(ctx.pubsub)
    ctx.input_audio_buffers[input_audio_buffer.id] = input_audio_buffer


# Server Events


@event_router.register("input_audio_buffer.speech_stopped")
def handle_input_audio_buffer_speech_stopped(ctx: SessionContext, event: InputAudioBufferSpeechStoppedEvent) -> None:
    input_audio_buffer = InputAudioBuffer(ctx.pubsub)
    ctx.input_audio_buffers[input_audio_buffer.id] = input_audio_buffer
    ctx.pubsub.publish_nowait(
        InputAudioBufferCommittedEvent(
            previous_item_id=next(reversed(ctx.conversation.items), None),  # FIXME
            item_id=event.item_id,
        )
    )


@event_router.register("input_audio_buffer.committed")
async def handle_input_audio_buffer_committed(ctx: SessionContext, event: InputAudioBufferCommittedEvent) -> None:
    input_audio_buffer = ctx.input_audio_buffers[event.item_id]

    transcriber = InputAudioBufferTranscriber(
        pubsub=ctx.pubsub,
        transcription_client=ctx.transcription_client,
        input_audio_buffer=input_audio_buffer,
        session=ctx.session,
        conversation=ctx.conversation,
    )
    transcriber.start()
    assert transcriber.task is not None
    await transcriber.task
