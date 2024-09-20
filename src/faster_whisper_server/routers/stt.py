from __future__ import annotations

import asyncio
from io import BytesIO
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import (
    APIRouter,
    Form,
    Query,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from fastapi.websockets import WebSocketState
from faster_whisper.vad import VadOptions, get_speech_timestamps
from faster_whisper_server.asr import FasterWhisperASR
from faster_whisper_server.audio import AudioStream, audio_samples_from_file
from faster_whisper_server.config import (
    SAMPLES_PER_SECOND,
    Language,
    ResponseFormat,
    Task,
    config,
)
from faster_whisper_server.core import Segment, segments_to_srt, segments_to_text, segments_to_vtt
from faster_whisper_server.logger import logger
from faster_whisper_server.model_manager import model_manager
from faster_whisper_server.server_models import (
    TranscriptionJsonResponse,
    TranscriptionVerboseJsonResponse,
)
from faster_whisper_server.transcriber import audio_transcriber
from pydantic import AfterValidator

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from faster_whisper.transcribe import TranscriptionInfo


router = APIRouter()


def segments_to_response(
    segments: Iterable[Segment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> Response:
    segments = list(segments)
    if response_format == ResponseFormat.TEXT:  # noqa: RET503
        return Response(segments_to_text(segments), media_type="text/plain")
    elif response_format == ResponseFormat.JSON:
        return Response(
            TranscriptionJsonResponse.from_segments(segments).model_dump_json(),
            media_type="application/json",
        )
    elif response_format == ResponseFormat.VERBOSE_JSON:
        return Response(
            TranscriptionVerboseJsonResponse.from_segments(segments, transcription_info).model_dump_json(),
            media_type="application/json",
        )
    elif response_format == ResponseFormat.VTT:
        return Response(
            "".join(segments_to_vtt(segment, i) for i, segment in enumerate(segments)), media_type="text/vtt"
        )
    elif response_format == ResponseFormat.SRT:
        return Response(
            "".join(segments_to_srt(segment, i) for i, segment in enumerate(segments)), media_type="text/plain"
        )


def format_as_sse(data: str) -> str:
    return f"data: {data}\n\n"


def segments_to_streaming_response(
    segments: Iterable[Segment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> StreamingResponse:
    def segment_responses() -> Generator[str, None, None]:
        for i, segment in enumerate(segments):
            if response_format == ResponseFormat.TEXT:
                data = segment.text
            elif response_format == ResponseFormat.JSON:
                data = TranscriptionJsonResponse.from_segments([segment]).model_dump_json()
            elif response_format == ResponseFormat.VERBOSE_JSON:
                data = TranscriptionVerboseJsonResponse.from_segment(segment, transcription_info).model_dump_json()
            elif response_format == ResponseFormat.VTT:
                data = segments_to_vtt(segment, i)
            elif response_format == ResponseFormat.SRT:
                data = segments_to_srt(segment, i)
            yield format_as_sse(data)

    return StreamingResponse(segment_responses(), media_type="text/event-stream")


def handle_default_openai_model(model_name: str) -> str:
    """Exists because some callers may not be able override the default("whisper-1") model name.

    For example, https://github.com/open-webui/open-webui/issues/2248#issuecomment-2162997623.
    """
    if model_name == "whisper-1":
        logger.info(f"{model_name} is not a valid model name. Using {config.whisper.model} instead.")
        return config.whisper.model
    return model_name


ModelName = Annotated[str, AfterValidator(handle_default_openai_model)]


@router.post(
    "/v1/audio/translations",
    response_model=str | TranscriptionJsonResponse | TranscriptionVerboseJsonResponse,
)
def translate_file(
    file: Annotated[UploadFile, Form()],
    model: Annotated[ModelName, Form()] = config.whisper.model,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = config.default_response_format,
    temperature: Annotated[float, Form()] = 0.0,
    stream: Annotated[bool, Form()] = False,
) -> Response | StreamingResponse:
    whisper = model_manager.load_model(model)
    segments, transcription_info = whisper.transcribe(
        file.file,
        task=Task.TRANSLATE,
        initial_prompt=prompt,
        temperature=temperature,
        vad_filter=True,
    )
    segments = Segment.from_faster_whisper_segments(segments)

    if stream:
        return segments_to_streaming_response(segments, transcription_info, response_format)
    else:
        return segments_to_response(segments, transcription_info, response_format)


# https://platform.openai.com/docs/api-reference/audio/createTranscription
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L8915
@router.post(
    "/v1/audio/transcriptions",
    response_model=str | TranscriptionJsonResponse | TranscriptionVerboseJsonResponse,
)
def transcribe_file(
    file: Annotated[UploadFile, Form()],
    model: Annotated[ModelName, Form()] = config.whisper.model,
    language: Annotated[Language | None, Form()] = config.default_language,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = config.default_response_format,
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        list[Literal["segment", "word"]],
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str | None, Form()] = None,
) -> Response | StreamingResponse:
    whisper = model_manager.load_model(model)
    segments, transcription_info = whisper.transcribe(
        file.file,
        task=Task.TRANSCRIBE,
        language=language,
        initial_prompt=prompt,
        word_timestamps="word" in timestamp_granularities,
        temperature=temperature,
        vad_filter=True,
        hotwords=hotwords,
    )
    segments = Segment.from_faster_whisper_segments(segments)

    if stream:
        return segments_to_streaming_response(segments, transcription_info, response_format)
    else:
        return segments_to_response(segments, transcription_info, response_format)


async def audio_receiver(ws: WebSocket, audio_stream: AudioStream) -> None:
    try:
        while True:
            bytes_ = await asyncio.wait_for(ws.receive_bytes(), timeout=config.max_no_data_seconds)
            logger.debug(f"Received {len(bytes_)} bytes of audio data")
            audio_samples = audio_samples_from_file(BytesIO(bytes_))
            audio_stream.extend(audio_samples)
            if audio_stream.duration - config.inactivity_window_seconds >= 0:
                audio = audio_stream.after(audio_stream.duration - config.inactivity_window_seconds)
                vad_opts = VadOptions(min_silence_duration_ms=500, speech_pad_ms=0)
                # NOTE: This is a synchronous operation that runs every time new data is received.
                # This shouldn't be an issue unless data is being received in tiny chunks or the user's machine is a potato.  # noqa: E501
                timestamps = get_speech_timestamps(audio.data, vad_opts)
                if len(timestamps) == 0:
                    logger.info(f"No speech detected in the last {config.inactivity_window_seconds} seconds.")
                    break
                elif (
                    # last speech end time
                    config.inactivity_window_seconds - timestamps[-1]["end"] / SAMPLES_PER_SECOND
                    >= config.max_inactivity_seconds
                ):
                    logger.info(f"Not enough speech in the last {config.inactivity_window_seconds} seconds.")
                    break
    except TimeoutError:
        logger.info(f"No data received in {config.max_no_data_seconds} seconds. Closing the connection.")
    except WebSocketDisconnect as e:
        logger.info(f"Client disconnected: {e}")
    audio_stream.close()


@router.websocket("/v1/audio/transcriptions")
async def transcribe_stream(
    ws: WebSocket,
    model: Annotated[ModelName, Query()] = config.whisper.model,
    language: Annotated[Language | None, Query()] = config.default_language,
    response_format: Annotated[ResponseFormat, Query()] = config.default_response_format,
    temperature: Annotated[float, Query()] = 0.0,
) -> None:
    await ws.accept()
    transcribe_opts = {
        "language": language,
        "temperature": temperature,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }
    whisper = model_manager.load_model(model)
    asr = FasterWhisperASR(whisper, **transcribe_opts)
    audio_stream = AudioStream()
    async with asyncio.TaskGroup() as tg:
        tg.create_task(audio_receiver(ws, audio_stream))
        async for transcription in audio_transcriber(asr, audio_stream):
            logger.debug(f"Sending transcription: {transcription.text}")
            if ws.client_state == WebSocketState.DISCONNECTED:
                break

            if response_format == ResponseFormat.TEXT:
                await ws.send_text(transcription.text)
            elif response_format == ResponseFormat.JSON:
                await ws.send_json(TranscriptionJsonResponse.from_transcription(transcription).model_dump())
            elif response_format == ResponseFormat.VERBOSE_JSON:
                await ws.send_json(TranscriptionVerboseJsonResponse.from_transcription(transcription).model_dump())

    if ws.client_state != WebSocketState.DISCONNECTED:
        logger.info("Closing the connection.")
        await ws.close()
