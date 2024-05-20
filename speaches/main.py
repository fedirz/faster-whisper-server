from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

from fastapi import (Depends, FastAPI, Response, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.websockets import WebSocketState
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

from speaches.asr import FasterWhisperASR, TranscribeOpts
from speaches.audio import AudioStream, audio_samples_from_file
from speaches.config import SAMPLES_PER_SECOND, Language, config
from speaches.core import Transcription
from speaches.logger import logger
from speaches.server_models import (ResponseFormat, TranscriptionResponse,
                                    TranscriptionVerboseResponse)
from speaches.transcriber import audio_transcriber

whisper: WhisperModel = None  # type: ignore


@asynccontextmanager
async def lifespan(_: FastAPI):
    global whisper
    logging.debug(f"Loading {config.whisper.model}")
    start = time.perf_counter()
    whisper = WhisperModel(
        config.whisper.model,
        device=config.whisper.inference_device,
        compute_type=config.whisper.compute_type,
    )
    end = time.perf_counter()
    logger.debug(f"Loaded {config.whisper.model} loaded in {end - start:.2f} seconds")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> Response:
    return Response(status_code=200, content="Everything is peachy!")


async def transcription_parameters(
    language: Language = Language.EN,
    vad_filter: bool = True,
    condition_on_previous_text: bool = False,
) -> TranscribeOpts:
    return TranscribeOpts(
        language=language,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
    )


TranscribeParams = Annotated[TranscribeOpts, Depends(transcription_parameters)]


@app.post("/v1/audio/transcriptions")
async def transcribe_file(
    file: UploadFile,
    transcription_opts: TranscribeParams,
    response_format: ResponseFormat = ResponseFormat.JSON,
) -> str:
    asr = FasterWhisperASR(whisper, transcription_opts)
    audio_samples = audio_samples_from_file(file.file)
    audio = AudioStream(audio_samples)
    transcription, _ = await asr.transcribe(audio)
    return format_transcription(transcription, response_format)


async def audio_receiver(ws: WebSocket, audio_stream: AudioStream) -> None:
    try:
        while True:
            bytes_ = await asyncio.wait_for(
                ws.receive_bytes(), timeout=config.max_no_data_seconds
            )
            logger.debug(f"Received {len(bytes_)} bytes of audio data")
            audio_samples = audio_samples_from_file(BytesIO(bytes_))
            audio_stream.extend(audio_samples)
            if audio_stream.duration - config.inactivity_window_seconds >= 0:
                audio = audio_stream.after(
                    audio_stream.duration - config.inactivity_window_seconds
                )
                vad_opts = VadOptions(min_silence_duration_ms=500, speech_pad_ms=0)
                # NOTE: This is a synchronous operation that runs every time new data is received.
                # This shouldn't be an issue unless data is being received in tiny chunks or the user's machine is a potato.
                timestamps = get_speech_timestamps(audio.data, vad_opts)
                if len(timestamps) == 0:
                    logger.info(
                        f"No speech detected in the last {config.inactivity_window_seconds} seconds."
                    )
                    break
                elif (
                    # last speech end time
                    config.inactivity_window_seconds
                    - timestamps[-1]["end"] / SAMPLES_PER_SECOND
                    >= config.max_inactivity_seconds
                ):
                    logger.info(
                        f"Not enough speech in the last {config.inactivity_window_seconds} seconds."
                    )
                    break
    except asyncio.TimeoutError:
        logger.info(
            f"No data received in {config.max_no_data_seconds} seconds. Closing the connection."
        )
    except WebSocketDisconnect as e:
        logger.info(f"Client disconnected: {e}")
    audio_stream.close()


def format_transcription(
    transcription: Transcription, response_format: ResponseFormat
) -> str:
    if response_format == ResponseFormat.TEXT:
        return transcription.text
    elif response_format == ResponseFormat.JSON:
        return TranscriptionResponse(text=transcription.text).model_dump_json()
    elif response_format == ResponseFormat.VERBOSE_JSON:
        return TranscriptionVerboseResponse(
            duration=transcription.duration,
            text=transcription.text,
            words=transcription.words,
        ).model_dump_json()


@app.websocket("/v1/audio/transcriptions")
async def transcribe_stream(
    ws: WebSocket,
    transcription_opts: TranscribeParams,
    response_format: ResponseFormat = ResponseFormat.JSON,
) -> None:
    await ws.accept()
    asr = FasterWhisperASR(whisper, transcription_opts)
    audio_stream = AudioStream()
    async with asyncio.TaskGroup() as tg:
        tg.create_task(audio_receiver(ws, audio_stream))
        async for transcription in audio_transcriber(asr, audio_stream):
            logger.debug(f"Sending transcription: {transcription.text}")
            if ws.client_state == WebSocketState.DISCONNECTED:
                break
            await ws.send_text(format_transcription(transcription, response_format))

    if not ws.client_state == WebSocketState.DISCONNECTED:
        logger.info("Closing the connection.")
        await ws.close()
