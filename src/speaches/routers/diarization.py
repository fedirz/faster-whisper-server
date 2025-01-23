from fastapi import APIRouter, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import StreamingResponse, Response
from speaches.dependencies import ConfigDependency, AudioFileDependency, ModelManagerDependency
from io import BytesIO
import logging
import numpy as np

from pydantic import BaseModel
from typing import Optional, List, Annotated
from speaches.api_types import (
    CreateTranscriptionResponseJson,
    CreateTranscriptionResponseVerboseJson,
    TimestampGranularities,
    TranscriptionSegment,
)
import asyncio
from faster_whisper.transcribe import BatchedInferencePipeline
from speaches.map_speakers import map_speakers_to_segments, DiarizationSegment, TranscriptionSegment
from speaches.config import Task, ResponseFormat
from speaches.routers.stt import ModelName, Language, get_timestamp_granularities, DEFAULT_TIMESTAMP_GRANULARITIES

import torchaudio

logger = logging.getLogger(__name__)
router = APIRouter(tags=['Diarization'])



class DiarizationResponse(BaseModel):
    diarization_segments: List[dict]
    success: bool
    error: Optional[str] = None


async def diarize_audio(
    config: ConfigDependency,
    audio: UploadFile = File(...),
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
):
    global diarization_pipeline
    try:
        logger.info(audio)
        # Perform diarization with the properly formatted audio input
        # Read the audio file
        audio_content = await audio.read()
        audio_stream = BytesIO(audio_content)
        
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(audio_stream)

        diarization = diarization_pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
            min_speakers=min_speakers or config.diarization.min_speakers,
            max_speakers=max_speakers or config.diarization.max_speakers,
        )
        
        # Convert diarization output to the expected format
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": float(turn.start),
                "end": float(turn.end)
            })
            
        return DiarizationResponse(
            diarization_segments=segments,
            success=True
        )
            
    except Exception as e:
        logger.exception("Diarization failed")
        return DiarizationResponse(
            diarization_segments=[],
            success=False,
            error=str(e)
        )
        
@router.post('/diarize', response_model=DiarizationResponse)
async def diarize(
    config: ConfigDependency,
    audio: UploadFile = File(...),
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None
):
    return await diarize_audio(config, audio, num_speakers, min_speakers, max_speakers)


@router.post(
    "/v1/audio/diarization",
    response_model=str | CreateTranscriptionResponseJson | CreateTranscriptionResponseVerboseJson,
)
async def diarize_file(
    config: ConfigDependency,
    model_manager: ModelManagerDependency,
    request: Request,
    audio: AudioFileDependency,
    model: Annotated[ModelName | None, Form()] = None,
    language: Annotated[Language | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat | None, Form()] = None,
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        TimestampGranularities,
        # WARN: `alias` doesn't actually work.
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    # stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str | None, Form()] = None,
    vad_filter: Annotated[bool, Form()] = False,
    num_speakers: Annotated[int | None, Form()] = None,
) -> Response | StreamingResponse:
    if model is None:
        model = config.whisper.model
    if language is None:
        language = config.default_language
    if response_format is None:
        response_format = config.default_response_format
    timestamp_granularities = asyncio.run(get_timestamp_granularities(request))
    if timestamp_granularities != DEFAULT_TIMESTAMP_GRANULARITIES and response_format != ResponseFormat.VERBOSE_JSON:
        logger.warning(
            "It only makes sense to provide `timestamp_granularities[]` when `response_format` is set to `verbose_json`. See https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-timestamp_granularities."  # noqa: E501
        )
    with model_manager.load_model(model) as whisper:
        whisper_model = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
        segments, transcription_info = whisper_model.transcribe(
            audio,
            task=Task.TRANSCRIBE,
            language=language,
            initial_prompt=prompt,
            word_timestamps="word" in timestamp_granularities,
            temperature=temperature,
            vad_filter=vad_filter,
            hotwords=hotwords,
        )
        # segments = TranscriptionSegment.from_faster_whisper_segments(segments)
        transcription_segments = list(TranscriptionSegment.from_faster_whisper_segments(segments))

    params = {'num_speakers': num_speakers} if num_speakers else {}
    
    try:
        # Convert numpy array to bytes
        audio_bytes = BytesIO()
        
        waveform = np.expand_dims(audio, axis=0)  # Добавляем размерность канала (channel dimension)
        # Save as WAV file
        torchaudio.save(
            audio_bytes,
            src=waveform,
            #torch.from_numpy(audio).unsqueeze(0),  # Add channel dimension
            sample_rate=16000,  # faster-whisper uses 16kHz
            format="wav"
        )
        audio_bytes.seek(0)  # Reset buffer position
        
        # Create files dictionary for requests
        files = {
            'audio': ('audio.wav', audio_bytes, 'audio/wav')
        }
        
        # Send request with proper file formatting
        diarization_response = await diarize(
            audio=audio_bytes,
            #files=files,
            **params
        )
        
        logger.info(diarization_data)
        if not diarization_response.success:
            raise HTTPException(
                status_code=500,
                detail=f"Diarization failed: {diarization_response.error}"
            )
            
        # Assign speakers to segments based on time overlap
        # Convert diarization segments to proper objects
        diarization_segments = [
                DiarizationSegment(
                    speaker=seg['speaker'],
                    start=float(seg['start']),
                    end=float(seg['end'])
                )
                for seg in diarization_response.diarization_segments
            ]
        logger.info(transcription_segments)
        logger.info(diarization_segments)
            # Map speakers to segments and return JSON response directly
        result = map_speakers_to_segments(transcription_segments, diarization_segments)
        return Response(
                content=result,  # result is already a JSON string from map_speakers_to_segments
                media_type="application/json"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to diarization service: {str(e)}"
        )
    # return map_speakers_to_segments(list(segments), diarization_segments)
    # return segments_to_response(segments, transcription_info, response_format)