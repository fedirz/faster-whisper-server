from collections.abc import Generator
import io
import logging
import time
from typing import Annotated, Literal, Self

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import numpy as np
from piper.voice import PiperVoice
from pydantic import BaseModel, BeforeValidator, Field, ValidationError, model_validator
import soundfile as sf

from speaches.dependencies import PiperModelManagerDependency
from speaches.hf_utils import (
    PiperModel,
    list_piper_models,
    read_piper_voices_config,
)

DEFAULT_MODEL = "piper"
# https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-response_format
DEFAULT_RESPONSE_FORMAT = "mp3"
DEFAULT_VOICE = "en_US-amy-medium"  # TODO: make configurable
DEFAULT_VOICE_SAMPLE_RATE = 22050  # NOTE: Dependant on the voice

# https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-model
# https://platform.openai.com/docs/models/tts
OPENAI_SUPPORTED_SPEECH_MODEL = ("tts-1", "tts-1-hd")

# https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-voice
# https://platform.openai.com/docs/guides/text-to-speech/voice-options
OPENAI_SUPPORTED_SPEECH_VOICE_NAMES = ("alloy", "echo", "fable", "onyx", "nova", "shimmer")

# https://platform.openai.com/docs/guides/text-to-speech/supported-output-formats
type ResponseFormat = Literal["mp3", "flac", "wav", "pcm"]
SUPPORTED_RESPONSE_FORMATS = ("mp3", "flac", "wav", "pcm")
UNSUPORTED_RESPONSE_FORMATS = ("opus", "aac")

MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000


logger = logging.getLogger(__name__)

router = APIRouter(tags=["speech-to-text"])


# aip 'Write a function `resample_audio` which would take in RAW PCM 16-bit signed, little-endian audio data represented as bytes (`audio_bytes`) and resample it (either downsample or upsample) from `sample_rate` to `target_sample_rate` using numpy'  # noqa: E501
def resample_audio(audio_bytes: bytes, sample_rate: int, target_sample_rate: int) -> bytes:
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    duration = len(audio_data) / sample_rate
    target_length = int(duration * target_sample_rate)
    resampled_data = np.interp(
        np.linspace(0, len(audio_data), target_length, endpoint=False), np.arange(len(audio_data)), audio_data
    )
    return resampled_data.astype(np.int16).tobytes()


def generate_audio(
    piper_tts: PiperVoice, text: str, *, speed: float = 1.0, sample_rate: int | None = None
) -> Generator[bytes, None, None]:
    if sample_rate is None:
        sample_rate = piper_tts.config.sample_rate
    start = time.perf_counter()
    for audio_bytes in piper_tts.synthesize_stream_raw(text, length_scale=1.0 / speed):
        if sample_rate != piper_tts.config.sample_rate:
            audio_bytes = resample_audio(audio_bytes, piper_tts.config.sample_rate, sample_rate)  # noqa: PLW2901
        yield audio_bytes
    logger.info(f"Generated audio for {len(text)} characters in {time.perf_counter() - start}s")


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
    converted_audio_bytes_buffer = io.BytesIO()
    sf.write(converted_audio_bytes_buffer, data, samplerate=sample_rate, format=audio_format)
    return converted_audio_bytes_buffer.getvalue()


def handle_openai_supported_model_ids(model_id: str) -> str:
    if model_id in OPENAI_SUPPORTED_SPEECH_MODEL:
        logger.warning(f"{model_id} is not a valid model name. Using '{DEFAULT_MODEL}' instead.")
        return DEFAULT_MODEL
    return model_id


ModelId = Annotated[
    Literal["piper"],
    BeforeValidator(handle_openai_supported_model_ids),
    Field(
        description=f"The ID of the model. The only supported model is '{DEFAULT_MODEL}'.",
        examples=[DEFAULT_MODEL],
    ),
]


def handle_openai_supported_voices(voice: str) -> str:
    if voice in OPENAI_SUPPORTED_SPEECH_VOICE_NAMES:
        logger.warning(f"{voice} is not a valid voice name. Using '{DEFAULT_VOICE}' instead.")
        return DEFAULT_VOICE
    return voice


Voice = Annotated[str, BeforeValidator(handle_openai_supported_voices)]  # TODO: description and examples


class CreateSpeechRequestBody(BaseModel):
    model: ModelId = DEFAULT_MODEL
    input: str = Field(
        ...,
        description="The text to generate audio for. ",
        examples=[
            "A rainbow is an optical phenomenon caused by refraction, internal reflection and dispersion of light in water droplets resulting in a continuous spectrum of light appearing in the sky. The rainbow takes the form of a multicoloured circular arc. Rainbows caused by sunlight always appear in the section of sky directly opposite the Sun. Rainbows can be caused by many forms of airborne water. These include not only rain, but also mist, spray, and airborne dew."  # noqa: E501
        ],
    )
    voice: Voice = DEFAULT_VOICE
    """
The last part of the voice name is the quality (x_low, low, medium, high).
Each quality has a different default sample rate:
- x_low: 16000 Hz
- low: 16000 Hz
- medium: 22050 Hz
- high: 22050 Hz
    """
    response_format: ResponseFormat = Field(
        DEFAULT_RESPONSE_FORMAT,
        description=f"The format to audio in. Supported formats are {", ".join(SUPPORTED_RESPONSE_FORMATS)}. {", ".join(UNSUPORTED_RESPONSE_FORMATS)} are not supported",  # noqa: E501
        examples=list(SUPPORTED_RESPONSE_FORMATS),
    )
    # https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-voice
    speed: float = Field(1.0, ge=0.25, le=4.0)
    """The speed of the generated audio. Select a value from 0.25 to 4.0. 1.0 is the default."""
    sample_rate: int | None = Field(None, ge=MIN_SAMPLE_RATE, le=MAX_SAMPLE_RATE)
    """Desired sample rate to convert the generated audio to. If not provided, the model's default sample rate will be used."""  # noqa: E501
    # TODO: document default sample rate for each voice quality

    # TODO: move into `Voice`
    @model_validator(mode="after")
    def verify_voice_is_valid(self) -> Self:
        valid_voices = read_piper_voices_config()
        if self.voice not in valid_voices:
            raise ValidationError(f"Voice '{self.voice}' is not supported. Supported voices: {valid_voices.keys()}")
        return self


# https://platform.openai.com/docs/api-reference/audio/createSpeech
@router.post("/v1/audio/speech")
def synthesize(
    piper_model_manager: PiperModelManagerDependency,
    body: CreateSpeechRequestBody,
) -> StreamingResponse:
    with piper_model_manager.load_model(body.voice) as piper_tts:
        audio_generator = generate_audio(piper_tts, body.input, speed=body.speed, sample_rate=body.sample_rate)
        if body.response_format != "pcm":
            audio_generator = (
                convert_audio_format(
                    audio_bytes, body.sample_rate or piper_tts.config.sample_rate, body.response_format
                )
                for audio_bytes in audio_generator
            )

        return StreamingResponse(audio_generator, media_type=f"audio/{body.response_format}")


@router.get("/v1/audio/speech/voices")
def list_voices() -> list[PiperModel]:
    return list(list_piper_models())
