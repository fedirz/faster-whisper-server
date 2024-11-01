import io
import platform

from openai import APIConnectionError, AsyncOpenAI, UnprocessableEntityError
import pytest
import soundfile as sf

platform_machine = platform.machine()
if platform_machine != "x86_64":
    pytest.skip("Only supported on x86_64", allow_module_level=True)

from faster_whisper_server.routers.speech import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_RESPONSE_FORMAT,
    DEFAULT_VOICE,
    SUPPORTED_RESPONSE_FORMATS,
    ResponseFormat,
)

DEFAULT_INPUT = "Hello, world!"


@pytest.mark.asyncio
@pytest.mark.skipif(platform_machine != "x86_64", reason="Only supported on x86_64")
@pytest.mark.parametrize("response_format", SUPPORTED_RESPONSE_FORMATS)
async def test_create_speech_formats(openai_client: AsyncOpenAI, response_format: ResponseFormat) -> None:
    await openai_client.audio.speech.create(
        model=DEFAULT_MODEL,
        voice=DEFAULT_VOICE,  # type: ignore  # noqa: PGH003
        input=DEFAULT_INPUT,
        response_format=response_format,
    )


GOOD_MODEL_VOICE_PAIRS: list[tuple[str, str]] = [
    ("tts-1", "alloy"),  # OpenAI and OpenAI
    ("tts-1-hd", "echo"),  # OpenAI and OpenAI
    ("tts-1", DEFAULT_VOICE),  # OpenAI and Piper
    (DEFAULT_MODEL, "echo"),  # Piper and OpenAI
    (DEFAULT_MODEL, DEFAULT_VOICE),  # Piper and Piper
]


@pytest.mark.asyncio
@pytest.mark.skipif(platform_machine != "x86_64", reason="Only supported on x86_64")
@pytest.mark.parametrize(("model", "voice"), GOOD_MODEL_VOICE_PAIRS)
async def test_create_speech_good_model_voice_pair(openai_client: AsyncOpenAI, model: str, voice: str) -> None:
    await openai_client.audio.speech.create(
        model=model,
        voice=voice,  # type: ignore  # noqa: PGH003
        input=DEFAULT_INPUT,
        response_format=DEFAULT_RESPONSE_FORMAT,
    )


BAD_MODEL_VOICE_PAIRS: list[tuple[str, str]] = [
    ("tts-1", "invalid"),  # OpenAI and invalid
    ("invalid", "echo"),  # Invalid and OpenAI
    (DEFAULT_MODEL, "invalid"),  # Piper and invalid
    ("invalid", DEFAULT_VOICE),  # Invalid and Piper
    ("invalid", "invalid"),  # Invalid and invalid
]


@pytest.mark.asyncio
@pytest.mark.skipif(platform_machine != "x86_64", reason="Only supported on x86_64")
@pytest.mark.parametrize(("model", "voice"), BAD_MODEL_VOICE_PAIRS)
async def test_create_speech_bad_model_voice_pair(openai_client: AsyncOpenAI, model: str, voice: str) -> None:
    # NOTE: not sure why `APIConnectionError` is sometimes raised
    with pytest.raises((UnprocessableEntityError, APIConnectionError)):
        await openai_client.audio.speech.create(
            model=model,
            voice=voice,  # type: ignore  # noqa: PGH003
            input=DEFAULT_INPUT,
            response_format=DEFAULT_RESPONSE_FORMAT,
        )


SUPPORTED_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]


@pytest.mark.asyncio
@pytest.mark.skipif(platform_machine != "x86_64", reason="Only supported on x86_64")
async def test_create_speech_with_varying_speed(openai_client: AsyncOpenAI) -> None:
    previous_size: int | None = None
    for speed in SUPPORTED_SPEEDS:
        res = await openai_client.audio.speech.create(
            model=DEFAULT_MODEL,
            voice=DEFAULT_VOICE,  # type: ignore  # noqa: PGH003
            input=DEFAULT_INPUT,
            response_format="pcm",
            speed=speed,
        )
        audio_bytes = res.read()
        if previous_size is not None:
            assert len(audio_bytes) * 1.5 < previous_size  # TODO: document magic number
        previous_size = len(audio_bytes)


UNSUPPORTED_SPEEDS = [0.1, 4.1]


@pytest.mark.asyncio
@pytest.mark.skipif(platform_machine != "x86_64", reason="Only supported on x86_64")
@pytest.mark.parametrize("speed", UNSUPPORTED_SPEEDS)
async def test_create_speech_with_unsupported_speed(openai_client: AsyncOpenAI, speed: float) -> None:
    with pytest.raises(UnprocessableEntityError):
        await openai_client.audio.speech.create(
            model=DEFAULT_MODEL,
            voice=DEFAULT_VOICE,  # type: ignore  # noqa: PGH003
            input=DEFAULT_INPUT,
            response_format="pcm",
            speed=speed,
        )


VALID_SAMPLE_RATES = [16000, 22050, 24000, 48000]


@pytest.mark.asyncio
@pytest.mark.skipif(platform_machine != "x86_64", reason="Only supported on x86_64")
@pytest.mark.parametrize("sample_rate", VALID_SAMPLE_RATES)
async def test_speech_valid_resample(openai_client: AsyncOpenAI, sample_rate: int) -> None:
    res = await openai_client.audio.speech.create(
        model=DEFAULT_MODEL,
        voice=DEFAULT_VOICE,  # type: ignore  # noqa: PGH003
        input=DEFAULT_INPUT,
        response_format="wav",
        extra_body={"sample_rate": sample_rate},
    )
    _, actual_sample_rate = sf.read(io.BytesIO(res.content))
    assert actual_sample_rate == sample_rate


INVALID_SAMPLE_RATES = [7999, 48001]


@pytest.mark.asyncio
@pytest.mark.skipif(platform_machine != "x86_64", reason="Only supported on x86_64")
@pytest.mark.parametrize("sample_rate", INVALID_SAMPLE_RATES)
async def test_speech_invalid_resample(openai_client: AsyncOpenAI, sample_rate: int) -> None:
    with pytest.raises(UnprocessableEntityError):
        await openai_client.audio.speech.create(
            model=DEFAULT_MODEL,
            voice=DEFAULT_VOICE,  # type: ignore  # noqa: PGH003
            input=DEFAULT_INPUT,
            response_format="wav",
            extra_body={"sample_rate": sample_rate},
        )


# TODO: implement the following test

# NUMBER_OF_MODELS = 1
# NUMBER_OF_VOICES = 124
#
#
# @pytest.mark.asyncio
# async def test_list_tts_models(openai_client: AsyncOpenAI) -> None:
#     raise NotImplementedError
