"""See `tests/openai_timestamp_granularities_test.py` to understand how OpenAI handles `response_type` and `timestamp_granularities`."""  # noqa: E501

from faster_whisper_server.server_models import TIMESTAMP_GRANULARITIES_COMBINATIONS, TimestampGranularities
from openai import AsyncOpenAI
import pytest


@pytest.mark.asyncio()
@pytest.mark.parametrize("timestamp_granularities", TIMESTAMP_GRANULARITIES_COMBINATIONS)
async def test_api_json_response_format_and_timestamp_granularities_combinations(
    openai_client: AsyncOpenAI,
    timestamp_granularities: TimestampGranularities,
) -> None:
    audio_file = open("audio.wav", "rb")  # noqa: SIM115, ASYNC230

    await openai_client.audio.transcriptions.create(
        file=audio_file, model="whisper-1", response_format="json", timestamp_granularities=timestamp_granularities
    )


@pytest.mark.asyncio()
@pytest.mark.parametrize("timestamp_granularities", TIMESTAMP_GRANULARITIES_COMBINATIONS)
async def test_api_verbose_json_response_format_and_timestamp_granularities_combinations(
    openai_client: AsyncOpenAI,
    timestamp_granularities: TimestampGranularities,
) -> None:
    audio_file = open("audio.wav", "rb")  # noqa: SIM115, ASYNC230

    transcription = await openai_client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=timestamp_granularities,
    )

    assert transcription.__pydantic_extra__
    if "word" in timestamp_granularities:
        assert transcription.__pydantic_extra__.get("segments") is not None
        assert transcription.__pydantic_extra__.get("words") is not None
    else:
        # Unless explicitly requested, words are not present
        assert transcription.__pydantic_extra__.get("segments") is not None
        assert transcription.__pydantic_extra__.get("words") is None
