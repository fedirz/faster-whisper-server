"""See `tests/openai_timestamp_granularities_test.py` to understand how OpenAI handles `response_type` and `timestamp_granularities`."""

from pathlib import Path

from openai import AsyncOpenAI
import pytest

from speaches.api_types import TIMESTAMP_GRANULARITIES_COMBINATIONS, TimestampGranularities


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamp_granularities", TIMESTAMP_GRANULARITIES_COMBINATIONS)
async def test_api_json_response_format_and_timestamp_granularities_combinations(
    openai_client: AsyncOpenAI,
    timestamp_granularities: TimestampGranularities,
) -> None:
    file_path = Path("audio.wav")

    await openai_client.audio.transcriptions.create(
        file=file_path, model="whisper-1", response_format="json", timestamp_granularities=timestamp_granularities
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamp_granularities", TIMESTAMP_GRANULARITIES_COMBINATIONS)
async def test_api_verbose_json_response_format_and_timestamp_granularities_combinations(
    openai_client: AsyncOpenAI,
    timestamp_granularities: TimestampGranularities,
) -> None:
    file_path = Path("audio.wav")

    transcription = await openai_client.audio.transcriptions.create(
        file=file_path,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=timestamp_granularities,
    )

    if "word" in timestamp_granularities:
        assert transcription.segments is not None
        assert transcription.words is not None
    else:
        # Unless explicitly requested, words are not present
        assert transcription.segments is not None
        assert transcription.words is None
