from collections.abc import AsyncGenerator, Generator
import logging
import os

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from openai import AsyncOpenAI
import pytest
import pytest_asyncio

from faster_whisper_server.main import create_app

disable_loggers = ["multipart.multipart", "faster_whisper"]


def pytest_configure() -> None:
    for logger_name in disable_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True


# NOTE: not being used. Keeping just in case
@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    os.environ["WHISPER__MODEL"] = "Systran/faster-whisper-tiny.en"
    with TestClient(create_app()) as client:
        yield client


@pytest_asyncio.fixture()
async def aclient() -> AsyncGenerator[AsyncClient, None]:
    os.environ["WHISPER__MODEL"] = "Systran/faster-whisper-tiny.en"
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as aclient:
        yield aclient


@pytest_asyncio.fixture()
def openai_client(aclient: AsyncClient) -> AsyncOpenAI:
    return AsyncOpenAI(api_key="cant-be-empty", http_client=aclient)


@pytest.fixture
def actual_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="https://api.openai.com/v1"
    )  # `base_url` is provided in case `OPENAI_API_BASE_URL` is set to a different value
