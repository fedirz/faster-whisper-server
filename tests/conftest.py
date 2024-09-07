from collections.abc import AsyncGenerator, Generator
import logging

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from openai import OpenAI
import pytest
import pytest_asyncio

disable_loggers = ["multipart.multipart", "faster_whisper"]


def pytest_configure() -> None:
    for logger_name in disable_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    from faster_whisper_server.main import app

    with TestClient(app) as client:
        yield client


@pytest_asyncio.fixture()
async def aclient() -> AsyncGenerator[AsyncClient, None]:
    from faster_whisper_server.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as aclient:
        yield aclient


@pytest.fixture()
def openai_client(client: TestClient) -> OpenAI:
    return OpenAI(api_key="cant-be-empty", http_client=client)
