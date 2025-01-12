from collections.abc import AsyncGenerator, Generator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
import logging
import os
from typing import Protocol

from fastapi.testclient import TestClient
import httpx
from httpx import ASGITransport, AsyncClient
from openai import AsyncOpenAI
import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from speaches.config import Config, WhisperConfig
from speaches.dependencies import get_config
from speaches.hf_utils import download_kokoro_model
from speaches.main import create_app

DISABLE_LOGGERS = ["multipart.multipart", "faster_whisper"]
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_WHISPER_MODEL = "Systran/faster-whisper-tiny.en"
# TODO: figure out a way to initialize the config without parsing environment variables, as those may interfere with the tests  # noqa: E501
DEFAULT_WHISPER_CONFIG = WhisperConfig(model=DEFAULT_WHISPER_MODEL, ttl=0)
DEFAULT_CONFIG = Config(
    whisper=DEFAULT_WHISPER_CONFIG,
    # disable the UI as it slightly increases the app startup time due to the imports it's doing
    enable_ui=False,
)
TIMEOUT = httpx.Timeout(15.0)


def pytest_configure() -> None:
    for logger_name in DISABLE_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.disabled = True


# NOTE: not being used. Keeping just in case. Needs to be modified to work similarly to `aclient_factory`
@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    os.environ["WHISPER__MODEL"] = "Systran/faster-whisper-tiny.en"
    with TestClient(create_app()) as client:
        yield client


# https://stackoverflow.com/questions/74890214/type-hint-callback-function-with-optional-parameters-aka-callable-with-optional
class AclientFactory(Protocol):
    def __call__(self, config: Config = DEFAULT_CONFIG) -> AbstractAsyncContextManager[AsyncClient]: ...


@pytest_asyncio.fixture()
async def aclient_factory(mocker: MockerFixture) -> AclientFactory:
    """Returns a context manager that provides an `AsyncClient` instance with `app` using the provided configuration."""

    @asynccontextmanager
    async def inner(config: Config = DEFAULT_CONFIG) -> AsyncGenerator[AsyncClient, None]:
        # NOTE: all calls to `get_config` should be patched. One way to test that this works is to update the original `get_config` to raise an exception and see if the tests fail  # noqa: E501
        mocker.patch("speaches.dependencies.get_config", return_value=config)
        mocker.patch("speaches.main.get_config", return_value=config)
        # NOTE: I couldn't get the following to work but it shouldn't matter
        # mocker.patch(
        #     "speaches.text_utils.Transcription._ensure_no_word_overlap.get_config", return_value=config
        # )

        app = create_app()
        # https://fastapi.tiangolo.com/advanced/testing-dependencies/
        app.dependency_overrides[get_config] = lambda: config
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=TIMEOUT) as aclient:
            yield aclient

    return inner


@pytest_asyncio.fixture()
async def aclient(aclient_factory: AclientFactory) -> AsyncGenerator[AsyncClient, None]:
    async with aclient_factory() as aclient:
        yield aclient


@pytest_asyncio.fixture()
def openai_client(aclient: AsyncClient) -> AsyncOpenAI:
    return AsyncOpenAI(api_key="cant-be-empty", http_client=aclient)


@pytest.fixture
def actual_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        # `base_url` is provided in case `OPENAI_BASE_URL` is set to a different value
        base_url=OPENAI_BASE_URL
    )


# TODO: remove the download after running the tests
# TODO: do not download when not needed
# @pytest.fixture(scope="session", autouse=True)
# def download_piper_voices() -> None:
#     # Only download `voices.json` and the default voice
#     snapshot_download("rhasspy/piper-voices", allow_patterns=["voices.json", "en/en_US/amy/**"])


@pytest.fixture(scope="session", autouse=True)
def download_kokoro() -> None:
    download_kokoro_model()
