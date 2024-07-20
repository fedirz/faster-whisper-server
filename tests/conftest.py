from collections.abc import Generator
import logging
import os

from fastapi.testclient import TestClient
from openai import OpenAI
import pytest

os.environ["WHISPER__MODEL"] = "Systran/faster-whisper-tiny.en"
from faster_whisper_server.main import app

disable_loggers = ["multipart.multipart", "faster_whisper"]


def pytest_configure() -> None:
    for logger_name in disable_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def openai_client(client: TestClient) -> OpenAI:
    return OpenAI(api_key="cant-be-empty", http_client=client)
