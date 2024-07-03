from collections.abc import Generator
import logging

from fastapi.testclient import TestClient
import pytest

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
