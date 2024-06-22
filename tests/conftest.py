import logging
import os
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# HACK
os.environ["WHISPER_MODEL"] = "Systran/faster-whisper-tiny.en"
from faster_whisper_server.main import app  # noqa: E402

disable_loggers = ["multipart.multipart", "faster_whisper"]


def pytest_configure():
    for logger_name in disable_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as client:
        yield client
