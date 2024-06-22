import json
import os

import pytest
from fastapi.testclient import TestClient
from httpx_sse import connect_sse

from faster_whisper_server.server_models import (
    TranscriptionJsonResponse,
    TranscriptionVerboseJsonResponse,
)

FILE_PATHS = ["audio.wav"]  # HACK
ENDPOINTS = [
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
]


parameters = [
    (file_path, endpoint) for endpoint in ENDPOINTS for file_path in FILE_PATHS
]


@pytest.mark.parametrize("file_path,endpoint", parameters)
def test_streaming_transcription_text(
    client: TestClient, file_path: str, endpoint: str
):
    extension = os.path.splitext(file_path)[1]
    with open(file_path, "rb") as f:
        data = f.read()
    kwargs = {
        "files": {"file": (f"audio.{extension}", data, f"audio/{extension}")},
        "data": {"response_format": "text", "stream": True},
    }
    with connect_sse(client, "POST", endpoint, **kwargs) as event_source:
        for event in event_source.iter_sse():
            print(event)
            assert (
                len(event.data) > 1
            )  # HACK: 1 because of the space character that's always prepended


@pytest.mark.parametrize("file_path,endpoint", parameters)
def test_streaming_transcription_json(
    client: TestClient, file_path: str, endpoint: str
):
    extension = os.path.splitext(file_path)[1]
    with open(file_path, "rb") as f:
        data = f.read()
    kwargs = {
        "files": {"file": (f"audio.{extension}", data, f"audio/{extension}")},
        "data": {"response_format": "json", "stream": True},
    }
    with connect_sse(client, "POST", endpoint, **kwargs) as event_source:
        for event in event_source.iter_sse():
            TranscriptionJsonResponse(**json.loads(event.data))


@pytest.mark.parametrize("file_path,endpoint", parameters)
def test_streaming_transcription_verbose_json(
    client: TestClient, file_path: str, endpoint: str
):
    extension = os.path.splitext(file_path)[1]
    with open(file_path, "rb") as f:
        data = f.read()
    kwargs = {
        "files": {"file": (f"audio.{extension}", data, f"audio/{extension}")},
        "data": {"response_format": "verbose_json", "stream": True},
    }
    with connect_sse(client, "POST", endpoint, **kwargs) as event_source:
        for event in event_source.iter_sse():
            TranscriptionVerboseJsonResponse(**json.loads(event.data))
