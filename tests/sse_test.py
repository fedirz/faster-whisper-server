import json
import os

from fastapi.testclient import TestClient
from httpx_sse import connect_sse
import pytest
import srt
import webvtt
import webvtt.vtt

from faster_whisper_server.server_models import (
    TranscriptionJsonResponse,
    TranscriptionVerboseJsonResponse,
)

FILE_PATHS = ["audio.wav"]  # HACK
ENDPOINTS = [
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
]


parameters = [(file_path, endpoint) for endpoint in ENDPOINTS for file_path in FILE_PATHS]


@pytest.mark.parametrize(("file_path", "endpoint"), parameters)
def test_streaming_transcription_text(client: TestClient, file_path: str, endpoint: str) -> None:
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
            assert len(event.data) > 1  # HACK: 1 because of the space character that's always prepended


@pytest.mark.parametrize(("file_path", "endpoint"), parameters)
def test_streaming_transcription_json(client: TestClient, file_path: str, endpoint: str) -> None:
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


@pytest.mark.parametrize(("file_path", "endpoint"), parameters)
def test_streaming_transcription_verbose_json(client: TestClient, file_path: str, endpoint: str) -> None:
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


def test_transcription_vtt(client: TestClient) -> None:
    with open("audio.wav", "rb") as f:
        data = f.read()
    kwargs = {
        "files": {"file": ("audio.wav", data, "audio/wav")},
        "data": {"response_format": "vtt", "stream": False},
    }
    response = client.post("/v1/audio/transcriptions", **kwargs)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/vtt; charset=utf-8"
    text = response.text
    webvtt.from_string(text)
    text = text.replace("WEBVTT", "YO")
    with pytest.raises(webvtt.vtt.MalformedFileError):
        webvtt.from_string(text)


def test_transcription_srt(client: TestClient) -> None:
    with open("audio.wav", "rb") as f:
        data = f.read()
    kwargs = {
        "files": {"file": ("audio.wav", data, "audio/wav")},
        "data": {"response_format": "srt", "stream": False},
    }
    response = client.post("/v1/audio/transcriptions", **kwargs)
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    text = response.text
    list(srt.parse(text))
    text = text.replace("1", "YO")
    with pytest.raises(srt.SRTParseError):
        list(srt.parse(text))
