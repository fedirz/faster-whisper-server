import json
import os
import threading
import time
from difflib import SequenceMatcher
from typing import Generator

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from faster_whisper_server.config import BYTES_PER_SECOND
from faster_whisper_server.main import app
from faster_whisper_server.server_models import TranscriptionVerboseJsonResponse

SIMILARITY_THRESHOLD = 0.97
AUDIO_FILES_LIMIT = 5
AUDIO_FILE_DIR = "tests/data"
TRANSCRIBE_ENDPOINT = "/v1/audio/transcriptions?response_format=verbose_json"


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def ws(client: TestClient) -> Generator[WebSocketTestSession, None, None]:
    with client.websocket_connect(TRANSCRIBE_ENDPOINT) as ws:
        yield ws


def get_audio_file_paths():
    file_paths = []
    directory = "tests/data"
    for filename in sorted(os.listdir(directory)[:AUDIO_FILES_LIMIT]):
        file_paths.append(os.path.join(directory, filename))
    return file_paths


file_paths = get_audio_file_paths()


def stream_audio_data(
    ws: WebSocketTestSession, data: bytes, *, chunk_size: int = 4000, speed: float = 1.0
):
    for i in range(0, len(data), chunk_size):
        ws.send_bytes(data[i : i + chunk_size])
        delay = len(data[i : i + chunk_size]) / BYTES_PER_SECOND / speed
        time.sleep(delay)


def transcribe_audio_data(
    client: TestClient, data: bytes
) -> TranscriptionVerboseJsonResponse:
    response = client.post(
        TRANSCRIBE_ENDPOINT,
        files={"file": ("audio.raw", data, "audio/raw")},
    )
    data = json.loads(response.json())  # TODO: figure this out
    return TranscriptionVerboseJsonResponse(**data)  # type: ignore


@pytest.mark.parametrize("file_path", file_paths)
def test_ws_audio_transcriptions(
    client: TestClient, ws: WebSocketTestSession, file_path: str
):
    with open(file_path, "rb") as file:
        data = file.read()

    streaming_transcription: TranscriptionVerboseJsonResponse = None  # type: ignore
    thread = threading.Thread(
        target=stream_audio_data, args=(ws, data), kwargs={"speed": 4.0}
    )
    thread.start()
    while True:
        try:
            streaming_transcription = TranscriptionVerboseJsonResponse(
                **ws.receive_json()
            )
        except WebSocketDisconnect:
            break
    file_transcription = transcribe_audio_data(client, data)
    s = SequenceMatcher(
        lambda x: x == " ", file_transcription.text, streaming_transcription.text
    )
    assert (
        s.ratio() > SIMILARITY_THRESHOLD
    ), f"\nExpected: {file_transcription.text}\nReceived: {streaming_transcription.text}"
