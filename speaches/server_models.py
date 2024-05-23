from __future__ import annotations

import enum

from faster_whisper.transcribe import Segment, Word
from pydantic import BaseModel

from speaches.core import Transcription


# https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-response_format
class ResponseFormat(enum.StrEnum):
    TEXT = "text"
    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    # VTT = "vtt"
    # SRT = "srt"


# https://platform.openai.com/docs/api-reference/audio/json-object
class TranscriptionJsonResponse(BaseModel):
    text: str

    @classmethod
    def from_transcription(
        cls, transcription: Transcription
    ) -> TranscriptionJsonResponse:
        return cls(text=transcription.text)


# https://platform.openai.com/docs/api-reference/audio/verbose-json-object
class TranscriptionVerboseJsonResponse(BaseModel):
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: list[Word]
    segments: list[Segment]

    @classmethod
    def from_transcription(
        cls, transcription: Transcription
    ) -> TranscriptionVerboseJsonResponse:
        return cls(
            language="english",  # FIX: hardcoded
            duration=transcription.duration,
            text=transcription.text,
            words=[
                Word(
                    start=word.start,
                    end=word.end,
                    word=word.text,
                    probability=word.probability,
                )
                for word in transcription.words
            ],
            segments=[],  # FIX: hardcoded
        )
