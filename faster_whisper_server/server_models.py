from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from faster_whisper_server import utils

if TYPE_CHECKING:
    from faster_whisper.transcribe import Segment, TranscriptionInfo, Word

    from faster_whisper_server.core import Transcription


# https://platform.openai.com/docs/api-reference/audio/json-object
class TranscriptionJsonResponse(BaseModel):
    text: str

    @classmethod
    def from_segments(cls, segments: list[Segment]) -> TranscriptionJsonResponse:
        return cls(text=utils.segments_text(segments))

    @classmethod
    def from_transcription(cls, transcription: Transcription) -> TranscriptionJsonResponse:
        return cls(text=transcription.text)


class WordObject(BaseModel):
    start: float
    end: float
    word: str
    probability: float

    @classmethod
    def from_word(cls, word: Word) -> WordObject:
        return cls(
            start=word.start,
            end=word.end,
            word=word.word,
            probability=word.probability,
        )


class SegmentObject(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float

    @classmethod
    def from_segment(cls, segment: Segment) -> SegmentObject:
        return cls(
            id=segment.id,
            seek=segment.seek,
            start=segment.start,
            end=segment.end,
            text=segment.text,
            tokens=segment.tokens,
            temperature=segment.temperature,
            avg_logprob=segment.avg_logprob,
            compression_ratio=segment.compression_ratio,
            no_speech_prob=segment.no_speech_prob,
        )


# https://platform.openai.com/docs/api-reference/audio/verbose-json-object
class TranscriptionVerboseJsonResponse(BaseModel):
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: list[WordObject]
    segments: list[SegmentObject]

    @classmethod
    def from_segment(cls, segment: Segment, transcription_info: TranscriptionInfo) -> TranscriptionVerboseJsonResponse:
        return cls(
            language=transcription_info.language,
            duration=segment.end - segment.start,
            text=segment.text,
            words=([WordObject.from_word(word) for word in segment.words] if isinstance(segment.words, list) else []),
            segments=[SegmentObject.from_segment(segment)],
        )

    @classmethod
    def from_segments(
        cls, segments: list[Segment], transcription_info: TranscriptionInfo
    ) -> TranscriptionVerboseJsonResponse:
        return cls(
            language=transcription_info.language,
            duration=transcription_info.duration,
            text=utils.segments_text(segments),
            segments=[SegmentObject.from_segment(segment) for segment in segments],
            words=[WordObject.from_word(word) for word in utils.words_from_segments(segments)],
        )

    @classmethod
    def from_transcription(cls, transcription: Transcription) -> TranscriptionVerboseJsonResponse:
        return cls(
            language="english",  # FIX: hardcoded
            duration=transcription.duration,
            text=transcription.text,
            words=[
                WordObject(
                    start=word.start,
                    end=word.end,
                    word=word.text,
                    probability=word.probability,
                )
                for word in transcription.words
            ],
            segments=[],  # FIX: hardcoded
        )


class ModelListResponse(BaseModel):
    data: list[ModelObject]
    object: Literal["list"] = "list"


class ModelObject(BaseModel):
    id: str
    """The model identifier, which can be referenced in the API endpoints."""
    created: int
    """The Unix timestamp (in seconds) when the model was created."""
    object_: Literal["model"] = Field(serialization_alias="object")
    """The object type, which is always "model"."""
    owned_by: str
    """The organization that owns the model."""
    language: list[str] = Field(default_factory=list)
    """List of ISO 639-3 supported by the model. It's possible that the list will be empty. This field is not a part of the OpenAI API spec and is added for convenience."""  # noqa: E501

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "Systran/faster-whisper-large-v3",
                    "created": 1700732060,
                    "object": "model",
                    "owned_by": "Systran",
                },
                {
                    "id": "Systran/faster-distil-whisper-large-v3",
                    "created": 1711378296,
                    "object": "model",
                    "owned_by": "Systran",
                },
                {
                    "id": "bofenghuang/whisper-large-v2-cv11-french-ct2",
                    "created": 1687968011,
                    "object": "model",
                    "owned_by": "bofenghuang",
                },
            ]
        },
    )
