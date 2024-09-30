from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from faster_whisper_server.text_utils import Transcription, canonicalize_word, segments_to_text

if TYPE_CHECKING:
    from collections.abc import Iterable

    import faster_whisper.transcribe


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10909
class TranscriptionWord(BaseModel):
    start: float
    end: float
    word: str
    probability: float

    @classmethod
    def from_segments(cls, segments: Iterable[TranscriptionSegment]) -> list[TranscriptionWord]:
        words: list[TranscriptionWord] = []
        for segment in segments:
            # NOTE: a temporary "fix" for https://github.com/fedirz/faster-whisper-server/issues/58.
            # TODO: properly address the issue
            assert (
                segment.words is not None
            ), "Segment must have words. If you are using an API ensure `timestamp_granularities[]=word` is set"
            words.extend(segment.words)
        return words

    def offset(self, seconds: float) -> None:
        self.start += seconds
        self.end += seconds

    @classmethod
    def common_prefix(cls, a: list[TranscriptionWord], b: list[TranscriptionWord]) -> list[TranscriptionWord]:
        i = 0
        while i < len(a) and i < len(b) and canonicalize_word(a[i].word) == canonicalize_word(b[i].word):
            i += 1
        return a[:i]


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10938
class TranscriptionSegment(BaseModel):
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
    words: list[TranscriptionWord] | None

    @classmethod
    def from_faster_whisper_segments(
        cls, segments: Iterable[faster_whisper.transcribe.Segment]
    ) -> Iterable[TranscriptionSegment]:
        for segment in segments:
            yield cls(
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
                words=[
                    TranscriptionWord(
                        start=word.start,
                        end=word.end,
                        word=word.word,
                        probability=word.probability,
                    )
                    for word in segment.words
                ]
                if segment.words is not None
                else None,
            )


# https://platform.openai.com/docs/api-reference/audio/json-object
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10924
class CreateTranscriptionResponseJson(BaseModel):
    text: str

    @classmethod
    def from_segments(cls, segments: list[TranscriptionSegment]) -> CreateTranscriptionResponseJson:
        return cls(text=segments_to_text(segments))

    @classmethod
    def from_transcription(cls, transcription: Transcription) -> CreateTranscriptionResponseJson:
        return cls(text=transcription.text)


# https://platform.openai.com/docs/api-reference/audio/verbose-json-object
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L11007
class CreateTranscriptionResponseVerboseJson(BaseModel):
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: list[TranscriptionWord] | None
    segments: list[TranscriptionSegment]

    @classmethod
    def from_segment(
        cls, segment: TranscriptionSegment, transcription_info: faster_whisper.transcribe.TranscriptionInfo
    ) -> CreateTranscriptionResponseVerboseJson:
        return cls(
            language=transcription_info.language,
            duration=segment.end - segment.start,
            text=segment.text,
            words=segment.words if transcription_info.transcription_options.word_timestamps else None,
            segments=[segment],
        )

    @classmethod
    def from_segments(
        cls, segments: list[TranscriptionSegment], transcription_info: faster_whisper.transcribe.TranscriptionInfo
    ) -> CreateTranscriptionResponseVerboseJson:
        return cls(
            language=transcription_info.language,
            duration=transcription_info.duration,
            text=segments_to_text(segments),
            segments=segments,
            words=TranscriptionWord.from_segments(segments)
            if transcription_info.transcription_options.word_timestamps
            else None,
        )

    @classmethod
    def from_transcription(cls, transcription: Transcription) -> CreateTranscriptionResponseVerboseJson:
        return cls(
            language="english",  # FIX: hardcoded
            duration=transcription.duration,
            text=transcription.text,
            words=transcription.words,
            segments=[],  # FIX: hardcoded
        )


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L8730
class ListModelsResponse(BaseModel):
    data: list[Model]
    object: Literal["list"] = "list"


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L11146
class Model(BaseModel):
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


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10909
TimestampGranularities = list[Literal["segment", "word"]]


DEFAULT_TIMESTAMP_GRANULARITIES: TimestampGranularities = ["segment"]
TIMESTAMP_GRANULARITIES_COMBINATIONS: list[TimestampGranularities] = [
    [],  # should be treated as ["segment"]. https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-timestamp_granularities
    ["segment"],
    ["word"],
    ["word", "segment"],
    ["segment", "word"],  # same as ["word", "segment"] but order is different
]
