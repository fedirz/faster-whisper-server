from __future__ import annotations

from faster_whisper.transcribe import Segment, TranscriptionInfo, Word
from pydantic import BaseModel

from speaches import utils
from speaches.core import Transcription


# https://platform.openai.com/docs/api-reference/audio/json-object
class TranscriptionJsonResponse(BaseModel):
    text: str

    @classmethod
    def from_segments(cls, segments: list[Segment]) -> TranscriptionJsonResponse:
        return cls(text=utils.segments_text(segments))

    @classmethod
    def from_transcription(
        cls, transcription: Transcription
    ) -> TranscriptionJsonResponse:
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
    def from_segment(
        cls, segment: Segment, transcription_info: TranscriptionInfo
    ) -> TranscriptionVerboseJsonResponse:
        return cls(
            language=transcription_info.language,
            duration=segment.end - segment.start,
            text=segment.text,
            words=(
                [WordObject.from_word(word) for word in segment.words]
                if isinstance(segment.words, list)
                else []
            ),
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
            words=[
                WordObject.from_word(word)
                for word in utils.words_from_segments(segments)
            ],
        )

    @classmethod
    def from_transcription(
        cls, transcription: Transcription
    ) -> TranscriptionVerboseJsonResponse:
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
