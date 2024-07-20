from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

from faster_whisper_server.config import config

if TYPE_CHECKING:
    from collections.abc import Iterable

    import faster_whisper.transcribe


class Word(BaseModel):
    start: float
    end: float
    word: str
    probability: float

    @classmethod
    def from_segments(cls, segments: Iterable[Segment]) -> list[Word]:
        words: list[Word] = []
        for segment in segments:
            assert segment.words is not None
            words.extend(segment.words)
        return words

    def offset(self, seconds: float) -> None:
        self.start += seconds
        self.end += seconds

    @classmethod
    def common_prefix(cls, a: list[Word], b: list[Word]) -> list[Word]:
        i = 0
        while i < len(a) and i < len(b) and canonicalize_word(a[i].word) == canonicalize_word(b[i].word):
            i += 1
        return a[:i]


class Segment(BaseModel):
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
    words: list[Word] | None

    @classmethod
    def from_faster_whisper_segments(cls, segments: Iterable[faster_whisper.transcribe.Segment]) -> Iterable[Segment]:
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
                    Word(
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


class Transcription:
    def __init__(self, words: list[Word] = []) -> None:
        self.words: list[Word] = []
        self.extend(words)

    @property
    def text(self) -> str:
        return " ".join(word.word for word in self.words).strip()

    @property
    def start(self) -> float:
        return self.words[0].start if len(self.words) > 0 else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if len(self.words) > 0 else 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def after(self, seconds: float) -> Transcription:
        return Transcription(words=[word for word in self.words if word.start > seconds])

    def extend(self, words: list[Word]) -> None:
        self._ensure_no_word_overlap(words)
        self.words.extend(words)

    def _ensure_no_word_overlap(self, words: list[Word]) -> None:
        if len(self.words) > 0 and len(words) > 0:
            if words[0].start + config.word_timestamp_error_margin <= self.words[-1].end:
                raise ValueError(
                    f"Words overlap: {self.words[-1]} and {words[0]}. Error margin: {config.word_timestamp_error_margin}"  # noqa: E501
                )
        for i in range(1, len(words)):
            if words[i].start + config.word_timestamp_error_margin <= words[i - 1].end:
                raise ValueError(f"Words overlap: {words[i - 1]} and {words[i]}. All words: {words}")


def is_eos(text: str) -> bool:
    if text.endswith("..."):
        return False
    return any(text.endswith(punctuation_symbol) for punctuation_symbol in ".?!")


def test_is_eos() -> None:
    assert not is_eos("Hello")
    assert not is_eos("Hello...")
    assert is_eos("Hello.")
    assert is_eos("Hello!")
    assert is_eos("Hello?")
    assert not is_eos("Hello. Yo")
    assert not is_eos("Hello. Yo...")
    assert is_eos("Hello. Yo.")


def to_full_sentences(words: list[Word]) -> list[list[Word]]:
    sentences: list[list[Word]] = [[]]
    for word in words:
        sentences[-1].append(word)
        if is_eos(word.word):
            sentences.append([])
    if len(sentences[-1]) == 0 or not is_eos(sentences[-1][-1].word):
        sentences.pop()
    return sentences


def tests_to_full_sentences() -> None:
    def word(text: str) -> Word:
        return Word(word=text, start=0.0, end=0.0, probability=0.0)

    assert to_full_sentences([]) == []
    assert to_full_sentences([word(text="Hello")]) == []
    assert to_full_sentences([word(text="Hello..."), word(" world")]) == []
    assert to_full_sentences([word(text="Hello..."), word(" world.")]) == [[word("Hello..."), word(" world.")]]
    assert to_full_sentences([word(text="Hello..."), word(" world."), word(" How")]) == [
        [word("Hello..."), word(" world.")],
    ]


def word_to_text(words: list[Word]) -> str:
    return "".join(word.word for word in words)


def words_to_text_w_ts(words: list[Word]) -> str:
    return "".join(f"{word.word}({word.start:.2f}-{word.end:.2f})" for word in words)


def segments_to_text(segments: Iterable[Segment]) -> str:
    return "".join(segment.text for segment in segments).strip()


def srt_format_timestamp(ts: float) -> str:
    hours = ts // 3600
    minutes = (ts % 3600) // 60
    seconds = ts % 60
    milliseconds = (ts * 1000) % 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def test_srt_format_timestamp() -> None:
    assert srt_format_timestamp(0.0) == "00:00:00,000"
    assert srt_format_timestamp(1.0) == "00:00:01,000"
    assert srt_format_timestamp(1.234) == "00:00:01,234"
    assert srt_format_timestamp(60.0) == "00:01:00,000"
    assert srt_format_timestamp(61.0) == "00:01:01,000"
    assert srt_format_timestamp(61.234) == "00:01:01,234"
    assert srt_format_timestamp(3600.0) == "01:00:00,000"
    assert srt_format_timestamp(3601.0) == "01:00:01,000"
    assert srt_format_timestamp(3601.234) == "01:00:01,234"
    assert srt_format_timestamp(23423.4234) == "06:30:23,423"


def vtt_format_timestamp(ts: float) -> str:
    hours = ts // 3600
    minutes = (ts % 3600) // 60
    seconds = ts % 60
    milliseconds = (ts * 1000) % 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"


def test_vtt_format_timestamp() -> None:
    assert vtt_format_timestamp(0.0) == "00:00:00.000"
    assert vtt_format_timestamp(1.0) == "00:00:01.000"
    assert vtt_format_timestamp(1.234) == "00:00:01.234"
    assert vtt_format_timestamp(60.0) == "00:01:00.000"
    assert vtt_format_timestamp(61.0) == "00:01:01.000"
    assert vtt_format_timestamp(61.234) == "00:01:01.234"
    assert vtt_format_timestamp(3600.0) == "01:00:00.000"
    assert vtt_format_timestamp(3601.0) == "01:00:01.000"
    assert vtt_format_timestamp(3601.234) == "01:00:01.234"
    assert vtt_format_timestamp(23423.4234) == "06:30:23.423"


def segments_to_vtt(segment: Segment, i: int) -> str:
    start = segment.start if i > 0 else 0.0
    result = f"{vtt_format_timestamp(start)} --> {vtt_format_timestamp(segment.end)}\n{segment.text}\n\n"

    if i == 0:
        return f"WEBVTT\n\n{result}"
    else:
        return result


def segments_to_srt(segment: Segment, i: int) -> str:
    return f"{i + 1}\n{srt_format_timestamp(segment.start)} --> {srt_format_timestamp(segment.end)}\n{segment.text}\n\n"


def canonicalize_word(text: str) -> str:
    text = text.lower()
    # Remove non-alphabetic characters using regular expression
    text = re.sub(r"[^a-z]", "", text)
    return text.lower().strip().strip(".,?!")


def test_canonicalize_word() -> None:
    assert canonicalize_word("ABC") == "abc"
    assert canonicalize_word("...ABC?") == "abc"
    assert canonicalize_word("... AbC  ...") == "abc"


def common_prefix(a: list[Word], b: list[Word]) -> list[Word]:
    i = 0
    while i < len(a) and i < len(b) and canonicalize_word(a[i].word) == canonicalize_word(b[i].word):
        i += 1
    return a[:i]


def test_common_prefix() -> None:
    def word(text: str) -> Word:
        return Word(word=text, start=0.0, end=0.0, probability=0.0)

    a = [word("a"), word("b"), word("c")]
    b = [word("a"), word("b"), word("c")]
    assert common_prefix(a, b) == [word("a"), word("b"), word("c")]

    a = [word("a"), word("b"), word("c")]
    b = [word("a"), word("b"), word("d")]
    assert common_prefix(a, b) == [word("a"), word("b")]

    a = [word("a"), word("b"), word("c")]
    b = [word("a")]
    assert common_prefix(a, b) == [word("a")]

    a = [word("a")]
    b = [word("a"), word("b"), word("c")]
    assert common_prefix(a, b) == [word("a")]

    a = [word("a")]
    b = []
    assert common_prefix(a, b) == []

    a = []
    b = [word("a")]
    assert common_prefix(a, b) == []

    a = [word("a"), word("b"), word("c")]
    b = [word("b"), word("c")]
    assert common_prefix(a, b) == []


def test_common_prefix_and_canonicalization() -> None:
    def word(text: str) -> Word:
        return Word(word=text, start=0.0, end=0.0, probability=0.0)

    a = [word("A...")]
    b = [word("a?"), word("b"), word("c")]
    assert common_prefix(a, b) == [word("A...")]

    a = [word("A..."), word("B?"), word("C,")]
    b = [word("a??"), word("  b"), word(" ,c")]
    assert common_prefix(a, b) == [word("A..."), word("B?"), word("C,")]
