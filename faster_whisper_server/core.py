# TODO: rename module
from __future__ import annotations

from dataclasses import dataclass
import re

from faster_whisper_server.config import config


# TODO: use the `Segment` from `faster-whisper.transcribe` instead
@dataclass
class Segment:
    text: str
    start: float = 0.0
    end: float = 0.0

    @property
    def is_eos(self) -> bool:
        if self.text.endswith("..."):
            return False
        return any(self.text.endswith(punctuation_symbol) for punctuation_symbol in ".?!")

    def offset(self, seconds: float) -> None:
        self.start += seconds
        self.end += seconds


# TODO: use the `Word` from `faster-whisper.transcribe` instead
@dataclass
class Word(Segment):
    probability: float = 0.0

    @classmethod
    def common_prefix(cls, a: list[Word], b: list[Word]) -> list[Word]:
        i = 0
        while i < len(a) and i < len(b) and canonicalize_word(a[i].text) == canonicalize_word(b[i].text):
            i += 1
        return a[:i]


class Transcription:
    def __init__(self, words: list[Word] = []) -> None:
        self.words: list[Word] = []
        self.extend(words)

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words).strip()

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


def test_segment_is_eos() -> None:
    assert not Segment("Hello").is_eos
    assert not Segment("Hello...").is_eos
    assert Segment("Hello.").is_eos
    assert Segment("Hello!").is_eos
    assert Segment("Hello?").is_eos
    assert not Segment("Hello. Yo").is_eos
    assert not Segment("Hello. Yo...").is_eos
    assert Segment("Hello. Yo.").is_eos


def to_full_sentences(words: list[Word]) -> list[Segment]:
    sentences: list[Segment] = [Segment("")]
    for word in words:
        sentences[-1] = Segment(
            start=sentences[-1].start,
            end=word.end,
            text=sentences[-1].text + word.text,
        )
        if word.is_eos:
            sentences.append(Segment(""))
    if len(sentences) > 0 and not sentences[-1].is_eos:
        sentences.pop()
    return sentences


def tests_to_full_sentences() -> None:
    assert to_full_sentences([]) == []
    assert to_full_sentences([Word(text="Hello")]) == []
    assert to_full_sentences([Word(text="Hello..."), Word(" world")]) == []
    assert to_full_sentences([Word(text="Hello..."), Word(" world.")]) == [Segment(text="Hello... world.")]
    assert to_full_sentences([Word(text="Hello..."), Word(" world."), Word(" How")]) == [
        Segment(text="Hello... world.")
    ]


def to_text(words: list[Word]) -> str:
    return "".join(word.text for word in words)


def to_text_w_ts(words: list[Word]) -> str:
    return "".join(f"{word.text}({word.start:.2f}-{word.end:.2f})" for word in words)


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
    while i < len(a) and i < len(b) and canonicalize_word(a[i].text) == canonicalize_word(b[i].text):
        i += 1
    return a[:i]


def test_common_prefix() -> None:
    def word(text: str) -> Word:
        return Word(text=text, start=0.0, end=0.0, probability=0.0)

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
        return Word(text=text, start=0.0, end=0.0, probability=0.0)

    a = [word("A...")]
    b = [word("a?"), word("b"), word("c")]
    assert common_prefix(a, b) == [word("A...")]

    a = [word("A..."), word("B?"), word("C,")]
    b = [word("a??"), word("  b"), word(" ,c")]
    assert common_prefix(a, b) == [word("A..."), word("B?"), word("C,")]
