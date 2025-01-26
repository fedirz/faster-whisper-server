from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable

    from speaches.api_types import TranscriptionSegment, TranscriptionWord


class Transcription:
    def __init__(self, words: list[TranscriptionWord] = []) -> None:
        self.words: list[TranscriptionWord] = []
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

    def extend(self, words: list[TranscriptionWord]) -> None:
        self._ensure_no_word_overlap(words)
        self.words.extend(words)

    def _ensure_no_word_overlap(self, words: list[TranscriptionWord]) -> None:
        from speaches.dependencies import get_config  # HACK: avoid circular import

        config = get_config()  # HACK
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


def to_full_sentences(words: list[TranscriptionWord]) -> list[list[TranscriptionWord]]:
    sentences: list[list[TranscriptionWord]] = [[]]
    for word in words:
        sentences[-1].append(word)
        if is_eos(word.word):
            sentences.append([])
    if len(sentences[-1]) == 0 or not is_eos(sentences[-1][-1].word):
        sentences.pop()
    return sentences


def word_to_text(words: list[TranscriptionWord]) -> str:
    return "".join(word.word for word in words)


def words_to_text_w_ts(words: list[TranscriptionWord]) -> str:
    return "".join(f"{word.word}({word.start:.2f}-{word.end:.2f})" for word in words)


def segments_to_text(segments: Iterable[TranscriptionSegment]) -> str:
    return "".join(segment.text for segment in segments).strip()


def srt_format_timestamp(ts: float) -> str:
    hours = ts // 3600
    minutes = (ts % 3600) // 60
    seconds = ts % 60
    milliseconds = (ts * 1000) % 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def vtt_format_timestamp(ts: float) -> str:
    hours = ts // 3600
    minutes = (ts % 3600) // 60
    seconds = ts % 60
    milliseconds = (ts * 1000) % 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"


def segments_to_vtt(segment: TranscriptionSegment, i: int) -> str:
    start = segment.start if i > 0 else 0.0
    result = f"{vtt_format_timestamp(start)} --> {vtt_format_timestamp(segment.end)}\n{segment.text}\n\n"

    if i == 0:
        return f"WEBVTT\n\n{result}"
    else:
        return result


def segments_to_srt(segment: TranscriptionSegment, i: int) -> str:
    return f"{i + 1}\n{srt_format_timestamp(segment.start)} --> {srt_format_timestamp(segment.end)}\n{segment.text}\n\n"


def canonicalize_word(text: str) -> str:
    text = text.lower()
    # Remove non-alphabetic characters using regular expression
    text = re.sub(r"[^a-z]", "", text)
    return text.lower().strip().strip(".,?!")


def common_prefix(a: list[TranscriptionWord], b: list[TranscriptionWord]) -> list[TranscriptionWord]:
    i = 0
    while i < len(a) and i < len(b) and canonicalize_word(a[i].word) == canonicalize_word(b[i].word):
        i += 1
    return a[:i]


# TODO: Add tests
# TODO: take into account various sentence endings like "..."
# TODO: maybe create MultiSentenceChunker to return multiple sentence (when available) at a time
# TODO: consider different handling of small sentences. i.e. if a sentence consist of only couple of words wait until more words are available  # noqa: E501
class SentenceChunker:
    def __init__(self) -> None:
        self._content = ""
        self._is_closed = False
        self._new_token_event = asyncio.Event()
        self._sentence_endings = {".", "!", "?"}
        self._processed_index = 0

    def add_token(self, token: str) -> None:
        """Add a token (text chunk) to the chunker."""
        if self._is_closed:
            raise RuntimeError("Cannot add tokens to a closed SentenceChunker")  # noqa: EM101

        self._content += token
        self._new_token_event.set()

    def close(self) -> None:
        """Close the chunker, preventing further token additions."""
        self._is_closed = True
        self._new_token_event.set()

    async def __aiter__(self) -> AsyncGenerator[str]:
        while True:
            # Find the next sentence ending after the last processed index
            next_end = -1
            for ending in self._sentence_endings:
                pos = self._content.find(ending, self._processed_index)
                if pos != -1 and (next_end == -1 or pos < next_end):
                    next_end = pos

            if next_end != -1:
                # We found a complete sentence
                sentence_end = next_end + 1
                sentence = self._content[self._processed_index : sentence_end]
                self._processed_index = sentence_end
                yield sentence
            else:
                # No complete sentence found
                if self._is_closed:
                    # If there's any remaining content, yield it
                    if self._processed_index < len(self._content):
                        yield self._content[self._processed_index :]
                    return

                # Wait for more content
                self._new_token_event.clear()
                await self._new_token_event.wait()
