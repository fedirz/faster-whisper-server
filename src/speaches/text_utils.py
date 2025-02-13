from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable

    from speaches.api_types import TranscriptionSegment


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


# TODO: Add tests
# TODO: take into account various sentence endings like "..."
# TODO: maybe create MultiSentenceChunker to return multiple sentence (when available) at a time
# TODO: consider different handling of small sentences. i.e. if a sentence consist of only couple of words wait until more words are available
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
