from __future__ import annotations

from typing import TYPE_CHECKING

from faster_whisper_server.audio import Audio, AudioStream
from faster_whisper_server.config import config
from faster_whisper_server.core import (
    Transcription,
    Word,
    common_prefix,
    to_full_sentences,
)
from faster_whisper_server.logger import logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from faster_whisper_server.asr import FasterWhisperASR


class LocalAgreement:
    def __init__(self) -> None:
        self.unconfirmed = Transcription()

    def merge(self, confirmed: Transcription, incoming: Transcription) -> list[Word]:
        # https://github.com/ufal/whisper_streaming/blob/main/whisper_online.py#L264
        incoming = incoming.after(confirmed.end - 0.1)
        prefix = common_prefix(incoming.words, self.unconfirmed.words)
        logger.debug(f"Confirmed: {confirmed.text}")
        logger.debug(f"Unconfirmed: {self.unconfirmed.text}")
        logger.debug(f"Incoming: {incoming.text}")

        if len(incoming.words) > len(prefix):
            self.unconfirmed = Transcription(incoming.words[len(prefix) :])
        else:
            self.unconfirmed = Transcription()

        return prefix

    @classmethod
    def prompt(cls, confirmed: Transcription) -> str | None:
        sentences = to_full_sentences(confirmed.words)
        if len(sentences) == 0:
            return None
        return sentences[-1].text

    # TODO: better name
    @classmethod
    def needs_audio_after(cls, confirmed: Transcription) -> float:
        full_sentences = to_full_sentences(confirmed.words)
        return full_sentences[-1].end if len(full_sentences) > 0 else 0.0


def needs_audio_after(confirmed: Transcription) -> float:
    full_sentences = to_full_sentences(confirmed.words)
    return full_sentences[-1].end if len(full_sentences) > 0 else 0.0


def prompt(confirmed: Transcription) -> str | None:
    sentences = to_full_sentences(confirmed.words)
    if len(sentences) == 0:
        return None
    return sentences[-1].text


async def audio_transcriber(
    asr: FasterWhisperASR,
    audio_stream: AudioStream,
) -> AsyncGenerator[Transcription, None]:
    local_agreement = LocalAgreement()
    full_audio = Audio()
    confirmed = Transcription()
    async for chunk in audio_stream.chunks(config.min_duration):
        full_audio.extend(chunk)
        audio = full_audio.after(needs_audio_after(confirmed))
        transcription, _ = await asr.transcribe(audio, prompt(confirmed))
        new_words = local_agreement.merge(confirmed, transcription)
        if len(new_words) > 0:
            confirmed.extend(new_words)
            yield confirmed
    logger.debug("Flushing...")
    confirmed.extend(local_agreement.unconfirmed.words)
    yield confirmed
    logger.info("Audio transcriber finished")
