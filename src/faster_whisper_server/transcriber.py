from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from faster_whisper_server.audio import Audio, AudioStream
from faster_whisper_server.text_utils import Transcription, common_prefix, to_full_sentences, word_to_text

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from faster_whisper_server.api_models import TranscriptionWord
    from faster_whisper_server.asr import FasterWhisperASR

logger = logging.getLogger(__name__)


class LocalAgreement:
    def __init__(self) -> None:
        self.unconfirmed = Transcription()

    def merge(self, confirmed: Transcription, incoming: Transcription) -> list[TranscriptionWord]:
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


# TODO: needs a better name
def needs_audio_after(confirmed: Transcription) -> float:
    full_sentences = to_full_sentences(confirmed.words)
    return full_sentences[-1][-1].end if len(full_sentences) > 0 else 0.0


def prompt(confirmed: Transcription) -> str | None:
    sentences = to_full_sentences(confirmed.words)
    return word_to_text(sentences[-1]) if len(sentences) > 0 else None


async def audio_transcriber(
    asr: FasterWhisperASR,
    audio_stream: AudioStream,
    min_duration: float,
) -> AsyncGenerator[Transcription, None]:
    local_agreement = LocalAgreement()
    full_audio = Audio()
    confirmed = Transcription()
    import time
    last_chunk_time = time.time()
    async for chunk in audio_stream.chunks(min_duration):
        last_chunk_time = time.time()
        full_audio.extend(chunk)
        audio = full_audio.after(needs_audio_after(confirmed))
        transcription, _ = await asr.transcribe(audio, prompt(confirmed))
        new_words = local_agreement.merge(confirmed, transcription)
        if len(new_words) > 0:
            confirmed.extend(new_words)
            yield confirmed
        else:
            logger.debug("No new words")
    logger.debug("Flushing...")
    confirmed.extend(local_agreement.unconfirmed.words)
    yield confirmed
    logger.info("Audio transcriber finished")
    logger.warning(f"Last chunk received at {last_chunk_time}, it has been {time.time() - last_chunk_time} seconds since then")
