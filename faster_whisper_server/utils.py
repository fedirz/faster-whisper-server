from faster_whisper.transcribe import Segment, Word


def segments_text(segments: list[Segment]) -> str:
    return "".join(segment.text for segment in segments).strip()


def words_from_segments(segments: list[Segment]) -> list[Word]:
    words = []
    for segment in segments:
        if segment.words is None:
            continue
        words.extend(segment.words)
    return words
