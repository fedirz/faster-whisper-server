from faster_whisper_server.api_models import TranscriptionWord
from faster_whisper_server.text_utils import (
    canonicalize_word,
    common_prefix,
    is_eos,
    srt_format_timestamp,
    to_full_sentences,
    vtt_format_timestamp,
)


def test_is_eos() -> None:
    assert not is_eos("Hello")
    assert not is_eos("Hello...")
    assert is_eos("Hello.")
    assert is_eos("Hello!")
    assert is_eos("Hello?")
    assert not is_eos("Hello. Yo")
    assert not is_eos("Hello. Yo...")
    assert is_eos("Hello. Yo.")


def tests_to_full_sentences() -> None:
    def word(text: str) -> TranscriptionWord:
        return TranscriptionWord(word=text, start=0.0, end=0.0, probability=0.0)

    assert to_full_sentences([]) == []
    assert to_full_sentences([word(text="Hello")]) == []
    assert to_full_sentences([word(text="Hello..."), word(" world")]) == []
    assert to_full_sentences([word(text="Hello..."), word(" world.")]) == [[word("Hello..."), word(" world.")]]
    assert to_full_sentences([word(text="Hello..."), word(" world."), word(" How")]) == [
        [word("Hello..."), word(" world.")],
    ]


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


def test_canonicalize_word() -> None:
    assert canonicalize_word("ABC") == "abc"
    assert canonicalize_word("...ABC?") == "abc"
    assert canonicalize_word("... AbC  ...") == "abc"


def test_common_prefix() -> None:
    def word(text: str) -> TranscriptionWord:
        return TranscriptionWord(word=text, start=0.0, end=0.0, probability=0.0)

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
    def word(text: str) -> TranscriptionWord:
        return TranscriptionWord(word=text, start=0.0, end=0.0, probability=0.0)

    a = [word("A...")]
    b = [word("a?"), word("b"), word("c")]
    assert common_prefix(a, b) == [word("A...")]

    a = [word("A..."), word("B?"), word("C,")]
    b = [word("a??"), word("  b"), word(" ,c")]
    assert common_prefix(a, b) == [word("A..."), word("B?"), word("C,")]
