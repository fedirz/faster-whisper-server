import enum

from pydantic import BaseModel

from speaches.core import Word


class ResponseFormat(enum.StrEnum):
    JSON = "json"
    TEXT = "text"
    VERBOSE_JSON = "verbose_json"


# https://platform.openai.com/docs/api-reference/audio/json-object
class TranscriptionResponse(BaseModel):
    text: str


# Subset of https://platform.openai.com/docs/api-reference/audio/verbose-json-object
class TranscriptionVerboseResponse(BaseModel):
    task: str = "transcribe"
    duration: float
    text: str
    words: list[
        Word
    ]  # Different from OpenAI's `words`. `Word.text` instead of `Word.word`
