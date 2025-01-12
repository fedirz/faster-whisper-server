from functools import lru_cache
import logging
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from httpx import ASGITransport, AsyncClient
from openai import AsyncOpenAI
from openai.resources.audio import AsyncSpeech, AsyncTranscriptions
from openai.resources.chat.completions import AsyncCompletions

from speaches.config import Config
from speaches.model_manager import KokoroModelManager, PiperModelManager, WhisperModelManager

logger = logging.getLogger(__name__)

# NOTE: `get_config` is called directly instead of using sub-dependencies so that these functions could be used outside of `FastAPI`  # noqa: E501


# https://fastapi.tiangolo.com/advanced/settings/?h=setti#creating-the-settings-only-once-with-lru_cache
# WARN: Any new module that ends up calling this function directly (not through `FastAPI` dependency injection) should be patched in `tests/conftest.py`  # noqa: E501
@lru_cache
def get_config() -> Config:
    return Config()


ConfigDependency = Annotated[Config, Depends(get_config)]


@lru_cache
def get_model_manager() -> WhisperModelManager:
    config = get_config()
    return WhisperModelManager(config.whisper)


ModelManagerDependency = Annotated[WhisperModelManager, Depends(get_model_manager)]


@lru_cache
def get_piper_model_manager() -> PiperModelManager:
    config = get_config()
    return PiperModelManager(config.whisper.ttl)  # HACK: should have its own config


PiperModelManagerDependency = Annotated[PiperModelManager, Depends(get_piper_model_manager)]


@lru_cache
def get_kokoro_model_manager() -> KokoroModelManager:
    config = get_config()
    return KokoroModelManager(config.whisper.ttl)  # HACK: should have its own config


KokoroModelManagerDependency = Annotated[KokoroModelManager, Depends(get_kokoro_model_manager)]

security = HTTPBearer()


async def verify_api_key(
    config: ConfigDependency, credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> None:
    if credentials.credentials != config.api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)


ApiKeyDependency = Depends(verify_api_key)


@lru_cache
def get_completion_client() -> AsyncCompletions:
    config = get_config()
    oai_client = AsyncOpenAI(base_url=config.chat_completion_base_url, api_key=config.chat_completion_api_key)
    return oai_client.chat.completions


CompletionClientDependency = Annotated[AsyncCompletions, Depends(get_completion_client)]


@lru_cache
def get_speech_client() -> AsyncSpeech:
    config = get_config()
    if config.speech_base_url is None:
        # this might not work as expected if `speech_router` won't have shared state (access to the same `model_manager`) with the main FastAPI `app`. TODO: verify  # noqa: E501
        from speaches.routers.speech import (
            router as speech_router,
        )

        http_client = AsyncClient(
            transport=ASGITransport(speech_router), base_url="http://test/v1"
        )  # NOTE: "test" can be replaced with any other value
        oai_client = AsyncOpenAI(http_client=http_client, api_key=config.speech_api_key)
    else:
        oai_client = AsyncOpenAI(base_url=config.speech_base_url, api_key=config.speech_api_key)
    return oai_client.audio.speech


SpeechClientDependency = Annotated[AsyncSpeech, Depends(get_speech_client)]


@lru_cache
def get_transcription_client() -> AsyncTranscriptions:
    config = get_config()
    if config.transcription_base_url is None:
        # this might not work as expected if `transcription_router` won't have shared state (access to the same `model_manager`) with the main FastAPI `app`. TODO: verify  # noqa: E501
        from speaches.routers.stt import (
            router as stt_router,
        )

        http_client = AsyncClient(
            transport=ASGITransport(stt_router), base_url="http://test/v1"
        )  # NOTE: "test" can be replaced with any other value

        oai_client = AsyncOpenAI(http_client=http_client, api_key=config.transcription_api_key)
    else:
        oai_client = AsyncOpenAI(base_url=config.transcription_base_url, api_key=config.transcription_api_key)
    return oai_client.audio.transcriptions


TranscriptionClientDependency = Annotated[AsyncTranscriptions, Depends(get_transcription_client)]
