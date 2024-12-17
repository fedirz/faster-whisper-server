from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from httpx import ASGITransport, AsyncClient
from openai import AsyncOpenAI
from openai.resources.audio import AsyncSpeech, AsyncTranscriptions
from openai.resources.chat.completions import AsyncCompletions

from faster_whisper_server.config import Config
from faster_whisper_server.model_manager import PiperModelManager, WhisperModelManager


@lru_cache
def get_config() -> Config:
    return Config()


ConfigDependency = Annotated[Config, Depends(get_config)]


@lru_cache
def get_model_manager() -> WhisperModelManager:
    config = get_config()  # HACK
    return WhisperModelManager(config.whisper)


ModelManagerDependency = Annotated[WhisperModelManager, Depends(get_model_manager)]


@lru_cache
def get_piper_model_manager() -> PiperModelManager:
    config = get_config()  # HACK
    return PiperModelManager(config.whisper.ttl)  # HACK


PiperModelManagerDependency = Annotated[PiperModelManager, Depends(get_piper_model_manager)]


security = HTTPBearer()


async def verify_api_key(
    config: ConfigDependency, credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> None:
    if credentials.credentials != config.api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)


ApiKeyDependency = Depends(verify_api_key)


@lru_cache
def get_completion_client() -> AsyncCompletions:
    config = get_config()  # HACK
    oai_client = AsyncOpenAI(base_url=config.chat_completion_base_url, api_key=config.chat_completion_api_key)
    return oai_client.chat.completions


CompletionClientDependency = Annotated[AsyncCompletions, Depends(get_completion_client)]


@lru_cache
def get_speech_client() -> AsyncSpeech:
    config = get_config()  # HACK
    if config.speech_base_url is None:
        # this might not work as expected if the `speech_router` won't have shared state with the main FastAPI `app`. TODO: verify  # noqa: E501
        from faster_whisper_server.routers.speech import (
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
        # this might not work as expected if the `transcription_router` won't have shared state with the main FastAPI `app`. TODO: verify  # noqa: E501
        from faster_whisper_server.routers.stt import (
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
