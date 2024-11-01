from functools import lru_cache
from typing import Annotated

from fastapi import Depends

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
