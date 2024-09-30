from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from faster_whisper_server.config import Config
from faster_whisper_server.model_manager import ModelManager


@lru_cache
def get_config() -> Config:
    return Config()


ConfigDependency = Annotated[Config, Depends(get_config)]


@lru_cache
def get_model_manager() -> ModelManager:
    config = get_config()  # HACK
    return ModelManager(config.whisper)


ModelManagerDependency = Annotated[ModelManager, Depends(get_model_manager)]
