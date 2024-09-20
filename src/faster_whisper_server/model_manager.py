from __future__ import annotations

from collections import OrderedDict
import gc
import time

from faster_whisper import WhisperModel

from faster_whisper_server.config import (
    config,
)
from faster_whisper_server.logger import logger


class ModelManager:
    def __init__(self) -> None:
        self.loaded_models: OrderedDict[str, WhisperModel] = OrderedDict()

    def load_model(self, model_name: str) -> WhisperModel:
        if model_name in self.loaded_models:
            logger.debug(f"{model_name} model already loaded")
            return self.loaded_models[model_name]
        if len(self.loaded_models) >= config.max_models:
            oldest_model_name = next(iter(self.loaded_models))
            logger.info(f"Max models ({config.max_models}) reached. Unloading the oldest model: {oldest_model_name}")
            del self.loaded_models[oldest_model_name]
            gc.collect()
        logger.debug(f"Loading {model_name}...")
        start = time.perf_counter()
        # NOTE: will raise an exception if the model name isn't valid. Should I do an explicit check?
        whisper = WhisperModel(
            model_name,
            device=config.whisper.inference_device,
            device_index=config.whisper.device_index,
            compute_type=config.whisper.compute_type,
            cpu_threads=config.whisper.cpu_threads,
            num_workers=config.whisper.num_workers,
        )
        logger.info(
            f"Loaded {model_name} loaded in {time.perf_counter() - start:.2f} seconds. {config.whisper.inference_device}({config.whisper.compute_type}) will be used for inference."  # noqa: E501
        )
        self.loaded_models[model_name] = whisper
        return whisper


model_manager = ModelManager()
