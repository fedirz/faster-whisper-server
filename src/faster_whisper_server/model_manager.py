from __future__ import annotations

from collections import OrderedDict
import gc
import logging
import time
from typing import TYPE_CHECKING

from faster_whisper import WhisperModel

if TYPE_CHECKING:
    from faster_whisper_server.config import (
        Config,
    )

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.loaded_models: OrderedDict[str, WhisperModel] = OrderedDict()

    def load_model(self, model_name: str) -> WhisperModel:
        if model_name in self.loaded_models:
            logger.debug(f"{model_name} model already loaded")
            return self.loaded_models[model_name]
        if len(self.loaded_models) >= self.config.max_models:
            oldest_model_name = next(iter(self.loaded_models))
            logger.info(
                f"Max models ({self.config.max_models}) reached. Unloading the oldest model: {oldest_model_name}"
            )
            del self.loaded_models[oldest_model_name]
            gc.collect()
        logger.debug(f"Loading {model_name}...")
        start = time.perf_counter()
        # NOTE: will raise an exception if the model name isn't valid. Should I do an explicit check?
        whisper = WhisperModel(
            model_name,
            device=self.config.whisper.inference_device,
            device_index=self.config.whisper.device_index,
            compute_type=self.config.whisper.compute_type,
            cpu_threads=self.config.whisper.cpu_threads,
            num_workers=self.config.whisper.num_workers,
        )
        logger.info(
            f"Loaded {model_name} loaded in {time.perf_counter() - start:.2f} seconds. {self.config.whisper.inference_device}({self.config.whisper.compute_type}) will be used for inference."  # noqa: E501
        )
        self.loaded_models[model_name] = whisper
        return whisper
