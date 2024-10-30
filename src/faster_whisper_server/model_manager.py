from __future__ import annotations

from collections import OrderedDict
import gc
import logging
import threading
import time
from typing import TYPE_CHECKING

import os

from faster_whisper import WhisperModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from faster_whisper_server.config import (
        WhisperConfig,
    )

logger = logging.getLogger(__name__)

# TODO: enable concurrent model downloads


class SelfDisposingWhisperModel:
    def __init__(
        self,
        model_id: str,
        whisper_config: WhisperConfig,
        *,
        on_unload: Callable[[str], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self.whisper_config = whisper_config
        self.on_unload = on_unload

        self.ref_count: int = 0
        self.rlock = threading.RLock()
        self.expire_timer: threading.Timer | None = None
        self.whisper: WhisperModel | None = None

    def unload(self) -> None:
        with self.rlock:
            if self.whisper is None:
                raise ValueError(f"Model {self.model_id} is not loaded. {self.ref_count=}")
            if self.ref_count > 0:
                raise ValueError(f"Model {self.model_id} is still in use. {self.ref_count=}")
            if self.expire_timer:
                self.expire_timer.cancel()
            self.whisper = None
            # WARN: ~300 MB of memory will still be held by the model. See https://github.com/SYSTRAN/faster-whisper/issues/992
            gc.collect()
            logger.info(f"Model {self.model_id} unloaded")
            if self.on_unload is not None:
                self.on_unload(self.model_id)

    def _load(self) -> None:
        with self.rlock:
            assert self.whisper is None
            start = time.perf_counter()
            if self.whisper_config.offline_models_root:
                model_size_or_path = os.path.join(self.whisper_config.offline_models_root, self.model_id)
            else:
                model_size_or_path = self.model_id
            logger.info(f"Loading model from {model_size_or_path}")
            self.whisper = WhisperModel(
                model_size_or_path=model_size_or_path,
                device=self.whisper_config.inference_device,
                device_index=self.whisper_config.device_index,
                compute_type=self.whisper_config.compute_type,
                cpu_threads=self.whisper_config.cpu_threads,
                num_workers=self.whisper_config.num_workers,
                download_root=self.whisper_config.download_root,
            )
            logger.info(f"Model {self.model_id} loaded in {time.perf_counter() - start:.2f}s")

    def _increment_ref(self) -> None:
        with self.rlock:
            self.ref_count += 1
            if self.expire_timer:
                logger.debug(f"Model was set to expire in {self.expire_timer.interval}s, cancelling")
                self.expire_timer.cancel()
            logger.debug(f"Incremented ref count for {self.model_id}, {self.ref_count=}")

    def _decrement_ref(self) -> None:
        with self.rlock:
            self.ref_count -= 1
            logger.debug(f"Decremented ref count for {self.model_id}, {self.ref_count=}")
            if self.ref_count <= 0:
                if self.whisper_config.ttl > 0:
                    logger.info(f"Model {self.model_id} is idle, scheduling offload in {self.whisper_config.ttl}s")
                    self.expire_timer = threading.Timer(self.whisper_config.ttl, self.unload)
                    self.expire_timer.start()
                elif self.whisper_config.ttl == 0:
                    logger.info(f"Model {self.model_id} is idle, unloading immediately")
                    self.unload()
                else:
                    logger.info(f"Model {self.model_id} is idle, not unloading")

    def __enter__(self) -> WhisperModel:
        with self.rlock:
            if self.whisper is None:
                self._load()
            self._increment_ref()
            assert self.whisper is not None
            return self.whisper

    def __exit__(self, *_args) -> None:  # noqa: ANN002
        self._decrement_ref()


class ModelManager:
    def __init__(self, whisper_config: WhisperConfig) -> None:
        self.whisper_config = whisper_config
        self.loaded_models: OrderedDict[str, SelfDisposingWhisperModel] = OrderedDict()
        self._lock = threading.Lock()

    def _handle_model_unload(self, model_name: str) -> None:
        with self._lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]

    def unload_model(self, model_name: str) -> None:
        with self._lock:
            model = self.loaded_models.get(model_name)
            if model is None:
                raise KeyError(f"Model {model_name} not found")
            self.loaded_models[model_name].unload()

    def load_model(self, model_name: str) -> SelfDisposingWhisperModel:
        with self._lock:
            if model_name in self.loaded_models:
                logger.debug(f"{model_name} model already loaded")
                return self.loaded_models[model_name]
            self.loaded_models[model_name] = SelfDisposingWhisperModel(
                model_name,
                self.whisper_config,
                on_unload=self._handle_model_unload,
            )
            return self.loaded_models[model_name]
