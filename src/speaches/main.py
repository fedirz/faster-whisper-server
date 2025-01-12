from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import platform
from typing import TYPE_CHECKING

from fastapi import (
    FastAPI,
)
from fastapi.middleware.cors import CORSMiddleware

from speaches.dependencies import ApiKeyDependency, get_config, get_model_manager
from speaches.logger import setup_logger
from speaches.routers.misc import (
    router as misc_router,
)
from speaches.routers.models import (
    router as models_router,
)
from speaches.routers.speech import (
    router as speech_router,
)
from speaches.routers.stt import (
    router as stt_router,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# https://swagger.io/docs/specification/v3_0/grouping-operations-with-tags/
# https://fastapi.tiangolo.com/tutorial/metadata/#metadata-for-tags
TAGS_METADATA = [
    {"name": "automatic-speech-recognition"},
    {"name": "speech-to-text"},
    {"name": "models"},
    {"name": "diagnostic"},
    {
        "name": "experimental",
        "description": "Not meant for public use yet. May change or be removed at any time.",
    },
]


def create_app() -> FastAPI:
    config = get_config()  # HACK
    setup_logger(config.log_level)
    logger = logging.getLogger(__name__)

    logger.debug(f"Config: {config}")

    if platform.machine() == "x86_64":
        logger.warning("`POST /v1/audio/speech` with `model=rhasspy/piper-voices` is only supported on x86_64 machines")

    model_manager = get_model_manager()  # HACK

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        for model_name in config.preload_models:
            model_manager.load_model(model_name)
        yield

    dependencies = []
    if config.api_key is not None:
        dependencies.append(ApiKeyDependency)

    app = FastAPI(lifespan=lifespan, dependencies=dependencies, openapi_tags=TAGS_METADATA)

    app.include_router(stt_router)
    app.include_router(models_router)
    app.include_router(misc_router)
    app.include_router(speech_router)

    if config.allow_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if config.enable_ui:
        import gradio as gr

        from speaches.gradio_app import create_gradio_demo

        app = gr.mount_gradio_app(app, create_gradio_demo(config), path="/")

    return app
