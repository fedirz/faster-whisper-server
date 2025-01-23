from __future__ import annotations

import logging
import platform
import uvicorn

from fastapi import (
    FastAPI,
)
from fastapi.middleware.cors import CORSMiddleware

from speaches.dependencies import ApiKeyDependency, get_config
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
from speaches.routers.vad import (
    router as vad_router,
)
from speaches.routers.diarization import (
    router as diarization_router,
)
from contextlib import asynccontextmanager
from speaches.config import CONFIG
import torch
from pyannote.audio import Pipeline

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Starting up')
    global diarization_pipeline
    diarization_pipeline = Pipeline.from_pretrained(
        CONFIG.diarization.model_name,
        use_auth_token=CONFIG.diarization.auth_token,
        
    ).to(torch.device(CONFIG.diarization.device))
    yield
    print('Shutting down')
    


def create_app() -> FastAPI:
    config = get_config()  # HACK
    setup_logger(config.log_level)
    logger = logging.getLogger(__name__)

    logger.debug(f"Config: {config}")

    if platform.machine() == "x86_64":
        logger.warning("`POST /v1/audio/speech` with `model=rhasspy/piper-voices` is only supported on x86_64 machines")

    dependencies = []
    if config.api_key is not None:
        dependencies.append(ApiKeyDependency)

    app = FastAPI(
        dependencies=dependencies, 
        openapi_tags=TAGS_METADATA, 
        lifespan=lifespan
    )
    
    routers = [
        stt_router, models_router, misc_router, speech_router, vad_router, diarization_router
    ]
    for router in routers:
        app.include_router(router)

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


if __name__ == "__main__":
    uvicorn.run(
        app=create_app,
        host="127.0.0.1",
        port=4000,
        factory=True,
    )