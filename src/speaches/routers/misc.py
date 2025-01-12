from __future__ import annotations

from fastapi import (
    APIRouter,
    Response,
)
import huggingface_hub
from huggingface_hub.hf_api import RepositoryNotFoundError

from speaches import hf_utils
from speaches.dependencies import ModelManagerDependency  # noqa: TC001

router = APIRouter()


@router.get("/health", tags=["diagnostic"])
def health() -> Response:
    return Response(status_code=200, content="OK")


@router.post(
    "/api/pull/{model_id:path}",
    tags=["experimental"],
    summary="Download a model from Hugging Face if it doesn't exist locally.",
)
def pull_model(model_id: str) -> Response:
    if hf_utils.does_local_model_exist(model_id):
        return Response(status_code=200, content=f"Model {model_id} already exists")
    try:
        huggingface_hub.snapshot_download(model_id, repo_type="model")
    except RepositoryNotFoundError as e:
        return Response(status_code=404, content=str(e))
    return Response(status_code=201, content=f"Model {model_id} downloaded")


@router.get("/api/ps", tags=["experimental"], summary="Get a list of loaded models.")
def get_running_models(
    model_manager: ModelManagerDependency,
) -> dict[str, list[str]]:
    return {"models": list(model_manager.loaded_models.keys())}


@router.post("/api/ps/{model_id:path}", tags=["experimental"], summary="Load a model into memory.")
def load_model_route(model_manager: ModelManagerDependency, model_id: str) -> Response:
    if model_id in model_manager.loaded_models:
        return Response(status_code=409, content="Model already loaded")
    with model_manager.load_model(model_id):
        pass
    return Response(status_code=201)


@router.delete("/api/ps/{model_id:path}", tags=["experimental"], summary="Unload a model from memory.")
def stop_running_model(model_manager: ModelManagerDependency, model_id: str) -> Response:
    try:
        model_manager.unload_model(model_id)
        return Response(status_code=204)
    except (KeyError, ValueError) as e:
        match e:
            case KeyError():
                return Response(status_code=404, content="Model not found")
            case ValueError():
                return Response(status_code=409, content=str(e))
