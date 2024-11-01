from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import (
    APIRouter,
    HTTPException,
    Path,
)
import huggingface_hub

from faster_whisper_server.api_models import (
    ListModelsResponse,
    Model,
)
from faster_whisper_server.hf_utils import list_whisper_models

if TYPE_CHECKING:
    from huggingface_hub.hf_api import ModelInfo

router = APIRouter()


@router.get("/v1/models")
def get_models() -> ListModelsResponse:
    whisper_models = list(list_whisper_models())
    return ListModelsResponse(data=whisper_models)


@router.get("/v1/models/{model_name:path}")
def get_model(
    # NOTE: `examples` doesn't work https://github.com/tiangolo/fastapi/discussions/10537
    model_name: Annotated[str, Path(example="Systran/faster-distil-whisper-large-v3")],
) -> Model:
    models = huggingface_hub.list_models(
        model_name=model_name, library="ctranslate2", tags="automatic-speech-recognition", cardData=True
    )
    models = list(models)
    models.sort(key=lambda model: model.downloads or -1, reverse=True)
    if len(models) == 0:
        raise HTTPException(status_code=404, detail="Model doesn't exists")
    exact_match: ModelInfo | None = None
    for model in models:
        if model.id == model_name:
            exact_match = model
            break
    if exact_match is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model doesn't exists. Possible matches: {', '.join([model.id for model in models])}",
        )
    assert exact_match.created_at is not None
    assert exact_match.card_data is not None
    assert exact_match.card_data.language is None or isinstance(exact_match.card_data.language, str | list)
    if exact_match.card_data.language is None:
        language = []
    elif isinstance(exact_match.card_data.language, str):
        language = [exact_match.card_data.language]
    else:
        language = exact_match.card_data.language
    return Model(
        id=exact_match.id,
        created=int(exact_match.created_at.timestamp()),
        object_="model",
        owned_by=exact_match.id.split("/")[0],
        language=language,
    )
