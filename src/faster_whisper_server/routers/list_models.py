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

if TYPE_CHECKING:
    from huggingface_hub.hf_api import ModelInfo

router = APIRouter()


@router.get("/v1/models")
def get_models() -> ListModelsResponse:
    models = huggingface_hub.list_models(library="ctranslate2", tags="automatic-speech-recognition", cardData=True)
    models = list(models)
    models.sort(key=lambda model: model.downloads, reverse=True)  # type: ignore  # noqa: PGH003
    transformed_models: list[Model] = []
    for model in models:
        assert model.created_at is not None
        assert model.card_data is not None
        assert model.card_data.language is None or isinstance(model.card_data.language, str | list)
        if model.card_data.language is None:
            language = []
        elif isinstance(model.card_data.language, str):
            language = [model.card_data.language]
        else:
            language = model.card_data.language
        transformed_model = Model(
            id=model.id,
            created=int(model.created_at.timestamp()),
            object_="model",
            owned_by=model.id.split("/")[0],
            language=language,
        )
        transformed_models.append(transformed_model)
    return ListModelsResponse(data=transformed_models)


@router.get("/v1/models/{model_name:path}")
# NOTE: `examples` doesn't work https://github.com/tiangolo/fastapi/discussions/10537
def get_model(
    model_name: Annotated[str, Path(example="Systran/faster-distil-whisper-large-v3")],
) -> Model:
    models = huggingface_hub.list_models(
        model_name=model_name, library="ctranslate2", tags="automatic-speech-recognition", cardData=True
    )
    models = list(models)
    models.sort(key=lambda model: model.downloads, reverse=True)  # type: ignore  # noqa: PGH003
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
