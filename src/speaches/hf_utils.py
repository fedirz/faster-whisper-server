from collections.abc import Generator
from functools import cached_property, lru_cache
import json
import logging
from pathlib import Path
import typing
from typing import Any, Literal

import huggingface_hub
from huggingface_hub.constants import HF_HUB_CACHE
from pydantic import BaseModel, Field, computed_field

from speaches.api_models import Model

logger = logging.getLogger(__name__)

LIBRARY_NAME = "ctranslate2"
TASK_NAME = "automatic-speech-recognition"


def does_local_model_exist(model_id: str) -> bool:
    return any(model_id == model.repo_id for model, _ in list_local_whisper_models())


def list_whisper_models() -> Generator[Model, None, None]:
    models = huggingface_hub.list_models(library="ctranslate2", tags="automatic-speech-recognition", cardData=True)
    models = list(models)
    models.sort(key=lambda model: model.downloads or -1, reverse=True)
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
        yield transformed_model


def list_local_whisper_models() -> (
    Generator[tuple[huggingface_hub.CachedRepoInfo, huggingface_hub.ModelCardData], None, None]
):
    hf_cache = huggingface_hub.scan_cache_dir()
    hf_models = [repo for repo in list(hf_cache.repos) if repo.repo_type == "model"]
    for model in hf_models:
        revision = next(iter(model.revisions))
        cached_readme_file = next((f for f in revision.files if f.file_name == "README.md"), None)
        if cached_readme_file:
            readme_file_path = Path(cached_readme_file.file_path)
        else:
            # NOTE: the README.md doesn't get downloaded when `WhisperModel` is called
            logger.debug(f"Model {model.repo_id} does not have a README.md file. Downloading it.")
            readme_file_path = Path(huggingface_hub.hf_hub_download(model.repo_id, "README.md"))

        model_card = huggingface_hub.ModelCard.load(readme_file_path)
        model_card_data = typing.cast(huggingface_hub.ModelCardData, model_card.data)
        if (
            model_card_data.library_name == LIBRARY_NAME
            and model_card_data.tags is not None
            and TASK_NAME in model_card_data.tags
        ):
            yield model, model_card_data


def get_whisper_models() -> Generator[Model, None, None]:
    models = huggingface_hub.list_models(library="ctranslate2", tags="automatic-speech-recognition", cardData=True)
    models = list(models)
    models.sort(key=lambda model: model.downloads or -1, reverse=True)
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
        yield transformed_model


PiperVoiceQuality = Literal["x_low", "low", "medium", "high"]
PIPER_VOICE_QUALITY_SAMPLE_RATE_MAP: dict[PiperVoiceQuality, int] = {
    "x_low": 16000,
    "low": 22050,
    "medium": 22050,
    "high": 22050,
}


class PiperModel(BaseModel):
    """Similar structure to the GET /v1/models response but with extra fields."""

    object: Literal["model"] = "model"
    created: int
    owned_by: Literal["rhasspy"] = "rhasspy"
    model_path: Path = Field(
        examples=[
            "/home/nixos/.cache/huggingface/hub/models--rhasspy--piper-voices/snapshots/3d796cc2f2c884b3517c527507e084f7bb245aea/en/en_US/amy/medium/en_US-amy-medium.onnx"
        ]
    )

    @computed_field(examples=["rhasspy/piper-voices/en_US-amy-medium"])
    @cached_property
    def id(self) -> str:
        return f"rhasspy/piper-voices/{self.model_path.name.removesuffix(".onnx")}"

    @computed_field(examples=["rhasspy/piper-voices/en_US-amy-medium"])
    @cached_property
    def voice(self) -> str:
        return self.model_path.name.removesuffix(".onnx")

    @computed_field
    @cached_property
    def config_path(self) -> Path:
        return Path(str(self.model_path) + ".json")

    @computed_field
    @cached_property
    def quality(self) -> PiperVoiceQuality:
        return self.id.split("-")[-1]  # pyright: ignore[reportReturnType]

    @computed_field
    @cached_property
    def sample_rate(self) -> int:
        return PIPER_VOICE_QUALITY_SAMPLE_RATE_MAP[self.quality]


def get_model_path(model_id: str, *, cache_dir: str | Path | None = None) -> Path | None:
    if cache_dir is None:
        cache_dir = HF_HUB_CACHE

    cache_dir = Path(cache_dir).expanduser().resolve()
    if not cache_dir.exists():
        raise huggingface_hub.CacheNotFound(
            f"Cache directory not found: {cache_dir}. Please use `cache_dir` argument or set `HF_HUB_CACHE` environment variable.",  # noqa: E501
            cache_dir=cache_dir,
        )

    if cache_dir.is_file():
        raise ValueError(
            f"Scan cache expects a directory but found a file: {cache_dir}. Please use `cache_dir` argument or set `HF_HUB_CACHE` environment variable."  # noqa: E501
        )

    for repo_path in cache_dir.iterdir():
        if not repo_path.is_dir():
            continue
        if repo_path.name == ".locks":  # skip './.locks/' folder
            continue
        repo_type, repo_id = repo_path.name.split("--", maxsplit=1)
        repo_type = repo_type[:-1]  # "models" -> "model"
        repo_id = repo_id.replace("--", "/")  # google--fleurs -> "google/fleurs"
        if repo_type != "model":
            continue
        if model_id == repo_id:
            return repo_path

    return None


def list_model_files(
    model_id: str, glob_pattern: str = "**/*", *, cache_dir: str | Path | None = None
) -> Generator[Path, None, None]:
    repo_path = get_model_path(model_id, cache_dir=cache_dir)
    if repo_path is None:
        return None
    snapshots_path = repo_path / "snapshots"
    if not snapshots_path.exists():
        return None
    yield from list(snapshots_path.glob(glob_pattern))


def list_piper_models() -> Generator[PiperModel, None, None]:
    model_weights_files = list_model_files("rhasspy/piper-voices", glob_pattern="**/*.onnx")
    for model_weights_file in model_weights_files:
        yield PiperModel(
            created=int(model_weights_file.stat().st_mtime),
            model_path=model_weights_file,
        )


# NOTE: It's debatable whether caching should be done here or by the caller. Should be revisited.


@lru_cache
def read_piper_voices_config() -> dict[str, Any]:
    voices_file = next(list_model_files("rhasspy/piper-voices", glob_pattern="**/voices.json"), None)
    if voices_file is None:
        raise FileNotFoundError("Could not find voices.json file")  # noqa: EM101
    return json.loads(voices_file.read_text())


@lru_cache
def get_piper_voice_model_file(voice: str) -> Path:
    model_file = next(list_model_files("rhasspy/piper-voices", glob_pattern=f"**/{voice}.onnx"), None)
    if model_file is None:
        raise FileNotFoundError(f"Could not find model file for '{voice}' voice")
    return model_file


class PiperVoiceConfigAudio(BaseModel):
    sample_rate: int
    quality: int


class PiperVoiceConfig(BaseModel):
    audio: PiperVoiceConfigAudio
    # NOTE: there are more fields in the config, but we don't care about them


@lru_cache
def read_piper_voice_config(voice: str) -> PiperVoiceConfig:
    model_config_file = next(list_model_files("rhasspy/piper-voices", glob_pattern=f"**/{voice}.onnx.json"), None)
    if model_config_file is None:
        raise FileNotFoundError(f"Could not find config file for '{voice}' voice")
    return PiperVoiceConfig.model_validate_json(model_config_file.read_text())
