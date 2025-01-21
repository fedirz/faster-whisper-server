from collections.abc import Generator
from functools import lru_cache
import json
import logging
from pathlib import Path
import typing
from typing import Any, Literal

import httpx
import huggingface_hub
from huggingface_hub.constants import HF_HUB_CACHE
from pydantic import BaseModel

from speaches.api_types import Model, Voice

logger = logging.getLogger(__name__)

LIBRARY_NAME = "ctranslate2"
TASK_NAME = "automatic-speech-recognition"

KOKORO_REVISION = "c97b7bbc3e60f447383c79b2f94fee861ff156ac"


def list_local_model_ids() -> list[str]:
    model_dirs = list(Path(HF_HUB_CACHE).glob("models--*"))
    return [model_id_from_path(model_dir) for model_dir in model_dirs]


def does_local_model_exist(model_id: str) -> bool:
    return model_id in list_local_model_ids()


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


def list_local_whisper_models() -> Generator[Model, None, None]:
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
            if model_card_data.language is None:
                language = []
            elif isinstance(model_card_data.language, str):
                language = [model_card_data.language]
            else:
                language = model_card_data.language
            transformed_model = Model(
                id=model.repo_id,
                created=int(model.last_modified),
                object_="model",
                owned_by=model.repo_id.split("/")[0],
                language=language,
            )
            yield transformed_model


def model_id_from_path(repo_path: Path) -> str:
    repo_type, repo_id = repo_path.name.split("--", maxsplit=1)
    repo_type = repo_type[:-1]  # "models" -> "model"
    assert repo_type == "model"
    repo_id = repo_id.replace("--", "/")  # google--fleurs -> "google/fleurs"
    return repo_id


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


def get_model_path(model_id: str, *, cache_dir: str | Path | None = None) -> Path | None:
    if cache_dir is None:
        cache_dir = HF_HUB_CACHE

    cache_dir = Path(cache_dir).expanduser().resolve()
    if not cache_dir.exists():
        raise huggingface_hub.CacheNotFound(
            f"Cache directory not found: {cache_dir}. Please use `cache_dir` argument or set `HF_HUB_CACHE` environment variable.",
            cache_dir=cache_dir,
        )

    if cache_dir.is_file():
        raise ValueError(
            f"Scan cache expects a directory but found a file: {cache_dir}. Please use `cache_dir` argument or set `HF_HUB_CACHE` environment variable."
        )

    for repo_path in cache_dir.iterdir():
        if not repo_path.is_dir():
            continue
        if repo_path.name == ".locks":  # skip './.locks/' folder
            continue
        if "--" not in repo_path.name:  # cache might contain unrelated custom files
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


def list_piper_models() -> Generator[Voice, None, None]:
    model_id = "rhasspy/piper-voices"
    model_weights_files = list_model_files(model_id, glob_pattern="**/*.onnx")
    for model_weights_file in model_weights_files:
        yield Voice(
            created=int(model_weights_file.stat().st_mtime),
            model_path=model_weights_file,
            voice_id=model_weights_file.name.removesuffix(".onnx"),
            model_id=model_id,
            owned_by=model_id.split("/")[0],
            sample_rate=PIPER_VOICE_QUALITY_SAMPLE_RATE_MAP[
                model_weights_file.name.removesuffix(".onnx").split("-")[-1]
            ],  # pyright: ignore[reportArgumentType]
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


def get_kokoro_model_path() -> Path:
    file_name = "kokoro-v0_19.onnx"
    onnx_files = list(list_model_files("hexgrad/Kokoro-82M", glob_pattern=f"**/{file_name}"))
    if len(onnx_files) == 0:
        raise ValueError(f"Could not find {file_name} file for 'hexgrad/Kokoro-82M' model")
    return onnx_files[0]


def download_kokoro_model() -> None:
    model_id = "hexgrad/Kokoro-82M"
    model_repo_path = Path(
        huggingface_hub.snapshot_download(
            model_id,
            repo_type="model",
            allow_patterns=["kokoro-v0_19.onnx"],
            revision=KOKORO_REVISION,
        )
    )
    # HACK
    res = httpx.get(
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin", follow_redirects=True
    ).raise_for_status()
    voices_path = model_repo_path / "voices.bin"
    voices_path.touch(exist_ok=True)
    voices_path.write_bytes(res.content)


# alternative implementation that uses `huggingface_hub.scan_cache_dir`. Slightly cleaner but much slower
# def list_local_model_ids() -> list[str]:
#     start = time.perf_counter()
#     hf_cache = huggingface_hub.scan_cache_dir()
#     logger.debug(f"Scanned HuggingFace cache in {time.perf_counter() - start:.2f} seconds")
#     hf_models = [repo for repo in list(hf_cache.repos) if repo.repo_type == "model"]
#     return [model.repo_id for model in hf_models]
