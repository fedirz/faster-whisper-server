from collections.abc import Generator
import logging
from pathlib import Path
import typing

import huggingface_hub

logger = logging.getLogger(__name__)

LIBRARY_NAME = "ctranslate2"
TASK_NAME = "automatic-speech-recognition"


def does_local_model_exist(model_id: str) -> bool:
    return any(model_id == model.repo_id for model, _ in list_local_models())


def list_local_models() -> Generator[tuple[huggingface_hub.CachedRepoInfo, huggingface_hub.ModelCardData], None, None]:
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
