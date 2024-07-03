import openai
from openai import OpenAI
import pytest

MODEL_THAT_EXISTS = "Systran/faster-whisper-tiny.en"
MODEL_THAT_DOES_NOT_EXIST = "i-do-not-exist"
MIN_EXPECTED_NUMBER_OF_MODELS = 70  # At the time of the test creation there are 89 models


def test_list_models(openai_client: OpenAI) -> None:
    models = openai_client.models.list().data
    assert len(models) > MIN_EXPECTED_NUMBER_OF_MODELS


def test_model_exists(openai_client: OpenAI) -> None:
    model = openai_client.models.retrieve(MODEL_THAT_EXISTS)
    assert model.id == MODEL_THAT_EXISTS


def test_model_does_not_exist(openai_client: OpenAI) -> None:
    with pytest.raises(openai.NotFoundError):
        openai_client.models.retrieve(MODEL_THAT_DOES_NOT_EXIST)
