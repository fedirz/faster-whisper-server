from typing import Generator

import pytest
from fastapi.testclient import TestClient

from faster_whisper_server.main import app
from faster_whisper_server.server_models import ModelObject

MODEL_THAT_EXISTS = "Systran/faster-whisper-tiny.en"
MODEL_THAT_DOES_NOT_EXIST = "i-do-not-exist"
MIN_EXPECTED_NUMBER_OF_MODELS = (
    200  # At the time of the test creation there are 228 models
)


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as client:
        yield client


# HACK: because ModelObject(**data) doesn't work
def model_dict_to_object(model_dict: dict) -> ModelObject:
    return ModelObject(
        id=model_dict["id"],
        created=model_dict["created"],
        object_=model_dict["object"],
        owned_by=model_dict["owned_by"],
    )


def test_list_models(client: TestClient):
    response = client.get("/v1/models")
    data = response.json()
    models = [model_dict_to_object(model_dict) for model_dict in data]
    assert len(models) > MIN_EXPECTED_NUMBER_OF_MODELS


def test_model_exists(client: TestClient):
    response = client.get(f"/v1/model/{MODEL_THAT_EXISTS}")
    data = response.json()
    model = model_dict_to_object(data)
    assert model.id == MODEL_THAT_EXISTS


def test_model_does_not_exist(client: TestClient):
    response = client.get(f"/v1/model/{MODEL_THAT_DOES_NOT_EXIST}")
    assert response.status_code == 404
