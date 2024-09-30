import asyncio
import os

import anyio
from httpx import ASGITransport, AsyncClient
import pytest

from faster_whisper_server.main import create_app


@pytest.mark.asyncio
async def test_model_unloaded_after_ttl() -> None:
    ttl = 5
    model = "Systran/faster-whisper-tiny.en"
    os.environ["WHISPER__TTL"] = str(ttl)
    os.environ["ENABLE_UI"] = "false"
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as aclient:
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 0
        await aclient.post(f"/api/ps/{model}")
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 1
        await asyncio.sleep(ttl + 1)
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 0


@pytest.mark.asyncio
async def test_ttl_resets_after_usage() -> None:
    ttl = 5
    model = "Systran/faster-whisper-tiny.en"
    os.environ["WHISPER__TTL"] = str(ttl)
    os.environ["ENABLE_UI"] = "false"
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as aclient:
        await aclient.post(f"/api/ps/{model}")
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 1
        await asyncio.sleep(ttl - 2)
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 1

        async with await anyio.open_file("audio.wav", "rb") as f:
            data = await f.read()
        res = (
            await aclient.post(
                "/v1/audio/transcriptions", files={"file": ("audio.wav", data, "audio/wav")}, data={"model": model}
            )
        ).json()
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 1
        await asyncio.sleep(ttl - 2)
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 1

        await asyncio.sleep(3)
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 0

        # test the model can be used again after being unloaded
        # this just ensures the model can be loaded again after being unloaded
        res = (
            await aclient.post(
                "/v1/audio/transcriptions", files={"file": ("audio.wav", data, "audio/wav")}, data={"model": model}
            )
        ).json()


@pytest.mark.asyncio
async def test_model_cant_be_unloaded_when_used() -> None:
    ttl = 0
    model = "Systran/faster-whisper-tiny.en"
    os.environ["WHISPER__TTL"] = str(ttl)
    os.environ["ENABLE_UI"] = "false"
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as aclient:
        async with await anyio.open_file("audio.wav", "rb") as f:
            data = await f.read()

        task = asyncio.create_task(
            aclient.post(
                "/v1/audio/transcriptions", files={"file": ("audio.wav", data, "audio/wav")}, data={"model": model}
            )
        )
        await asyncio.sleep(0.01)
        res = await aclient.delete(f"/api/ps/{model}")
        assert res.status_code == 409

        await task
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 0


@pytest.mark.asyncio
async def test_model_cant_be_loaded_twice() -> None:
    ttl = -1
    model = "Systran/faster-whisper-tiny.en"
    os.environ["ENABLE_UI"] = "false"
    os.environ["WHISPER__TTL"] = str(ttl)
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as aclient:
        res = await aclient.post(f"/api/ps/{model}")
        assert res.status_code == 201
        res = await aclient.post(f"/api/ps/{model}")
        assert res.status_code == 409
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 1


@pytest.mark.asyncio
async def test_model_is_unloaded_after_request_when_ttl_is_zero() -> None:
    ttl = 0
    os.environ["WHISPER__MODEL"] = "Systran/faster-whisper-tiny.en"
    os.environ["WHISPER__TTL"] = str(ttl)
    os.environ["ENABLE_UI"] = "false"
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as aclient:
        async with await anyio.open_file("audio.wav", "rb") as f:
            data = await f.read()
        res = await aclient.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", data, "audio/wav")},
            data={"model": "Systran/faster-whisper-tiny.en"},
        )
        res = (await aclient.get("/api/ps")).json()
        assert len(res["models"]) == 0
