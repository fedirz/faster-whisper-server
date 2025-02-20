import gradio as gr
import httpx
from openai import AsyncOpenAI

from speaches.config import Config

TIMEOUT_SECONDS = 180
TIMEOUT = httpx.Timeout(timeout=TIMEOUT_SECONDS)


def base_url_from_gradio_req(request: gr.Request, config: Config) -> str:
    if config.loopback_host_url is not None and len(config.loopback_host_url) > 0:
        return config.loopback_host_url
    # NOTE: `request.request.url` seems to always have a path of "/gradio_api/queue/join"
    assert request.request is not None
    return f"{request.request.url.scheme}://{request.request.url.netloc}"


def http_client_from_gradio_req(request: gr.Request, config: Config) -> httpx.AsyncClient:
    base_url = base_url_from_gradio_req(request, config)
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=TIMEOUT,
        headers={"Authorization": f"Bearer {config.api_key}"} if config.api_key else None,
    )


def openai_client_from_gradio_req(request: gr.Request, config: Config) -> AsyncOpenAI:
    base_url = base_url_from_gradio_req(request, config)
    return AsyncOpenAI(
        base_url=f"{base_url}/v1", api_key=config.api_key.get_secret_value() if config.api_key else "cant-be-empty"
    )
