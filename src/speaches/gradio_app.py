from collections.abc import AsyncGenerator
from pathlib import Path
import platform

import gradio as gr
import httpx
from httpx_sse import aconnect_sse
from openai import AsyncOpenAI

from speaches.config import Config, Task
from speaches.hf_utils import PiperModel

TRANSCRIPTION_ENDPOINT = "/v1/audio/transcriptions"
TRANSLATION_ENDPOINT = "/v1/audio/translations"
TIMEOUT_SECONDS = 180
TIMEOUT = httpx.Timeout(timeout=TIMEOUT_SECONDS)

# NOTE: `gr.Request` seems to be passed in as the last positional (not keyword) argument


def base_url_from_gradio_req(request: gr.Request) -> str:
    # NOTE: `request.request.url` seems to always have a path of "/gradio_api/queue/join"
    assert request.request is not None
    return f"{request.request.url.scheme}://{request.request.url.netloc}"


def http_client_from_gradio_req(request: gr.Request, config: Config) -> httpx.AsyncClient:
    base_url = base_url_from_gradio_req(request)
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=TIMEOUT,
        headers={"Authorization": f"Bearer {config.api_key}"} if config.api_key else None,
    )


def openai_client_from_gradio_req(request: gr.Request, config: Config) -> AsyncOpenAI:
    base_url = base_url_from_gradio_req(request)
    return AsyncOpenAI(base_url=f"{base_url}/v1", api_key=config.api_key if config.api_key else "cant-be-empty")


def create_gradio_demo(config: Config) -> gr.Blocks:  # noqa: C901, PLR0915
    async def whisper_handler(
        file_path: str, model: str, task: Task, temperature: float, stream: bool, request: gr.Request
    ) -> AsyncGenerator[str, None]:
        http_client = http_client_from_gradio_req(request, config)
        if task == Task.TRANSCRIBE:
            endpoint = TRANSCRIPTION_ENDPOINT
        elif task == Task.TRANSLATE:
            endpoint = TRANSLATION_ENDPOINT

        if stream:
            previous_transcription = ""
            async for transcription in streaming_audio_task(http_client, file_path, endpoint, temperature, model):
                previous_transcription += transcription
                yield previous_transcription
        else:
            yield await audio_task(http_client, file_path, endpoint, temperature, model)

    async def audio_task(
        http_client: httpx.AsyncClient, file_path: str, endpoint: str, temperature: float, model: str
    ) -> str:
        with Path(file_path).open("rb") as file:  # noqa: ASYNC230
            response = await http_client.post(
                endpoint,
                files={"file": file},
                data={
                    "model": model,
                    "response_format": "text",
                    "temperature": temperature,
                },
            )

        response.raise_for_status()
        return response.text

    async def streaming_audio_task(
        http_client: httpx.AsyncClient, file_path: str, endpoint: str, temperature: float, model: str
    ) -> AsyncGenerator[str, None]:
        with Path(file_path).open("rb") as file:  # noqa: ASYNC230
            kwargs = {
                "files": {"file": file},
                "data": {
                    "response_format": "text",
                    "temperature": temperature,
                    "model": model,
                    "stream": True,
                },
            }
            async with aconnect_sse(http_client, "POST", endpoint, **kwargs) as event_source:
                async for event in event_source.aiter_sse():
                    yield event.data

    async def update_whisper_model_dropdown(request: gr.Request) -> gr.Dropdown:
        openai_client = openai_client_from_gradio_req(request, config)
        models = (await openai_client.models.list()).data
        model_names: list[str] = [model.id for model in models]
        assert config.whisper.model in model_names
        recommended_models = {model for model in model_names if model.startswith("Systran")}
        other_models = [model for model in model_names if model not in recommended_models]
        model_names = list(recommended_models) + other_models
        return gr.Dropdown(
            choices=model_names,
            label="Model",
            value=config.whisper.model,
        )

    async def update_piper_voices_dropdown(request: gr.Request) -> gr.Dropdown:
        http_client = http_client_from_gradio_req(request, config)
        res = (await http_client.get("/v1/audio/speech/voices")).raise_for_status()
        piper_models = [PiperModel.model_validate(x) for x in res.json()]
        return gr.Dropdown(choices=[model.voice for model in piper_models], label="Voice", value=DEFAULT_VOICE)

    async def handle_audio_speech(
        text: str, voice: str, response_format: str, speed: float, sample_rate: int | None, request: gr.Request
    ) -> Path:
        openai_client = openai_client_from_gradio_req(request, config)
        res = await openai_client.audio.speech.create(
            input=text,
            model="piper",
            voice=voice,  # pyright: ignore[reportArgumentType]
            response_format=response_format,  # pyright: ignore[reportArgumentType]
            speed=speed,
            extra_body={"sample_rate": sample_rate},
        )
        audio_bytes = res.response.read()
        file_path = Path(f"audio.{response_format}")
        with file_path.open("wb") as file:  # noqa: ASYNC230
            file.write(audio_bytes)
        return file_path

    with gr.Blocks(title="Speaches Playground") as demo:
        gr.Markdown(
            "### Consider supporting the project by starring the [repository on GitHub](https://github.com/speaches-ai/speaches)."
        )
        with gr.Tab(label="Transcribe/Translate"):
            audio = gr.Audio(type="filepath")
            model_dropdown = gr.Dropdown(
                choices=[config.whisper.model],
                label="Model",
                value=config.whisper.model,
            )
            task_dropdown = gr.Dropdown(
                choices=[task.value for task in Task],
                label="Task",
                value=Task.TRANSCRIBE,
            )
            temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.0)
            stream_checkbox = gr.Checkbox(label="Stream", value=True)
            button = gr.Button("Generate")

            output = gr.Textbox()

            # NOTE: the inputs order must match the `whisper_handler` signature
            button.click(
                whisper_handler, [audio, model_dropdown, task_dropdown, temperature_slider, stream_checkbox], output
            )

        with gr.Tab(label="Speech Generation"):
            if platform.machine() == "x86_64":
                from speaches.routers.speech import (
                    DEFAULT_VOICE,
                    MAX_SAMPLE_RATE,
                    MIN_SAMPLE_RATE,
                    SUPPORTED_RESPONSE_FORMATS,
                )

                text = gr.Textbox(label="Input Text")
                voice_dropdown = gr.Dropdown(
                    choices=["en_US-amy-medium"],
                    label="Voice",
                    value="en_US-amy-medium",
                    info="""
The last part of the voice name is the quality (x_low, low, medium, high).
Each quality has a different default sample rate:
- x_low: 16000 Hz
- low: 16000 Hz
- medium: 22050 Hz
- high: 22050 Hz
""",
                )
                response_fromat_dropdown = gr.Dropdown(
                    choices=SUPPORTED_RESPONSE_FORMATS,
                    label="Response Format",
                    value="wav",
                )
                speed_slider = gr.Slider(minimum=0.25, maximum=4.0, step=0.05, label="Speed", value=1.0)
                sample_rate_slider = gr.Number(
                    minimum=MIN_SAMPLE_RATE,
                    maximum=MAX_SAMPLE_RATE,
                    label="Desired Sample Rate",
                    info="""
Setting this will resample the generated audio to the desired sample rate.
You may want to set this if you are going to use voices of different qualities but want to keep the same sample rate.
Default: None (No resampling)
""",
                    value=lambda: None,
                )
                button = gr.Button("Generate Speech")
                output = gr.Audio(type="filepath")
                button.click(
                    handle_audio_speech,
                    [text, voice_dropdown, response_fromat_dropdown, speed_slider, sample_rate_slider],
                    output,
                )
                demo.load(update_piper_voices_dropdown, inputs=None, outputs=voice_dropdown)
            else:
                gr.Textbox("Speech generation is only supported on x86_64 machines.")

        demo.load(update_whisper_model_dropdown, inputs=None, outputs=model_dropdown)
    return demo
