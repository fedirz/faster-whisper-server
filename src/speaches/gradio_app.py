from collections.abc import AsyncGenerator
from pathlib import Path
import platform

import gradio as gr
import httpx
from httpx_sse import aconnect_sse
from openai import AsyncOpenAI

from speaches import kokoro_utils
from speaches.api_types import Voice
from speaches.config import Config, Task
from speaches.routers.speech import (
    MAX_SAMPLE_RATE,
    MIN_SAMPLE_RATE,
    SUPPORTED_RESPONSE_FORMATS,
)

TRANSCRIPTION_ENDPOINT = "/v1/audio/transcriptions"
TRANSLATION_ENDPOINT = "/v1/audio/translations"
TIMEOUT_SECONDS = 180
TIMEOUT = httpx.Timeout(timeout=TIMEOUT_SECONDS)
DEFAULT_TEXT = "A rainbow is an optical phenomenon caused by refraction, internal reflection and dispersion of light in water droplets resulting in a continuous spectrum of light appearing in the sky."  # noqa: E501

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

    async def update_voices_and_language_dropdown(model_id: str | None, request: gr.Request) -> dict:
        params = httpx.QueryParams({"model_id": model_id})
        http_client = http_client_from_gradio_req(request, config)
        res = (await http_client.get("/v1/audio/speech/voices", params=params)).raise_for_status()
        voice_ids = [Voice.model_validate(x).voice_id for x in res.json()]
        return {
            voice_dropdown: gr.update(choices=voice_ids, value=voice_ids[0]),
            language_dropdown: gr.update(visible=model_id == "hexgrad/Kokoro-82M"),
        }

    async def handle_audio_speech(
        text: str,
        model: str,
        voice: str,
        language: str | None,
        response_format: str,
        speed: float,
        sample_rate: int | None,
        request: gr.Request,
    ) -> Path:
        openai_client = openai_client_from_gradio_req(request, config)
        res = await openai_client.audio.speech.create(
            input=text,
            model=model,
            voice=voice,  # pyright: ignore[reportArgumentType]
            response_format=response_format,  # pyright: ignore[reportArgumentType]
            speed=speed,
            extra_body={"language": language, "sample_rate": sample_rate},
        )
        audio_bytes = res.response.read()
        file_path = Path(f"audio.{response_format}")
        with file_path.open("wb") as file:  # noqa: ASYNC230
            file.write(audio_bytes)
        return file_path

    with gr.Blocks(title="Speaches Playground") as demo:
        gr.Markdown("# Speaches Playground")
        gr.Markdown(
            "### Consider supporting the project by starring the [speaches-ai/speaches repository on GitHub](https://github.com/speaches-ai/speaches)."
        )
        gr.Markdown("### Documentation Website: https://speaches-ai.github.io/speaches")
        gr.Markdown(
            "### For additional details regarding the parameters, see the [API Documentation](https://speaches-ai.github.io/speaches/api)"
        )

        with gr.Tab(label="Speech-to-Text"):
            audio = gr.Audio(type="filepath")
            whisper_model_dropdown = gr.Dropdown(
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
                whisper_handler,
                [audio, whisper_model_dropdown, task_dropdown, temperature_slider, stream_checkbox],
                output,
            )

        with gr.Tab(label="Text-to-Speech"):
            model_dropdown_choices = ["hexgrad/Kokoro-82M", "rhasspy/piper-voices"]
            if platform.machine() != "x86_64":
                model_dropdown_choices.remove("rhasspy/piper-voices")
                gr.Textbox("Speech generation using `rhasspy/piper-voices` model is only supported on x86_64 machines.")

            text = gr.Textbox(
                label="Input Text",
                value=DEFAULT_TEXT,
            )
            stt_model_dropdown = gr.Dropdown(
                choices=model_dropdown_choices,
                label="Model",
                value="hexgrad/Kokoro-82M",
            )
            voice_dropdown = gr.Dropdown(
                choices=["af"],
                label="Voice",
                value="af",
            )
            language_dropdown = gr.Dropdown(
                choices=kokoro_utils.LANGUAGES, label="Language", value="en-us", visible=True
            )
            stt_model_dropdown.change(
                update_voices_and_language_dropdown,
                inputs=[stt_model_dropdown],
                outputs=[voice_dropdown, language_dropdown],
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
You may want to set this if you are going to use 'rhasspy/piper-voices' with voices of different qualities but want to keep the same sample rate.
Default: None (No resampling)
""",  # noqa: E501
                value=lambda: None,
            )
            button = gr.Button("Generate Speech")
            output = gr.Audio(type="filepath")
            button.click(
                handle_audio_speech,
                [
                    text,
                    stt_model_dropdown,
                    voice_dropdown,
                    language_dropdown,
                    response_fromat_dropdown,
                    speed_slider,
                    sample_rate_slider,
                ],
                output,
            )

        demo.load(update_whisper_model_dropdown, inputs=None, outputs=whisper_model_dropdown)
        demo.load(
            update_voices_and_language_dropdown,
            inputs=[stt_model_dropdown],
            outputs=[voice_dropdown, language_dropdown],
        )
    return demo
