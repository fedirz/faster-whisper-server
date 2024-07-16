from collections.abc import Generator
import os

import gradio as gr
import httpx
from httpx_sse import connect_sse
from openai import OpenAI

from faster_whisper_server.config import Config, Task

TRANSCRIPTION_ENDPOINT = "/v1/audio/transcriptions"
TRANSLATION_ENDPOINT = "/v1/audio/translations"
TIMEOUT_SECONDS = 180


def create_gradio_demo(config: Config) -> gr.Blocks:
    host = os.getenv("UVICORN_HOST", "0.0.0.0")
    port = int(os.getenv("UVICORN_PORT", "8000"))
    # NOTE: worth looking into generated clients
    http_client = httpx.Client(base_url=f"http://{host}:{port}", timeout=httpx.Timeout(timeout=TIMEOUT_SECONDS))
    openai_client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="cant-be-empty")

    def handler(file_path: str, model: str, task: Task, temperature: float, stream: bool) -> Generator[str, None, None]:
        if stream:
            previous_transcription = ""
            for transcription in transcribe_audio_streaming(file_path, task, temperature, model):
                previous_transcription += transcription
                yield previous_transcription
        else:
            yield transcribe_audio(file_path, task, temperature, model)

    def transcribe_audio(file_path: str, task: Task, temperature: float, model: str) -> str:
        if task == Task.TRANSCRIBE:
            endpoint = TRANSCRIPTION_ENDPOINT
        elif task == Task.TRANSLATE:
            endpoint = TRANSLATION_ENDPOINT

        with open(file_path, "rb") as file:
            response = http_client.post(
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

    def transcribe_audio_streaming(
        file_path: str, task: Task, temperature: float, model: str
    ) -> Generator[str, None, None]:
        with open(file_path, "rb") as file:
            kwargs = {
                "files": {"file": file},
                "data": {
                    "response_format": "text",
                    "temperature": temperature,
                    "model": model,
                    "stream": True,
                },
            }
            endpoint = TRANSCRIPTION_ENDPOINT if task == Task.TRANSCRIBE else TRANSLATION_ENDPOINT
            with connect_sse(http_client, "POST", endpoint, **kwargs) as event_source:
                for event in event_source.iter_sse():
                    yield event.data

    def update_model_dropdown() -> gr.Dropdown:
        models = openai_client.models.list().data
        model_names: list[str] = [model.id for model in models]
        assert config.whisper.model in model_names
        recommended_models = {model for model in model_names if model.startswith("Systran")}
        other_models = [model for model in model_names if model not in recommended_models]
        model_names = list(recommended_models) + other_models
        return gr.Dropdown(
            # no idea why it's complaining
            choices=model_names,  # pyright: ignore[reportArgumentType]
            label="Model",
            value=config.whisper.model,
        )

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
    with gr.Interface(
        title="Whisper Playground",
        description="""Consider supporting the project by starring the <a href="https://github.com/fedirz/faster-whisper-server">repository on GitHub</a>.""",  # noqa: E501
        inputs=[
            gr.Audio(type="filepath"),
            model_dropdown,
            task_dropdown,
            temperature_slider,
            stream_checkbox,
        ],
        fn=handler,
        outputs="text",
    ) as demo:
        demo.load(update_model_dropdown, inputs=None, outputs=model_dropdown)
    return demo
