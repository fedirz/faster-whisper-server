import os
from typing import Generator

import gradio as gr
import httpx
from httpx_sse import connect_sse

from faster_whisper_server.config import Config, Task

TRANSCRIPTION_ENDPOINT = "/v1/audio/transcriptions"
TRANSLATION_ENDPOINT = "/v1/audio/translations"


def create_gradio_demo(config: Config) -> gr.Blocks:
    host = os.getenv("UVICORN_HOST", "0.0.0.0")
    port = os.getenv("UVICORN_PORT", 8000)
    # NOTE: worth looking into generated clients
    http_client = httpx.Client(base_url=f"http://{host}:{port}", timeout=None)

    def handler(
        file_path: str, model: str, task: Task, temperature: float, stream: bool
    ) -> Generator[str, None, None]:
        if stream:
            previous_transcription = ""
            for transcription in transcribe_audio_streaming(
                file_path, task, temperature, model
            ):
                previous_transcription += transcription
                yield previous_transcription
        else:
            yield transcribe_audio(file_path, task, temperature, model)

    def transcribe_audio(
        file_path: str, task: Task, temperature: float, model: str
    ) -> str:
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
            endpoint = (
                TRANSCRIPTION_ENDPOINT
                if task == Task.TRANSCRIBE
                else TRANSLATION_ENDPOINT
            )
            with connect_sse(http_client, "POST", endpoint, **kwargs) as event_source:
                for event in event_source.iter_sse():
                    yield event.data

    def update_model_dropdown() -> gr.Dropdown:
        res = http_client.get("/v1/models")
        res_data = res.json()
        models: list[str] = [model["id"] for model in res_data]
        assert config.whisper.model in models
        recommended_models = set(
            model for model in models if model.startswith("Systran")
        )
        other_models = [model for model in models if model not in recommended_models]
        models = list(recommended_models) + other_models
        model_dropdown = gr.Dropdown(
            # no idea why it's complaining
            choices=models,  # type: ignore
            label="Model",
            value=config.whisper.model,
        )
        return model_dropdown

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
    temperature_slider = gr.Slider(
        minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.0
    )
    stream_checkbox = gr.Checkbox(label="Stream", value=True)
    with gr.Interface(
        title="Whisper Playground",
        description="""Consider supporting the project by starring the <a href="https://github.com/fedirz/faster-whisper-server">repository on GitHub</a>.""",
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
