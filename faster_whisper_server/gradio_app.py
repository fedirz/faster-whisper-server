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
        file_path: str | None, model: str, task: Task, temperature: float, stream: bool
    ) -> Generator[str, None, None]:
        if file_path is None:
            yield ""
            return
        if stream:
            yield from transcribe_audio_streaming(file_path, task, temperature, model)
        yield transcribe_audio(file_path, task, temperature, model)

    def transcribe_audio(
        file_path: str, task: Task, temperature: float, model: str
    ) -> str:
        if task == Task.TRANSCRIPTION:
            endpoint = TRANSCRIPTION_ENDPOINT
        elif task == Task.TRANSLATION:
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
                if task == Task.TRANSCRIPTION
                else TRANSLATION_ENDPOINT
            )
            with connect_sse(http_client, "POST", endpoint, **kwargs) as event_source:
                for event in event_source.iter_sse():
                    yield event.data

    model_dropdown = gr.Dropdown(
        # TODO: use output from /v1/models
        choices=[config.whisper.model],
        label="Model",
        value=config.whisper.model,
    )
    task_dropdown = gr.Dropdown(
        choices=[task.value for task in Task],
        label="Task",
        value=Task.TRANSCRIPTION,
    )
    temperature_slider = gr.Slider(
        minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.0
    )
    stream_checkbox = gr.Checkbox(label="Stream", value=True)
    demo = gr.Interface(
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
    )
    return demo
