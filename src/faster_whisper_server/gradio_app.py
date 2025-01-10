from collections.abc import Generator
from pathlib import Path

import gradio as gr
import httpx
from httpx_sse import connect_sse
from openai import OpenAI

from faster_whisper_server.config import Config, Task
from faster_whisper_server.hf_utils import PiperModel

# FIX: this won't work on ARM
from faster_whisper_server.routers.speech import (
    DEFAULT_VOICE,
    MAX_SAMPLE_RATE,
    MIN_SAMPLE_RATE,
    SUPPORTED_RESPONSE_FORMATS,
)

TRANSCRIPTION_ENDPOINT = "/v1/audio/transcriptions"
TRANSLATION_ENDPOINT = "/v1/audio/translations"
TIMEOUT_SECONDS = 180
TIMEOUT = httpx.Timeout(timeout=TIMEOUT_SECONDS)


def create_gradio_demo(config: Config) -> gr.Blocks:  # noqa: C901, PLR0915
    base_url = f"http://{config.host}:{config.port}"
    http_client = httpx.Client(base_url=base_url, timeout=TIMEOUT)
    openai_client = OpenAI(base_url=f"{base_url}/v1", api_key="cant-be-empty")

    # TODO: make async
    def whisper_handler(
        file_path: str, model: str, task: Task, temperature: float, stream: bool
    ) -> Generator[str, None, None]:
        if task == Task.TRANSCRIBE:
            endpoint = TRANSCRIPTION_ENDPOINT
        elif task == Task.TRANSLATE:
            endpoint = TRANSLATION_ENDPOINT

        if stream:
            previous_transcription = ""
            for transcription in streaming_audio_task(file_path, endpoint, temperature, model):
                previous_transcription += transcription
                yield previous_transcription
        else:
            yield audio_task(file_path, endpoint, temperature, model)

    def audio_task(file_path: str, endpoint: str, temperature: float, model: str) -> str:
        with Path(file_path).open("rb") as file:
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

    def streaming_audio_task(
        file_path: str, endpoint: str, temperature: float, model: str
    ) -> Generator[str, None, None]:
        with Path(file_path).open("rb") as file:
            kwargs = {
                "files": {"file": file},
                "data": {
                    "response_format": "text",
                    "temperature": temperature,
                    "model": model,
                    "stream": True,
                },
            }
            with connect_sse(http_client, "POST", endpoint, **kwargs) as event_source:
                for event in event_source.iter_sse():
                    yield event.data

    def update_whisper_model_dropdown() -> gr.Dropdown:
        models = openai_client.models.list().data
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

    def update_piper_voices_dropdown() -> gr.Dropdown:
        res = http_client.get("/v1/audio/speech/voices").raise_for_status()
        piper_models = [PiperModel.model_validate(x) for x in res.json()]
        return gr.Dropdown(choices=[model.voice for model in piper_models], label="Voice", value=DEFAULT_VOICE)

    # TODO: make async
    def handle_audio_speech(text: str, voice: str, response_format: str, speed: float, sample_rate: int | None) -> Path:
        res = openai_client.audio.speech.create(
            input=text,
            model="piper",
            voice=voice,  # pyright: ignore[reportArgumentType]
            response_format=response_format,  # pyright: ignore[reportArgumentType]
            speed=speed,
            extra_body={"sample_rate": sample_rate},
        )
        audio_bytes = res.response.read()
        file_path = Path(f"audio.{response_format}")
        with file_path.open("wb") as file:
            file.write(audio_bytes)
        return file_path

    with gr.Blocks(title="faster-whisper-server Playground") as demo:
        gr.Markdown(
            "### Consider supporting the project by starring the [repository on GitHub](https://github.com/fedirz/faster-whisper-server)."
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
            # TODO: add warning about ARM
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

        demo.load(update_whisper_model_dropdown, inputs=None, outputs=model_dropdown)
        demo.load(update_piper_voices_dropdown, inputs=None, outputs=voice_dropdown)
    return demo
