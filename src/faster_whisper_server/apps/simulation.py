import asyncio
import time
from collections import deque
from urllib.parse import urlencode

import gradio as gr
import httpx
import numpy as np
import websockets
from openai import OpenAI
from pydub import AudioSegment
from pyexpat import model

from faster_whisper_server.config import Config

# Audio parameters
SAMPLES_RATE = 16000
CHUNK_TIME = 100  # Chunk size in ms


def create_gradio_demo(config: Config) -> gr.Blocks:
    base_url = f"http://{config.host}:{config.port}"
    openai_client = OpenAI(base_url=f"{base_url}/v1", api_key="cant-be-empty")
    WEBSOCKET_URL_BASE = f"ws://{config.host}:{config.port}/v1/audio/transcriptions"

    async def receive_responses(ws):
        """Receive responses from the WebSocket server asynchronously."""
        try:
            while True:
                response = await ws.recv()
                yield response
        except websockets.ConnectionClosed:
            return

    async def stream_audio(ws, audio_file_path):
        """Stream audio from the specified file to the WebSocket server."""
        audio = AudioSegment.from_file(audio_file_path).set_frame_rate(SAMPLES_RATE)
        start = time.perf_counter()
        expect_time_per_chunk = CHUNK_TIME / 1000
        for i, audio_start in enumerate(range(0, len(audio), CHUNK_TIME)):
            audio_chunk = audio[audio_start : audio_start + CHUNK_TIME]
            await ws.send(audio_chunk.raw_data)
            time_to_sleep = (i + 1) * expect_time_per_chunk - (time.perf_counter() - start)
            await asyncio.sleep(time_to_sleep)

    async def websocket_stream(audio_file_path, base, queries):
        url = f"{base}?{urlencode(queries)}"
        async with websockets.connect(url) as ws:
            stream_task = asyncio.create_task(stream_audio(ws, audio_file_path))

            # Yield responses live as they arrive
            async for response in receive_responses(ws):
                yield response

    async def stream_audio_file(audio_file, model, language, temperature):
        queries = {
            "response_format": "text",
            "model": model,
            "temperature": temperature,
        }
        if language != "auto":
            queries["language"] = language
        async for response in websocket_stream(audio_file, WEBSOCKET_URL_BASE, queries):
            yield response  # Yield each WebSocket response to Gradio

    def update_model_dropdown() -> gr.Dropdown:
        model_data = openai_client.models.list().data
        models = {model.id: model for model in model_data}
        model_names = list(models.keys())
        dropdown = gr.Dropdown(
            value="deepdml/faster-whisper-large-v3-turbo-ct2",
            choices=model_names,
            label="Model",
        )
        return dropdown, models

    def update_language_dropdown(model, models) -> gr.Dropdown:
        model_data = models[model]
        languages = model_data.language
        value = "auto"
        if len(languages) == 1:
            value = languages[0]
            languages = [value]
        else:
            mapping = {"en": 0, "zh": 1, "ar": 2}
            languages.sort(key=lambda x: mapping.get(x, 3))
            languages = ["auto"] + languages
        dropdown = gr.Dropdown(
            value=value,
            choices=languages,
            label="Language",
        )
        return dropdown

    def fn_preload_models(model):
        response = httpx.post(f"http://192.168.5.32:9080/api/ps/{model}")
        if response.is_success:
            return "Models preloaded successfully"
        elif response.status_code == 409:
            return "Model already loaded"
        else:
            return "Failed to preload models"

    async def fn_stream(audio_stream, model, language, temperature):
        async for stream in audio_stream:
            print(stream)

    BUFFER_INTERVAL = 0.5

    class SessionAudioStreamer:
        """Handles audio streaming to a WebSocket server for a single session."""

        def __init__(self):
            self.buffer = deque()
            self.buffer_lock = asyncio.Lock()
            self.ws_connection = None
            self.ws_established = asyncio.Event()
            self.flush_task = None

        async def initialize_ws_connection(self, model, language, temperature):
            """Initialize the WebSocket connection based on session parameters."""

            queries = {
                "response_format": "text",
                "model": model,
                "temperature": temperature,
            }
            if language != "auto":
                queries["language"] = language

            url = f"{WEBSOCKET_URL_BASE}?{urlencode(queries)}"
            self.ws_connection = await websockets.connect(url)
            self.ws_established.set()
            print("WebSocket connection established for session")

        async def buffer_audio_chunk(self, audio_chunk):
            """Buffer incoming audio chunks and start flush task if not running."""
            print("Buffering audio chunk")
            async with self.buffer_lock:
                self.buffer.append(audio_chunk)
                print("Buffer size:", len(self.buffer))

            # Start the flush task if not already running
            if self.flush_task is None:
                self.flush_task = asyncio.create_task(self._buffer_flush())
            print("Buffered audio chunk")

        async def _buffer_flush(self):
            return
            """Flush the buffer periodically by sending data to WebSocket."""
            await self.ws_established.wait()
            while True:
                if self.ws_connection and self.buffer:
                    async with self.buffer_lock:
                        combined_data = np.concatenate(self.buffer, axis=0).tobytes()
                        self.buffer.clear()

                    try:
                        await self.ws_connection.send(combined_data)
                        response = await self.ws_connection.recv()
                        print("Server response:", response)
                    except websockets.ConnectionClosed:
                        print("WebSocket connection closed unexpectedly")
                        break

                await asyncio.sleep(BUFFER_INTERVAL)

        async def close(self):
            """Close the WebSocket connection and clear the buffer."""
            if self.ws_connection:
                await self.ws_connection.close()
                print("WebSocket connection closed for session")
            if self.flush_task:
                self.flush_task.cancel()
            self.buffer.clear()

    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        models = gr.State({})
        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    label="Model",
                )
                language_dropdown = gr.Dropdown(
                    label="Language",
                )
                temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.0)
                with gr.Row():
                    preload_models = gr.Button("Preload models")
                    preload_models_reponse = gr.Textbox(show_label=False)
                iface.load(update_model_dropdown, inputs=[], outputs=[model_dropdown, models])
                model_dropdown.change(
                    update_language_dropdown, inputs=[model_dropdown, models], outputs=[language_dropdown]
                )
                preload_models.click(fn_preload_models, inputs=[model_dropdown], outputs=[preload_models_reponse])
                # with gr.Tab("Live Stream Audio"):
                #     audio_file = gr.Audio(label="Audio", streaming=True)
                #     streamer_state = gr.State(SessionAudioStreamer())

                #     async def on_audio_start_recording(model, language, temperature, streamer_state):
                #         await streamer_state.initialize_ws_connection(model, language, temperature)
                #         return streamer_state

                #     async def on_audio_stream(audio_chunk, streamer_state):
                #         await streamer_state.buffer_audio_chunk(audio_chunk)

                #     audio_file.start_recording(on_audio_start_recording,
                #                                inputs=[model_dropdown, language_dropdown, temperature_slider, streamer_state],
                #                                outputs=[streamer_state])
                #     audio_file.stream(on_audio_stream, inputs=[audio_file, streamer_state], outputs=[])
            with gr.Column():
                with gr.Tab("Audio Live Simulation"):
                    audio_file = gr.Audio(label="Audio", sources=["upload"], type="filepath")
                    transcription = gr.Textbox(label="Transcription")
                    audio_file.play(
                        stream_audio_file,
                        inputs=[audio_file, model_dropdown, language_dropdown, temperature_slider],
                        outputs=[transcription],
                    )
    return iface
