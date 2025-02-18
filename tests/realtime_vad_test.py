import asyncio
import base64
import logging
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openai.types.beta.realtime.conversation_item_content_param import ConversationItemContentParam
from openai.types.beta.realtime.conversation_item_param import ConversationItemParam
from openai.types.beta.realtime.session_update_event_param import Session, SessionTurnDetection
import pytest
import soundfile as sf
import websockets

from speaches.audio import resample_audio

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
BYTERATE = SAMPLE_RATE * SAMPLE_WIDTH  # like "bitrate" but in bytes

WS_BASE_URL = "ws://localhost:8000/v1"
MODEL = "gpt-4o-mini"

RESPONSE_SESSION = Session(turn_detection=SessionTurnDetection(create_response=True))
NO_RESPONSE_SESSION = Session(turn_detection=SessionTurnDetection(create_response=False))


async def audio_sender(
    conn: AsyncRealtimeConnection, audio_bytes: bytes, chunks_per_second: int = 10, speed: int = 1
) -> None:
    chunk_size = BYTERATE // chunks_per_second
    try:
        async with asyncio.TaskGroup() as tg:
            for i in range(0, len(audio_bytes), chunk_size):
                logger.info(f"Sending audio chunk from {i} to {i + chunk_size} of {len(audio_bytes)}")
                audio_chunk = audio_bytes[i : i + chunk_size]
                tg.create_task(conn.input_audio_buffer.append(audio=base64.b64encode(audio_chunk).decode("utf-8")))
                await asyncio.sleep(1 / chunks_per_second / speed)
    except* websockets.exceptions.ConnectionClosedError:
        logger.info("Connection closed")


async def print_events(conn: AsyncRealtimeConnection, final_event: str | None = None) -> None:
    try:
        async for event in conn:
            if event.type == "response.audio.delta":
                size = len(base64.b64decode(event.delta))
                event.delta = f"base64 encoded audio of size {size} bytes"
            print(event.model_dump_json())
            if final_event is not None and event.type == final_event:
                break
    except websockets.exceptions.ConnectionClosedError:
        logger.info("Connection closed")


data, samplerate = sf.read(Path("1_2_3_4_5_6_7_8.wav"), dtype="int16")
pcm_audio_bytes = data.tobytes()
audio_bytes = resample_audio(pcm_audio_bytes, samplerate, 24000)
quite_audio = np.zeros(SAMPLE_RATE * 3, dtype=np.int16).tobytes()
audio_bytes = audio_bytes + quite_audio


@pytest.mark.asyncio
@pytest.mark.requires_openai
async def test_realtime_vad_openai() -> None:
    realtime_client = AsyncOpenAI(websocket_base_url=WS_BASE_URL).beta.realtime
    async with asyncio.TaskGroup() as tg, realtime_client.connect(model=MODEL) as conn:
        print_events_task = tg.create_task(
            print_events(conn, final_event="conversation.item.input_audio_transcription.completed")
        )
        await conn.session.update(session=NO_RESPONSE_SESSION)
        audio_sender_task = tg.create_task(audio_sender(conn, audio_bytes))
        await audio_sender_task
        await print_events_task
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.requires_openai
async def test_realtime_response() -> None:
    realtime_client = AsyncOpenAI(websocket_base_url=WS_BASE_URL).beta.realtime
    async with asyncio.TaskGroup() as tg, realtime_client.connect(model=MODEL) as conn:
        print_events_task = tg.create_task(print_events(conn, final_event=None))
        await conn.session.update(session=RESPONSE_SESSION)
        audio_sender_task = tg.create_task(audio_sender(conn, audio_bytes))
        await audio_sender_task
        await print_events_task
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.requires_openai
async def test_realtime_create_conversation_item() -> None:
    realtime_client = AsyncOpenAI(websocket_base_url=WS_BASE_URL).beta.realtime
    async with asyncio.TaskGroup() as tg, realtime_client.connect(model=MODEL) as conn:
        print_events_task = tg.create_task(print_events(conn, final_event="response.done"))
        await conn.session.update(session=NO_RESPONSE_SESSION)
        await conn.conversation.item.create(
            item=ConversationItemParam(
                role="user", type="message", content=[ConversationItemContentParam(type="input_text", text="Hello")]
            )
        )
        await conn.response.create()
        await print_events_task
        await conn.close()
