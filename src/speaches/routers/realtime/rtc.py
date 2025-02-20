import asyncio
import base64
import json
import logging
from pathlib import Path
import time
from typing import Annotated

from aiortc import RTCDataChannel, RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from aiortc.sdp import SessionDescription
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler
from fastapi import (
    APIRouter,
    Query,
    Request,
    Response,
)
import numpy as np
from openai import AsyncOpenAI
from openai.types.beta.realtime.error_event import Error
from pydantic import ValidationError

from speaches.dependencies import (
    ConfigDependency,
    TranscriptionClientDependency,
)
from speaches.realtime.context import SessionContext
from speaches.realtime.conversation_event_router import event_router as conversation_event_router
from speaches.realtime.event_router import EventRouter
from speaches.realtime.input_audio_buffer_event_router import (
    event_router as input_audio_buffer_event_router,
)
from speaches.realtime.response_event_router import event_router as response_event_router
from speaches.realtime.rtc.audio_stream_track import AudioStreamTrack
from speaches.realtime.session import create_session_object_configuration
from speaches.realtime.session_event_router import event_router as session_event_router
from speaches.routers.realtime.ws import event_listener
from speaches.types.realtime import (
    SERVER_EVENT_TYPES,
    ErrorEvent,
    InputAudioBufferAppendEvent,
    SessionCreatedEvent,
    client_event_type_adapter,
    server_event_type_adapter,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["realtime"])

event_router = EventRouter()
event_router.include_router(conversation_event_router)
event_router.include_router(input_audio_buffer_event_router)
event_router.include_router(response_event_router)
event_router.include_router(session_event_router)

# TODO: limit session duration
# TODO: faster session initialization with web rtc

# https://stackoverflow.com/questions/77560930/cant-create-audio-frame-with-from-nd-array

rtc_tasks: set[asyncio.Task[None]] = set()
pcs = set()


async def rtc_datachannel_sender(ctx: SessionContext, channel: RTCDataChannel) -> None:
    logger.info("Sender task started")
    q = ctx.pubsub.subscribe()
    try:
        while True:
            # logger.debug("Waiting for event")
            event = await q.get()
            if event.type not in SERVER_EVENT_TYPES:
                continue
            server_event = server_event_type_adapter.validate_python(event)
            if server_event.type == "response.audio.delta":
                logger.debug("Skipping response.audio.delta event")
                continue
            logger.debug(f"Sending {event.type} event")
            channel.send(server_event.model_dump_json())
            logger.info(f"Sent {event.type} event")
            # try:
            # except fastapi.WebSocketDisconnect:
            #     logger.info("Failed to send message due to disconnect")
            #     break
    except BaseException:
        logger.exception("Sender task failed")
        ctx.pubsub.subscribers.remove(q)
        raise


def message_handler(ctx: SessionContext, message: str) -> None:
    logger.info(f"Message received: {message}")
    try:
        event = client_event_type_adapter.validate_json(message)
    except ValidationError as e:
        ctx.pubsub.publish_nowait(ErrorEvent(error=Error(type="invalid_request_error", message=str(e))))
        logger.exception(f"Received an invalid client event: {message}")
        return

    logger.debug(f"Received {event.type} event")
    ctx.pubsub.publish_nowait(event)
    # asyncio.create_task(event_router.dispatch(ctx, event))


async def audio_receiver(ctx: SessionContext, track: RemoteStreamTrack) -> None:
    # NOTE: IMPORTANT! 24Khz because that's what the `input_audio_buffer.append` handler expects
    desired_sample_rate = 24000
    min_buffer_duration_ms = 200
    buffer_size = int(desired_sample_rate * min_buffer_duration_ms / 1000)

    # Initialize buffer to store audio data
    buffer = np.array([], dtype=np.int16)

    while True:
        frames = await track.recv()
        # ensure that the received frames are of expected format
        assert isinstance(frames, AudioFrame)
        assert frames.sample_rate == 48000
        assert frames.layout.name == "stereo"
        assert frames.format.name == "s16"

        resampler = AudioResampler(format="s16", layout="mono", rate=desired_sample_rate)
        frames = resampler.resample(frames)

        # Accumulate audio data
        for frame in frames:
            arr = frame.to_ndarray()
            buffer = np.append(buffer, arr.flatten())  # Flatten and append to buffer

            # When buffer reaches or exceeds target size, emit event
            while len(buffer) >= buffer_size:
                # Take BUFFER_SIZE samples
                output_chunk = buffer[:buffer_size]
                # Keep remaining samples in buffer
                buffer = buffer[buffer_size:]

                # Convert to bytes and emit event
                audio_bytes = output_chunk.tobytes()
                assert len(audio_bytes) == len(output_chunk) * 2, "Audio sample width is not 2 bytes"
                ctx.pubsub.publish_nowait(
                    InputAudioBufferAppendEvent(
                        type="input_audio_buffer.append",
                        audio=base64.b64encode(audio_bytes).decode(),
                    )
                )


def datachannel_handler(ctx: SessionContext, channel: RTCDataChannel) -> None:
    logger.info(f"Data channel created: {channel}")
    channel.send(SessionCreatedEvent(session=ctx.session).model_dump_json())

    rtc_tasks.add(asyncio.create_task(rtc_datachannel_sender(ctx, channel)))

    channel.on("message")(lambda message: message_handler(ctx, message))


def iceconnectionstatechange_handler(ctx: SessionContext, pc: RTCPeerConnection) -> None:
    logger.info(f"ICE connection state changed to {pc.iceConnectionState}")
    if pc.iceConnectionState in ["failed", "closed"]:
        pcs.discard(pc)
        logger.info("Peer connection closed")

        with Path(f"sessions/{ctx.session.id}.json").open("w") as f:
            logger.info(f"Dumping events to file {ctx.session.id}")
            f.write(json.dumps([m.model_dump() for m in ctx.pubsub.events], indent=2))


def track_handler(ctx: SessionContext, track: RemoteStreamTrack) -> None:
    logger.info(f"Track received: kind={track.kind}")
    if track.kind == "audio":
        # Start a task to log audio data
        rtc_tasks.add(asyncio.create_task(audio_receiver(ctx, track)))
    track.on("ended")(lambda: logger.info(f"Track ended: kind={track.kind}"))


@router.post("/v1/realtime")
async def realtime_webrtc(
    request: Request,
    model: Annotated[str, Query(...)],
    config: ConfigDependency,
    transcription_client: TranscriptionClientDependency,
) -> Response:
    completion_client = AsyncOpenAI(
        base_url=f"http://{config.host}:{config.port}/v1",
        api_key=config.api_key.get_secret_value() if config.api_key else None,
    ).chat.completions
    ctx = SessionContext(
        transcription_client=transcription_client,
        completion_client=completion_client,
        session=create_session_object_configuration(model),
    )

    # TODO: handle both application/sdp and application/json
    sdp = (await request.body()).decode("utf-8")
    # session_description = SessionDescription.parse(sdp)
    # for media in session_description.media:
    #     logger.info(f"offer media: {media}")
    offer = RTCSessionDescription(sdp=sdp, type="offer")
    logger.info(f"Received offer: {offer.sdp[:5]}")

    # Create a new RTCPeerConnection
    # configuration = RTCConfiguration(
    #     iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")],
    # )
    pc = RTCPeerConnection()
    pcs.add(pc)

    pc.on("datachannel", lambda channel: datachannel_handler(ctx, channel))
    pc.on("iceconnectionstatechange", lambda: iceconnectionstatechange_handler(ctx, pc))
    pc.on("track", lambda track: track_handler(ctx, track))
    pc.on(
        "icegatheringstatechange",
        lambda: logger.info(f"ICE gathering state changed to {pc.iceGatheringState}"),
    )
    pc.on(
        "icecandidate",
        lambda *args, **kwargs: logger.info(f"ICE candidate: {args}, {kwargs}. {pc.iceGatheringState}"),
    )

    logger.info("Created peer connection")

    # NOTE: is relay needed?
    audio_track = AudioStreamTrack(ctx)
    pc.addTrack(audio_track)

    # Set the remote description and create an answer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    assert answer is not None
    answer_session_description = SessionDescription.parse(answer.sdp)

    # remove all codecs except opus. This **should** ensure that we only receive opus audio
    for media in answer_session_description.media:
        if media.kind != "audio":
            continue
        logger.info(f"Codec before: {media.rtp.codecs}")
        media.rtp.codecs = [codec for codec in media.rtp.codecs if codec.name == "opus"]
        logger.info(f"Codec after: {media.rtp.codecs}")

    # logger.info(f"Created answer: {answer_session_description}")
    start = time.perf_counter()
    await pc.setLocalDescription(
        answer
        # RTCSessionDescription(str(answer_session_description), type="answer")
    )  # NOTE: this takes ~5 secondd: could be relevant https://github.com/aiortc/aiortc/issues/1183
    logger.info(f"Set local description in {time.perf_counter() - start:.3f} seconds")

    rtc_tasks.add(asyncio.create_task(event_listener(ctx)))

    return Response(content=pc.localDescription.sdp, media_type="text/plain charset=utf-8")
