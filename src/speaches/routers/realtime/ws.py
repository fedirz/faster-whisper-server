import asyncio
import logging

from fastapi import (
    APIRouter,
    WebSocket,
)
from openai import AsyncOpenAI

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
from speaches.realtime.message_manager import WsServerMessageManager
from speaches.realtime.response_event_router import event_router as response_event_router
from speaches.realtime.session import OPENAI_REALTIME_SESSION_DURATION_SECONDS, create_session_object_configuration
from speaches.realtime.session_event_router import event_router as session_event_router
from speaches.realtime.utils import task_done_callback
from speaches.types.realtime import SessionCreatedEvent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["realtime"])

event_router = EventRouter()
event_router.include_router(conversation_event_router)
event_router.include_router(input_audio_buffer_event_router)
event_router.include_router(response_event_router)
event_router.include_router(session_event_router)


async def event_listener(ctx: SessionContext) -> None:
    try:
        async with asyncio.TaskGroup() as tg:
            async for event in ctx.pubsub.poll():
                # logger.debug(f"Received event: {event.type}")

                task = tg.create_task(event_router.dispatch(ctx, event))
                task.add_done_callback(task_done_callback)
    except asyncio.CancelledError:
        logger.info("Event listener task cancelled")
        raise
    finally:
        logger.info("Event listener task finished")


@router.websocket("/v1/realtime")
async def realtime(
    ws: WebSocket,
    model: str,
    config: ConfigDependency,
    transcription_client: TranscriptionClientDependency,
) -> None:
    await ws.accept()
    logger.info("Accepted websocket connection")

    completion_client = AsyncOpenAI(
        base_url=f"http://{config.host}:{config.port}/v1",
        api_key=config.api_key.get_secret_value() if config.api_key else "cant-be-empty",
        max_retries=1,
    ).chat.completions
    ctx = SessionContext(
        transcription_client=transcription_client,
        completion_client=completion_client,
        session=create_session_object_configuration(model),
    )
    message_manager = WsServerMessageManager(ctx.pubsub)
    async with asyncio.TaskGroup() as tg:
        event_listener_task = tg.create_task(event_listener(ctx), name="event_listener")
        async with asyncio.timeout(OPENAI_REALTIME_SESSION_DURATION_SECONDS):
            mm_task = asyncio.create_task(message_manager.run(ws))
            # HACK: a tiny delay to ensure the message_manager.run() task is started. Otherwise, the `SessionCreatedEvent` will not be sent, as it's published before the `sender` task subscribes to the pubsub.
            await asyncio.sleep(0.001)
            ctx.pubsub.publish_nowait(SessionCreatedEvent(session=ctx.session))
            await mm_task
        event_listener_task.cancel()

    logger.info(f"Finished handling '{ctx.session.id}' session")
