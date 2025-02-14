import asyncio
import logging

from fastapi import (
    APIRouter,
    WebSocket,
)
from openai.types.beta.realtime.session_created_event import SessionCreatedEvent

from speaches.dependencies import (
    CompletionClientDependency,
    SpeechClientDependency,
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
from speaches.realtime.session import OPENAI_REALTIME_SESSION_DURATION_SECONDS, create_session_configuration
from speaches.realtime.session_event_router import event_router as session_event_router
from speaches.realtime.utils import generate_event_id
from speaches.types.realtime import (
    Session,
)

logger = logging.getLogger(__name__)

router = APIRouter()

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

                def task_done_callback(task: asyncio.Task[None]) -> None:
                    try:
                        task.result()
                    except asyncio.CancelledError:
                        logger.info(f"Task {task.get_name()} cancelled")
                    except Exception:
                        logger.exception("Task failed")

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
    transcription_client: TranscriptionClientDependency,
    completion_client: CompletionClientDependency,
    speech_client: SpeechClientDependency,
) -> None:
    await ws.accept()
    logger.info("Accepted websocket connection")
    ctx = SessionContext(
        transcription_client=transcription_client,
        speech_client=speech_client,
        completion_client=completion_client,
        configuration=create_session_configuration(model),
    )
    message_manager = WsServerMessageManager(ctx.pubsub)
    async with asyncio.TaskGroup() as tg:
        event_listener_task = tg.create_task(event_listener(ctx), name="event_listener")
        async with asyncio.timeout(OPENAI_REALTIME_SESSION_DURATION_SECONDS):
            mm_task = asyncio.create_task(message_manager.run(ws))
            await asyncio.sleep(0.1)  # HACK
            ctx.pubsub.publish_nowait(
                SessionCreatedEvent(
                    type="session.created",
                    event_id=generate_event_id(),
                    session=Session(id=ctx.session_id, **ctx.configuration.model_dump(exclude={"id"})),
                )
            )
            await mm_task
        event_listener_task.cancel()

    logger.info(f"Finished handling '{ctx.session_id}' session")
