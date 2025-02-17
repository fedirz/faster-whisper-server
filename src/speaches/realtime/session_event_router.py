import logging

from openai.types.beta.realtime import ErrorEvent
from openai.types.beta.realtime.error_event import Error

from speaches.realtime.context import SessionContext
from speaches.realtime.event_router import EventRouter
from speaches.realtime.utils import generate_event_id
from speaches.types.realtime import Session, SessionUpdatedEvent, SessionUpdateEvent

logger = logging.getLogger(__name__)

event_router = EventRouter()


def _update_dict(original: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict):
            original[key] = _update_dict(original.get(key, {}), value)
        else:
            original[key] = value
    return original


@event_router.register("session.update")
def handle_session_update_event(ctx: SessionContext, event: SessionUpdateEvent) -> None:
    if event.session.input_audio_format is None and ctx.configuration.input_audio_format != "pcm16":
        msg = "Using a custom value for `input_audio_format` is not supported. The configuration value will remain as `pcm16`."
        ctx.pubsub.publish_nowait(
            ErrorEvent(
                type="error",
                event_id=generate_event_id(),
                error=Error(type="invalid_request_error", message=msg),
            )
        )
        logger.warning(msg)

    if event.session.output_audio_format is None and ctx.configuration.output_audio_format != "pcm16":
        msg = "Using a custom value for `output_audio_format` is not supported. The configuration value will remain as `pcm16`."
        ctx.pubsub.publish_nowait(
            ErrorEvent(
                type="error",
                event_id=generate_event_id(),
                error=Error(type="invalid_request_error", message=msg),
            )
        )
        logger.warning(msg)

    if (
        event.session.turn_detection is not None
        and event.session.turn_detection.prefix_padding_ms != ctx.configuration.turn_detection.prefix_padding_ms
    ):
        msg = f"Using a custom value for `turn_detection.prefix_padding_ms` is not supported. The configuration value will remain as `{ctx.configuration.turn_detection.prefix_padding_ms}`."
        ctx.pubsub.publish_nowait(
            ErrorEvent(
                type="error",
                event_id=generate_event_id(),
                error=Error(type="invalid_request_error", message=msg),
            )
        )
        logger.warning(msg)

    session_dict = ctx.configuration.model_dump()
    session_update_dict = event.session.model_dump(
        exclude_none=True,
        # https://docs.pydantic.dev/latest/concepts/serialization/#advanced-include-and-exclude
        exclude={"input_audio_format": True, "output_audio_format": True, "turn_detection": {"prefix_padding_ms"}},
    )
    logger.debug(f"Applying session update: {session_update_dict}")
    updated_session = _update_dict(session_dict, session_update_dict)
    ctx.configuration = Session(**updated_session)

    ctx.pubsub.publish_nowait(
        SessionUpdatedEvent(
            type="session.updated",
            event_id=generate_event_id(),
            session=Session(id=ctx.session_id, **ctx.configuration.model_dump(exclude={"id"})),
        )
    )
