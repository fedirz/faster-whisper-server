from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai.types.beta.realtime.error_event import Error

from speaches.realtime.event_router import EventRouter
from speaches.types.realtime import ErrorEvent, Session, SessionUpdatedEvent, SessionUpdateEvent

if TYPE_CHECKING:
    from speaches.realtime.context import SessionContext

logger = logging.getLogger(__name__)

event_router = EventRouter()


def update_dict(original: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict):
            original[key] = update_dict(original.get(key, {}), value)
        else:
            original[key] = value
    return original


def unsupported_field_error(field: str) -> ErrorEvent:
    return ErrorEvent(
        error=Error(
            type="invalid_request_error",
            message=f"Specifying `{field}` is not supported. The server either does not support this field or it is not configurable.",
        )
    )


@event_router.register("session.update")
def handle_session_update_event(ctx: SessionContext, event: SessionUpdateEvent) -> None:
    if event.session.input_audio_format is not None:
        ctx.pubsub.publish_nowait(unsupported_field_error("session.input_audio_format"))
    if event.session.output_audio_format is not None:
        ctx.pubsub.publish_nowait(unsupported_field_error("session.output_audio_format"))
    if event.session.turn_detection is not None and event.session.turn_detection.prefix_padding_ms is not None:
        ctx.pubsub.publish_nowait(unsupported_field_error("session.turn_detection.prefix_padding_ms"))

    session_dict = ctx.session.model_dump()
    session_update_dict = event.session.model_dump(
        exclude_none=True,
        # https://docs.pydantic.dev/latest/concepts/serialization/#advanced-include-and-exclude
        exclude={"input_audio_format": True, "output_audio_format": True, "turn_detection": {"prefix_padding_ms"}},
    )

    logger.debug(f"Applying session configuration update: {session_update_dict}")
    logger.debug(f"Session configuration before update: {session_dict}")
    updated_session = update_dict(session_dict, session_update_dict)
    logger.debug(f"Session configuration after update: {updated_session}")
    ctx.session = Session(**updated_session)

    ctx.pubsub.publish_nowait(SessionUpdatedEvent(session=ctx.session))
