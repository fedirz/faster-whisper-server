import logging

from speaches.realtime.context import SessionContext
from speaches.realtime.event_router import EventRouter
from speaches.realtime.utils import generate_event_id
from speaches.types.realtime import Session, SessionUpdatedEvent, SessionUpdateEvent

logger = logging.getLogger(__name__)

event_router = EventRouter()


@event_router.register("session.update")
def handle_session_update_event(ctx: SessionContext, event: SessionUpdateEvent) -> None:  # noqa: ARG001
    # if event.session.input_audio_transcription is None or event.session.input_audio_transcription.model == "whisper-1":
    #     logger.warning("Invalid input_audio_transcription model")  # TODO
    #     event.session.input_audio_transcription = SessionInputAudioTranscription(
    #         model="Systran/faster-distil-whisper-large-v3"
    #     )
    # if (
    #     event.session.turn_detection is not None
    #     # and event.session.turn_detection.prefix_padding_ms != DEFAULT_TURN_DETECTION.prefix_padding_ms
    # ):
    #     logger.warning("Using a custom value for `turn_detection.prefix_padding_ms` is not supported.")
    #     event.session.turn_detection = DEFAULT_TURN_DETECTION
    # if event.session.instructions != ctx.configuration.instructions:
    #     logger.warning("Changing `instructions` is not supported.")
    #     event.session.instructions = DEFAULT_REALTIME_SESSION_INSTRUCTIONS
    # if event.session.input_audio_transcription is None or event.session.input_audio_transcription.model == "whisper-1":
    #     logger.warning("Invalid input_audio_transcription model")

    # NOTE: the updated `openai-realtime-console` sends partial `session.update.config` data which I don't currently support
    # TODO: figure out how to apply session updates and what to do with the above checks
    # ctx.configuration = event.session  # pyright: ignore[reportAttributeAccessIssue]
    ctx.pubsub.publish_nowait(
        SessionUpdatedEvent(
            type="session.updated",
            event_id=generate_event_id(),
            session=Session(id=ctx.session_id, **ctx.configuration.model_dump(exclude={"id"})),
        )
    )
