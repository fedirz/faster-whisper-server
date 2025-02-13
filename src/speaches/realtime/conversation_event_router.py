from io import BytesIO
import logging

from openai.types.beta.realtime.error_event import Error

from speaches.realtime.context import SessionContext
from speaches.realtime.event_router import EventRouter
from speaches.realtime.input_audio_buffer import InputAudioBuffer
from speaches.realtime.utils import generate_event_id, generate_item_id, generate_response_id
from speaches.types.realtime import (
    ConversationItem,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ErrorEvent,
    RealtimeResponse,
    ResponseCreatedEvent,
)

logger = logging.getLogger(__name__)

event_router = EventRouter()


def last_committed_audio_buffer(ctx: SessionContext) -> InputAudioBuffer:
    input_audio_buffers_iter = reversed(ctx.input_audio_buffers)
    next(input_audio_buffers_iter)  # skip the last buffer
    return ctx.input_audio_buffers[next(input_audio_buffers_iter)]


async def transcription_flow(
    ctx: SessionContext,
    item: ConversationItem,
) -> None:
    assert ctx.configuration.input_audio_transcription is not None  # HACK
    assert ctx.configuration.input_audio_transcription.model is not None
    file = BytesIO()
    transcription = await ctx.transcription_client.create(
        file=file,
        model=ctx.configuration.input_audio_transcription.model,
    )
    assert item.content[0].type == "input_audio"  # HACK?
    item.content[0].transcript = transcription.text
    ctx.conversation[item.id] = item
    ctx.pubsub.publish_nowait(
        ConversationItemInputAudioTranscriptionCompletedEvent(
            type="conversation.item.input_audio_transcription.completed",
            event_id=generate_event_id(),
            content_index=0,
            item_id=item.id,
            transcript=transcription.text,
        )
    )


# Client Events
@event_router.register("conversation.item.create")
def handle_conversation_item_create_event(ctx: SessionContext, event: ConversationItemCreateEvent) -> None:
    # TODO: What should happen if this get's called when a response is being generated?
    # TODO: Test what happens when `previous_item_id` is passed in but isn't the last item.
    if event.previous_item_id is not None:
        raise NotImplementedError
    # if event.item.id in ctx.conversation:
    #     # TODO: Weirdly OpenAI's API allows creating an item with an already existing ID! Do their implementation replace the item?  # noqa: E501
    #     raise NotImplementedError
    # TODO: should we assign the previous item's id when it hasn't been specified in the request?
    if event.item.id is None:
        event.item.id = generate_item_id()
    item = ConversationItem(**event.item.model_dump())  # ConversationItem from OpenAIConversationItem
    ctx.conversation[item.id] = item
    ctx.pubsub.publish_nowait(
        ConversationItemCreatedEvent(
            type="conversation.item.created",
            event_id=generate_event_id(),
            previous_item_id=None,  # TODO
            item=item,
        )
    )


# @event_router.register("conversation.item.truncate") # TODO
# def handle_conversation_item_truncate_event(ctx: SessionContext, event: ConversationItemTruncateEvent) -> None:
#     pass


@event_router.register("conversation.item.delete")
def handle_conversation_item_delete_event(ctx: SessionContext, event: ConversationItemDeleteEvent) -> None:
    if event.item_id not in ctx.conversation:
        ctx.pubsub.publish_nowait(
            ErrorEvent(
                type="error",
                event_id=generate_event_id(),
                error=Error(
                    type="invalid_request_error",
                    message=f"Error deleting item: the item with id '{event.item_id}' does not exist.",
                ),
            )
        )
    else:
        # TODO: What should be done if this a conversation that's being currently genererated?
        del ctx.conversation[event.item_id]
        ctx.pubsub.publish_nowait(
            ConversationItemDeletedEvent(
                type="conversation.item.deleted", event_id=generate_event_id(), item_id=event.item_id
            )
        )


# Server Events


@event_router.register("conversation.item.created")
async def handle_conversation_item_created_event(ctx: SessionContext, event: ConversationItemCreatedEvent) -> None:
    item = ctx.conversation[event.item.id]
    if item.type == "message" and item.role == "user" and item.content[0].type == "input_audio":
        # NOTE: we aren't passing in `event.item` directly since `event.item` is a copy of the original item, meaning we won't be able to update the original item in the context.  # noqa: E501
        await transcription_flow(ctx, item)


@event_router.register("conversation.item.input_audio_transcription.completed")
def handle_conversation_item_input_audio_transcription_completed_event(
    ctx: SessionContext, _event: ConversationItemInputAudioTranscriptionCompletedEvent
) -> None:
    # ONLY run if turn detection is on (and maybe check if response isn't being generated already)
    if ctx.configuration.turn_detection is not None and ctx.configuration.turn_detection.create_response:
        response = RealtimeResponse(
            id=generate_response_id(),
            output=[],
        )
        ctx.responses[response.id] = response
        ctx.pubsub.publish_nowait(
            ResponseCreatedEvent(type="response.created", event_id=generate_event_id(), response=response)
        )
        logger.info("Created a new response because of conversation.item.input_audio_transcription.completed")


# @event_router.register("conversation.item.input_audio_transcription.failed") # TODO
