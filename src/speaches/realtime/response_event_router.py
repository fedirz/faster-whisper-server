import asyncio
import base64
import logging
import time

from openai.types.beta.realtime import response_content_part_added_event, response_content_part_done_event
from openai.types.beta.realtime.error_event import Error

from speaches.dependencies import get_config
from speaches.realtime.chat_utils import (
    conversation_item_from_chunk,
    conversation_to_chat_messages,
    generate_reply_message,
    message_content_from_chunk,
    tool_call_argument_delta_from_chunk,
)
from speaches.realtime.context import SessionContext
from speaches.realtime.event_router import EventRouter
from speaches.realtime.utils import generate_event_id, generate_response_id
from speaches.types.realtime import (
    ConversationItemContent,
    ConversationItemCreatedEvent,
    RealtimeResponse,
    RealtimeResponseStatus,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

logger = logging.getLogger(__name__)

event_router = EventRouter()

# TODO: start using this error
conversation_already_has_active_response_error = Error(
    type="invalid_request_error",
    message="Conversation already has an active response",
)


async def audio_generation_flow(ctx: SessionContext, text: str) -> None:
    response_id = next(reversed(ctx.responses))
    response = ctx.responses[response_id]
    assert response.output is not None and len(response.output) == 1, response
    item = response.output[0]
    assert item.content[0].type == "audio"  # HACK

    start = time.perf_counter()
    first_chunk_timestamp = None
    config = get_config()
    async with ctx.speech_client.with_streaming_response.create(
        input=text,
        model=config.speech_model,
        voice=ctx.configuration.voice,  # pyright: ignore  # noqa: PGH003
        response_format="pcm",
        extra_body=config.speech_extra_body,
    ) as speech_streamed_response:
        chunk = await speech_streamed_response.read()
        if first_chunk_timestamp is None:
            first_chunk_timestamp = time.perf_counter()
            print(f"First chunk took: {first_chunk_timestamp - start}")
        ctx.pubsub.publish_nowait(
            ResponseAudioDeltaEvent(
                type="response.audio.delta",
                event_id=generate_event_id(),
                response_id=response.id,
                item_id=item.id,
                content_index=0,
                output_index=0,
                delta=base64.b64encode(chunk).decode(),
            )
        )
    print(f"Audio generation took: {time.perf_counter() - start}")


def post_response_flow(ctx: SessionContext) -> None:  # TODO: on event
    response_id = next(reversed(ctx.responses))
    response = ctx.responses[response_id]
    assert response.output is not None and len(response.output) == 1, response
    item = response.output[0]
    cancelled = False  # TODO
    item.status = "completed" if not cancelled else "incomplete"
    ctx.pubsub.publish_nowait(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            event_id=generate_event_id(),
            response_id=response.id,
            item=item,
            output_index=0,
        )
    )
    response.status = "completed" if not cancelled else "incomplete"
    # response.output.append(item) # FIXME: this VERY LIKELY shouldn't be commented out. I need to verify if output is appended anywhere else  # noqa: E501
    status_detail = None if not cancelled else RealtimeResponseStatus()  # XXX: this likely needs to be populated
    response.status_details = status_detail
    ctx.pubsub.publish_nowait(ResponseDoneEvent(type="response.done", event_id=generate_event_id(), response=response))
    if not cancelled:  # HACK
        ctx.conversation[item.id] = item


async def text_generation_flow(ctx: SessionContext) -> None:  # noqa: C901, PLR0912, PLR0915
    response_id = next(reversed(ctx.responses))
    response = ctx.responses[response_id]
    assert response.output is not None, response
    # assert len(response.output) == 1, response
    # item = response.output[0]
    config = get_config()  # HACK
    try:
        start = time.perf_counter()
        item = None
        async for chunk in generate_reply_message(
            ctx.completion_client,
            config.chat_completion_model,  # type: ignore  # noqa: PGH003
            list(conversation_to_chat_messages(ctx.conversation)),
            ctx.configuration,
        ):
            if item is None:
                item = conversation_item_from_chunk(chunk)
                response.output.append(item)
                ctx.pubsub.publish_nowait(
                    ResponseOutputItemAddedEvent(
                        type="response.output_item.added",
                        event_id=generate_event_id(),
                        output_index=0,
                        response_id=response.id,
                        item=item,
                    )
                )
                ctx.conversation[item.id] = item
                ctx.pubsub.publish_nowait(
                    ConversationItemCreatedEvent(
                        type="conversation.item.created", event_id=generate_event_id(), previous_item_id=None, item=item
                    )  # TODO: previous_item_id
                )
                # continue  # NOTE: this might only make sense to do for OpenAI since other implemetation might actually provide useful info in the first chunk. OpenAI doesn't  # noqa: E501

            if chunk.usage is not None:
                pass
                # TODO: set usage

            assert item.type is not None and item.type != "function_call_output", item
            match item.type:
                case "message":
                    if len(item.content) == 0:
                        if ctx.configuration.modalities == ["text"]:
                            content_item = ConversationItemContent(id=generate_response_id(), text="")
                        else:
                            content_item = ConversationItemContent(id=generate_response_id(), transcript="")
                        item.content.append(content_item)
                        ctx.pubsub.publish_nowait(
                            ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                event_id=generate_event_id(),
                                content_index=0,
                                output_index=0,
                                response_id=response.id,
                                item_id=item.id,
                                part=response_content_part_added_event.Part(
                                    **content_item.model_dump(exclude_none=True, exclude={"id"})
                                ),
                            )
                        )
                    content_item = item.content[0]
                    delta = message_content_from_chunk(chunk)
                    if ctx.configuration.modalities == ["text"]:
                        assert content_item.type == "text" and content_item.text is not None, content_item
                        content_item.text += delta
                        delta_event = ResponseTextDeltaEvent(
                            type="response.text.delta",
                            event_id=generate_event_id(),
                            content_index=0,
                            output_index=0,
                            response_id=response.id,
                            item_id=item.id,
                            delta=delta,
                        )
                    else:
                        assert content_item.type == "input_audio" and content_item.transcript is not None, content_item
                        content_item.transcript += delta
                        delta_event = ResponseAudioTranscriptDeltaEvent(
                            type="response.audio_transcript.delta",
                            event_id=generate_event_id(),
                            content_index=0,
                            output_index=0,
                            response_id=response.id,
                            item_id=item.id,
                            delta=delta,
                        )
                case "function_call":
                    delta = tool_call_argument_delta_from_chunk(chunk)
                    if delta is None:
                        continue
                    assert item.type == "function_call" and item.arguments is not None and item.call_id is not None, (
                        item
                    )
                    item.arguments += delta
                    delta_event = ResponseFunctionCallArgumentsDeltaEvent(
                        type="response.function_call_arguments.delta",
                        event_id=generate_event_id(),
                        output_index=0,
                        response_id=response.id,
                        item_id=item.id,
                        call_id=item.call_id,
                        delta=delta,
                        # TODO: check why there's no `name`
                    )

            ctx.pubsub.publish_nowait(delta_event)

        logger.info(f"Response generation took: {time.perf_counter() - start}")

        assert item is not None
        assert item.type is not None and item.type != "function_call_output", item
        match item.type:
            case "message":
                assert item.content[0].text is not None, item
                if ctx.configuration.modalities == ["text"]:
                    done_event = ResponseTextDoneEvent(
                        type="response.text.done",
                        event_id=generate_event_id(),
                        content_index=0,
                        output_index=0,
                        response_id=response.id,
                        item_id=item.id,
                        text=item.content[0].text,
                    )
                else:
                    assert item.content[0].transcript is not None, item
                    done_event = ResponseAudioTranscriptDoneEvent(
                        type="response.audio_transcript.done",
                        event_id=generate_event_id(),
                        content_index=0,
                        output_index=0,
                        response_id=response.id,
                        item_id=item.id,
                        transcript=item.content[0].transcript,
                    )

            case "function_call":
                assert item.call_id is not None and item.arguments is not None, item
                done_event = ResponseFunctionCallArgumentsDoneEvent(
                    type="response.function_call_arguments.done",
                    event_id=generate_event_id(),
                    output_index=0,
                    response_id=response.id,
                    item_id=item.id,
                    call_id=item.call_id,
                    arguments=item.arguments,
                )
        item.status = "completed"
        ctx.pubsub.publish_nowait(done_event)
    except (
        asyncio.CancelledError
    ):  # Woudln't this also get triggered when some error occurs? Should we handle this differently?
        logger.exception("Response generation cancelled")
        raise
        # TODO: handle this
        # cancelled = True


@event_router.register("response.create")
def handle_response_create_event(ctx: SessionContext, _event: ResponseCreateEvent) -> None:
    response = RealtimeResponse(
        id=generate_response_id(),
        output=[],  # TODO: should this be None
    )
    ctx.responses[response.id] = response
    ctx.pubsub.publish_nowait(
        ResponseCreatedEvent(type="response.created", event_id=generate_event_id(), response=response)
    )
    logger.info("Created a new response because of response.create")


@event_router.register("response.created")
async def handle_response_created_event(ctx: SessionContext, _event: ResponseCreatedEvent) -> None:
    logger.info("Response created")
    await text_generation_flow(ctx)


@event_router.register("response.text.done")
def handle_response_text_done_event(ctx: SessionContext, event: ResponseTextDoneEvent) -> None:
    item = ctx.conversation[event.item_id]
    assert item.type == "message"
    item_content = item.content[0]
    assert item_content.type == "text"
    ctx.pubsub.publish_nowait(
        ResponseContentPartDoneEvent(
            type="response.content_part.done",
            event_id=generate_event_id(),
            content_index=0,
            output_index=0,
            response_id=event.response_id,
            item_id=event.item_id,
            part=response_content_part_done_event.Part(**item_content.model_dump(exclude_none=True, exclude={"id"})),
        )
    )
    post_response_flow(ctx)


@event_router.register("response.function_call_arguments.done")
def handle_response_function_call_arguments_done_event(
    ctx: SessionContext, _event: ResponseFunctionCallArgumentsDoneEvent
) -> None:
    post_response_flow(ctx)


@event_router.register("response.audio_transcript.done")
async def handle_response_audio_transcript_done_event(
    ctx: SessionContext, event: ResponseAudioTranscriptDoneEvent
) -> None:
    item = ctx.conversation[event.item_id]
    assert item.type == "message" and item.content[0].type == "audio", item
    await audio_generation_flow(ctx, item.content[0].transcript)
    ctx.pubsub.publish_nowait(
        ResponseAudioDoneEvent(
            type="response.audio.done",
            event_id=generate_event_id(),
            content_index=0,
            output_index=0,
            response_id=event.response_id,
            item_id=item.id,
        )
    )


@event_router.register("response.audio.done")
def handle_response_audio_done_event(ctx: SessionContext, event: ResponseAudioDoneEvent) -> None:
    item = ctx.conversation[event.item_id]
    assert item.type == "message"
    content_part = item.content[0]
    assert content_part.type == "audio"
    ctx.pubsub.publish_nowait(
        ResponseContentPartDoneEvent(
            type="response.content_part.done",
            event_id=generate_event_id(),
            content_index=0,
            output_index=0,
            response_id=event.response_id,
            item_id=event.item_id,
            part=content_part,
        )
    )
    post_response_flow(ctx)


@event_router.register("response.cancel")
def handle_response_cancel_event(ctx: SessionContext, _event: ResponseCancelEvent) -> None:
    # If there's  no response task, then it's a no-op. OpenAI's API should be monitored to see if the behaviour changes.  # noqa: E501
    pass
