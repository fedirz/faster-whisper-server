from collections import OrderedDict
from collections.abc import AsyncGenerator, Generator
import logging
from typing import Literal

from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.shared_params.function_definition import FunctionDefinition

from speaches.types.realtime import ConversationItem, Session

logger = logging.getLogger(__name__)


def message_content_from_chunk(chunk: ChatCompletionChunk) -> str:
    if len(chunk.choices) == 0:
        logger.warning(f"Chunk has no choices: {chunk}")
        return ""
    elif len(chunk.choices) > 1:
        logger.warning(f"Chunk has more than one choice: {chunk}")
        return ""
    # assert chunk.choices[0].delta.content is not None # doesn't work with groq
    if chunk.choices[0].delta.content is None:
        logger.warning(f"Chunk has no content: {chunk}")
        return ""

    return chunk.choices[0].delta.content


async def generate_reply_message(
    client: AsyncCompletions,
    model_id: str,
    messages: list[ChatCompletionMessageParam],
    session_config: Session,
    # TODO: response_config: ResponseConfiguration,
) -> AsyncGenerator[ChatCompletionChunk, None]:
    kwargs = {}  # NOTE: passing these directly causes type checker to complaint as None is not compatible NotGiven
    if session_config.tools is not None and len(session_config.tools) > 0:
        tools: list[ChatCompletionToolParam] = []
        for tool in session_config.tools:
            assert tool.name is not None and tool.description is not None and isinstance(tool.parameters, dict)
            tools.append(
                ChatCompletionToolParam(
                    type="function",
                    function=FunctionDefinition(
                        name=tool.name, description=tool.description, parameters=tool.parameters
                    ),
                )
            )
        kwargs["tools"] = tools
        if session_config.tool_choice is not None:
            kwargs["tool_choice"] = session_config.tool_choice

    assert session_config.instructions is not None  # FIXME: propbably not a valid assert
    async for chat_completion_chunk in await client.create(
        model=model_id,
        messages=[
            ChatCompletionSystemMessageParam(
                role="system",
                content=session_config.instructions,
            ),
            *messages,
        ],
        stream=True,
        temperature=session_config.temperature,
        max_tokens=session_config.max_response_output_tokens
        if isinstance(session_config.max_response_output_tokens, int)
        else None,
        stream_options={"include_usage": True},
        **kwargs,
    ):
        yield chat_completion_chunk

    # TODO: cleanup this function


def conversation_to_chat_messages(  # noqa: C901
    conversation: OrderedDict[str, ConversationItem],
) -> Generator[ChatCompletionMessageParam, None, None]:
    for item in conversation.values():
        match item.type:
            case "message":
                content_list = item.content
                assert content_list is not None and len(content_list) == 1, conversation
                content = content_list[0]
                if item.status != "completed":
                    logger.warning(f"Item {item} is not completed. Skipping.")
                    continue
                match content.type:
                    case "text":
                        assert content.text, content
                        yield ChatCompletionAssistantMessageParam(role="assistant", content=content.text)
                    case "audio":
                        # TODO: why is this code unreachable
                        assert content.transcript, content
                        yield ChatCompletionAssistantMessageParam(role="assistant", content=content.transcript)
                    case "input_text":
                        assert content.text, content
                        yield ChatCompletionUserMessageParam(role="user", content=content.text)
                    case "input_audio":
                        if not content.transcript:
                            # TODO: look into why this is happening
                            #     assert content.transcript, content
                            # AssertionError: type='input_audio' transcript=''
                            logger.error(f"Skipping user input audio conversation item without transcript: {item}")
                            continue
                        yield ChatCompletionUserMessageParam(role="user", content=content.transcript)
            case "function_call":
                assert item.call_id and item.name and item.arguments and item.status == "completed", item

                yield ChatCompletionAssistantMessageParam(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageToolCallParam(
                            id=item.call_id,  # XXX: or should this be `item.id`
                            type="function",
                            function=Function(
                                name=item.name,
                                arguments=item.arguments,
                            ),
                        )
                    ],
                )
            case "function_call_output":
                assert item.call_id and item.output, item
                yield ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=item.call_id,
                    content=item.output,
                )


def get_chunk_output_type(chunk: ChatCompletionChunk) -> Literal["message", "function_call"]:
    assert len(chunk.choices) == 1
    delta = chunk.choices[0].delta
    if delta.content is not None:
        return "message"
    elif delta.tool_calls is not None:
        return "function_call"
    raise ValueError(f"Unknown chunk content type: {chunk}")


def conversation_item_from_chunk(
    chunk: ChatCompletionChunk,
) -> ConversationItem:
    output_type = get_chunk_output_type(chunk)
    match output_type:
        case "message":
            # TODO: why is content empty?
            item = ConversationItem(role="assistant", content=[], type="message")
        case "function_call":
            assert len(chunk.choices) == 1
            delta = chunk.choices[0].delta
            assert delta.tool_calls is not None and len(delta.tool_calls) == 1
            tool_call = delta.tool_calls[0]
            assert tool_call.type == "function" and tool_call.function is not None
            assert tool_call.function.name is not None and tool_call.function.arguments is not None

            item = ConversationItem(
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
                # TODO: add `type` and other args
            )
    return item


def tool_call_argument_delta_from_chunk(chunk: ChatCompletionChunk) -> str | None:
    if len(chunk.choices) == 0:
        assert chunk.usage is not None, chunk
        return None
    assert len(chunk.choices) == 1, chunk
    choice = chunk.choices[0]
    delta = choice.delta
    if delta.tool_calls is None:
        assert choice.finish_reason is not None, chunk
        return None

    assert delta.tool_calls is not None and len(delta.tool_calls) == 1, chunk
    tool_call = delta.tool_calls[0]
    assert tool_call.function is not None and tool_call.function.arguments is not None, chunk
    return tool_call.function.arguments
