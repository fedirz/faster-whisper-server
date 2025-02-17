import logging

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionAudioParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsStreaming,
)
from openai.types.shared_params.function_definition import FunctionDefinition

from speaches.types.realtime import ConversationItem, Response

logger = logging.getLogger(__name__)


def create_completion_params(
    model_id: str, messages: list[ChatCompletionMessageParam], response: Response
) -> CompletionCreateParamsStreaming:
    assert response.output_audio_format == "pcm16"  # HACK

    max_tokens = None if response.max_response_output_tokens == "inf" else response.max_response_output_tokens
    tools = [
        ChatCompletionToolParam(
            type=tool.type,
            function=FunctionDefinition(name=tool.name, description=tool.description, parameters=tool.parameters),
        )
        for tool in response.tools
    ]

    return CompletionCreateParamsStreaming(
        model=model_id,
        messages=[
            ChatCompletionSystemMessageParam(
                role="system",
                content=response.instructions,
            ),
            *messages,
        ],
        stream=True,
        modalities=response.modalities,
        audio=ChatCompletionAudioParam(
            voice=response.voice,  # pyright: ignore[reportArgumentType]
            format=response.output_audio_format,
        ),
        temperature=response.temperature,
        max_tokens=max_tokens,
        stream_options=ChatCompletionStreamOptionsParam(include_usage=True),
        **kwargs,
    )


def conversation_item_to_chat_message(  # noqa: PLR0911
    item: ConversationItem,
) -> ChatCompletionMessageParam | None:
    match item.type:
        case "message":
            content_list = item.content
            assert content_list is not None and len(content_list) == 1, item
            content = content_list[0]
            if item.status != "completed":
                logger.warning(f"Item {item} is not completed. Skipping.")
                return None
            match content.type:
                case "text":
                    assert content.text, content
                    return ChatCompletionAssistantMessageParam(role="assistant", content=content.text)
                case "audio":
                    assert content.transcript, content
                    return ChatCompletionAssistantMessageParam(role="assistant", content=content.transcript)
                case "input_text":
                    assert content.text, content
                    return ChatCompletionUserMessageParam(role="user", content=content.text)
                case "input_audio":
                    if not content.transcript:
                        logger.error(f"Conversation item doesn't have a non-empty transcript: {item}")
                        return None
                    return ChatCompletionUserMessageParam(role="user", content=content.transcript)
        case "function_call":
            assert item.call_id and item.name and item.arguments and item.status == "completed", item
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                tool_calls=[
                    ChatCompletionMessageToolCallParam(
                        id=item.call_id,
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
            return ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=item.call_id,
                content=item.output,
            )


def items_to_chat_messages(items: list[ConversationItem]) -> list[ChatCompletionMessageParam]:
    return [
        chat_message
        for chat_message in (conversation_item_to_chat_message(item) for item in items)
        if chat_message is not None
    ]
