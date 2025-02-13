from __future__ import annotations

from typing import Annotated

from openai.types.beta.realtime import (
    ConversationCreatedEvent,
    ConversationItemContent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
    ConversationItemTruncatedEvent,
    ConversationItemTruncateEvent,
    ErrorEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearedEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferCommittedEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RateLimitsUpdatedEvent,
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
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SessionUpdateEvent,
)
from openai.types.beta.realtime import (
    ConversationItem as OpenAIConversationItem,
)
from openai.types.beta.realtime import (
    ConversationItemCreatedEvent as OpenAIConversationItemCreatedEvent,
)
from openai.types.beta.realtime import (
    RealtimeResponse as OpenAIRealtimeResponse,
)
from openai.types.beta.realtime import (
    RealtimeResponseStatus as RealtimeResponseStatus,  # noqa: PLC0414
)
from openai.types.beta.realtime import (
    Session as OpenAISession,
)
from pydantic import BaseModel, Discriminator
from pydantic.type_adapter import TypeAdapter


# https://github.com/microsoft/pyright/issues/6270
class ConversationItem(OpenAIConversationItem):
    # Make `id` non-nullable
    id: str  # pyright: ignore[reportGeneralTypeIssues, reportIncompatibleVariableOverride]
    # Make `content` non-nullable
    content: list[ConversationItemContent]  # pyright: ignore[reportGeneralTypeIssues, reportIncompatibleVariableOverride]


class ConversationItemCreatedEvent(OpenAIConversationItemCreatedEvent):
    # Change `OpenAIConversationItem` to `ConversationItem`
    item: ConversationItem  # pyright: ignore[reportIncompatibleVariableOverride]
    # TODO: it's not yet clear if this one should be nullable.
    # This needs to be verified by looking at what OpenAI's API does when a response creation is triggered without any messages.
    # Even if this shouldn't be nullable I'm keeping it nullable as I don't have proper handling of previous_item_id yet.
    previous_item_id: str | None  # pyright: ignore[reportIncompatibleVariableOverride]


class RealtimeResponse(OpenAIRealtimeResponse):
    # Make `id` non-nullable
    id: str  # pyright: ignore[reportGeneralTypeIssues, reportIncompatibleVariableOverride]
    # Change `OpenAIConversationItem` to `ConversationItem`
    output: list[ConversationItem] | None = None  # pyright: ignore[reportIncompatibleVariableOverride]


# Same as openai.types.beta.realtime.session_update_event.SessionTurnDetection but with all the fields made non-nullable
class TurnDetection(BaseModel):
    create_response: bool = True
    """Whether or not to automatically generate a response when VAD is enabled.

    `true` by default.
    """

    prefix_padding_ms: int = 300
    """Amount of audio to include before the VAD detected speech (in milliseconds).

    Defaults to 300ms.
    """

    silence_duration_ms: int = 500  # # NOTE: sometimes differs. Other values seen: 800
    """Duration of silence to detect speech stop (in milliseconds).

    Defaults to 500ms. With shorter values the model will respond more quickly, but
    may jump in on short pauses from the user.
    """

    threshold: float = 0.5
    """Activation threshold for VAD (0.0 to 1.0), this defaults to 0.5.

    A higher threshold will require louder audio to activate the model, and thus
    might perform better in noisy environments.
    """

    type: str = "server_vad"
    """Type of turn detection, only `server_vad` is currently supported."""


# Same as openai.types.beta.realtime.session.Session but with `turn_detection` using the model defined above
class Session(OpenAISession):
    turn_detection: TurnDetection = TurnDetection()  # pyright: ignore[reportIncompatibleVariableOverride]


type SessionClientEvent = SessionUpdateEvent  # TODO: session.create
type SessionServerEvent = SessionCreatedEvent | SessionUpdatedEvent


type InputAudioBufferClientEvent = (
    InputAudioBufferAppendEvent | InputAudioBufferCommitEvent | InputAudioBufferClearEvent
)

type InputAudioBufferServerEvent = (
    InputAudioBufferCommittedEvent
    | InputAudioBufferClearedEvent
    | InputAudioBufferSpeechStartedEvent
    | InputAudioBufferSpeechStoppedEvent
)

type ConversationClientEvent = ConversationItemCreateEvent | ConversationItemTruncateEvent | ConversationItemDeleteEvent

type ConversationServerEvent = (
    ConversationCreatedEvent
    | ConversationItemCreatedEvent
    | ConversationItemInputAudioTranscriptionCompletedEvent
    | ConversationItemInputAudioTranscriptionFailedEvent
    | ConversationItemTruncatedEvent
    | ConversationItemDeletedEvent
)


type ResponseClientEvent = ResponseCreateEvent | ResponseCancelEvent


type ResponseServerEvent = (
    ResponseCreatedEvent
    | ResponseOutputItemAddedEvent
    | ResponseContentPartAddedEvent
    | ResponseTextDeltaEvent
    | ResponseTextDoneEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
    | ResponseAudioTranscriptDeltaEvent
    | ResponseAudioTranscriptDoneEvent
    | ResponseAudioDeltaEvent
    | ResponseAudioDoneEvent
    | ResponseContentPartDoneEvent
    | ResponseOutputItemDoneEvent
    | ResponseDoneEvent
)

# https://platform.openai.com/docs/guides/realtime/overview#events
CLIENT_EVENT_TYPES = {
    "session.update",
    "input_audio_buffer.append",
    "input_audio_buffer.commit",
    "input_audio_buffer.clear",
    "conversation.item.create",
    "conversation.item.truncate",
    "conversation.item.delete",
    "response.create",
    "response.cancel",
}
SERVER_EVENT_TYPES = {
    "error",
    "session.created",
    "session.updated",
    "conversation.created",
    "input_audio_buffer.committed",
    "input_audio_buffer.cleared",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "conversation.item.created",
    "conversation.item.input_audio_transcription.completed",
    "conversation.item.input_audio_transcription.failed",
    "conversation.item.truncated",
    "conversation.item.deleted",
    "response.created",
    "response.done",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.text.delta",
    "response.text.done",
    "response.audio_transcript.delta",
    "response.audio_transcript.done",
    "response.audio.delta",
    "response.audio.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "rate_limits.updated",
}

ClientEvent = Annotated[
    SessionUpdateEvent | InputAudioBufferClientEvent | ConversationClientEvent | ResponseClientEvent,
    Discriminator("type"),
]

client_event_type_adapter = TypeAdapter[ClientEvent](ClientEvent)

ServerEvent = Annotated[
    SessionServerEvent
    | InputAudioBufferServerEvent
    | ConversationServerEvent
    | ResponseServerEvent
    | ErrorEvent
    | RateLimitsUpdatedEvent,
    Discriminator("type"),
]

server_event_type_adapter = TypeAdapter[ServerEvent](ServerEvent)

Event = ClientEvent | ServerEvent
