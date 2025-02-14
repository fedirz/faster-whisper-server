from collections import OrderedDict

from openai.resources.audio import AsyncSpeech, AsyncTranscriptions
from openai.resources.chat.completions import AsyncCompletions

from speaches.realtime.input_audio_buffer import InputAudioBuffer
from speaches.realtime.pubsub import EventPubSub
from speaches.realtime.utils import (
    generate_session_id,
)
from speaches.types.realtime import ConversationItem, RealtimeResponse, Session


class SessionContext:
    def __init__(
        self,
        transcription_client: AsyncTranscriptions,
        completion_client: AsyncCompletions,
        speech_client: AsyncSpeech,
        configuration: Session,
    ) -> None:
        self.transcription_client = transcription_client
        self.speech_client = speech_client
        self.completion_client = completion_client

        self.session_id = generate_session_id()
        self.configuration = configuration

        self.conversation = OrderedDict[
            str, ConversationItem
        ]()  # TODO: should probaly be implemented as a linked list like structure
        self.responses = OrderedDict[str, RealtimeResponse]()
        self.pubsub = EventPubSub()

        input_audio_buffer = InputAudioBuffer()
        self.input_audio_buffers = OrderedDict[str, InputAudioBuffer]({input_audio_buffer.id: input_audio_buffer})
