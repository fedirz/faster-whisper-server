!!! note

    This project was previously named `faster-whisper-server`. I've decided to change the name from `faster-whisper-server`, as the project has evolved to support more than just transcription.

!!! note

    These docs are a work in progress. If you have any questions, suggestions, or find a bug, please create an issue.

TODO: add HuggingFace Space URL

# Speaches

`speaches` is an OpenAI API-compatible server supporting transcription, translation, and speech generation. For transcription/translation it uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and for text-to-speech [piper](https://github.com/rhasspy/piper) is used.

## Features:

- GPU and CPU support.
- [Deployable via Docker Compose / Docker](./installation.md)
- [Highly configurable](./configuration.md)
- OpenAI API compatible. All tools and SDKs that work with OpenAI's API should work with `speaches`.
- Streaming support (transcription is sent via [SSE](https://en.wikipedia.org/wiki/Server-sent_events) as the audio is transcribed. You don't need to wait for the audio to fully be transcribed before receiving it).
- Live transcription support (audio is sent via websocket as it's generated).
- Dynamic model loading / offloading. Just specify which model you want to use in the request and it will be loaded automatically. It will then be unloaded after a period of inactivity.
- [Text-to-speech (TTS) via `piper`]
- (Coming soon) Audio generation (chat completions endpoint) | [OpenAI Documentation](https://platform.openai.com/docs/guides/realtime)
  - Generate a spoken audio summary of a body of text (text in, audio out)
  - Perform sentiment analysis on a recording (audio in, text out)
  - Async speech to speech interactions with a model (audio in, audio out)
- (Coming soon) Realtime API | [OpenAI Documentation](https://platform.openai.com/docs/guides/realtime)

Please create an issue if you find a bug, have a question, or a feature suggestion.

## OpenAI API Compatibility ++

See [OpenAI API reference](https://platform.openai.com/docs/api-reference/audio) for more information.

- Audio file transcription via `POST /v1/audio/transcriptions` endpoint.
  - Unlike OpenAI's API, `speaches` also supports streaming transcriptions (and translations). This is useful for when you want to process large audio files and would rather receive the transcription in chunks as they are processed, rather than waiting for the whole file to be transcribed. It works similarly to chat messages when chatting with LLMs.
- Audio file translation via `POST /v1/audio/translations` endpoint.
- Live audio transcription via `WS /v1/audio/transcriptions` endpoint.
  - LocalAgreement2 ([paper](https://aclanthology.org/2023.ijcnlp-demo.3.pdf) | [original implementation](https://github.com/ufal/whisper_streaming)) algorithm is used for live transcription.
  - Only transcription of a single channel, 16000 sample rate, raw, 16-bit little-endian audio is supported.

TODO: add a note about gradio ui
TODO: add a note about hf space
