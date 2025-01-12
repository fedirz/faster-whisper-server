!!! note

    This project was previously named `faster-whisper-server`. I've decided to change the name from `faster-whisper-server`, as the project has evolved to support more than just transcription.

!!! note

    These docs are a work in progress.

# Speaches

`speaches` is an OpenAI API-compatible server supporting streaming transcription, translation, and speech generation. Speach-to-Text is powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and for Text-to-Speech [piper](https://github.com/rhasspy/piper) and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) are used. This project aims to be Ollama, but for TTS/STT models.

Try it out on the [HuggingFace Space](https://huggingface.co/spaces/speaches-ai/speaches)

## Features:

- GPU and CPU support.
- [Deployable via Docker Compose / Docker](https://speaches-ai.github.io/speaches/installation/)
- [Highly configurable](https://speaches-ai.github.io/speaches/configuration/)
- OpenAI API compatible. All tools and SDKs that work with OpenAI's API should work with `speaches`.
- Streaming support (transcription is sent via SSE as the audio is transcribed. You don't need to wait for the audio to fully be transcribed before receiving it).

  - LocalAgreement2 ([paper](https://aclanthology.org/2023.ijcnlp-demo.3.pdf) | [original implementation](https://github.com/ufal/whisper_streaming)) algorithm is used for live transcription.

- Live transcription support (audio is sent via websocket as it's generated).
- Dynamic model loading / offloading. Just specify which model you want to use in the request and it will be loaded automatically. It will then be unloaded after a period of inactivity.
- Text-to-Speech via `kokoro`(Ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper` models.
- [Coming soon](https://github.com/speaches-ai/speaches/issues/231): Audio generation (chat completions endpoint) | [OpenAI Documentation](https://platform.openai.com/docs/guides/realtime)
  - Generate a spoken audio summary of a body of text (text in, audio out)
  - Perform sentiment analysis on a recording (audio in, text out)
  - Async speech to speech interactions with a model (audio in, audio out)
- [Coming soon](https://github.com/speaches-ai/speaches/issues/115): Realtime API | [OpenAI Documentation](https://platform.openai.com/docs/guides/realtime)

Please create an issue if you find a bug, have a question, or a feature suggestion.

## Demo

### Streaming Transcription

TODO

### Speech Generation

TODO

### Live Transcription (using WebSockets)

<video width="100%" controls>
  <source src="https://github.com/fedirz/faster-whisper-server/assets/76551385/e334c124-af61-41d4-839c-874be150598f" type="video/mp4">
</video>
