!!! warning

    Under development. I don't yet recommend using these docs as reference for now.

# Faster Whisper Server

`faster-whisper-server` is an OpenAI API-compatible transcription server which uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) as its backend.
Features:

- GPU and CPU support.
- Easily deployable using Docker.
- **Configurable through environment variables (see [config.py](./src/faster_whisper_server/config.py))**.
- OpenAI API compatible.
- Streaming support (transcription is sent via [SSE](https://en.wikipedia.org/wiki/Server-sent_events) as the audio is transcribed. You don't need to wait for the audio to fully be transcribed before receiving it).
- Live transcription support (audio is sent via websocket as it's generated).
- Dynamic model loading / offloading. Just specify which model you want to use in the request and it will be loaded automatically. It will then be unloaded after a period of inactivity.

Please create an issue if you find a bug, have a question, or a feature suggestion.

## OpenAI API Compatibility ++

See [OpenAI API reference](https://platform.openai.com/docs/api-reference/audio) for more information.

- Audio file transcription via `POST /v1/audio/transcriptions` endpoint.
  - Unlike OpenAI's API, `faster-whisper-server` also supports streaming transcriptions (and translations). This is useful for when you want to process large audio files and would rather receive the transcription in chunks as they are processed, rather than waiting for the whole file to be transcribed. It works similarly to chat messages when chatting with LLMs.
- Audio file translation via `POST /v1/audio/translations` endpoint.
- Live audio transcription via `WS /v1/audio/transcriptions` endpoint.
  - LocalAgreement2 ([paper](https://aclanthology.org/2023.ijcnlp-demo.3.pdf) | [original implementation](https://github.com/ufal/whisper_streaming)) algorithm is used for live transcription.
  - Only transcription of a single channel, 16000 sample rate, raw, 16-bit little-endian audio is supported.

TODO: add a note about gradio ui
TODO: add a note about hf space
