# WARN: WIP(code is ugly, may have bugs, test files aren't included, etc.)
# Intro
`speaches` is a webserver that supports real-time transcription using WebSockets.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) is used as the backend. Both GPU and CPU inference is supported.
- LocalAgreement2([paper](https://aclanthology.org/2023.ijcnlp-demo.3.pdf)|[original implementation](https://github.com/ufal/whisper_streaming)) algorithm is used for real-time transcription.
- Can be deployed using Docker (Compose configuration can be found in (compose.yaml[./compose.yaml])).
- All configuration is done through environment variables. See [config.py](./speaches/config.py).
- NOTE: only transcription of single channel, 16000 sample rate, raw, 16-bit little-endian audio is supported.
- NOTE: this isn't really meant to be used as a standalone tool but rather to add transcription features to other applications
Please create an issue if you find a bug, have a question, or a feature suggestion.
# Quick Start
NOTE: You'll need to install [websocat](https://github.com/vi/websocat?tab=readme-ov-file#installation) or an alternative.
Spinning up a `speaches` web-server
```bash
docker run --detach --gpus=all --publish 8000:8000 --mount ~/.cache/huggingface:/root/.cache/huggingface --name speaches fedirz/speaches:cuda
# or
docker run --detach --publish 8000:8000 --mount ~/.cache/huggingface:/root/.cache/huggingface --name speaches fedirz/speaches:cpu
```
Sending audio data via websocket
```bash
arecord -f S16_LE -c1 -r 16000 -t raw -D default | websocat --binary ws://localhost:8000/v1/audio/transcriptions
# or
ffmpeg -f alsa -ac 1 -ar 16000 -sample_fmt s16le -i default | websocat --binary ws://localhost:8000/v1/audio/transcriptions
```
# Example
