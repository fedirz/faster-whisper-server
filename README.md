## Faster Whisper Server
`faster-whisper-server` is a web server that supports real-time transcription using WebSockets.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) is used as the backend. Both GPU and CPU inference are supported.
- LocalAgreement2 ([paper](https://aclanthology.org/2023.ijcnlp-demo.3.pdf) | [original implementation](https://github.com/ufal/whisper_streaming)) algorithm is used for real-time transcription.
- Can be deployed using Docker (Compose configuration can be found in [compose.yaml](./compose.yaml)).
- All configuration is done through environment variables. See [config.py](./faster_whisper_server/config.py).
- NOTE: only transcription of single channel, 16000 sample rate, raw, 16-bit little-endian audio is supported.
- NOTE: this isn't really meant to be used as a standalone tool but rather to add transcription features to other applications.
Please create an issue if you find a bug, have a question, or a feature suggestion.
# Quick Start
Using Docker
```bash
docker run --gpus=all --publish 8000:8000 --volume ~/.cache/huggingface:/root/.cache/huggingface fedirz/faster-whisper-server:cuda
# or
docker run --publish 8000:8000 --volume ~/.cache/huggingface:/root/.cache/huggingface fedirz/faster-whisper-server:cpu
```
Using Docker Compose
```bash
curl -sO https://raw.githubusercontent.com/fedirz/faster-whisper-server/master/compose.yaml
docker compose up --detach up faster-whisper-server-cuda
# or
docker compose up --detach up faster-whisper-server-cpu
```
## Usage
Streaming audio data from a microphone. [websocat](https://github.com/vi/websocat?tab=readme-ov-file#installation) installation is required.
```bash
ffmpeg -loglevel quiet -f alsa -i default -ac 1 -ar 16000 -f s16le - | websocat --binary ws://0.0.0.0:8000/v1/audio/transcriptions
# or
arecord -f S16_LE -c1 -r 16000 -t raw -D default 2>/dev/null | websocat --binary ws://0.0.0.0:8000/v1/audio/transcriptions
```
Streaming audio data from a file.
```bash
ffmpeg -loglevel quiet -f alsa -i default -ac 1 -ar 16000 -f s16le - > output.raw
# send all data at once
cat output.raw | websocat --no-close --binary ws://0.0.0.0:8000/v1/audio/transcriptions
# Output: {"text":"One,"}{"text":"One,  two,  three,  four,  five."}{"text":"One,  two,  three,  four,  five."}%
# streaming 16000 samples per second. each sample is 2 bytes
cat output.raw | pv -qL 32000 | websocat --no-close --binary ws://0.0.0.0:8000/v1/audio/transcriptions
# Output: {"text":"One,"}{"text":"One,  two,"}{"text":"One,  two,  three,"}{"text":"One,  two,  three,  four,  five."}{"text":"One,  two,  three,  four,  five.  one."}%
```
Transcribing a file
```bash
# convert the file if it has a different format
ffmpeg -i output.wav -ac 1 -ar 16000 -f s16le output.raw
curl -X POST -F "file=@output.raw" http://0.0.0.0:8000/v1/audio/transcriptions
# Output: "{\"text\":\"One,  two,  three,  four,  five.\"}"%
```
## Roadmap
- [ ] Support file transcription (non-streaming) of multiple formats.
- [ ] CLI client.
- [ ] Separate the web server related code from the "core", and publish "core" as a package.
- [ ] Additional documentation and code comments.
- [ ] Write benchmarks for measuring streaming transcription performance. Possible metrics:
    - Latency (time when transcription is sent - time between when audio has been received)
    - Accuracy (already being measured when testing but the process can be improved)
    - Total seconds of audio transcribed / audio duration (since each audio chunk is being processed at least twice)
- [ ] Get the API response closer to the format used by OpenAI.
- [ ] Integrations...
