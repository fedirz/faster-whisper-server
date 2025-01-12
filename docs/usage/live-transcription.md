TODO: live vs streaming

## Live Transcription (using WebSocket)

!!! note

    More content will be added here soon.

From [live-audio](https://github.com/speaches-ai/speaches/tree/master/examples/live-audio) example

<video width="100%" controls>
  <source src="https://github.com/fedirz/faster-whisper-server/assets/76551385/e334c124-af61-41d4-839c-874be150598f" type="video/mp4">
</video>

[websocat](https://github.com/vi/websocat?tab=readme-ov-file#installation) installation is required.
Live transcription of audio data from a microphone.

```bash
ffmpeg -loglevel quiet -f alsa -i default -ac 1 -ar 16000 -f s16le - | websocat --binary ws://localhost:8000/v1/audio/transcriptions
```
