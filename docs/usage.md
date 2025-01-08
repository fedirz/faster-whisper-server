TODO: break this down into: transcription/translation, streaming transcription/translation, live transcription, audio generation, model listing
TODO: add video demos for all
TODO: add a note about OPENAI_API_KEY

## Curl

```bash
curl http://localhost:8000/v1/audio/transcriptions -F "file=@audio.wav"
```

## Python

=== "httpx"

    ```python
    import httpx

    with open('audio.wav', 'rb') as f:
        files = {'file': ('audio.wav', f)}
        response = httpx.post('http://localhost:8000/v1/audio/transcriptions', files=files)

    print(response.text)
    ```

## OpenAI SDKs

=== "Python"

    ```python
    import httpx

    with open('audio.wav', 'rb') as f:
        files = {'file': ('audio.wav', f)}
        response = httpx.post('http://localhost:8000/v1/audio/transcriptions', files=files)

    print(response.text)
    ```

=== "CLI"

    ```bash
    export OPENAI_BASE_URL=http://localhost:8000/v1/
    export OPENAI_API_KEY="cant-be-empty"
    openai api audio.transcriptions.create -m Systran/faster-whisper-small -f audio.wav --response-format text
    ```

=== "Other"

    See [OpenAI libraries](https://platform.openai.com/docs/libraries) and [OpenAI speech-to-text usage](https://platform.openai.com/docs/guides/speech-to-text).

## Open WebUI

### Using the UI

1. Go to the [Admin Settings](http://localhost:8080/admin/settings) page
2. Click on the "Audio" tab
3. Update settings
   - Speech-to-Text Engine: OpenAI
   - API Base URL: http://faster-whisper-server:8000/v1
   - API Key: does-not-matter-what-you-put-but-should-not-be-empty
   - Model: Systran/faster-distil-whisper-large-v3
4. Click "Save"

### Using environment variables (Docker Compose)

!!! warning

    This doesn't seem to work when you've previously used the UI to set the STT engine.

```yaml
# NOTE: Some parts of the file are omitted for brevity.
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ...
    environment:
      ...
      # Environment variables are documented here https://docs.openwebui.com/getting-started/env-configuration#speech-to-text
      AUDIO_STT_ENGINE: "openai"
      AUDIO_STT_OPENAI_API_BASE_URL: "http://faster-whisper-server:8000/v1"
      AUDIO_STT_OPENAI_API_KEY: "does-not-matter-what-you-put-but-should-not-be-empty"
      AUDIO_STT_MODEL: "Systran/faster-distil-whisper-large-v3"
  faster-whisper-server:
    image: fedirz/faster-whisper-server:latest-cuda
    ...
```
