!!! warning

    This feature not supported on ARM devices only x86_64. I was unable to build [piper-phonemize](https://github.com/rhasspy/piper-phonemize)(my [fork](https://github.com/fedirz/piper-phonemize))

https://platform.openai.com/docs/api-reference/audio/createSpeech
https://platform.openai.com/docs/guides/text-to-speech
http://localhost:8001/faster-whisper-server/api/
TODO: add a note about automatic downloads
TODO: add a note about api-key
TODO: add a demo

## Prerequisite

Download the piper voices from [HuggingFace model repository](https://huggingface.co/rhasspy/piper-voices)

```bash
# Download all voices (~15 minutes / 7.7 Gbs)
docker exec -it faster-whisper-server huggingface-cli download rhasspy/piper-voices
# Download all English voices (~4.5 minutes)
docker exec -it faster-whisper-server huggingface-cli download rhasspy/piper-voices --include 'en/**/*' 'voices.json'
# Download all qualities of a specific voice (~4 seconds)
docker exec -it faster-whisper-server huggingface-cli download rhasspy/piper-voices --include 'en/en_US/amy/**/*' 'voices.json'
# Download specific quality of a specific voice (~2 seconds)
docker exec -it faster-whisper-server huggingface-cli download rhasspy/piper-voices --include 'en/en_US/amy/medium/*' 'voices.json'
```

!!! note

    You can find audio samples of all the available voices [here](https://rhasspy.github.io/piper-samples/)

## Curl

```bash
# Generate speech from text using the default values (response_format="mp3", speed=1.0, voice="en_US-amy-medium", etc.)
curl http://localhost:8000/v1/audio/speech --header "Content-Type: application/json" --data '{"input": "Hello World!"}' --output audio.mp3
# Specifying the output format
curl http://localhost:8000/v1/audio/speech --header "Content-Type: application/json" --data '{"input": "Hello World!", "response_format": "wav"}' --output audio.wav
# Specifying the audio speed
curl http://localhost:8000/v1/audio/speech --header "Content-Type: application/json" --data '{"input": "Hello World!", "speed": 2.0}' --output audio.mp3

# List available (downloaded) voices
curl http://localhost:8000/v1/audio/speech/voices
# List just the voice names
curl http://localhost:8000/v1/audio/speech/voices | jq --raw-output '.[] | .voice'
# List just the voices in your language
curl --silent http://localhost:8000/v1/audio/speech/voices | jq --raw-output '.[] | select(.voice | startswith("en")) | .voice'

curl http://localhost:8000/v1/audio/speech --header "Content-Type: application/json" --data '{"input": "Hello World!", "voice": "en_US-ryan-high"}' --output audio.mp3
```

## Python

=== "httpx"

    ```python
    from pathlib import Path

    import httpx

    client = httpx.Client(base_url="http://localhost:8000/")
    res = client.post(
        "v1/audio/speech",
        json={
            "model": "piper",
            "voice": "en_US-amy-medium",
            "input": "Hello, world!",
            "response_format": "mp3",
            "speed": 1,
        },
    ).raise_for_status()
    with Path("output.mp3").open("wb") as f:
        f.write(res.read())
    ```

## OpenAI SDKs

=== "Python"

    ```python
    from pathlib import Path

    from openai import OpenAI

    openai = OpenAI(base_url="http://localhost:8000/v1", api_key="cant-be-empty")
    res = openai.audio.speech.create(
        model="piper",
        voice="en_US-amy-medium",  # pyright: ignore[reportArgumentType]
        input="Hello, world!",
        response_format="mp3",
        speed=1,
    )
    with Path("output.mp3").open("wb") as f:
        f.write(res.response.read())
    ```

=== "Other"

    See [OpenAI libraries](https://platform.openai.com/docs/libraries)
