#!/usr/bin/env bash

set -e

# The audio file was copied from the `youtube` example and converted to a raw, single channel, 16000 sample rate, 16-bit little-endian PCM audio file.
# cp ../youtube/the-evolution-of-the-operating-system.mp3 ./audio.mp3
# ffmpeg -y -hide_banner -loglevel quiet -i audio.mp3 -ac 1 -ar 16000 -f s16le -acodec pcm_s16le audio.pcm
# rm -f audio.mp3

export WHISPER_MODEL=Systran/faster-distil-whisper-large-v3 # or Systran/faster-whisper-tiny.en if you are running on a CPU for a faster inference.

# Ensure you have `faster-whisper-server` running. If this is your first time running it expect to wait up-to a minute for the model to be downloaded and loaded into memory. You can run `curl localhost:8000/health` to check if the server is ready or watch the logs with `docker logs -f <container_id>`.
docker run --detach --gpus=all --publish 8000:8000 --volume ~/.cache/huggingface:/root/.cache/huggingface --env WHISPER_MODEL=$WHISPER_MODEL fedirz/faster-whisper-server:latest-cuda
# or you can run it on a CPU
# docker run --detach --publish 8000:8000 --volume ~/.cache/huggingface:/root/.cache/huggingface --env WHISPER_MODEL=$WHISPER_MODEL fedirz/faster-whisper-server:latest-cpu

# `pv` is used to limit the rate at which the audio is streamed to the server. Audio is being streamed at a rate of 32kb/s(16000 sample rate * 16-bit sample / 8 bits per byte = 32000 bytes per second). This emulutes live audio input from a microphone: `ffmpeg -loglevel quiet -f alsa -i default -ac 1 -ar 16000 -f s16le`
# shellcheck disable=SC2002
cat audio.pcm | pv -qL 32000 | websocat --no-close --binary 'ws://localhost:8000/v1/audio/transcriptions?language=en'
