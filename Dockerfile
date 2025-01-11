ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
# hadolint ignore=DL3006
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/fedirz/faster-whisper-server"
LABEL org.opencontainers.image.licenses="MIT"
# `ffmpeg` is installed because without it `gradio` won't work with mp3(possible others as well) files
# hadolint ignore=DL3008
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg python3.12 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# "ubuntu" is the default user on ubuntu images with UID=1000. This user is used for two reasons:
#   1. It's generally a good practice to run containers as non-root users. See https://www.docker.com/blog/understanding-the-docker-user-instruction/
#   2. Docker Spaces on HuggingFace don't support running containers as root. See https://huggingface.co/docs/hub/en/spaces-sdks-docker#permissions
USER ubuntu
ENV HOME=/home/ubuntu \
    PATH=/home/ubuntu/.local/bin:$PATH
WORKDIR $HOME/faster-whisper-server
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --chown=ubuntu --from=ghcr.io/astral-sh/uv:0.5.18 /uv /bin/uv
# https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
# https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
# TODO: figure out if `/home/ubuntu/.cache/uv` should be used instead of `/root/.cache/uv`
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --compile-bytecode --no-install-project
COPY --chown=ubuntu ./src ./pyproject.toml ./uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --compile-bytecode --extra ui
# Creating a directory for the cache to avoid the following error:
# PermissionError: [Errno 13] Permission denied: '/home/ubuntu/.cache/huggingface/hub'
# This error occurs because the volume is mounted as root and the `ubuntu` user doesn't have permission to write to it. Pre-creating the directory solves this issue.
RUN mkdir -p $HOME/.cache/huggingface/hub
ENV WHISPER__MODEL=Systran/faster-whisper-large-v3
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
ENV PATH="$HOME/faster-whisper-server/.venv/bin:$PATH"
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhubenablehftransfer
# NOTE: I've disabled this because it doesn't inside of Docker container. I couldn't pinpoint the exact reason. This doesn't happen when running the server locally.
# RuntimeError: An error occurred while downloading using `hf_transfer`. Consider disabling HF_HUB_ENABLE_HF_TRANSFER for better error handling.
ENV HF_HUB_ENABLE_HF_TRANSFER=0
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#donottrack
# https://www.reddit.com/r/StableDiffusion/comments/1f6asvd/gradio_sends_ip_address_telemetry_by_default/
ENV DO_NOT_TRACK=1
EXPOSE 8000
CMD ["uvicorn", "--factory", "faster_whisper_server.main:create_app"]
