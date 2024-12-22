ARG BASE_IMAGE=nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/fedirz/faster-whisper-server"
# `ffmpeg` is installed because without it `gradio` won't work with mp3(possible others as well) files
# hadolint ignore=DL3008
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg python3.12 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
USER ubuntu
ENV HOME=/home/ubuntu \
    PATH=/home/ubuntu/.local/bin:$PATH
WORKDIR $HOME/faster-whisper-server
COPY --chown=ubuntu --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv
# https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
# https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --compile-bytecode --no-install-project
COPY --chown=ubuntu ./src ./pyproject.toml ./uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --compile-bytecode --extra ui --extra opentelemetry
ENV WHISPER__MODEL=Systran/faster-whisper-large-v3
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
EXPOSE 8000
CMD ["uv", "run", "opentelemetry-instrument", "uvicorn", "--factory", "faster_whisper_server.main:create_app"]
