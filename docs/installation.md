## Docker Compose (Recommended)

!!! note

    I'm using newer Docker Compsose features. If you are using an older version of Docker Compose, you may need need to update.

Download the necessary Docker Compose files

=== "CUDA"

    ```bash
    curl --silent --remote-name https://raw.githubusercontent.com/fedirz/faster-whisper-server/master/compose.yaml
    curl --silent --remote-name https://raw.githubusercontent.com/fedirz/faster-whisper-server/master/compose.cuda.yaml
    export COMPOSE_FILE=compose.cuda.yaml
    ```

=== "CUDA (with CDI feature enabled)"

    ```bash
    curl --silent --remote-name https://raw.githubusercontent.com/fedirz/faster-whisper-server/master/compose.yaml
    curl --silent --remote-name https://raw.githubusercontent.com/fedirz/faster-whisper-server/master/compose.cuda.yaml
    curl --silent --remote-name https://raw.githubusercontent.com/fedirz/faster-whisper-server/master/compose.cuda-cdi.yaml
    export COMPOSE_FILE=compose.cuda-cdi.yaml
    ```

=== "CPU"

    ```bash
    curl --silent --remote-name https://raw.githubusercontent.com/fedirz/faster-whisper-server/master/compose.yaml
    curl --silent --remote-name https://raw.githubusercontent.com/fedirz/faster-whisper-server/master/compose.cpu.yaml
    export COMPOSE_FILE=compose.cpu.yaml
    ```

Start the service

```bash
docker compose up --detach
```

??? note "Build from source"

    ```bash
    # NOTE: you need to install and enable [buildx](https://github.com/docker/buildx) for multi-platform builds

    # Build image with CUDA support
    docker compose --file compose.cuda.yaml build

    # Build image without CUDA support
    docker compose --file compose.cpu.yaml build
    ```

## Docker

=== "CUDA"

    ```bash
    docker run \
      --rm \
      --detach \
      --publish 8000:8000 \
      --name faster-whisper-server \
      --volume hf-hub-cache:/home/ubuntu/.cache/huggingface/hub \
      --gpus=all \
      fedirz/faster-whisper-server:latest-cuda
    ```

=== "CUDA (with CDI feature enabled)"

    ```bash
    docker run \
      --rm \
      --detach \
      --publish 8000:8000 \
      --name faster-whisper-server \
      --volume hf-hub-cache:/home/ubuntu/.cache/huggingface/hub \
      --device=nvidia.com/gpu=all \
      fedirz/faster-whisper-server:latest-cuda
    ```

=== "CPU"

    ```bash
    docker run \
      --rm \
      --detach \
      --publish 8000:8000 \
      --name faster-whisper-server \
      --volume hf-hub-cache:/home/ubuntu/.cache/huggingface/hub \
      fedirz/faster-whisper-server:latest-cpu
    ```

??? note "Build from source"

    ```bash
    docker build --tag faster-whisper-server .

    # NOTE: you need to install and enable [buildx](https://github.com/docker/buildx) for multi-platform builds
    # Build image for both amd64 and arm64
    docker buildx build --tag faster-whisper-server --platform linux/amd64,linux/arm64 .

    # Build image without CUDA support
    docker build --tag faster-whisper-server --build-arg BASE_IMAGE=ubuntu:24.04 .
    ```

## Python (requires Python 3.12+ and `uv` package manager)

```bash
git clone https://github.com/fedirz/faster-whisper-server.git
cd faster-whisper-server
uv venv
sourve .venv/bin/activate
uv sync --all-extras
uvicorn --factory --host 0.0.0.0 faster_whisper_server.main:create_app
```
