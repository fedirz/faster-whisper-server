## Docker Compose (Recommended)

TODO: just reference the existing compose file in the repo
=== "CUDA"

    ```yaml
    # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
    services:
      faster-whisper-server:
        image: fedirz/faster-whisper-server:latest-cuda
        name: faster-whisper-server
        restart: unless-stopped
        ports:
          - 8000:8000
        volumes:
          - hf-hub-cache:/home/ubuntu/.cache/huggingface/hub
        deploy:
          resources:
            reservations:
              devices:
                - capabilities: ["gpu"]
    volumes:
      hf-hub-cache:
    ```

=== "CUDA (with CDI feature enabled)"

    ```yaml
    # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
    services:
      faster-whisper-server:
        image: fedirz/faster-whisper-server:latest-cuda
        name: faster-whisper-server
        restart: unless-stopped
        ports:
          - 8000:8000
        volumes:
          - hf-hub-cache:/home/ubuntu/.cache/huggingface/hub
        deploy:
          resources:
            reservations:
              # https://docs.docker.com/reference/cli/dockerd/#enable-cdi-devices
              # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
              devices:
                - driver: cdi
                  device_ids:
                  - nvidia.com/gpu=all
    volumes:
      hf-hub-cache:
    ```

=== "CPU"

    ```yaml
    services:
      faster-whisper-server:
        image: fedirz/faster-whisper-server:latest-cpu
        name: faster-whisper-server
        restart: unless-stopped
        ports:
          - 8000:8000
        volumes:
          - hf-hub-cache:/home/ubuntu/.cache/huggingface/hub
    volumes:
      hf-hub-cache:
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

## Kubernetes

WARNING: it was written few months ago and may be outdated.
Please refer to this [blog post](https://substratus.ai/blog/deploying-faster-whisper-on-k8s)

## Python (requires Python 3.12+ and `uv` package manager)

```bash
git clone https://github.com/fedirz/faster-whisper-server.git
cd faster-whisper-server
uv venv
sourve .venv/bin/activate
uv sync --all-extras
uvicorn --factory --host 0.0.0.0 faster_whisper_server.main:create_app
```
