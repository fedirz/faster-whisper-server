#! /bin/bash

export COMPOSE_FILE=compose.cpu.yaml
docker compose down
docker compose up -d --build
