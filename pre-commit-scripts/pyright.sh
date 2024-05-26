#!/usr/bin/env bash
# shellcheck disable=SC1091
source "$(poetry env info --path)"/bin/activate
pyright
