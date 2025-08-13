#!/bin/bash

set -e

# rm -f uv.lock
# uv sync --extra cpu
# uv pip install --group dev
uv sync --group dev --no-default-groups
