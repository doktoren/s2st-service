#!/bin/bash

set -euo pipefail

# Avoid MIOpen workspace warnings when running on ROCm.
export MIOPEN_DEBUG_DISABLE_WORKSPACE=1

uv run uvicorn app.vad:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
echo "uvicorn exited with code $?"
