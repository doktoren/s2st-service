#!/bin/bash

set -euo pipefail

echo "Starting backend using Azure Speech Translation."
echo "Note that port numbers on the test page should be updated to point to port 8003"
uv run uvicorn app.vad_and_translate:app --host 0.0.0.0 --port 8003 --workers 1 --log-level info
echo "uvicorn exited with code $?"
