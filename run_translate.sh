#!/bin/bash

set -euo pipefail

# Avoid MIOpen workspace warnings when running on ROCm.
export MIOPEN_DEBUG_DISABLE_WORKSPACE=1

export HSA_ENABLE_SDMA=0
export MIOPEN_FIND_MODE=1        # stable conv solution finding
export MIOPEN_DISABLE_CACHE=1    # avoid stale find-db

# Force PyTorch to avoid SDPA fast paths that can hit bad kernels on AMD
export PYTORCH_SDP_DISABLE_FLASH_ATTENTION=1
export PYTORCH_SDP_DISABLE_MEM_EFFICIENT=1   # fall back to math SDPA

# If you have xformers or Triton installed, force-disable their attention kernels
export XFORMERS_FORCE_DISABLE_TRITON=1
export XFORMERS_DISABLE_FLASH_ATTN=1

uv run uvicorn app.translate:app --host 0.0.0.0 --port 8001 --workers 1 --log-level debug
echo "uvicorn exited with code $?"

