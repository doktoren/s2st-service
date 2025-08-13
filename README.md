`pyproject.toml` assumes a Linux machine with a AMD Radeon XTX 7900 and rocm 6.3
- this is the machine I'm using for testing and "production".

Requires [uv](https://github.com/astral-sh/uv)

Setup: `uv sync --extra gpu-rocm`

Setup, chatgpt.com/codex: `uv sync --extra cpu`

Run: `./run.sh`
