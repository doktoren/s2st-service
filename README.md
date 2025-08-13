# Seamless Speech-to-Speech Translation Service

This repository hosts a small FastAPI based server that exposes Meta's
SeamlessM4T-V2-Large model over a WebSocket interface.  The goal is to run
the full speech-to-speech translation pipeline locally without depending on
any third-party cloud service.  Clients stream audio frames to the `/ws`
endpoint and receive translated audio in real time.

## Features

* Accepts audio in `g711_ulaw` or `pcm16` with sample rates of 8, 16 or 24 kHz.
* Two ingestion strategies:
  * `server_vad` – send 20 ms frames and let the server perform VAD based
    segmentation.
  * `client_chunked` – the client explicitly signals when a segment ends.
* Streams back translated audio chunks followed by an `end_of_audio` marker.
* Python 3.13 with strict typing, `ruff` linting and `mypy` type checking.
* Runs on CPU only or on ROCm 6.3 GPUs.

## Installing dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency
management.  First install `uv` and then choose a backend:

```bash
# CPU only
uv sync --extra cpu

# AMD ROCm 6.3 GPU
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/rocm6.3 \
    uv sync --extra gpu-rocm
```

Development tools such as `ruff` and `mypy` live in the `dev` dependency
group.  Install them when contributing code:

```bash
uv pip install --group dev
```

## Running the server

The `run.sh` script launches `uvicorn` and also suppresses noisy MIOpen
workspace warnings on ROCm systems:

```bash
./run.sh
```

Once running, connect a client to `ws://localhost:8000/ws` and start sending
audio messages as described in `app/main.py`.

## Development workflow

* `./lint.sh` – install dev dependencies, run `ruff` and `mypy`.
* The main implementation lives in `app/main.py`.
* `run.sh` starts the server with a single worker and verbose logging.

Contributions are welcome.  Please ensure that code remains thoroughly typed
and linted and that documentation is updated alongside code changes.

