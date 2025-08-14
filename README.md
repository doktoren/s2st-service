# Seamless Speech-to-Speech Translation Service

This repository hosts a small FastAPI based server that exposes Meta's
SeamlessM4T-V2-Large model over a WebSocket interface.  The goal is to run
the full speech-to-speech translation pipeline locally without depending on
any third-party cloud service.  Clients stream audio frames to the `/ws`
endpoint and receive translated audio in real time.

## Features

* Accepts audio in `g711_ulaw` or `pcm16` with sample rates of 8, 16 or 24 kHz.
* Ingestion strategy: send 20 ms frames and let the server perform VAD based
  segmentation.
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
uv sync --extra gpu-rocm
```

Development tools such as `ruff` and `mypy` live in the `dev` dependency
group.  Install them when contributing code:

```bash
uv pip install --group dev
```

## Intended streaming behavior

* Once the test page is started it opens a WebSocket connection and streams audio in 20 ms chunks to the server.
* The backend performs voice activity detection on the incoming frames.
* When a pause is detected, the buffered speech is forwarded to the model for translation.
* As soon as translated audio is available—even partially—the server streams it back to the browser.  The server may send audio faster than real time; the client should play it at normal speed until finished.
* While a translation is running the server should avoid beginning another one by pausing WebSocket reads or otherwise ensuring that only one translation is in flight.
* Clicking stop on the test page should send roughly one second of silence so the final segment triggers VAD processing.

## Development workflow

* `./lint.sh` – install dev dependencies, run `ruff` and `mypy`.
* The translation implementation lives in `app/translate.py`.
* `run_translate.sh` starts the server with a single worker and verbose logging.
* The vad implementation lives in `app/vad.py`.
* `run_vad.sh` starts the server with a single worker and verbose logging.

Contributions are welcome.  Please ensure that code remains thoroughly typed
and linted and that documentation is updated alongside code changes.


### Target platform

The author has focused on his own platform and in some cases even hard coded values:

* Operating system: Ubuntu 24.04.3 LTS
* GPU: Sapphire Radeon RX7900XTX Pulse Gaming OC 24B
* CPU: 11th Gen Intel Core i5-1135G7 x 8
* RAM: 48 GB

The test server (`cd test && ./run.sh`) assumes that endpoints are exposed using the authors tailscale DNS name:

```
./tailscale_serve.sh
```