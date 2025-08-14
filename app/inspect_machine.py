"""
Inspect machine.

```
jesper@doktoren-ThinkPad-T14:~/workspace/s2st-service$ uv run python app/inspect_machine.py
python 3.13.5 (main, Jun 12 2025, 12:40:22) [Clang 20.1.4 ]
torch 2.8.0+rocm6.3 hip 6.3.42131-fa1d09cbd cuda_available True
device Radeon RX 7900 XTX
transformers 4.55.1
SDPBackend: <class 'torch.nn.attention._SDPBackend'>
```
"""

import os
import sys

import torch
import transformers

print("python", sys.version)
print("torch", torch.__version__, "hip", torch.version.hip, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
# SDPA backend visibility (works on ROCm via CUDA APIs)
try:
    from torch.nn.attention import SDPBackend

    print("SDPBackend:", SDPBackend)
except Exception as e:
    print("sdpa_kernel unavailable:", e)


if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("env ENABLE_COMPILE:", os.environ.get("ENABLE_COMPILE"))
print("env ENABLE_AMP:", os.environ.get("ENABLE_AMP"))
