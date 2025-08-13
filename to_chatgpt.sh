#!/bin/bash

for f in $(
    find . -type f \
    -not -path "*/.venv/*" \
    -not -path "*/.mypy_cache/*" \
    -not -path "*/.ruff_cache/*" \
    -not -path "*/__pycache__/*" \
    -not -path "*/.egg-info/*" \
    -not -path "*/.git/*" \
    -not -name "tmp.md" \
    -not -name "uv.lock"
)
do
    echo $f
    echo "===== BEGIN ====="
    cat $f
    echo "===== END ====="
done
