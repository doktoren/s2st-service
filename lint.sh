#!/bin/bash

set -e

uv pip install --group dev

echo "Running ruff linting..."
uv run ruff check

echo "Running mypy type checking..."
uv run mypy --config-file mypy.ini .
