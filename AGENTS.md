# AI Development Guidelines

This project includes AI components. Follow these best practices to ensure consistency, efficiency, and ethical AI development.

Ignore any TODO comments unless you've been told otherwise.

## General Principles

Always generate code for the version of Python and 3rd party packages specified in `./pyproject.toml`
Prefer using syntax from newer versions. For example prefer `my_list: list[str]` over `my_list: List[str]`.

If possible, you should verify your changes by running `./lint.sh`.
Note that `./lint.sh` may take a long time to finish during its first invocation.
You may choose to only do the first part (`uv run ruff check`).

The code must pass strict type checking (`strict = True`). See `./mypy.ini` for details.
This means that you must add full type annotations to e.g. function arguments and return types.
That is; `list` is not good enough. Use e.g. `list[str]`.
When you're unable to specify an exact type prefer `object` over `Any` whenever the object type suffices.

The code must also pass strict linting. See `./ruff.toml` for details.X

Ensure that all modules, classes, methods and functions have concise and meaningful docstrings.
Even when e.g. the function name makes the usage clear you need to add this repeated information to please the linting tool.

It is not enought for code to be correct. The code must be as pythonic as possible.
If it's possible to simplify then do so.

## Testing

You can assume that `chatgpt_codex_setup_script.sh` has been executed successfully.
Note that this is NOT installing all dependencies for the Python backend.
Therefore, do NOT try to run the python program - only do linting.
