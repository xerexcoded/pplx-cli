# Development

Use the pinned Nix shell for development:

```sh
nix develop
poetry env use "$(command -v python3.12)"
poetry install --with test
PYTHONDONTWRITEBYTECODE=1 poetry run pytest -q
```

The shell supports Apple Silicon/Intel macOS and ARM64/x86-64 Linux. Python dependencies come from `poetry.lock`; the shell toolchain comes from `flake.lock`. Dependency installation is explicit and requires network access on a cold cache.

Model downloads are separate from dependency installation. Tests use temporary databases and deterministic model doubles unless a test is explicitly marked as a real-model integration check. Do not point tests at personal notes or wiki data.

The [confirmed local research design](design/ollama-local-research.md) defines the Ollama work. The shared `local_only_mode()` scope carries its no-remote/no-download requirement through retrieval and generation. Nested scopes cannot relax that requirement.
