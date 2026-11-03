# Development

Use the pinned Nix shell for development:

```sh
nix develop
poetry env use "$(command -v python3.12)"
poetry install --with test
PYTHONDONTWRITEBYTECODE=1 poetry run pytest -q
```

The shell defines Apple Silicon/Intel macOS and ARM64/x86-64 Linux environments. Python dependencies come from `poetry.lock`; the shell toolchain comes from `flake.lock`. Linux environments expose the C++ and zlib runtime libraries needed by binary Python wheels. Dependency installation is explicit and requires network access on a cold cache. The Apple Silicon environment has been exercised locally; the other shell outputs still need runtime validation on their target platforms.

Model downloads are separate from dependency installation. Tests use temporary databases and deterministic model doubles unless a test is explicitly marked as a real-model integration check. Do not point tests at personal notes or wiki data.

The [confirmed local research design](design/ollama-local-research.md) defines the Ollama work. The shared `local_only_mode()` scope is an interface for the follow-on retrieval and generation implementations to consult; by itself, it does not enforce network restrictions. Nested scopes cannot relax that flag. The foundation PR does not enable local inference or claim offline enforcement.
