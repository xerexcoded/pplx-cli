# Local research verification

Verified on 2026-11-03 using the pinned Nix development environment: Python 3.12.15, sentence-transformers 3.4.1, Torch 2.7.1, and Ollama 0.35.1 on an Apple M4 Mac with 24 GB RAM.

## Automated regressions

The integrated suite passed **297 tests** at implementation revision `1dd5402`. Coverage includes hosted-provider compatibility, configuration preservation, rejected unsafe daemon/model states, frozen connection settings, cache-only loaders, source lifecycle, bounded evidence, citation correction, generated-page protection, and CLI integration.

Hosted-provider tests use HTTP/model doubles. They do not claim live hosted-service validation. The latest pull-request test checks are the authoritative results for later revisions.

## Real model workflow

`scripts/verify-local-research.py` completed these commands with the cached `gemma3:4b` model:

| Command | Result |
| --- | --- |
| `ask` | Passed; history saving disabled for the fixture |
| `ask-notes` | Passed with note citations and inspectable evidence |
| `wiki query` | Passed with local source references and excerpts |
| `wiki compile` | Passed with source footnotes and supporting evidence |

Both the temporary daemon and the Python verification process ran under a macOS profile denying external outbound networking and allowing loopback. The verifier first required an `EPERM`/`EACCES` result for an external connection attempt. Hugging Face offline modes were also enabled; BGE-small and MiniLM caches had been prepared explicitly before this run. The daemon reported cloud features disabled.

Fixtures included Markdown, plain text, a PDF containing text, a cached web source, and a real temporary notes database. The web fixture was captured from a local HTTP server, which was stopped before answer generation. Source-file bytes and a pre-existing manual wiki note remained unchanged; wiki lint reported no errors.

All configuration and content fixtures lived in temporary directories. Personal notes, provider settings and chat history were not used as test data. The optional CPU quantization attempt fell back to the unquantized embedding model on this build; this did not affect the workflow results.

## Reproduction

Install dependencies through the [Nix development setup](../development.md). Explicitly prepare the small local embeddings with `rag-config --embedding-provider local --model small` followed by `prepare-models`, and have `gemma3:4b` installed before running the check.

In a dedicated terminal, start a temporary daemon on an unused loopback port:

```sh
sandbox-exec -p '(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))' \
  env OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NO_CLOUD=1 OLLAMA_NOPRUNE=1 ollama serve
```

From the project Nix shell in another terminal:

```sh
sandbox-exec -p '(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))' \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  poetry run python scripts/verify-local-research.py --endpoint http://127.0.0.1:11435 --model gemma3:4b
```

Stop that temporary daemon after verification. This profile is a macOS test method; the application's runtime guarantee remains verification of reported local policy, with OS-level isolation outside the product scope. Citation checks establish identifier integrity, while semantic answer quality still requires evaluation against the intended corpus.
