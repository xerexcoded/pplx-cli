# Ollama local research

Status: design confirmed by the user on 2026-11-03. Implementation has not started.

## Goal

Use Ollama to answer questions about local documents and CLI notes, with traceable citations for source-backed answers. Build on the existing retrieval and wiki workflows. Ordinary `ask` remains general model interaction; selecting Ollama does not give an ungrounded answer a source citation.

The supporting [landscape research](../research/local-model-cli-landscape-2026-11-03.md) compares LLM, AIChat, Mods, and Vane. The requirements below supersede its initial rollout suggestions.

## User workflows

- Support Ollama through `ask`, `ask-notes`, `wiki query`, and `wiki compile`.
- Support Markdown (`.md` and `.markdown`), plain text (`.txt`), text-based PDFs, existing captured wiki sources, and CLI notes. Plain-text wiki ingestion is a new requirement.
- Keep notes and wiki sources separate. Continue using `ask-notes` and `wiki query` for their respective collections; preserve existing storage and history behavior.
- Preserve current hosted-provider defaults until the user explicitly selects Ollama as the default. Continue supporting per-command provider/model selection.
- Officially support Ollama as the first local runtime. Keep the provider interface extensible without promising other runtimes in this release.

Illustrative future invocation, unavailable until implementation:

```sh
perplexity wiki query "Why did we choose SQLite?" \
  --dir ~/research --provider ollama --model "your-installed-chat-model"
```

Selecting Ollama activates local-only enforcement automatically; users do not need to remember a separate offline flag.

## Setup and configuration

Ollama setup lists installed chat-capable models and lets the user choose one. Persist the selected model and local endpoint, with a suitable configurable timeout for local generation. Local inference must not require a hosted-provider API key. Preserve existing keys and settings when reading or saving the expanded configuration.

Downloads and service startup are deliberate user actions. Provide an explicit preparation path for missing generation or embedding models. Ordinary queries must not pull models, start the daemon, sign in, or change daemon configuration. Missing required components produce actionable errors.

Retain the existing local BGE embeddings for wiki/RAG and the separate MiniLM embeddings for legacy notes initially. An unavailable optional CrossEncoder reranker causes a visible fallback to hybrid retrieval, without a download. Missing essential models stop the affected workflow.

## Local-only contract

Before transmitting user content or running a component that might access a remote service:

1. Require a loopback Ollama endpoint. Prevent HTTP proxies or redirects from rerouting local requests.
2. Verify that the running daemon reports its cloud features disabled. Treat an unavailable, unsupported, malformed, or unrecognized status response as a compatibility error, and stop before sending user content.
3. Verify that the chosen generation model is installed locally and is not a cloud-routed model. A familiar model name or a localhost address alone is insufficient.
4. Require cached local embedding models for the selected workflow. Reject remote embedding providers. Load cached models without network access and without silently changing the selected embedding model.
5. Reject live web research, remote generation, automatic model downloads, and cloud/provider fallback in Ollama-backed workflows.

The official daemon configuration is `OLLAMA_NO_CLOUD=1` or `disable_ollama_cloud: true` in its configuration, followed by a daemon restart. Setting an environment variable only in the CLI does not reconfigure an already-running daemon. [Ollama FAQ](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features)

Current Ollama source exposes `GET /api/status` with `cloud.disabled` and marks the status type experimental. Current model metadata also exposes `remote_model` and `remote_host`. Treat these as version-dependent capabilities, test their response contracts, and reject unverifiable versions rather than guessing. [Routes](https://github.com/ollama/ollama/blob/main/server/routes.go), [response types](https://github.com/ollama/ollama/blob/main/api/types.go), [cloud policy](https://github.com/ollama/ollama/blob/main/internal/cloud/policy.go)

The guarantee covers CLI behavior and verification of daemon-reported local policy. It does not establish OS-level network isolation or prove that an independently managed daemon emits no external traffic. Isolating both processes at the OS/network layer is outside this release's scope.

## Evidence and output

- Source-backed answers use inline references and expose inspectable supporting excerpts with source locators.
- Explicitly report insufficient evidence instead of inventing a supported conclusion.
- Validate that output citation IDs belong to the evidence supplied to the writer. On invalid IDs, make one correction attempt. If they remain invalid, report failure and show the retrieved evidence.
- Citation-ID validation establishes reference integrity, not semantic proof that every claim follows from its source. Document that limit.
- Improve `ask-notes` to meet this evidence contract; its current selected-note list does not provide inline citations.
- Preserve the wiki's existing protections for source files, manual notes, and marked generated sections. Invalid output must not be presented as a successful compilation.

## Context handling

Fit retrieved evidence and generation instructions into the selected model's context budget, reserving room for output. Disclose omitted evidence instead of silently cutting off the assembled prompt. Process large wiki compilations in bounded steps. The same budgeting applies to citation-correction requests.

## Acceptance criteria

- Complete `ask`, `ask-notes`, `wiki query`, and `wiki compile` with a real installed local model while external networking is unavailable.
- Exercise each supported input type, source-backed answer inspection, and the separate notes/wiki workflows.
- Verify local-only enforcement for remote endpoints, proxies/redirects, cloud-enabled or unverifiable daemons, cloud model metadata, remote embeddings, web requests, and missing caches. No user content should be sent before the necessary local checks pass.
- Verify actionable missing-service/model errors and the visible optional-reranker fallback.
- Verify valid citations, insufficient-evidence output, the single correction attempt, and failure with inspectable retrieved evidence.
- Exercise oversized inputs and bounded compilation; preserve source files and manual wiki content.
- Regress existing hosted providers, configuration compatibility, provider defaults, and command-specific persistence behavior.
- Use Nix for any new development environment, following the user's workspace instruction.

These are validation requirements, not results already obtained. A specific model's answer quality and runtime performance still need to be measured during implementation.

## Deferred

Other local runtimes, live web search with local synthesis, OCR/scanned PDFs, additional document formats, unified notes/wiki search, and OS-level network isolation.
