import json
from pathlib import Path

import numpy as np
import pytest

from pplx_cli.api import ChatCompletion
from pplx_cli.wiki import WikiWorkspace


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        def embed(text):
            lowered = str(text).lower()
            return np.array(
                [
                    1.0 if "vector" in lowered else 0.0,
                    1.0 if "citation" in lowered else 0.0,
                    1.0 if "python" in lowered else 0.0,
                    0.5,
                ],
                dtype=np.float32,
            )

        if isinstance(texts, str):
            return embed(texts)
        return np.vstack([embed(text) for text in texts])

    def get_model_info(self):
        return {"provider": "local", "model_name": "fake-bge"}


@pytest.fixture
def workspace(tmp_path):
    return WikiWorkspace(tmp_path, embedding_model=FakeEmbeddingModel())


def test_sync_indexes_markdown_and_is_idempotent(workspace, tmp_path):
    source = tmp_path / "retrieval.md"
    source.write_text(
        "# Retrieval\n\nVector search finds relevant evidence.\n\n"
        "## Citations\n\nAnswers cite original sources.",
        encoding="utf-8",
    )

    assert workspace.sync()["indexed"] == 1
    assert workspace.sync()["unchanged"] == 1
    status = workspace.status()
    assert status["sources_by_type"] == {"markdown": 1}
    assert status["chunks"] == 2

    results = workspace.search("vector citation", limit=2)
    assert len(results) == 2
    assert results[0].title == "Retrieval"
    assert "Heading:" in results[0].locator
    assert "sha256=" in results[0].citation
    source = workspace.get_source(results[0].source_id)
    assert source["chunks"]
    assert source["checksum"] == results[0].checksum


def test_sync_updates_and_deactivates_sources(workspace, tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n\nVector retrieval", encoding="utf-8")
    first = workspace.sync()
    assert first["indexed"] == 1

    source.write_text("# Notes\n\nCitations are mandatory", encoding="utf-8")
    assert workspace.sync()["updated"] == 1
    results = workspace.search("citations", limit=5)
    assert results
    assert all("Vector retrieval" not in result.content for result in results)

    source.unlink()
    assert workspace.sync()["removed"] == 1
    assert workspace.status()["sources"] == 0


def test_ingest_web_page_captures_a_rebuildable_snapshot(workspace, monkeypatch):
    class Response:
        text = "<html><title>Web Evidence</title><body><article>Vector evidence from a web page.</article></body></html>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr("pplx_cli.wiki.workspace.requests.get", lambda *args, **kwargs: Response())
    result = workspace.ingest("https://example.test/evidence")[0]

    assert result.status == "indexed"
    assert list(workspace.web_cache_dir.glob("*.md"))
    assert workspace.search("vector", limit=1)[0].source_type == "web"


def test_explicit_source_outside_workspace_is_supported(workspace, tmp_path):
    external = tmp_path.parent / "external-source.md"
    external.write_text("# External\n\nExternal vector evidence.", encoding="utf-8")
    try:
        result = workspace.ingest(external)[0]
        assert result.status == "indexed"
        assert workspace.search("external vector", limit=1)[0].title == "External"
    finally:
        external.unlink(missing_ok=True)


def test_ingest_pdf_keeps_page_locator_and_tags(workspace, tmp_path, monkeypatch):
    class Page:
        def extract_text(self):
            return "PDF vector evidence."

    class Reader:
        metadata = None
        pages = [Page()]

    monkeypatch.setattr("pypdf.PdfReader", lambda path: Reader())
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"not a real pdf; reader is mocked")

    result = workspace.ingest(pdf, tags=["research"])[0]

    assert result.status == "indexed"
    found = workspace.search("vector", limit=1, tags=["research"])[0]
    assert found.source_type == "pdf"
    assert found.locator == "Page 1"


def test_compile_preserves_manual_content_and_lint_detects_staleness(workspace, tmp_path, monkeypatch):
    (tmp_path / "source.md").write_text("# Source\n\nVector claims need citations.", encoding="utf-8")
    workspace.sync()
    monkeypatch.setattr(
        "pplx_cli.wiki.workspace.query_chat_completion",
        lambda *args, **kwargs: ChatCompletion(content="A cited summary.", model="fake"),
    )

    pages = workspace.compile()
    overview = workspace.wiki_dir / "overview.md"
    overview.write_text(overview.read_text(encoding="utf-8") + "\nMy manual context.\n", encoding="utf-8")
    workspace.compile()

    assert pages
    assert "My manual context." in overview.read_text(encoding="utf-8")
    assert workspace.lint()["errors"] == []

    source = tmp_path / "source.md"
    source.write_text("# Source\n\nA changed claim.", encoding="utf-8")
    workspace.sync()
    assert workspace.lint()["warnings"]


def test_query_uses_raw_source_evidence_and_keeps_web_response_distinct(workspace, tmp_path, monkeypatch):
    (tmp_path / "source.md").write_text("# Source\n\nVector evidence is local.", encoding="utf-8")
    workspace.sync()
    monkeypatch.setattr(workspace, "_rerank", lambda query, results: None)
    completions = iter(
        [
            ChatCompletion(content="Local answer [L1]", model="writer"),
            ChatCompletion(content="Web answer [1]", model="sonar", citations=["https://example.test/web"]),
        ]
    )
    monkeypatch.setattr("pplx_cli.wiki.workspace.query_chat_completion", lambda *args, **kwargs: next(completions))

    answer = workspace.query("What evidence is local?", web=True)

    assert answer.content == "Local answer [L1]"
    assert answer.local_results[0].source_type == "markdown"
    assert answer.web_completion.citations == ["https://example.test/web"]


def test_evaluate_reports_retrieval_metrics(workspace, tmp_path):
    (tmp_path / "source.md").write_text("# Python\n\nPython has a vector ecosystem.", encoding="utf-8")
    workspace.sync()
    source_id = workspace.search("python", limit=1)[0].source_id
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(json.dumps({"query": "python vector", "relevant_source_ids": [source_id]}) + "\n", encoding="utf-8")

    outcome = workspace.evaluate(dataset)
    assert outcome["queries"] == 1
    assert outcome["recall_at_k"] == 1.0
    assert outcome["mrr"] == 1.0


def test_embedding_backend_mismatch_requires_reindex(tmp_path):
    source = tmp_path / "source.md"
    source.write_text("# Source\n\nVector evidence.", encoding="utf-8")
    first = WikiWorkspace(tmp_path, embedding_model=FakeEmbeddingModel())
    first.sync()

    class DifferentEmbeddings(FakeEmbeddingModel):
        def encode(self, texts, **kwargs):
            def embed(text):
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)

            if isinstance(texts, str):
                return embed(texts)
            return np.vstack([embed(text) for text in texts])

        def get_model_info(self):
            return {"provider": "local", "model_name": "different-model"}

    source.write_text("# Source\n\nChanged vector evidence.", encoding="utf-8")
    second = WikiWorkspace(tmp_path, embedding_model=DifferentEmbeddings())
    outcome = second.sync()
    assert outcome["failed"] == 1
    assert "Run a full reindex" in outcome["errors"][0]


def test_unreadable_pdf_is_reported_without_stopping_ingestion(workspace, tmp_path, monkeypatch):
    monkeypatch.setattr("pypdf.PdfReader", lambda path: (_ for _ in ()).throw(ValueError("bad PDF")))
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"bad")

    result = workspace.ingest(pdf)[0]

    assert result.status == "error"
    assert "bad PDF" in result.error
