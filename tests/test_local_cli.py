from types import SimpleNamespace

import pytest
import numpy as np
from typer.testing import CliRunner

from pplx_cli import config
from pplx_cli.cli import app
from pplx_cli.config import Config, save_ollama_settings
from pplx_cli.notes import NotesDB


@pytest.fixture
def local_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    Config._instance = None
    save_ollama_settings("http://127.0.0.1:11435", "test-local")
    monkeypatch.setattr("pplx_cli.cli.verify_model", lambda model: "test-local")
    yield CliRunner(), tmp_path
    Config._instance = None


def test_notes_citations_and_excerpts_without_changing_storage(local_cli, monkeypatch):
    runner, directory = local_cli
    model = SimpleNamespace(encode=lambda *a, **kw: np.array([1.0, 0.0], dtype=np.float32))
    monkeypatch.setattr(NotesDB, "model", property(lambda self: model))
    db = NotesDB(directory)
    note_id = db.add_note("Storage", "SQLite keeps the ledger local.")
    before = db.db_path.read_bytes()
    monkeypatch.setattr("pplx_cli.cli.get_context_budget", lambda model: 8192)
    monkeypatch.setattr("pplx_cli.cli.get_output_budget", lambda model: 1024)
    monkeypatch.setattr("pplx_cli.cli.query_chat", lambda *args, **kwargs: f"The ledger stays local. [N{note_id}]")
    result = runner.invoke(app, ["ask-notes", "Why SQLite?", "--dir", str(directory)])
    assert result.exit_code == 0, result.output
    assert f"[N{note_id}]" in result.output
    assert f"notes.db#note-{note_id}" in result.output
    assert "SQLite keeps the ledger local." in result.output
    assert db.db_path.read_bytes() == before


@pytest.mark.parametrize("command", ["ask-notes", "wiki query", "wiki compile"])
def test_missing_cached_models_are_actionable(local_cli, monkeypatch, command):
    runner, directory = local_cli
    def unavailable(*args, **kwargs):
        raise RuntimeError("Required embedding model is missing; run prepare-models.")
    monkeypatch.setattr("pplx_cli.cli.NotesDB", unavailable)
    monkeypatch.setattr("pplx_cli.cli._wiki_workspace", unavailable)
    arguments = command.split()
    if command != "wiki compile":
        arguments.append("question")
    result = runner.invoke(app, arguments + ["--dir", str(directory)])
    assert result.exit_code == 1
    assert "prepare-models" in result.output
    assert "Traceback" not in result.output


def test_wiki_answer_prints_supplied_excerpt_and_omissions(local_cli, monkeypatch):
    runner, directory = local_cli
    answer = SimpleNamespace(
        content="A local answer. [L1]",
        local_results=[SimpleNamespace(citation="[L1] file:///notes.md#L1", content="Evidence excerpt.")],
        omitted_evidence=2, web_completion=None,
    )
    monkeypatch.setattr("pplx_cli.cli._wiki_workspace", lambda path: SimpleNamespace(query=lambda *a, **kw: answer))
    result = runner.invoke(app, ["wiki", "query", "question", "--dir", str(directory)])
    assert result.exit_code == 0, result.output
    assert "Evidence excerpt." in result.output
    assert "Omitted 2" in result.output


def test_preparation_is_explicit_and_reports_models(local_cli, monkeypatch):
    runner, _ = local_cli
    requested = []
    monkeypatch.setattr("pplx_cli.preparation.prepare_local_models", lambda include_reranker: requested.append(include_reranker) or ["BGE", "MiniLM"])
    result = runner.invoke(app, ["prepare-models", "--include-reranker"])
    assert result.exit_code == 0, result.output
    assert requested == [True]
    assert "Prepared: BGE" in result.output


@pytest.mark.parametrize("arguments", [["wiki", "init"], ["wiki", "sync"], ["wiki", "search", "question"], ["wiki", "ingest", "source.txt"]])
def test_default_local_indexing_never_starts_outside_cache_policy(local_cli, monkeypatch, arguments):
    from pplx_cli.local import local_only_enabled
    runner, directory = local_cli
    def inspect_policy(path):
        assert local_only_enabled()
        raise RuntimeError("cache policy checked")
    monkeypatch.setattr("pplx_cli.cli._wiki_workspace", inspect_policy)
    result = runner.invoke(app, arguments + ["--dir", str(directory)])
    assert result.exit_code == 1
    assert "cache policy checked" in result.output


def test_default_local_ingest_rejects_live_urls_before_models(local_cli, monkeypatch):
    runner, directory = local_cli
    monkeypatch.setattr("pplx_cli.cli._wiki_workspace", lambda path: pytest.fail("must reject before workspace access"))
    result = runner.invoke(app, ["wiki", "ingest", "https://example.com/article", "--dir", str(directory)])
    assert result.exit_code == 2
    assert "Live URL capture" in result.output


def test_invalid_citations_show_evidence_explicitly(local_cli, monkeypatch):
    from pplx_cli.evidence import CitationError, Evidence
    runner, _ = local_cli
    item = Evidence("N1", "notes.db#note-1", "Inspectable private note excerpt.")
    error = CitationError("Invalid citations.", [item])
    assert item.excerpt not in str(error)
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr("pplx_cli.cli._wiki_workspace", fail)
    result = runner.invoke(app, ["wiki", "query", "question", "--dir", "/tmp"])
    assert result.exit_code == 1
    assert "Retrieved evidence:" in result.output
    assert item.locator in result.output
    assert item.excerpt in result.output
