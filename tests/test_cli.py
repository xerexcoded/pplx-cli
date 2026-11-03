import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pplx_cli.cli import app
from pplx_cli.config import (
    Config,
    PerplexityModel,
    Provider,
    get_provider_settings,
    load_api_key,
    load_default_provider,
    save_api_key,
)
from pplx_cli.notes import NotesDB


def response_json(content="Mocked response"):
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_config(tmp_path, monkeypatch):
    monkeypatch.setattr("pplx_cli.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("pplx_cli.config.CONFIG_FILE", tmp_path / "config.json")
    for provider in Provider:
        monkeypatch.delenv(get_provider_settings(provider).environment_variable, raising=False)
    Config._instance = None
    yield tmp_path
    Config._instance = None


def test_list_models(runner, mock_config):
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    for model in PerplexityModel:
        assert model.value in result.stdout


def test_list_models_describes_openrouter_native_models(runner, mock_config):
    result = runner.invoke(app, ["list-models", "--provider", "openrouter"])
    assert result.exit_code == 0
    assert "openrouter/free" in result.stdout
    assert "--model <model-id>" in result.stdout


def test_ask_without_api_key(runner, mock_config):
    result = runner.invoke(app, ["ask", "test question"])
    assert result.exit_code == 1
    assert "No Perplexity API key found" in result.output
    assert "PERPLEXITY_API_KEY" in result.output


def test_ask_with_saved_perplexity_api_key(runner, mock_config, requests_mock):
    requests_mock.post(Config.API_ENDPOINT, json=response_json())
    save_api_key("test-api-key")

    result = runner.invoke(app, ["ask", "--no-save-history", "test question"])

    assert result.exit_code == 0
    assert "Mocked response" in result.stdout


def test_perplexity_model_selection_remains_compatible(
    runner, mock_config, requests_mock
):
    requests_mock.post(Config.API_ENDPOINT, json=response_json("Test response"))
    save_api_key("test-api-key")

    result = runner.invoke(
        app, ["ask", "--model", "small", "--no-save-history", "test"]
    )

    assert result.exit_code == 0
    assert requests_mock.last_request.json()["model"] == "sonar"


def test_ask_provider_override_uses_nvidia_environment_key(
    runner, mock_config, monkeypatch, requests_mock
):
    settings = get_provider_settings(Provider.NVIDIA)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
    requests_mock.post(settings.api_endpoint, json=response_json("NVIDIA response"))

    result = runner.invoke(
        app,
        [
            "ask",
            "--provider",
            "nvidia",
            "--model",
            "meta/llama-3.3-70b-instruct",
            "--no-save-history",
            "test",
        ],
    )

    assert result.exit_code == 0
    assert "NVIDIA response" in result.stdout
    assert requests_mock.last_request.json()["model"] == "meta/llama-3.3-70b-instruct"


def test_ask_uses_saved_openrouter_default_and_custom_model(
    runner, mock_config, requests_mock
):
    settings = get_provider_settings(Provider.OPENROUTER)
    save_api_key("openrouter-key", Provider.OPENROUTER)
    requests_mock.post(settings.api_endpoint, json=response_json("OpenRouter response"))

    result = runner.invoke(
        app,
        [
            "ask",
            "--model",
            "openai/gpt-oss-20b",
            "--no-save-history",
            "test",
        ],
    )

    assert result.exit_code == 0
    assert "OpenRouter response" in result.stdout
    assert requests_mock.last_request.json()["model"] == "openai/gpt-oss-20b"


def test_setup_prompts_for_provider_and_saves_its_key(
    runner, mock_config, monkeypatch
):
    monkeypatch.setattr("pplx_cli.cli.get_masked_input", lambda _: "nvidia-key")

    result = runner.invoke(app, ["setup"], input="nvidia\n")

    assert result.exit_code == 0
    assert load_default_provider() == Provider.NVIDIA
    assert load_api_key(Provider.NVIDIA) == "nvidia-key"
    assert "NVIDIA NIM API key saved successfully" in result.stdout


def test_setup_provider_option_bypasses_provider_prompt(
    runner, mock_config, monkeypatch
):
    monkeypatch.setattr("pplx_cli.cli.get_masked_input", lambda _: "router-key")

    result = runner.invoke(app, ["setup", "--provider", "openrouter"])

    assert result.exit_code == 0
    assert load_default_provider() == Provider.OPENROUTER
    assert load_api_key(Provider.OPENROUTER) == "router-key"


def test_ask_notes_routes_to_selected_provider(
    runner, mock_config, monkeypatch, requests_mock
):
    class FakeNotesDB:
        def __init__(self, directory):
            self.directory = directory

        def search_similar_notes(self, query, top_k):
            return [
                (
                    {"id": 1, "title": "Test note", "content": "A saved fact"},
                    0.95,
                )
            ]

    monkeypatch.setattr("pplx_cli.cli.NotesDB", FakeNotesDB)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    settings = get_provider_settings(Provider.OPENROUTER)
    requests_mock.post(settings.api_endpoint, json=response_json("RAG response"))

    result = runner.invoke(
        app,
        [
            "ask-notes",
            "--provider",
            "openrouter",
            "--model",
            "openrouter/free",
            "question",
        ],
    )

    assert result.exit_code == 0
    assert "RAG response" in result.stdout
    assert requests_mock.last_request.json()["model"] == "openrouter/free"


def test_help_command(runner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Perplexity CLI" in result.stdout
    assert "Commands" in result.stdout or "commands" in result.stdout.lower()
    assert "ask" in result.stdout


def test_wiki_commands_are_exposed(runner):
    result = runner.invoke(app, ["wiki", "--help"])
    assert result.exit_code == 0
    for command in ["init", "ingest", "sync", "compile", "query", "mcp"]:
        assert command in result.stdout


def test_wiki_init_creates_open_workspace(runner, tmp_path):
    result = runner.invoke(app, ["wiki", "init", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / ".pplx" / "index.sqlite3").exists()
    assert (tmp_path / "wiki").is_dir()


def test_rag_search_syntax_remains_compatible_with_nested_eval(runner, monkeypatch):
    class EmptySearchEngine:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, *args, **kwargs):
            return []

    monkeypatch.setattr("pplx_cli.cli.get_rag_db", lambda: object())
    monkeypatch.setattr("pplx_cli.cli.HybridSearchEngine", EmptySearchEngine)

    result = runner.invoke(app, ["rag", "semantic retrieval"])

    assert result.exit_code == 0
    assert "No results found" in result.stdout


def test_version_flag(runner):
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Perplexity CLI version" in result.stdout


def test_note_commands_create_list_view_and_report_missing(
    runner, mock_config, monkeypatch
):
    monkeypatch.setattr(NotesDB, "_generate_embedding", lambda self, text: b"")
    directory = str(mock_config)
    created = runner.invoke(
        app,
        [
            "note",
            "--title",
            "Research",
            "--content",
            "Evidence goes here.",
            "--tag",
            "work",
            "--dir",
            directory,
        ],
    )
    listed = runner.invoke(app, ["list-notes", "--tag", "work", "--dir", directory])
    viewed = runner.invoke(app, ["view-note", "1", "--dir", directory])
    missing = runner.invoke(app, ["view-note", "99", "--dir", directory])
    empty = runner.invoke(app, ["list-notes", "--tag", "missing", "--dir", directory])

    assert created.exit_code == 0
    assert "ID: 1" in created.stdout
    assert listed.exit_code == 0 and "Research" in listed.stdout
    assert "Research" not in empty.stdout
    assert viewed.exit_code == 0 and "Evidence goes here." in viewed.stdout
    assert missing.exit_code == 1 and "not found" in missing.stdout
    assert empty.exit_code == 0 and "No notes found" in empty.stdout


def test_rag_stats_formats_search_and_database_configuration(runner, mock_config, monkeypatch):
    class FakeSearchEngine:
        def __init__(self, rag_db):
            assert rag_db == "rag-db"

        def get_search_stats(self):
            return {
                "database_stats": {
                    "total_documents": 3,
                    "by_content_type": {
                        "note": {"chunks": 2, "unique_sources": 1}
                    },
                },
                "search_config": {
                    "default_mode": "hybrid",
                    "rrf_k": 60,
                    "vector_weight": 1.0,
                    "keyword_weight": 1.0,
                },
                "embedding_model": {
                    "model_name": "fake-model",
                    "device": "cpu",
                    "embedding_dim": 4,
                    "quantized": False,
                },
            }

    monkeypatch.setattr("pplx_cli.cli.get_rag_db", lambda: "rag-db")
    monkeypatch.setattr("pplx_cli.cli.HybridSearchEngine", FakeSearchEngine)

    result = runner.invoke(app, ["rag-stats"])

    assert result.exit_code == 0
    assert "Total Documents: 3" in result.stdout
    assert "Chunks: 2" in result.stdout
    assert "Model: fake-model" in result.stdout


def test_rag_index_confirmation_and_result_messages(runner, monkeypatch):
    calls = []

    class FakeIndexer:
        def __init__(self, rag_db):
            assert rag_db == "rag-db"

        def reindex_all(self, clear_first):
            calls.append(clear_first)
            return False

    monkeypatch.setattr("pplx_cli.cli.get_rag_db", lambda: "rag-db")
    monkeypatch.setattr("pplx_cli.cli.BatchIndexer", FakeIndexer)

    cancelled = runner.invoke(app, ["rag-index"], input="n\n")
    completed = runner.invoke(app, ["rag-index", "--clear"], input="y\n")

    assert cancelled.exit_code == 0 and "cancelled" in cancelled.stdout.lower()
    assert completed.exit_code == 0 and "some errors" in completed.stdout.lower()
    assert calls == [True]


def test_rag_config_show_and_update_settings(runner, monkeypatch, tmp_path):
    class FakeEmbeddingModel:
        def get_model_info(self):
            return {
                "model_name": "fake-model",
                "device": "cpu",
                "embedding_dim": 4,
                "quantized": False,
                "cache_size": 10,
                "max_sequence_length": 128,
            }

    monkeypatch.setattr("pplx_cli.cli.get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    shown = runner.invoke(app, ["rag-config", "--show"])
    updated = runner.invoke(
        app,
        [
            "rag-config",
            "--model",
            "small",
            "--device",
            "cpu",
            "--embedding-provider",
            "perplexity",
        ],
    )

    assert shown.exit_code == 0 and "fake-model" in shown.stdout
    assert updated.exit_code == 0 and "Configuration saved" in updated.stdout
    saved = json.loads((tmp_path / ".config/perplexity/rag_config.json").read_text())
    assert saved == {"model": "small", "device": "cpu", "embedding_provider": "perplexity"}
