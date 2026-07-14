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
