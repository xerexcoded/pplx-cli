import json

import pytest
import requests

from pplx_cli import config
from pplx_cli.api import query_chat_completion
from pplx_cli.config import Provider
from pplx_cli.local import local_only_mode
from pplx_cli.ollama import OllamaError, temporary_settings, verify_model


@pytest.fixture
def setup_ollama(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    config.Config._instance = None
    yield
    config.Config._instance = None


def register(mock, status=None, tags=None, show=None):
    root = "http://127.0.0.1:11434"
    mock.get(root + "/api/status", json=status if status is not None else {"cloud": {"disabled": True}})
    mock.get(root + "/api/tags", json=tags if tags is not None else {"models": [{"name": "test:latest"}]})
    mock.post(root + "/api/show", json=show if show is not None else {"capabilities": ["completion"]})
    mock.post(root + "/api/chat", json={"model": "test:latest", "message": {"content": "answer"}})


def test_chat_verifies_before_sending_and_ignores_proxy(setup_ollama, requests_mock, monkeypatch):
    config.save_ollama_settings("http://127.0.0.1:11434", "test", context=8192)
    register(requests_mock)
    monkeypatch.setenv("HTTP_PROXY", "http://malicious.invalid:80")
    result = query_chat_completion("private", provider=Provider.OLLAMA)
    assert result.content == "answer"
    assert [r.path for r in requests_mock.request_history] == ["/api/status", "/api/tags", "/api/show", "/api/chat"]
    assert requests_mock.last_request.json()["options"]["num_ctx"] == 8192


@pytest.mark.parametrize("status", [{}, {"cloud": {}}, {"cloud": {"disabled": False}}, {"cloud": {"disabled": 1}}])
def test_unverifiable_cloud_stops_before_prompt(setup_ollama, requests_mock, status):
    register(requests_mock, status=status)
    with pytest.raises(OllamaError):
        query_chat_completion("secret", model="test", provider=Provider.OLLAMA)
    assert [r.path for r in requests_mock.request_history] == ["/api/status"]


@pytest.mark.parametrize("show", [{}, {"capabilities": ["embedding"]}, {"capabilities": ["completion"], "remote_host": "example.com"}])
def test_unverifiable_or_remote_model_stops_before_prompt(setup_ollama, requests_mock, show):
    register(requests_mock, show=show)
    with pytest.raises(OllamaError):
        query_chat_completion("secret", model="test", provider=Provider.OLLAMA)
    assert all(r.path != "/api/chat" for r in requests_mock.request_history)


@pytest.mark.parametrize("endpoint", ["http://localhost:11434", "http://example.com:11434", "http://127.0.0.1:11434/path", "http://user@127.0.0.1:11434", "https://127.0.0.1:11434"])
def test_reject_nonlocal_endpoint(setup_ollama, endpoint):
    with temporary_settings({"endpoint": endpoint}):
        with pytest.raises(OllamaError):
            verify_model("test")


def test_redirect_refused(setup_ollama, requests_mock):
    requests_mock.get("http://127.0.0.1:11434/api/status", status_code=302, headers={"Location": "http://example.com"})
    with pytest.raises(OllamaError):
        verify_model("test")
    assert len(requests_mock.request_history) == 1


def test_settings_preserve_keys_and_provider(setup_ollama):
    config.save_api_key("key", Provider.NVIDIA)
    config.save_ollama_settings("http://127.0.0.1:11434", "test", make_default=False)
    config.save_api_key("other", Provider.OPENROUTER)
    data = json.loads(config.CONFIG_FILE.read_text())
    assert data["api_keys"]["nvidia"] == "key"
    assert data["ollama"]["model"] == "test"
    assert data["provider"] == "openrouter"


def test_local_policy_blocks_remote_generation(setup_ollama):
    with local_only_mode():
        with pytest.raises(ValueError, match="Remote generation"):
            query_chat_completion("secret", provider=Provider.PERPLEXITY)


def test_cli_verifies_before_constructing_workspace(setup_ollama, monkeypatch):
    from typer.testing import CliRunner
    from pplx_cli.cli import app
    events = []
    def verify(model):
        events.append("verify")
        raise OllamaError("unverified")
    monkeypatch.setattr("pplx_cli.cli.verify_model", verify)
    monkeypatch.setattr("pplx_cli.cli._wiki_workspace", lambda path: events.append("workspace"))
    result = CliRunner().invoke(app, ["wiki", "query", "private", "--dir", "/tmp", "--provider", "ollama", "--model", "test"])
    assert result.exit_code != 0
    assert events == ["verify"]


def test_cli_workspace_runs_inside_local_policy(setup_ollama, monkeypatch):
    from typer.testing import CliRunner
    from pplx_cli.cli import app
    from pplx_cli.local import local_only_enabled
    monkeypatch.setattr("pplx_cli.cli.verify_model", lambda model: None)
    def workspace(path):
        assert local_only_enabled()
        raise RuntimeError("checked")
    monkeypatch.setattr("pplx_cli.cli._wiki_workspace", workspace)
    result = CliRunner().invoke(app, ["wiki", "query", "private", "--dir", "/tmp", "--provider", "ollama", "--model", "test"])
    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "checked"


def test_cli_rejects_web_before_verification(setup_ollama, monkeypatch):
    from typer.testing import CliRunner
    from pplx_cli.cli import app
    monkeypatch.setattr("pplx_cli.cli.verify_model", lambda model: pytest.fail("unexpected verification"))
    result = CliRunner().invoke(app, ["wiki", "query", "private", "--dir", "/tmp", "--provider", "ollama", "--web"])
    assert result.exit_code != 0
    assert "--web is unavailable" in result.output
