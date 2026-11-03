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
    mock.post(root + "/api/show", json=show if show is not None else {"capabilities": ["completion"], "model_info": {"test.context_length": 16384}})
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


@pytest.mark.parametrize("show,reason", [
    ({"model_info": {"test.context_length": 4096}}, "capabilities"),
    ({"capabilities": ["embedding"], "model_info": {"test.context_length": 4096}}, "not chat-capable"),
    ({"capabilities": ["completion"], "model_info": {"test.context_length": 4096}, "remote_host": "example.com"}, "cloud-routed"),
])
def test_unverifiable_or_remote_model_stops_before_prompt(setup_ollama, requests_mock, show, reason):
    register(requests_mock, show=show)
    with pytest.raises(OllamaError, match=reason):
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
    assert "Ollama error: unverified" in result.output


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
    assert result.exit_code == 2
    assert "web" in result.output and "unavailable" in result.output


@pytest.mark.parametrize("marker", [[], {}, True, None, 42])
def test_malformed_remote_markers_fail_closed(setup_ollama, requests_mock, marker):
    register(requests_mock, tags={"models": [{"name": "test:latest", "remote_host": marker}]})
    with pytest.raises(OllamaError, match="Malformed Ollama remote_host"):
        verify_model("test")
    assert [r.path for r in requests_mock.request_history] == ["/api/status", "/api/tags"]


def test_discovery_skips_cloud_without_show_and_propagates_malformed(setup_ollama, requests_mock):
    from pplx_cli.ollama import list_chat_models
    register(requests_mock, tags={"models": [{"name": "cloud:latest", "remote_host": "cloud.invalid"}, {"name": "test:latest"}]}, show={"capabilities": "invalid"})
    with pytest.raises(OllamaError, match="capabilities"):
        list_chat_models()
    shows = [r.json()["model"] for r in requests_mock.request_history if r.path == "/api/show"]
    assert shows == ["test:latest"]


def test_context_clamps_to_model_and_bounds_output(setup_ollama, requests_mock):
    from pplx_cli.ollama import get_context_budget, get_output_budget
    config.save_ollama_settings("http://127.0.0.1:11434", "test", context=8192)
    register(requests_mock, show={"capabilities": ["completion"], "model_info": {"test.context_length": 2048}})
    assert get_context_budget("test") == 2048
    assert get_output_budget("test") == 512
    query_chat_completion("private", model="test", provider=Provider.OLLAMA)
    assert requests_mock.last_request.json()["options"] == {"num_ctx": 2048, "num_predict": 512}


@pytest.mark.parametrize("model_info", [None, {}, {"test.context_length": "4096"}, {"test.context_length": True}])
def test_missing_model_context_fails_before_prompt(setup_ollama, requests_mock, model_info):
    register(requests_mock, show={"capabilities": ["completion"], "model_info": model_info})
    with pytest.raises(OllamaError, match="context metadata"):
        query_chat_completion("secret", model="test", provider=Provider.OLLAMA)
    assert all(r.path != "/api/chat" for r in requests_mock.request_history)


def test_cli_lists_verified_models_and_saved_default(setup_ollama, monkeypatch):
    from typer.testing import CliRunner
    from pplx_cli.cli import app
    config.save_ollama_settings("http://127.0.0.1:11434", "saved", make_default=False)
    monkeypatch.setattr("pplx_cli.cli.list_chat_models", lambda: ["test:latest"])
    result = CliRunner().invoke(app, ["list-models", "--provider", "ollama"])
    assert result.exit_code == 0
    assert "saved" in result.output and "test:latest" in result.output


@pytest.mark.parametrize("arguments", [
    ["ask", "private", "--no-save-history"],
    ["ask-notes", "private"],
    ["wiki", "query", "private", "--dir", "/tmp"],
    ["wiki", "compile", "--dir", "/tmp"],
])
def test_cli_verification_failure_is_actionable(setup_ollama, monkeypatch, arguments):
    from typer.testing import CliRunner
    from pplx_cli.cli import app
    def fail(model):
        raise OllamaError("Start the daemon with cloud disabled")
    monkeypatch.setattr("pplx_cli.cli.verify_model", fail)
    result = CliRunner().invoke(app, arguments + ["--provider", "ollama", "--model", "test"])
    assert result.exit_code == 1
    assert "Start the daemon with cloud disabled" in result.output
    assert "Traceback" not in result.output


def test_session_disables_proxy_environment(monkeypatch):
    from pplx_cli import ollama
    class FakeSession:
        def __init__(self):
            self.trust_env = True
    monkeypatch.setattr(ollama.requests, "Session", FakeSession)
    assert ollama._session().trust_env is False


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), True, 0])
def test_invalid_timeout_rejected_at_save_and_use(setup_ollama, timeout):
    with pytest.raises(ValueError):
        config.save_ollama_settings("http://127.0.0.1:11434", "test", timeout=timeout)
    with temporary_settings({"timeout": timeout}):
        with pytest.raises(OllamaError, match="timeout"):
            verify_model("test")


def test_local_settings_preserve_hosted_defaults_and_endpoint(setup_ollama):
    config.save_api_key("secret", Provider.NVIDIA)
    config.save_ollama_settings("http://127.0.0.2:1234", "model:latest", timeout=240, context=8192, make_default=False)
    config.Config._instance = None
    reloaded = config.Config.get_instance()
    assert reloaded.provider == Provider.NVIDIA
    assert reloaded.get_api_key(Provider.NVIDIA) == "secret"
    assert reloaded.get_api_endpoint(Provider.OLLAMA) == "http://127.0.0.2:1234"
    assert reloaded.get_default_model(Provider.OLLAMA) == "model:latest"
    assert config.load_ollama_settings()["timeout"] == 240
    assert config.load_ollama_settings()["context"] == 8192


@pytest.mark.parametrize("arguments, constructor", [
    (["ask-notes", "secret"], "NotesDB"),
    (["wiki", "query", "secret", "--dir", "/tmp"], "_wiki_workspace"),
    (["wiki", "compile", "--dir", "/tmp"], "_wiki_workspace"),
])
def test_verify_precedes_retrieval_construction(setup_ollama, monkeypatch, arguments, constructor):
    from typer.testing import CliRunner
    from pplx_cli.cli import app
    def fail(model):
        raise OllamaError("daemon unavailable")
    monkeypatch.setattr("pplx_cli.cli.verify_model", fail)
    monkeypatch.setattr("pplx_cli.cli." + constructor, lambda *args: pytest.fail("constructed before verification"))
    result = CliRunner().invoke(app, arguments + ["--provider", "ollama", "--model", "test"])
    assert result.exit_code == 1
    assert "daemon unavailable" in result.output


def test_cloud_tag_rejected_before_show(setup_ollama, requests_mock):
    register(requests_mock, tags={"models": [{"name": "test:latest", "remote_model": "model", "remote_host": "cloud.example"}]})
    with pytest.raises(OllamaError, match="cloud-routed"):
        verify_model("test")
    assert [request.path for request in requests_mock.request_history] == ["/api/status", "/api/tags"]


def test_incomplete_generation_is_an_error(setup_ollama, requests_mock):
    register(requests_mock)
    requests_mock.post("http://127.0.0.1:11434/api/chat", json={"done_reason": "length", "message": {"content": "incomplete"}})
    with pytest.raises(OllamaError, match="generation limit.*incomplete"):
        query_chat_completion("private", model="test", provider=Provider.OLLAMA)


def test_oversized_utf8_input_rejected_before_chat(setup_ollama, requests_mock):
    config.save_ollama_settings("http://127.0.0.1:11434", "test", context=1024)
    register(requests_mock)
    with pytest.raises(OllamaError, match="Input is too large"):
        query_chat_completion("é" * 400, model="test", provider=Provider.OLLAMA)
    assert all(request.path != "/api/chat" for request in requests_mock.request_history)


@pytest.mark.parametrize("model,context", [(None, 4096), (42, 4096), ("test", True), ("test", 4096.0)])
def test_invalid_save_types_rejected(setup_ollama, model, context):
    with pytest.raises(ValueError):
        config.save_ollama_settings("http://127.0.0.1:11434", model, context=context)


def test_discovery_skips_models_without_usable_context(setup_ollama, requests_mock):
    from pplx_cli.ollama import list_chat_models
    register(requests_mock, tags={"models": [{"name": "odd:latest"}, {"name": "test:latest"}]})
    requests_mock.post("http://127.0.0.1:11434/api/show", [
        {"json": {"capabilities": ["completion"], "model_info": {}}},
        {"json": {"capabilities": ["completion"], "model_info": {"test.context_length": 4096}}},
    ])
    assert list_chat_models() == ["test:latest"]


def test_unqualified_name_uses_verified_canonical_name(setup_ollama, requests_mock):
    register(requests_mock)
    query_chat_completion("hello", model="test", provider=Provider.OLLAMA)
    assert [r.json()["model"] for r in requests_mock.request_history if r.path in ("/api/show", "/api/chat")] == ["test:latest", "test:latest"]
