import json
from contextlib import contextmanager

import pytest
import requests

from pplx_cli import config
from pplx_cli.api import query_chat_completion
from pplx_cli.config import Provider
from pplx_cli.local import local_only_mode
from pplx_cli.ollama import OllamaError, temporary_settings, verify_model


@contextmanager
def _mock_operation(fn, model):
    fn(model)
    yield


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
    monkeypatch.setattr("pplx_cli.cli.verified_operation", lambda model: _mock_operation(verify, model))
    monkeypatch.setattr("pplx_cli.cli._wiki_workspace", lambda path: events.append("workspace"))
    result = CliRunner().invoke(app, ["wiki", "query", "private", "--dir", "/tmp", "--provider", "ollama", "--model", "test"])
    assert result.exit_code != 0
    assert events == ["verify"]
    assert "Error: unverified" in result.output


def test_cli_workspace_runs_inside_local_policy(setup_ollama, monkeypatch):
    from typer.testing import CliRunner
    from pplx_cli.cli import app
    from pplx_cli.local import local_only_enabled
    monkeypatch.setattr("pplx_cli.cli.verified_operation", lambda model: _mock_operation(lambda _: None, model))
    def workspace(path):
        assert local_only_enabled()
        raise RuntimeError("checked")
    monkeypatch.setattr("pplx_cli.cli._wiki_workspace", workspace)
    result = CliRunner().invoke(app, ["wiki", "query", "private", "--dir", "/tmp", "--provider", "ollama", "--model", "test"])
    assert result.exit_code == 1
    assert "checked" in result.output
    assert isinstance(result.exception, SystemExit)


def test_cli_rejects_web_before_verification(setup_ollama, monkeypatch):
    from typer.testing import CliRunner
    from pplx_cli.cli import app
    monkeypatch.setattr("pplx_cli.cli.verified_operation", lambda model: _mock_operation(lambda _: pytest.fail("unexpected verification"), model))
    result = CliRunner().invoke(app, ["wiki", "query", "private", "--dir", "/tmp", "--provider", "ollama", "--web"])
    assert result.exit_code == 2
    assert "web" in result.output and "unavailable" in result.output


@pytest.mark.parametrize("marker", [[], {}, True, None, 42])
def test_malformed_remote_markers_fail_closed(setup_ollama, requests_mock, marker):
    register(requests_mock, tags={"models": [{"name": "test:latest", "remote_host": marker}]})
    with pytest.raises(OllamaError, match="Malformed Ollama remote_host"):
        verify_model("test")
    assert [r.path for r in requests_mock.request_history] == ["/api/status", "/api/tags"]


def test_discovery_skips_cloud_without_show_and_warns_for_malformed(setup_ollama, requests_mock, caplog):
    from pplx_cli.ollama import list_chat_models
    register(requests_mock, tags={"models": [{"name": "cloud:latest", "remote_host": "cloud.invalid"}, {"name": "test:latest"}]}, show={"capabilities": "invalid"})
    with caplog.at_level("WARNING"):
        assert list_chat_models() == []
    shows = [r.json()["model"] for r in requests_mock.request_history if r.path == "/api/show"]
    assert shows == ["test:latest"]
    assert "cloud:latest" not in caplog.text
    assert "test:latest" in caplog.text


def test_context_clamps_to_model_and_bounds_output(setup_ollama, requests_mock):
    from pplx_cli.ollama import get_context_budget, get_output_budget
    config.save_ollama_settings("http://127.0.0.1:11434", "test", context=8192)
    register(requests_mock, show={"capabilities": ["completion"], "model_info": {"test.context_length": 2048}})
    assert get_context_budget("test") == 2048
    request_count = len(requests_mock.request_history)
    assert get_output_budget(2048) == 512
    assert len(requests_mock.request_history) == request_count
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
    monkeypatch.setattr("pplx_cli.cli.verified_operation", lambda model: _mock_operation(fail, model))
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
    monkeypatch.setattr("pplx_cli.cli.verified_operation", lambda model: _mock_operation(fail, model))
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


def test_discovery_filters_ineligible_models_and_lists_valid(setup_ollama, requests_mock):
    from pplx_cli.ollama import list_chat_models
    register(requests_mock, tags={"models": [
        {"name": "cloud:latest", "remote_host": "cloud.example"},
        {"name": "missing:latest"}, {"name": "small:latest"},
        {"name": "embedding:latest"}, {"name": "good:latest"},
    ]})
    requests_mock.post("http://127.0.0.1:11434/api/show", [
        {"json": {"capabilities": ["completion"], "model_info": {}}},
        {"json": {"capabilities": ["completion"], "model_info": {"x.context_length": 512}}},
        {"json": {"capabilities": ["embedding"], "model_info": {"x.context_length": 4096}}},
        {"json": {"capabilities": ["completion"], "model_info": {"x.context_length": 4096}}},
    ])
    assert list_chat_models() == ["good:latest"]
    shows = [request.json()["model"] for request in requests_mock.request_history if request.path == "/api/show"]
    assert shows == ["missing:latest", "small:latest", "embedding:latest", "good:latest"]


@pytest.mark.parametrize("value", [True, "4096", -1, 0])
def test_discovery_skips_malformed_context_with_warning(setup_ollama, requests_mock, value, caplog):
    from pplx_cli.ollama import list_chat_models
    register(requests_mock, show={"capabilities": ["completion"], "model_info": {"x.context_length": value}})
    with caplog.at_level("WARNING"):
        assert list_chat_models() == []
    assert "test:latest" in caplog.text
    assert "context metadata is malformed" in caplog.text


def test_discovery_skips_malformed_capabilities_with_warning(setup_ollama, requests_mock, caplog):
    from pplx_cli.ollama import list_chat_models
    register(requests_mock, show={"capabilities": "completion", "model_info": {"x.context_length": 4096}})
    with caplog.at_level("WARNING"):
        assert list_chat_models() == []
    assert "test:latest" in caplog.text
    assert "capabilities are unavailable" in caplog.text


@pytest.mark.parametrize("context", [True, 1023, 4096.0, "4096", None])
def test_output_budget_rejects_invalid_context(context):
    from pplx_cli.ollama import get_output_budget
    with pytest.raises(ValueError, match="Context window"):
        get_output_budget(context)


def test_operation_snapshots_endpoint_timeout_and_context(setup_ollama, requests_mock, monkeypatch):
    from pplx_cli import ollama
    state = {"endpoint": "http://127.0.0.1:11434", "timeout": 123, "context": 4096}
    reads = []
    def settings():
        reads.append(True)
        return dict(state)
    monkeypatch.setattr(ollama, "load_ollama_settings", settings)
    register(requests_mock)
    def change_settings(request, response):
        state.update(endpoint="http://127.0.0.2:11434", timeout=1, context=1024)
        return {"cloud": {"disabled": True}}
    requests_mock.get("http://127.0.0.1:11434/api/status", json=change_settings)
    result = query_chat_completion("private", model="test", provider=Provider.OLLAMA)
    assert result.content == "answer"
    assert len(reads) == 1
    assert [r.path for r in requests_mock.request_history] == ["/api/status", "/api/tags", "/api/show", "/api/chat"]
    assert all(r.hostname == "127.0.0.1" and r.timeout == 123 for r in requests_mock.request_history)
    assert requests_mock.last_request.json()["options"]["num_ctx"] == 4096


def test_verified_operation_reuses_first_preflight_and_refreshes_next_chat(setup_ollama, requests_mock):
    from pplx_cli.ollama import verified_operation, get_context_budget
    register(requests_mock)
    with verified_operation("test"):
        verify_model("test")
        assert get_context_budget("test") == 4096
        query_chat_completion("one", model="test", provider=Provider.OLLAMA)
        assert [r.path for r in requests_mock.request_history] == ["/api/status", "/api/tags", "/api/show", "/api/chat"]
        query_chat_completion("two", model="test", provider=Provider.OLLAMA)
    assert [r.path for r in requests_mock.request_history] == ["/api/status", "/api/tags", "/api/show", "/api/chat"] * 2
    query_chat_completion("three", model="test", provider=Provider.OLLAMA)
    assert [r.path for r in requests_mock.request_history][-4:] == ["/api/status", "/api/tags", "/api/show", "/api/chat"]


def test_nested_operation_restores_state_on_exception(setup_ollama, requests_mock):
    from pplx_cli.ollama import verified_operation
    register(requests_mock)
    with pytest.raises(RuntimeError, match="abort"):
        with verified_operation("test"):
            with verified_operation(None):
                with pytest.raises(OllamaError, match="different Ollama model"):
                    with verified_operation("other"):
                        pass
                raise RuntimeError("abort")
    query_chat_completion("later", model="test", provider=Provider.OLLAMA)
    assert [r.path for r in requests_mock.request_history] == ["/api/status", "/api/tags", "/api/show", "/api/status", "/api/tags", "/api/show", "/api/chat"]


def test_discovery_warns_for_malformed_models_and_keeps_valid(setup_ollama, requests_mock, caplog):
    from pplx_cli.ollama import list_chat_models
    register(requests_mock, tags={"models": [{"name": "broken:latest"}, {"name": "good:latest"}]})
    requests_mock.post("http://127.0.0.1:11434/api/show", [
        {"json": {"model_info": {"x.context_length": 4096}}},
        {"json": {"capabilities": ["completion"], "model_info": {"x.context_length": 4096}}},
    ])
    with caplog.at_level("WARNING"):
        assert list_chat_models() == ["good:latest"]
    assert "broken:latest" in caplog.text and "capabilities" in caplog.text


def test_discovery_warns_for_malformed_show_but_not_transport_failure(setup_ollama, requests_mock, caplog):
    from pplx_cli.ollama import list_chat_models
    register(requests_mock, show=[])
    with caplog.at_level("WARNING"):
        assert list_chat_models() == []
    assert "malformed Ollama /api/show" in caplog.text
    requests_mock.post("http://127.0.0.1:11434/api/show", status_code=500)
    with pytest.raises(OllamaError, match="HTTP 500"):
        list_chat_models()


def test_cli_ask_reuses_operation_preflight(setup_ollama, requests_mock):
    from typer.testing import CliRunner
    from pplx_cli.cli import app
    register(requests_mock)
    result = CliRunner().invoke(app, ["ask", "hello", "--provider", "ollama", "--model", "test", "--no-save-history"])
    assert result.exit_code == 0
    assert [r.path for r in requests_mock.request_history] == ["/api/status", "/api/tags", "/api/show", "/api/chat"]


def test_operation_freezes_default_model_when_config_changes(setup_ollama, requests_mock):
    from pplx_cli.ollama import verified_operation
    config.save_ollama_settings("http://127.0.0.1:11434", "test")
    register(requests_mock)
    with verified_operation():
        config.save_ollama_settings("http://127.0.0.1:11434", "other")
        result = query_chat_completion("private", provider=Provider.OLLAMA)
        assert result.content == "answer"
    assert requests_mock.last_request.json()["model"] == "test:latest"


def test_verified_operation_closes_session_on_failure(setup_ollama, requests_mock, monkeypatch):
    from pplx_cli import ollama
    from unittest.mock import Mock
    register(requests_mock)
    session = requests.Session()
    session.close = Mock(wraps=session.close)
    monkeypatch.setattr(ollama.requests, "Session", lambda: session)
    with pytest.raises(RuntimeError, match="abort"):
        with ollama.verified_operation("test"):
            raise RuntimeError("abort")
    session.close.assert_called_once_with()


def test_empty_model_reuses_active_default_in_nested_scopes(setup_ollama, requests_mock):
    from pplx_cli.ollama import chat, verified_operation
    config.save_ollama_settings("http://127.0.0.1:11434", "test")
    register(requests_mock)
    with verified_operation(""):
        config.save_ollama_settings("http://127.0.0.1:11434", "other")
        with verified_operation(""):
            assert chat([{"role": "user", "content": "hello"}], model="")[0] == "answer"
    assert requests_mock.last_request.json()["model"] == "test:latest"
    assert [r.path for r in requests_mock.request_history] == ["/api/status", "/api/tags", "/api/show", "/api/chat"]


@pytest.mark.parametrize("first_response", [
    {"status_code": 500},
    {"json": {"done_reason": "length", "message": {"content": "incomplete"}}},
])
def test_failed_generation_requires_reverification(setup_ollama, requests_mock, first_response):
    from pplx_cli.ollama import chat, verified_operation
    register(requests_mock)
    requests_mock.post("http://127.0.0.1:11434/api/chat", [
        first_response, {"json": {"message": {"content": "answer"}}},
    ])
    with verified_operation("test"):
        with pytest.raises(OllamaError):
            chat([{"role": "user", "content": "first"}], model="test")
        assert chat([{"role": "user", "content": "second"}], model="test")[0] == "answer"
    assert [r.path for r in requests_mock.request_history] == ["/api/status", "/api/tags", "/api/show", "/api/chat"] * 2
