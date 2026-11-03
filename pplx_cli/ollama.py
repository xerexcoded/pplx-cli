"""Fail-closed transport for a locally configured Ollama daemon."""
import ipaddress
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import urlsplit

import requests

from .config import Config, Provider, load_ollama_settings


_override = ContextVar("ollama_settings", default=None)


@contextmanager
def temporary_settings(settings):
    token = _override.set(settings)
    try:
        yield
    finally:
        _override.reset(token)


class OllamaError(RuntimeError):
    """The local daemon or model could not be verified."""


def _settings():
    settings = _override.get()
    if settings is None:
        settings = load_ollama_settings()
    endpoint = settings.get("endpoint", "http://127.0.0.1:11434")
    if not isinstance(endpoint, str):
        raise OllamaError("Ollama endpoint must be a numeric loopback HTTP URL")
    try:
        url = urlsplit(endpoint)
        address = ipaddress.ip_address(url.hostname or "")
        port = url.port
    except ValueError as error:
        raise OllamaError("Ollama endpoint must be a numeric loopback HTTP URL") from error
    if (url.scheme != "http" or not address.is_loopback or url.username or url.password
            or url.path not in ("", "/") or url.query or url.fragment or not port):
        raise OllamaError("Ollama endpoint must be a numeric loopback HTTP URL with a port")
    timeout = settings.get("timeout", 300)
    context = settings.get("context", 4096)
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0
            or isinstance(context, bool) or not isinstance(context, int) or context < 1024):
        raise OllamaError("Invalid Ollama timeout or context setting; run setup again")
    return endpoint.rstrip("/"), timeout, context


def _request(session, method, path, *, body=None):
    endpoint, timeout, _ = _settings()
    try:
        response = session.request(method, endpoint + path, json=body, timeout=timeout,
                                   allow_redirects=False)
        if response.status_code != 200:
            raise OllamaError(f"Ollama {path} returned HTTP {response.status_code}; verify your local daemon and model")
        value = response.json()
    except (requests.RequestException, ValueError) as error:
        raise OllamaError(f"Cannot verify local Ollama at {endpoint}; start a compatible daemon with cloud disabled: {error}") from error
    if not isinstance(value, dict):
        raise OllamaError(f"Unsupported or malformed Ollama {path} response")
    return value


def _session():
    session = requests.Session()
    session.trust_env = False
    return session


def _status(session):
    status = _request(session, "GET", "/api/status")
    cloud = status.get("cloud")
    if not isinstance(cloud, dict) or cloud.get("disabled") is not True:
        raise OllamaError("Ollama cloud status is enabled or unverifiable. Disable cloud on the daemon (OLLAMA_NO_CLOUD=1 or disable_ollama_cloud in server.json), restart it, and use a version supporting /api/status.")


def _models(session):
    payload = _request(session, "GET", "/api/tags")
    models = payload.get("models")
    if not isinstance(models, list) or any(not isinstance(item, dict) or not isinstance(item.get("name"), str) for item in models):
        raise OllamaError("Unsupported or malformed Ollama model list")
    return models


def _show(session, model, entry):
    info = _request(session, "POST", "/api/show", body={"model": model})
    capabilities = info.get("capabilities")
    if not isinstance(capabilities, list) or not all(isinstance(c, str) for c in capabilities):
        raise OllamaError("Ollama model capabilities are unavailable; upgrade the daemon")
    if "completion" not in capabilities:
        raise OllamaError(f"Ollama model {model!r} is not chat-capable")
    for obj in (entry, info):
        if obj.get("remote_model") or obj.get("remote_host"):
            raise OllamaError(f"Ollama model {model!r} is cloud-routed and cannot be used locally")
    return info


def list_chat_models():
    """Return installed, locally routed chat models after checking daemon policy."""
    with _session() as session:
        _status(session)
        result = []
        for entry in _models(session):
            try:
                _show(session, entry["name"], entry)
            except OllamaError:
                continue
            result.append(entry["name"])
        return result


def _verify(session, model):
    _status(session)
    entries = _models(session)
    # Ollama resolves an omitted tag to :latest.
    matches = [item for item in entries if item["name"] == model or
               (":" not in model and item["name"] == model + ":latest")]
    if not matches:
        raise OllamaError(f"Ollama model {model!r} is not installed. Install it explicitly with 'ollama pull {model}' and retry.")
    _show(session, model, matches[0])


def verify_model(model=None):
    model = model or Config.get_instance().get_default_model(Provider.OLLAMA)
    if not isinstance(model, str) or not model:
        raise OllamaError("No Ollama model configured; run 'perplexity setup --provider ollama'.")
    with _session() as session:
        _verify(session, model)
    return model


def get_context_budget(model=None):
    """Configured context window in tokens; the generation request enforces it."""
    return _settings()[2]


def chat(messages, model=None):
    model = model or Config.get_instance().get_default_model(Provider.OLLAMA)
    if not model:
        raise OllamaError("No Ollama model configured; run 'perplexity setup --provider ollama'.")
    with _session() as session:
        _verify(session, model)
        payload = _request(session, "POST", "/api/chat", body={"model": model, "messages": messages,
                           "stream": False, "options": {"num_ctx": get_context_budget(model)}})
    message = payload.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise OllamaError("Malformed Ollama chat response")
    return message["content"], payload
