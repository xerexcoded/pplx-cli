"""Fail-closed transport for a locally configured Ollama daemon.

The trust boundary is daemon-reported policy and model metadata at verification
time; it is not OS-level isolation or an atomic guarantee against daemon changes.
"""
import ipaddress
import logging
import math
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import urlsplit

import requests

from .config import Config, Provider, load_ollama_settings


_logger = logging.getLogger(__name__)

_override = ContextVar("ollama_settings", default=None)
_operation = ContextVar("ollama_verified_operation", default=None)


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
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0
            or isinstance(context, bool) or not isinstance(context, int) or context < 1024):
        raise OllamaError("Invalid Ollama timeout or context setting; run setup again")
    return endpoint.rstrip("/"), timeout, context


def _request(session, method, path, *, body=None):
    endpoint, timeout, _ = session._pplx_settings
    try:
        response = session.request(method, endpoint + path, json=body, timeout=timeout,
                                   allow_redirects=False)
        if response.status_code != 200:
            raise OllamaError(f"Ollama {path} returned HTTP {response.status_code}; verify your local daemon and model")
    except requests.RequestException as error:
        raise OllamaError(f"Cannot verify local Ollama at {endpoint}; start a compatible daemon with cloud disabled: {error}") from error
    try:
        value = response.json()
    except ValueError as error:
        error_type = _MalformedModel if path == "/api/show" else OllamaError
        raise error_type(f"Unsupported or malformed Ollama {path} response") from error
    if not isinstance(value, dict):
        error_type = _MalformedModel if path == "/api/show" else OllamaError
        raise error_type(f"Unsupported or malformed Ollama {path} response")
    return value


def _session():
    settings = _settings()
    session = requests.Session()
    session._pplx_settings = settings
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


class _MalformedModel(OllamaError):
    """The model's own metadata cannot be interpreted safely."""


class _IneligibleModel(OllamaError):
    """An installed model is explicitly non-chat or cloud-routed."""


def _check_remote_markers(obj, model):
    # Local models omit these optional fields in current Ollama responses.
    # Nonempty values identify remote routing; unexpected types are unverifiable.
    # https://github.com/ollama/ollama/blob/main/api/types.go
    for field in ("remote_model", "remote_host"):
        if field in obj:
            value = obj[field]
            if not isinstance(value, str):
                raise _MalformedModel(f"Malformed Ollama {field} metadata for {model!r}")
            if value:
                raise _IneligibleModel(f"Ollama model {model!r} is cloud-routed and cannot be used locally")


def _model_context(info):
    model_info = info.get("model_info")
    if model_info is None:
        raise _IneligibleModel("Ollama model context metadata is unavailable; select a compatible model")
    if not isinstance(model_info, dict):
        raise _MalformedModel("Ollama model context metadata is malformed")
    values = [value for key, value in model_info.items()
              if isinstance(key, str) and key.endswith(".context_length")]
    if not values:
        raise _IneligibleModel("Ollama model context metadata is missing")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise _MalformedModel("Ollama model context metadata is malformed")
    if any(value <= 0 for value in values):
        raise _MalformedModel("Ollama model context metadata is malformed")
    if any(value < 1024 for value in values):
        raise _IneligibleModel("Ollama model context metadata advertises an unsupported context")
    return min(values)


def _show(session, model, entry):
    _check_remote_markers(entry, model)
    info = _request(session, "POST", "/api/show", body={"model": model})
    _check_remote_markers(info, model)
    capabilities = info.get("capabilities")
    if not isinstance(capabilities, list) or not all(isinstance(c, str) for c in capabilities):
        raise _MalformedModel("Ollama model capabilities are unavailable; upgrade the daemon")
    if "completion" not in capabilities:
        raise _IneligibleModel(f"Ollama model {model!r} is not chat-capable")
    _model_context(info)
    return info


def list_chat_models():
    """Return installed, locally routed chat models after checking daemon policy."""
    with _session() as session:
        _status(session)
        result = []
        for entry in _models(session):
            try:
                _show(session, entry["name"], entry)
            except _IneligibleModel as error:
                _logger.debug("Skipping incompatible Ollama model %s: %s", entry["name"], error)
                continue
            except _MalformedModel as error:
                _logger.warning("Skipping incompatible Ollama model %s: %s", entry["name"], error)
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
    resolved = matches[0]["name"]
    return resolved, _show(session, resolved, matches[0])


def _selected_model(model=None):
    """Treat an omitted or empty override as the frozen operation/default model."""
    active = _operation.get()
    if active is not None:
        if model and model not in (active["requested"], active["resolved"]):
            raise OllamaError("A different Ollama model cannot be used inside an active verified operation")
        return active["requested"]
    selected = model or Config.get_instance().get_default_model(Provider.OLLAMA)
    if not isinstance(selected, str) or not selected:
        raise OllamaError("No Ollama model configured; run 'perplexity setup --provider ollama'.")
    return selected


@contextmanager
def verified_operation(model=None):
    """Scope preflight reuse to one operation, refreshing before each generation."""
    selected = _selected_model(model)
    existing = _operation.get()
    if existing is not None:
        yield
        return
    with _session() as session:
        resolved, info = _verify(session, selected)
        state = {"requested": selected, "resolved": resolved, "info": info, "session": session}
        token = _operation.set(state)
        try:
            yield
        finally:
            _operation.reset(token)


def _ready(state):
    if state["info"] is None:
        state["resolved"], state["info"] = _verify(state["session"], state["requested"])
    return state["info"]


def verify_model(model=None):
    selected = _selected_model(model)
    with verified_operation(selected):
        _ready(_operation.get())
    return selected


def get_context_budget(model=None):
    """Return the model-capped window, verifying if no scoped preflight is available."""
    selected = _selected_model(model)
    with verified_operation(selected):
        state = _operation.get()
        return min(state["session"]._pplx_settings[2], _model_context(_ready(state)))


def get_output_budget(context_tokens):
    """Calculate the output reserve from an already verified context window."""
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int) or context_tokens < 1024:
        raise ValueError("Context window must be an integer of at least 1024 tokens")
    return min(1024, context_tokens // 4)


def chat(messages, model=None):
    selected = _selected_model(model)
    with verified_operation(selected):
        state = _operation.get()
        session = state["session"]
        info = _ready(state)
        model = state["resolved"]
        context = min(session._pplx_settings[2], _model_context(info))
        output_budget = get_output_budget(context)
        # UTF-8 bytes conservatively bound ordinary byte-tokenizer input.
        input_budget = sum(len(item["content"].encode("utf-8")) + 32 for item in messages)
        if input_budget + output_budget > context:
            raise OllamaError(
                "Input is too large for the configured Ollama context and output reserve. "
                "Shorten the question or evidence, or increase the configured context within the model limit."
            )
        # Consume before sending: even a failed generation must be reverified.
        state["info"] = None
        payload = _request(session, "POST", "/api/chat", body={"model": model, "messages": messages,
                           "stream": False, "options": {"num_ctx": context,
                           "num_predict": output_budget}})
    if payload.get("done_reason") == "length":
        raise OllamaError(
            "Ollama reached the generation limit and returned an incomplete response. "
            "Ask for a shorter answer or split the request into smaller parts."
        )
    message = payload.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise OllamaError("Malformed Ollama chat response")
    payload.setdefault("model", model)
    return message["content"], payload
