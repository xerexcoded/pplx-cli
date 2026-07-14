"""Provider-aware chat completion requests."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import requests

from .config import (
    Config,
    PerplexityModel,
    Provider,
    get_provider_settings,
    normalize_provider,
)


@dataclass(frozen=True)
class ChatCompletion:
    """The provider response retained for callers that need grounding metadata."""

    content: str
    model: str
    citations: List[str] = field(default_factory=list)
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)


def query_chat_completion(
    prompt: str,
    model: Optional[Union[PerplexityModel, str]] = None,
    provider: Optional[Union[Provider, str]] = None,
    system_prompt: Optional[str] = None,
) -> ChatCompletion:
    """Send a completion request and retain provider citations and usage details."""
    config = Config.get_instance()
    selected_provider = normalize_provider(provider or config.provider)
    settings = get_provider_settings(selected_provider)
    api_key = config.get_api_key(selected_provider)

    if not api_key:
        raise ValueError(
            f"API key not found for {settings.display_name}. "
            f"Please set {settings.environment_variable} or run "
            f"'perplexity setup --provider {selected_provider.value}'."
        )

    selected_model = model.value if isinstance(model, PerplexityModel) else model
    selected_model = selected_model or config.get_default_model(selected_provider)
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = requests.post(
        config.get_api_endpoint(selected_provider),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": selected_model, "messages": messages},
        timeout=config.timeout,
    )

    if response.status_code == 401:
        raise ValueError(f"Invalid {settings.display_name} API key")
    if response.status_code != 200:
        raise RuntimeError(
            f"{settings.display_name} API request failed: "
            f"{response.status_code} - {response.text}"
        )

    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Unexpected response from {settings.display_name} API"
        ) from error

    citations = payload.get("citations") or []
    search_results = payload.get("search_results") or []
    return ChatCompletion(
        content=content,
        model=payload.get("model") or str(selected_model),
        citations=[citation for citation in citations if isinstance(citation, str)],
        search_results=[item for item in search_results if isinstance(item, dict)],
        usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
    )


def query_chat(
    prompt: str,
    model: Optional[Union[PerplexityModel, str]] = None,
    provider: Optional[Union[Provider, str]] = None,
) -> str:
    """Send a non-streaming chat-completions request to a selected provider."""
    return query_chat_completion(prompt, model=model, provider=provider).content


def query_perplexity(
    prompt: str, model: Optional[PerplexityModel] = None
) -> str:
    """Backward-compatible helper for callers that explicitly use Perplexity."""
    return query_chat(prompt, model=model, provider=Provider.PERPLEXITY)
