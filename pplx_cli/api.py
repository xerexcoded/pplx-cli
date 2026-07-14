"""Provider-aware chat completion requests."""

from typing import Optional, Union

import requests

from .config import (
    Config,
    PerplexityModel,
    Provider,
    get_provider_settings,
    normalize_provider,
)


def query_chat(
    prompt: str,
    model: Optional[Union[PerplexityModel, str]] = None,
    provider: Optional[Union[Provider, str]] = None,
) -> str:
    """Send a non-streaming chat-completions request to a selected provider."""
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
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": selected_model,
        "messages": [{"role": "user", "content": prompt}],
    }

    response = requests.post(
        config.get_api_endpoint(selected_provider),
        headers=headers,
        json=data,
        timeout=config.timeout,
    )

    if response.status_code == 200:
        try:
            return response.json()["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Unexpected response from {settings.display_name} API"
            ) from error
    if response.status_code == 401:
        raise ValueError(f"Invalid {settings.display_name} API key")

    raise RuntimeError(
        f"{settings.display_name} API request failed: "
        f"{response.status_code} - {response.text}"
    )


def query_perplexity(
    prompt: str, model: Optional[PerplexityModel] = None
) -> str:
    """Backward-compatible helper for callers that explicitly use Perplexity."""
    return query_chat(prompt, model=model, provider=Provider.PERPLEXITY)
