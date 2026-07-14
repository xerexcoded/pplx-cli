import pytest

from pplx_cli.api import query_chat, query_chat_completion, query_perplexity
from pplx_cli.config import (
    Config,
    PerplexityModel,
    Provider,
    get_provider_settings,
)


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr("pplx_cli.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("pplx_cli.config.CONFIG_FILE", tmp_path / "config.json")
    for provider in Provider:
        monkeypatch.delenv(get_provider_settings(provider).environment_variable, raising=False)
    Config._instance = None
    yield
    Config._instance = None


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        (Provider.PERPLEXITY, PerplexityModel.SONAR),
        (Provider.NVIDIA, "meta/llama-3.3-70b-instruct"),
        (Provider.OPENROUTER, "openrouter/free"),
    ],
)
def test_query_chat_uses_provider_endpoint_headers_and_model(
    requests_mock, provider, model
):
    config = Config.get_instance()
    config.provider = provider
    config.api_key = f"{provider.value}-key"
    settings = get_provider_settings(provider)
    requests_mock.post(
        settings.api_endpoint,
        json={"choices": [{"message": {"content": f"{provider.value} response"}}]},
    )

    response = query_chat("test question", model=model, provider=provider)

    assert response == f"{provider.value} response"
    request = requests_mock.last_request
    assert request.headers["Authorization"] == f"Bearer {provider.value}-key"
    assert request.json() == {
        "model": model.value if isinstance(model, PerplexityModel) else model,
        "messages": [{"role": "user", "content": "test question"}],
    }


@pytest.mark.parametrize("provider", list(Provider))
def test_query_chat_uses_provider_default_model(requests_mock, provider):
    config = Config.get_instance()
    config.provider = provider
    config.api_key = "test-api-key"
    settings = get_provider_settings(provider)
    requests_mock.post(
        settings.api_endpoint,
        json={"choices": [{"message": {"content": "Test response"}}]},
    )

    assert query_chat("test", provider=provider) == "Test response"
    assert requests_mock.last_request.json()["model"] == settings.default_model


@pytest.mark.parametrize("provider", list(Provider))
def test_query_chat_no_api_key(provider):
    config = Config.get_instance()
    config.provider = provider
    config.api_key = None
    settings = get_provider_settings(provider)

    with pytest.raises(ValueError, match=settings.environment_variable):
        query_chat("test", provider=provider)


def test_query_chat_invalid_api_key_is_provider_specific(requests_mock):
    config = Config.get_instance()
    config.provider = Provider.NVIDIA
    config.api_key = "test-api-key"
    settings = get_provider_settings(Provider.NVIDIA)
    requests_mock.post(settings.api_endpoint, status_code=401)

    with pytest.raises(ValueError, match="Invalid NVIDIA NIM API key"):
        query_chat("test", provider=Provider.NVIDIA)


def test_query_chat_rejects_unexpected_response_shape(requests_mock):
    config = Config.get_instance()
    config.api_key = "test-api-key"
    requests_mock.post(Config.API_ENDPOINT, json={"choices": []})

    with pytest.raises(RuntimeError, match="Unexpected response from Perplexity API"):
        query_chat("test", provider=Provider.PERPLEXITY)


def test_query_perplexity_remains_backwards_compatible(requests_mock):
    config = Config.get_instance()
    config.provider = Provider.PERPLEXITY
    config.api_key = "test-api-key"
    requests_mock.post(
        Config.API_ENDPOINT,
        json={"choices": [{"message": {"content": "Test response"}}]},
    )

    assert query_perplexity("test", PerplexityModel.SONAR) == "Test response"


def test_query_chat_completion_retains_grounding_metadata(requests_mock):
    config = Config.get_instance()
    config.api_key = "test-api-key"
    requests_mock.post(
        Config.API_ENDPOINT,
        json={
            "model": "sonar",
            "choices": [{"message": {"content": "Grounded answer"}}],
            "citations": ["https://example.com/source"],
            "search_results": [{"title": "Source", "url": "https://example.com/source"}],
            "usage": {"total_tokens": 12},
        },
    )

    completion = query_chat_completion("test", provider=Provider.PERPLEXITY)

    assert completion.content == "Grounded answer"
    assert completion.citations == ["https://example.com/source"]
    assert completion.search_results[0]["title"] == "Source"
    assert completion.usage["total_tokens"] == 12
