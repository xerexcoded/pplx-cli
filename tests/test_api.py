import pytest
import requests_mock
from pplx_cli.api import query_perplexity
from pplx_cli.config import PerplexityModel, Config

@pytest.fixture(autouse=True)
def mock_config(monkeypatch):
    config = Config.get_instance()
    config.api_key = "test-api-key"
    return config

def test_query_perplexity(requests_mock):
    requests_mock.post(
        Config.API_ENDPOINT,
        json={
            "choices": [
                {
                    "message": {
                        "content": "Test response"
                    }
                }
            ]
        }
    )
    response = query_perplexity("test question", PerplexityModel.SONAR)
    assert response == "Test response"

@pytest.mark.parametrize("model", [
    PerplexityModel.SONAR,
    PerplexityModel.SONAR_REASONING,
    PerplexityModel.SONAR_DEEP_RESEARCH,
    None
])
def test_query_perplexity_models(requests_mock, model):
    requests_mock.post(
        Config.API_ENDPOINT,
        json={
            "choices": [
                {
                    "message": {
                        "content": "Test response"
                    }
                }
            ]
        }
    )
    response = query_perplexity("test", model)
    assert response == "Test response"

def test_query_perplexity_no_api_key(monkeypatch):
    config = Config.get_instance()
    config.api_key = None
    with pytest.raises(ValueError, match="API key not found"):
        query_perplexity("test")