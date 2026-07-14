import json

import pytest

from pplx_cli.config import (
    Config,
    PerplexityModel,
    Provider,
    get_provider_settings,
    load_api_key,
    load_default_provider,
    save_api_key,
)


@pytest.fixture
def temp_config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("pplx_cli.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("pplx_cli.config.CONFIG_FILE", tmp_path / "config.json")
    for provider in Provider:
        monkeypatch.delenv(get_provider_settings(provider).environment_variable, raising=False)
    Config._instance = None
    return tmp_path


@pytest.fixture(autouse=True)
def reset_config_singleton():
    Config._instance = None
    yield
    Config._instance = None


@pytest.mark.parametrize("provider", list(Provider))
def test_load_api_key_from_provider_environment(monkeypatch, temp_config_dir, provider):
    settings = get_provider_settings(provider)
    monkeypatch.setenv(settings.environment_variable, f"{provider.value}-env-key")

    assert load_api_key(provider) == f"{provider.value}-env-key"


def test_environment_key_overrides_saved_key(monkeypatch, temp_config_dir):
    save_api_key("saved-nvidia-key", Provider.NVIDIA)
    monkeypatch.setenv("NVIDIA_API_KEY", "environment-nvidia-key")

    assert load_api_key(Provider.NVIDIA) == "environment-nvidia-key"


def test_legacy_config_is_read_as_perplexity_and_migrated(temp_config_dir):
    config_file = temp_config_dir / "config.json"
    config_file.write_text(json.dumps({"api_key": "legacy-perplexity-key"}))

    assert load_default_provider() == Provider.PERPLEXITY
    assert load_api_key() == "legacy-perplexity-key"

    save_api_key("new-nvidia-key", Provider.NVIDIA)
    saved_config = json.loads(config_file.read_text())
    assert saved_config == {
        "provider": "nvidia",
        "api_keys": {
            "perplexity": "legacy-perplexity-key",
            "nvidia": "new-nvidia-key",
        },
    }


def test_save_api_key_sets_default_provider_and_keeps_existing_keys(temp_config_dir):
    save_api_key("perplexity-key")
    save_api_key("openrouter-key", Provider.OPENROUTER)

    assert load_default_provider() == Provider.OPENROUTER
    assert load_api_key(Provider.PERPLEXITY) == "perplexity-key"
    assert load_api_key(Provider.OPENROUTER) == "openrouter-key"


def test_model_enum():
    assert PerplexityModel.SONAR.value == "sonar"
    assert PerplexityModel.SONAR_REASONING.value == "sonar-reasoning"
    assert PerplexityModel.SONAR_DEEP_RESEARCH.value == "sonar-deep-research"


def test_config_initialization(temp_config_dir):
    config = Config()
    assert config.provider == Provider.PERPLEXITY
    assert config.model == PerplexityModel.SONAR
    assert config.get_default_model(Provider.NVIDIA) == "meta/llama-3.3-70b-instruct"
    assert config.get_default_model(Provider.OPENROUTER) == "openrouter/free"


def test_file_permissions(temp_config_dir):
    save_api_key("test-api-key", Provider.NVIDIA)
    assert (temp_config_dir / "config.json").stat().st_mode & 0o777 == 0o600


def test_singleton_instance(temp_config_dir):
    config1 = Config.get_instance()
    config2 = Config.get_instance()
    assert config1 is config2
