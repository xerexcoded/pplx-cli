import os
import pytest
from pathlib import Path
import tempfile
from pplx_cli.config import Config, PerplexityModel, save_api_key, load_api_key

@pytest.fixture
def temp_config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("pplx_cli.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("pplx_cli.config.CONFIG_FILE", tmp_path / "config.json")
    # Reset singleton instance
    if hasattr(Config, '_instance'):
        delattr(Config, '_instance')
    return tmp_path



def test_load_api_key_from_env(monkeypatch, temp_config_dir):
    test_key = "test-env-api-key"
    monkeypatch.setenv("PERPLEXITY_API_KEY", test_key)
    loaded_key = load_api_key()
    assert loaded_key == test_key

def test_model_enum():
    assert PerplexityModel.SONAR.value == "sonar"
    assert PerplexityModel.SONAR_REASONING.value == "sonar-reasoning"
    assert PerplexityModel.SONAR_DEEP_RESEARCH.value == "sonar-deep-research"

def test_config_initialization():
    config = Config()
    assert isinstance(config.model, PerplexityModel)
    assert config.model == PerplexityModel.SONAR
    assert hasattr(config, 'api_key')

def test_file_permissions(temp_config_dir):
    test_key = "test-api-key"
    save_api_key(test_key)
    assert (temp_config_dir / "config.json").stat().st_mode & 0o777 == 0o600

def test_singleton_instance():
    config1 = Config.get_instance()
    config2 = Config.get_instance()
    assert config1 is config2