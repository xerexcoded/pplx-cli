import pytest
from typer.testing import CliRunner
from pplx_cli.cli import app
from pplx_cli.config import PerplexityModel, save_api_key, Config, load_api_key, get_version
import os
from pathlib import Path

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def mock_config(tmp_path, monkeypatch):
    config_dir = tmp_path
    monkeypatch.setattr("pplx_cli.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("pplx_cli.config.CONFIG_FILE", config_dir / "config.json")
    # Reset singleton instance
    if hasattr(Config, '_instance'):
        delattr(Config, '_instance')
    return config_dir

def test_list_models(runner):
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    for model in PerplexityModel:
        assert model.value in result.stdout

def test_ask_without_api_key(runner, mock_config, monkeypatch):
    # Ensure no API key is set
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    if hasattr(Config, '_instance'):
        delattr(Config, '_instance')
    result = runner.invoke(app, ["ask", "test question"])
    assert result.exit_code == 1
    assert "No API key found" in result.stdout

def test_ask_with_api_key(runner, mock_config, requests_mock):
    # Mock the API response
    requests_mock.post(
        Config.API_ENDPOINT,
        json={
            "choices": [
                {
                    "message": {
                        "content": "Mocked response"
                    }
                }
            ]
        }
    )
    
    # Set up API key
    save_api_key("test-api-key")
    
    result = runner.invoke(app, ["ask", "test question"])
    assert result.exit_code == 0
    assert "Mocked response" in result.stdout

def test_model_selection(runner, mock_config, requests_mock):
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
    save_api_key("test-api-key")
    
    result = runner.invoke(app, ["ask", "--model", "small", "test"])
    assert result.exit_code == 0
    assert "Test response" in result.stdout

def test_help_command(runner):
    """Test that --help flag works correctly."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Perplexity CLI" in result.stdout
    assert "Commands" in result.stdout or "commands" in result.stdout.lower()
    assert "ask" in result.stdout

def test_version_flag(runner):
    """Test that --version flag works correctly."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Perplexity CLI version" in result.stdout
    version_str = get_version()
    assert version_str in result.stdout
