"""Configuration and provider definitions for the Perplexity CLI."""

import json
import os
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import toml
from dotenv import load_dotenv

try:
    from importlib.metadata import version as get_package_version
except ImportError:
    # Python < 3.8
    from importlib_metadata import version as get_package_version


load_dotenv()

CONFIG_DIR = Path.home() / ".config" / "perplexity"
CONFIG_FILE = CONFIG_DIR / "config.json"


class Provider(str, Enum):
    """Supported hosted chat-completion providers."""

    PERPLEXITY = "perplexity"
    NVIDIA = "nvidia"
    OPENROUTER = "openrouter"


@dataclass(frozen=True)
class ProviderSettings:
    """Static connection settings for one provider."""

    display_name: str
    api_endpoint: str
    environment_variable: str
    default_model: str


PROVIDER_SETTINGS = {
    Provider.PERPLEXITY: ProviderSettings(
        display_name="Perplexity",
        api_endpoint="https://api.perplexity.ai/chat/completions",
        environment_variable="PERPLEXITY_API_KEY",
        default_model="sonar",
    ),
    Provider.NVIDIA: ProviderSettings(
        display_name="NVIDIA NIM",
        api_endpoint="https://integrate.api.nvidia.com/v1/chat/completions",
        environment_variable="NVIDIA_API_KEY",
        default_model="meta/llama-3.3-70b-instruct",
    ),
    Provider.OPENROUTER: ProviderSettings(
        display_name="OpenRouter",
        api_endpoint="https://openrouter.ai/api/v1/chat/completions",
        environment_variable="OPENROUTER_API_KEY",
        default_model="openrouter/free",
    ),
}


def normalize_provider(provider: Union[Provider, str]) -> Provider:
    """Return a Provider enum, raising ValueError for unsupported names."""
    if isinstance(provider, Provider):
        return provider
    try:
        return Provider(provider.lower())
    except (AttributeError, ValueError) as error:
        valid_providers = ", ".join(item.value for item in Provider)
        raise ValueError(f"Provider must be one of: {valid_providers}") from error


def get_provider_settings(provider: Union[Provider, str]) -> ProviderSettings:
    """Return the static settings for a configured provider."""
    return PROVIDER_SETTINGS[normalize_provider(provider)]


def _read_config() -> dict:
    """Read config data, converting the legacy single-key format in memory."""
    if not CONFIG_FILE.exists():
        return {}

    try:
        with open(CONFIG_FILE, encoding="utf-8") as config_file:
            data = json.load(config_file)
    except (json.JSONDecodeError, IOError):
        return {}

    if not isinstance(data, dict):
        return {}

    api_keys = data.get("api_keys")
    if not isinstance(api_keys, dict):
        api_keys = {}

    # Config files created before provider support held one Perplexity API key.
    legacy_api_key = data.get("api_key")
    if isinstance(legacy_api_key, str) and legacy_api_key:
        api_keys.setdefault(Provider.PERPLEXITY.value, legacy_api_key)

    return {
        "provider": data.get("provider", Provider.PERPLEXITY.value),
        "api_keys": api_keys,
    }


def load_default_provider() -> Provider:
    """Load the saved default provider, preserving Perplexity as the fallback."""
    provider_name = _read_config().get("provider", Provider.PERPLEXITY.value)
    try:
        return normalize_provider(provider_name)
    except ValueError:
        return Provider.PERPLEXITY


def load_api_key(provider: Union[Provider, str] = Provider.PERPLEXITY) -> Optional[str]:
    """Load a provider key, preferring that provider's environment variable."""
    selected_provider = normalize_provider(provider)
    settings = get_provider_settings(selected_provider)

    api_key = os.getenv(settings.environment_variable)
    if api_key:
        return api_key

    api_keys = _read_config().get("api_keys", {})
    stored_key = api_keys.get(selected_provider.value)
    return stored_key if isinstance(stored_key, str) and stored_key else None


def save_api_key(
    api_key: str, provider: Union[Provider, str] = Provider.PERPLEXITY
) -> None:
    """Save a provider API key and make that provider the default."""
    selected_provider = normalize_provider(provider)
    config = _read_config()
    api_keys = config.get("api_keys", {})
    api_keys[selected_provider.value] = api_key

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as config_file:
        json.dump(
            {"provider": selected_provider.value, "api_keys": api_keys},
            config_file,
        )

    # API keys must only be readable by the current user.
    CONFIG_FILE.chmod(0o600)


def get_version() -> str:
    """Get the version from package metadata or pyproject.toml."""
    try:
        return get_package_version("pplx-cli")
    except Exception:
        try:
            project_root = Path(__file__).parent.parent
            pyproject_path = project_root / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, encoding="utf-8") as pyproject_file:
                    pyproject_data = toml.load(pyproject_file)
                    return pyproject_data.get("tool", {}).get("poetry", {}).get(
                        "version", "unknown"
                    )
        except Exception:
            pass

        return "unknown"


class PerplexityModel(str, Enum):
    SONAR = "sonar"
    SONAR_REASONING = "sonar-reasoning"
    SONAR_DEEP_RESEARCH = "sonar-deep-research"

    @classmethod
    def get_friendly_name(cls, model: "PerplexityModel") -> str:
        return model.name.lower()


class Config:
    """Application settings and backwards-compatible Perplexity properties."""

    API_ENDPOINT = PROVIDER_SETTINGS[Provider.PERPLEXITY].api_endpoint
    TIMEOUT = 30  # seconds
    DEFAULT_MODEL = PerplexityModel.SONAR

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.provider = load_default_provider()
        self.api_key = load_api_key(self.provider)
        self.model = self.DEFAULT_MODEL
        self.notes_dir = Path.home() / ".local" / "share" / "perplexity" / "notes"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def reload(self) -> None:
        """Reload the saved default provider and its selected API key."""
        self.provider = load_default_provider()
        self.api_key = load_api_key(self.provider)

    def get_api_key(self, provider: Union[Provider, str]) -> Optional[str]:
        """Get an API key for a provider, retaining direct legacy overrides."""
        selected_provider = normalize_provider(provider)
        if selected_provider == self.provider and self.api_key:
            return self.api_key
        return load_api_key(selected_provider)

    def get_api_endpoint(self, provider: Union[Provider, str]) -> str:
        return get_provider_settings(provider).api_endpoint

    def get_default_model(self, provider: Union[Provider, str]) -> str:
        return get_provider_settings(provider).default_model

    @property
    def api_endpoint(self):
        return self.get_api_endpoint(self.provider)

    @property
    def timeout(self):
        return self.TIMEOUT

    @property
    def API_KEY(self):
        return self.api_key

    @API_KEY.setter
    def API_KEY(self, value):
        self.api_key = value

    @classmethod
    def get_model_info(cls, model: PerplexityModel) -> dict:
        model_info = {
            PerplexityModel.SONAR: {
                "type": "lightweight",
                "description": "Cost-effective search model with grounding",
            },
            PerplexityModel.SONAR_REASONING: {
                "type": "reasoning",
                "description": "Fast, real-time reasoning model for quick problem-solving with search",
            },
            PerplexityModel.SONAR_DEEP_RESEARCH: {
                "type": "research",
                "description": "Expert-level research model conducting exhaustive searches and comprehensive reports",
            },
        }
        return model_info.get(model, {})
