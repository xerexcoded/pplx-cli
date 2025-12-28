from enum import Enum
from typing import Optional
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import toml

try:
    from importlib.metadata import version as get_package_version
except ImportError:
    # Python < 3.8
    from importlib_metadata import version as get_package_version

load_dotenv()

CONFIG_DIR = Path.home() / ".config" / "perplexity"
CONFIG_FILE = CONFIG_DIR / "config.json"

def load_api_key() -> Optional[str]:
    # First try environment variable
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if api_key:
        return api_key

    # Then try config file
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)
                return config.get("api_key")
        except (json.JSONDecodeError, IOError):
            return None
    
    return None

def save_api_key(api_key: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {"api_key": api_key}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)
    # Set file permissions to be readable only by the user
    CONFIG_FILE.chmod(0o600)

def get_version() -> str:
    """Get the version from package metadata or pyproject.toml."""
    try:
        # First try to get version from installed package metadata
        return get_package_version("pplx-cli")
    except Exception:
        # Fallback: try to read from pyproject.toml (for development/editable installs)
        try:
            project_root = Path(__file__).parent.parent
            pyproject_path = project_root / "pyproject.toml"
            
            if pyproject_path.exists():
                with open(pyproject_path, 'r') as f:
                    pyproject_data = toml.load(f)
                    return pyproject_data.get("tool", {}).get("poetry", {}).get("version", "unknown")
        except Exception:
            pass
        
        return "unknown"

class PerplexityModel(str, Enum):
    SONAR = "sonar"
    SONAR_REASONING = "sonar-reasoning"
    SONAR_DEEP_RESEARCH = "sonar-deep-research"

    @classmethod
    def get_friendly_name(cls, model: 'PerplexityModel') -> str:
        return model.name.lower()

class Config:
    API_ENDPOINT = "https://api.perplexity.ai/chat/completions"
    TIMEOUT = 30  # seconds
    DEFAULT_MODEL = PerplexityModel.SONAR

    def __init__(self):
        self.api_key = load_api_key()
        self.model = self.DEFAULT_MODEL
        self.notes_dir = Path.home() / ".local" / "share" / "perplexity" / "notes"

    @classmethod
    def get_instance(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

    @property
    def api_endpoint(self):
        return self.API_ENDPOINT

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
            PerplexityModel.SONAR: {"type": "lightweight", "description": "Cost-effective search model with grounding"},
            PerplexityModel.SONAR_REASONING: {"type": "reasoning", "description": "Fast, real-time reasoning model for quick problem-solving with search"},
            PerplexityModel.SONAR_DEEP_RESEARCH: {"type": "research", "description": "Expert-level research model conducting exhaustive searches and comprehensive reports"}
        }
        return model_info.get(model, {})