from enum import Enum
from typing import Optional
import os
import json
from pathlib import Path
from dotenv import load_dotenv

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

class PerplexityModel(str, Enum):
    SMALL = "llama-3.1-sonar-small-128k-online"
    LARGE = "llama-3.1-sonar-large-128k-online"
    HUGE = "llama-3.1-sonar-huge-128k-online"

    @classmethod
    def get_friendly_name(cls, model: 'PerplexityModel') -> str:
        return model.name.lower()

class Config:
    API_ENDPOINT = "https://api.perplexity.ai/chat/completions"
    API_KEY = load_api_key()
    DEFAULT_MODEL = PerplexityModel.LARGE
    TIMEOUT = 30  # seconds

    @classmethod
    def get_model_info(cls, model: PerplexityModel) -> dict:
        model_info = {
            PerplexityModel.SMALL: {"parameters": "8B", "context_length": 127072},
            PerplexityModel.LARGE: {"parameters": "70B", "context_length": 127072},
            PerplexityModel.HUGE: {"parameters": "405B", "context_length": 127072},
        }
        return model_info.get(model, {})