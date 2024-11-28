import requests
from .config import Config, PerplexityModel
from typing import Optional

def query_perplexity(prompt: str, model: Optional[PerplexityModel] = None) -> str:
    config = Config.get_instance()
    if not config.api_key:
        raise ValueError("API key not found. Please set PERPLEXITY_API_KEY in your environment.")

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model.value if model else config.model.value,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(
        config.api_endpoint, 
        headers=headers, 
        json=data, 
        timeout=config.timeout
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    elif response.status_code == 401:
        raise ValueError("Invalid API key")
    else:
        raise RuntimeError(f"API request failed: {response.status_code} - {response.text}")