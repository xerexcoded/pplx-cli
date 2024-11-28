import requests
from .config import Config, PerplexityModel
from typing import Optional

def query_perplexity(prompt: str, model: Optional[PerplexityModel] = None) -> str:
    if not Config.API_KEY:
        raise ValueError("API key not found. Please set PERPLEXITY_API_KEY in your environment.")

    headers = {
        "Authorization": f"Bearer {Config.API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model.value if model else Config.DEFAULT_MODEL.value,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(
        Config.API_ENDPOINT, 
        headers=headers, 
        json=data, 
        timeout=Config.TIMEOUT
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"