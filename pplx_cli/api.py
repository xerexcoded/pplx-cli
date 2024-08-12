import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_ENDPOINT = "https://api.perplexity.ai/chat/completions"
API_KEY = os.getenv("PERPLEXITY_API_KEY")

def query_perplexity(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(API_ENDPOINT, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"
