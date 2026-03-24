import requests
import os
from dotenv import load_dotenv
load_dotenv('backend/.env')

API_KEY = os.getenv("OPENROUTER_API_KEY")

models = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-2-9b-it:free",
    "qwen/qwen3-4b:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "google/gemma-3-4b-it:free",
    "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "qwen/qwen2.5-coder-32b-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free"
]

for m in models:
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": m,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10
        }
    )
    if resp.status_code == 200:
        print(f"SUCCESS: {m}")
    else:
        print(f"FAILED {resp.status_code}: {m} - {resp.text}")
