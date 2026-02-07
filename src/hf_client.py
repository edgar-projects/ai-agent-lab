import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# TODO: refactor HF client initialization

if not HF_TOKEN:
    raise RuntimeError("Missing HUGGINGFACE_API_TOKEN in .env")

_client = InferenceClient(token=HF_TOKEN)

def chat(prompt: str, *, max_tokens: int = 300, temperature: float = 0.0) -> str:
    resp = _client.chat_completion(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content
