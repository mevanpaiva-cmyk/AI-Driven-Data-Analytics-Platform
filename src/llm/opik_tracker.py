import os
from opik import track
from .ollama_client import call_ollama

os.environ["OPIK_URL_OVERRIDE"] = "http://localhost:5173/api"

@track(name="ollama_ai_generated_insights")
def generate_ai_insights(prompt: str) -> str:
    return call_ollama(prompt)
