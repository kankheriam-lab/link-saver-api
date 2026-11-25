import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
OLLAMA_URL = "http://localhost:11434/api/generate"


class SummarizeIn(BaseModel):
    text: str


class SummarizeOut(BaseModel):
    summary: str


@app.post("/summarize", response_model=SummarizeOut)
async def summarize(body: SummarizeIn) -> SummarizeOut:
    payload = {
        "model": "llama3.2:3b",
        "prompt": f"Summarize this in 2 sentences:\n\n{body.text}",
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
    return SummarizeOut(summary=r.json()["response"])
