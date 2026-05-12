"""Embedding client for vLLM's OpenAI-compatible /v1/embeddings endpoint."""

import json
import urllib.request


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def embed(endpoint: str, model_id: str, texts: list[str]) -> list[list[float]]:
    """Return embeddings for *texts* using the vLLM embedding endpoint."""
    data = _post_json(
        f"{endpoint.rstrip('/')}/v1/embeddings",
        {"model": model_id, "input": texts},
        60,
    )
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]


def embed_one(endpoint: str, model_id: str, text: str) -> list[float]:
    return embed(endpoint, model_id, [text])[0]


def make_embedder(endpoint: str, model_id: str):
    def _embed(texts: list[str]) -> list[list[float]]:
        return embed(endpoint, model_id, texts)

    return _embed
