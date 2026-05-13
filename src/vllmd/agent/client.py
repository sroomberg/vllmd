"""HTTP client for the vllmd agent daemon."""

from __future__ import annotations

import httpx


class AgentClient:
    """Async httpx client wrapping the agent daemon API."""

    def __init__(self, base_url: str, api_key: str = "") -> None:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30)

    async def health(self) -> dict:
        r = await self._client.get("/health")
        r.raise_for_status()
        return r.json()

    async def get_gpus(self) -> dict:
        r = await self._client.get("/gpus")
        r.raise_for_status()
        return r.json()

    async def list_models(self) -> list[dict]:
        r = await self._client.get("/models")
        r.raise_for_status()
        return r.json()

    async def get_model(self, name: str) -> dict:
        r = await self._client.get(f"/models/{name}")
        r.raise_for_status()
        return r.json()

    async def start_model(self, name: str, spec: dict) -> dict:
        r = await self._client.post(f"/models/{name}/start", json=spec)
        r.raise_for_status()
        return r.json()

    async def stop_model(self, name: str) -> dict:
        r = await self._client.post(f"/models/{name}/stop")
        r.raise_for_status()
        return r.json()

    async def get_logs(self, name: str, tail: int = 100) -> dict:
        r = await self._client.get(f"/models/{name}/logs", params={"tail": tail})
        r.raise_for_status()
        return r.json()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AgentClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
