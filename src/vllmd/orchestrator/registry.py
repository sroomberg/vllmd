"""In-memory registry mapping model names to their live endpoints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelEndpoint:
    node: str
    endpoint: str
    healthy: bool = True


class ModelRegistry:
    def __init__(self) -> None:
        self._endpoints: dict[str, list[ModelEndpoint]] = {}

    def register(self, model: str, node: str, endpoint: str) -> None:
        existing = self._endpoints.setdefault(model, [])
        for ep in existing:
            if ep.node == node:
                ep.endpoint = endpoint
                ep.healthy = True
                return
        existing.append(ModelEndpoint(node=node, endpoint=endpoint))

    def deregister(self, model: str, node: str) -> None:
        self._endpoints[model] = [
            ep for ep in self._endpoints.get(model, []) if ep.node != node
        ]

    def get_endpoints(self, model: str) -> list[ModelEndpoint]:
        return list(self._endpoints.get(model, []))

    def mark_healthy(self, model: str, node: str, healthy: bool) -> None:
        for ep in self._endpoints.get(model, []):
            if ep.node == node:
                ep.healthy = healthy
                return

    def all_models(self) -> list[str]:
        return list(self._endpoints.keys())
