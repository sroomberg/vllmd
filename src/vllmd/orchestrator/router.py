"""Endpoint selection logic for the orchestrator."""

from __future__ import annotations

from fastapi import HTTPException

from .registry import ModelEndpoint, ModelRegistry

_last_index: dict[str, int] = {}


def pick_endpoint(
    model: str,
    registry: ModelRegistry,
    *,
    node_pin: str | None = None,
) -> ModelEndpoint:
    endpoints = registry.get_endpoints(model)
    healthy = [ep for ep in endpoints if ep.healthy]

    if not healthy:
        raise HTTPException(
            status_code=503,
            detail=f"No healthy endpoints for model '{model}'.",
        )

    if node_pin is not None:
        for ep in healthy:
            if ep.node == node_pin:
                return ep
        raise HTTPException(
            status_code=503,
            detail=f"Node '{node_pin}' is not healthy for model '{model}'.",
        )

    # Round-robin
    idx = _last_index.get(model, -1) + 1
    _last_index[model] = idx % len(healthy)
    return healthy[_last_index[model]]
