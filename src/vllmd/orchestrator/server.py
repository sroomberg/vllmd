"""vllmd orchestrator — cluster control plane and OpenAI-compatible proxy."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..agent.client import AgentClient
from ..cluster.config import ClusterConfig, ModelSpec
from .registry import ModelRegistry
from .router import pick_endpoint

_config: ClusterConfig | None = None
_registry = ModelRegistry()
_api_key: str = ""


def _check_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    if not _api_key:
        return
    if authorization != f"Bearer {_api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")


Auth = Annotated[None, Depends(_check_auth)]


# ---------------------------------------------------------------------------
# Startup / background health polling
# ---------------------------------------------------------------------------


async def _poll_agents() -> None:
    """Background task: refresh registry every 30 s from all agents."""
    while True:
        await asyncio.sleep(30)
        if _config is None:
            continue
        await _refresh_registry(_config)


async def _refresh_registry(cfg: ClusterConfig) -> None:
    for node in cfg.nodes:
        async with AgentClient(node.agent_url, cfg.api_key) as client:
            try:
                containers = await client.list_models()
                for c in containers:
                    model_id = c.get("model_id", "")
                    port = c.get("port")
                    if model_id and port:
                        endpoint = f"http://{node.host}:{port}"
                        _registry.register(model_id, node.name, endpoint)
                        _registry.mark_healthy(model_id, node.name, True)
            except Exception:
                for spec in cfg.models_for_node(node.name):
                    _registry.mark_healthy(spec.name, node.name, False)


@asynccontextmanager
async def _lifespan(_application: FastAPI) -> AsyncIterator[None]:
    if _config is not None:
        await _refresh_registry(_config)
    task = asyncio.create_task(_poll_agents())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="vllmd-orchestrator", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Cluster management
# ---------------------------------------------------------------------------


@app.get("/cluster/status")
async def cluster_status(_: Auth) -> dict:
    if _config is None:
        raise HTTPException(status_code=503, detail="No cluster config loaded.")
    result = []
    for node in _config.nodes:
        async with AgentClient(node.agent_url, _config.api_key) as client:
            try:
                await client.health()
                models = await client.list_models()
                result.append(
                    {
                        "node": node.name,
                        "host": node.host,
                        "healthy": True,
                        "models": models,
                    }
                )
            except Exception:
                result.append(
                    {
                        "node": node.name,
                        "host": node.host,
                        "healthy": False,
                        "models": [],
                    }
                )
    return {"nodes": result}


async def _start_spec_on_node(
    spec: ModelSpec, node_name: str, cfg: ClusterConfig
) -> dict:
    node = cfg.node(node_name)
    if node is None:
        return {"node": node_name, "model": spec.name, "error": "node not found"}
    async with AgentClient(node.agent_url, cfg.api_key) as client:
        try:
            payload = {
                "model_path": spec.model_path,
                "port": spec.port,
                "dtype": spec.dtype,
                "gpu_count": spec.gpu_count,
                "gpu_memory_utilization": spec.gpu_memory_utilization,
                "extra_args": spec.extra_args,
            }
            if spec.max_model_len is not None:
                payload["max_model_len"] = spec.max_model_len
            if spec.lora_path is not None:
                payload["lora_path"] = spec.lora_path
            result = await client.start_model(spec.name, payload)
            endpoint = f"http://{node.host}:{spec.port}"
            _registry.register(spec.name, node_name, endpoint)
            return {"node": node_name, "model": spec.name, **result}
        except Exception as exc:
            return {"node": node_name, "model": spec.name, "error": str(exc)}


@app.post("/cluster/up")
async def cluster_up(_: Auth) -> dict:
    if _config is None:
        raise HTTPException(status_code=503, detail="No cluster config loaded.")
    tasks = [
        _start_spec_on_node(spec, node_name, _config)
        for spec in _config.models
        for node_name in spec.nodes
    ]
    results = await asyncio.gather(*tasks)
    return {"results": list(results)}


@app.post("/cluster/down")
async def cluster_down(_: Auth) -> dict:
    if _config is None:
        raise HTTPException(status_code=503, detail="No cluster config loaded.")
    tasks = []
    for spec in _config.models:
        for node_name in spec.nodes:
            node = _config.node(node_name)
            if node is None:
                continue

            async def _stop(n=node, s=spec) -> dict:
                async with AgentClient(n.agent_url, _config.api_key) as client:
                    try:
                        r = await client.stop_model(s.name)
                        _registry.deregister(s.name, n.name)
                        return {"node": n.name, "model": s.name, **r}
                    except Exception as exc:
                        return {"node": n.name, "model": s.name, "error": str(exc)}

            tasks.append(_stop())
    results = await asyncio.gather(*tasks)
    return {"results": list(results)}


@app.post("/cluster/up/{model}")
async def cluster_up_model(model: str, _: Auth) -> dict:
    if _config is None:
        raise HTTPException(status_code=503, detail="No cluster config loaded.")
    spec = _config.model(model)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not in config.")
    tasks = [_start_spec_on_node(spec, node_name, _config) for node_name in spec.nodes]
    results = await asyncio.gather(*tasks)
    return {"results": list(results)}


@app.post("/cluster/down/{model}")
async def cluster_down_model(model: str, _: Auth) -> dict:
    if _config is None:
        raise HTTPException(status_code=503, detail="No cluster config loaded.")
    spec = _config.model(model)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not in config.")
    tasks = []
    for node_name in spec.nodes:
        node = _config.node(node_name)
        if node is None:
            continue

        async def _stop(n=node, s=spec) -> dict:
            async with AgentClient(n.agent_url, _config.api_key) as client:
                try:
                    r = await client.stop_model(s.name)
                    _registry.deregister(s.name, n.name)
                    return {"node": n.name, "model": s.name, **r}
                except Exception as exc:
                    return {"node": n.name, "model": s.name, "error": str(exc)}

        tasks.append(_stop())
    results = await asyncio.gather(*tasks)
    return {"results": list(results)}


# ---------------------------------------------------------------------------
# OpenAI-compatible proxy
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models(_: Auth) -> dict:
    seen: dict[str, dict] = {}
    if _config is None:
        return {"object": "list", "data": []}
    for node in _config.nodes:
        async with AgentClient(node.agent_url, _config.api_key) as client:
            try:
                containers = await client.list_models()
                for c in containers:
                    m = c.get("model_id", "")
                    if m and m not in seen:
                        seen[m] = {"id": m, "object": "model"}
            except Exception:
                pass
    return {"object": "list", "data": list(seen.values())}


async def _proxy_stream(url: str, body: bytes, headers: dict) -> AsyncIterator[bytes]:
    async with (
        httpx.AsyncClient(timeout=None) as client,
        client.stream("POST", url, content=body, headers=headers) as r,
    ):
        async for chunk in r.aiter_bytes():
            yield chunk


async def _proxy_request(
    request: Request,
    path: str,
    node_pin: str | None,
) -> Response:
    body = await request.body()
    try:
        data = json.loads(body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc

    model = data.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model' field.")

    endpoint = pick_endpoint(model, _registry, node_pin=node_pin)

    target_url = f"{endpoint.endpoint}{path}"
    proxy_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    proxy_headers["content-type"] = "application/json"

    streaming = data.get("stream", False)

    if streaming:
        return StreamingResponse(
            _proxy_stream(target_url, body, proxy_headers),
            media_type="text/event-stream",
            headers={"X-Vllmd-Node": endpoint.node},
        )

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(target_url, content=body, headers=proxy_headers)
    return JSONResponse(
        content=r.json(),
        status_code=r.status_code,
        headers={"X-Vllmd-Node": endpoint.node},
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_vllmd_node: Annotated[str | None, Header()] = None,
    _: Auth = None,
) -> Response:
    return await _proxy_request(request, "/v1/chat/completions", x_vllmd_node)


@app.post("/v1/completions")
async def completions(
    request: Request,
    x_vllmd_node: Annotated[str | None, Header()] = None,
    _: Auth = None,
) -> Response:
    return await _proxy_request(request, "/v1/completions", x_vllmd_node)


@app.post("/v1/embeddings")
async def embeddings(
    request: Request,
    x_vllmd_node: Annotated[str | None, Header()] = None,
    _: Auth = None,
) -> Response:
    return await _proxy_request(request, "/v1/embeddings", x_vllmd_node)


# ---------------------------------------------------------------------------
# Entry point helper
# ---------------------------------------------------------------------------


def create_app(config: ClusterConfig) -> FastAPI:
    global _config, _api_key
    _config = config
    _api_key = config.api_key
    return app
