"""vllmd orchestrator — cluster control plane and OpenAI-compatible proxy."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..agent.client import AgentClient
from ..cluster.config import ClusterConfig, ModelSpec, NodeConfig
from .registry import ModelRegistry
from .router import pick_endpoint

log = logging.getLogger(__name__)


class _State:
    def __init__(self) -> None:
        self.config: ClusterConfig | None = None
        self.registry: ModelRegistry = ModelRegistry()
        self.api_key: str = ""
        self.persist_path: Path = (
            Path.home() / ".local" / "share" / "vllmd" / "registry.json"
        )


_state = _State()


def _save_registry() -> None:
    with contextlib.suppress(Exception):
        _state.persist_path.parent.mkdir(parents=True, exist_ok=True)
        _state.persist_path.write_text(json.dumps(_state.registry.dump(), indent=2))


def _load_registry() -> None:
    with contextlib.suppress(Exception):
        data = json.loads(_state.persist_path.read_text())
        _state.registry.load(data)


def _check_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    if not _state.api_key:
        return
    if authorization != f"Bearer {_state.api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")


Auth = Annotated[None, Depends(_check_auth)]


# ---------------------------------------------------------------------------
# Startup / background health polling
# ---------------------------------------------------------------------------


async def _poll_agents() -> None:
    """Background task: refresh registry every 30 s from all agents."""
    while True:
        await asyncio.sleep(30)
        if _state.config is None:
            continue
        await _refresh_registry(_state.config)


async def _refresh_registry(cfg: ClusterConfig) -> None:
    changed = False
    for node in cfg.nodes:
        async with AgentClient(node.agent_url, cfg.api_key) as client:
            try:
                containers = await client.list_models()
                for c in containers:
                    model_id = c.get("model_id", "")
                    port = c.get("port")
                    if model_id and port:
                        endpoint = f"http://{node.host}:{port}"
                        _state.registry.register(model_id, node.name, endpoint)
                        _state.registry.mark_healthy(model_id, node.name, True)
                        changed = True
            except Exception:
                log.debug("Agent at %s unreachable", node.agent_url, exc_info=True)
                for spec in cfg.models_for_node(node.name):
                    _state.registry.mark_healthy(spec.name, node.name, False)
                    changed = True
    if changed:
        _save_registry()


@asynccontextmanager
async def _lifespan(_application: FastAPI) -> AsyncIterator[None]:
    if _state.config is None:
        from ..cluster.config import load_cluster_config

        _state.config = load_cluster_config()
    _load_registry()
    await _refresh_registry(_state.config)
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
    if _state.config is None:
        raise HTTPException(status_code=503, detail="No cluster config loaded.")
    result = []
    for node in _state.config.nodes:
        async with AgentClient(node.agent_url, _state.config.api_key) as client:
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
                log.debug("Agent at %s unreachable", node.agent_url, exc_info=True)
                result.append(
                    {
                        "node": node.name,
                        "host": node.host,
                        "healthy": False,
                        "models": [],
                    }
                )
    return {"nodes": result}


async def _stop_model_on_node(
    node: NodeConfig, spec: ModelSpec, cfg: ClusterConfig
) -> dict:
    async with AgentClient(node.agent_url, cfg.api_key) as client:
        try:
            r = await client.stop_model(spec.name)
            _state.registry.deregister(spec.name, node.name)
            return {"node": node.name, "model": spec.name, **r}
        except Exception as exc:
            return {"node": node.name, "model": spec.name, "error": str(exc)}


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
            _state.registry.register(spec.name, node_name, endpoint)
            return {"node": node_name, "model": spec.name, **result}
        except Exception as exc:
            return {"node": node_name, "model": spec.name, "error": str(exc)}


@app.post("/cluster/up")
async def cluster_up(_: Auth) -> dict:
    if _state.config is None:
        raise HTTPException(status_code=503, detail="No cluster config loaded.")
    tasks = [
        _start_spec_on_node(spec, node_name, _state.config)
        for spec in _state.config.models
        for node_name in spec.nodes
    ]
    results = await asyncio.gather(*tasks)
    _save_registry()
    return {"results": list(results)}


@app.post("/cluster/down")
async def cluster_down(_: Auth) -> dict:
    if _state.config is None:
        raise HTTPException(status_code=503, detail="No cluster config loaded.")
    tasks = []
    for spec in _state.config.models:
        for node_name in spec.nodes:
            node = _state.config.node(node_name)
            if node is None:
                continue
            tasks.append(_stop_model_on_node(node, spec, _state.config))
    results = await asyncio.gather(*tasks)
    _save_registry()
    return {"results": list(results)}


@app.post("/cluster/up/{model}")
async def cluster_up_model(model: str, _: Auth) -> dict:
    if _state.config is None:
        raise HTTPException(status_code=503, detail="No cluster config loaded.")
    spec = _state.config.model(model)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not in config.")
    tasks = [
        _start_spec_on_node(spec, node_name, _state.config) for node_name in spec.nodes
    ]
    results = await asyncio.gather(*tasks)
    _save_registry()
    return {"results": list(results)}


@app.post("/cluster/down/{model}")
async def cluster_down_model(model: str, _: Auth) -> dict:
    if _state.config is None:
        raise HTTPException(status_code=503, detail="No cluster config loaded.")
    spec = _state.config.model(model)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not in config.")
    tasks = []
    for node_name in spec.nodes:
        node = _state.config.node(node_name)
        if node is None:
            continue
        tasks.append(_stop_model_on_node(node, spec, _state.config))
    results = await asyncio.gather(*tasks)
    _save_registry()
    return {"results": list(results)}


# ---------------------------------------------------------------------------
# OpenAI-compatible proxy
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models(_: Auth) -> dict:
    seen: dict[str, dict] = {}
    if _state.config is None:
        return {"object": "list", "data": []}
    for node in _state.config.nodes:
        async with AgentClient(node.agent_url, _state.config.api_key) as client:
            try:
                containers = await client.list_models()
                for c in containers:
                    m = c.get("model_id", "")
                    if m and m not in seen:
                        seen[m] = {"id": m, "object": "model"}
            except Exception:
                log.debug("Agent at %s unreachable", node.agent_url, exc_info=True)
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

    endpoint = pick_endpoint(model, _state.registry, node_pin=node_pin)

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
    _state.config = config
    _state.api_key = config.api_key
    return app
