"""vllmd agent daemon — manages Docker containers on a single node."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from ..runner import (
    CONTAINER_RUNTIME,
    RunConfig,
    build_docker_run_cmd,
    container_exists,
    detect_lora_rank,
    list_containers,
    status,
    stop,
)

app = FastAPI(title="vllmd-agent")

_api_key: str = ""


def _check_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    if not _api_key:
        return
    if authorization != f"Bearer {_api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")


Auth = Annotated[None, Depends(_check_auth)]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health(_: Auth) -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GPU inventory
# ---------------------------------------------------------------------------


def _allocated_gpu_devices() -> list[int]:
    """Return GPU device IDs currently in use by vllmd containers."""
    allocated: list[int] = []
    for c in list_containers():
        allocated.extend(c.get("gpu_devices", []))
    return allocated


@app.get("/gpus")
def get_gpus(_: Auth) -> dict:
    node_gpus_env = os.environ.get("VLLMD_NODE_GPUS", "")
    if node_gpus_env:
        node_gpus = [int(x) for x in node_gpus_env.split(",") if x.strip().isdigit()]
    else:
        node_gpus = []

    allocated = _allocated_gpu_devices()
    free = [g for g in node_gpus if g not in allocated]
    return {"node_gpus": node_gpus, "allocated": allocated, "free": free}


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


@app.get("/models")
def get_models(_: Auth) -> list[dict]:
    return list_containers()


@app.get("/models/{name}")
def get_model(name: str, _: Auth) -> dict:
    return status(name)


class StartModelRequest(BaseModel):
    model_path: str
    port: int = 8001
    dtype: str = "auto"
    max_model_len: int | None = None
    lora_path: str | None = None
    max_lora_rank: int | None = None
    gpu_count: int = 1
    gpu_memory_utilization: float = 0.9
    extra_args: list[str] = []


@app.post("/models/{name}/start")
def start_model(name: str, req: StartModelRequest, _: Auth) -> dict:
    if container_exists(name):
        raise HTTPException(
            status_code=409, detail=f"Container '{name}' already exists."
        )

    # GPU allocation
    node_gpus_env = os.environ.get("VLLMD_NODE_GPUS", "")
    if node_gpus_env:
        node_gpus = [int(x) for x in node_gpus_env.split(",") if x.strip().isdigit()]
        allocated = _allocated_gpu_devices()
        free = [g for g in node_gpus if g not in allocated]
        if len(free) < req.gpu_count:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Insufficient GPUs: need {req.gpu_count}, have {len(free)} free."
                ),
            )
        assigned = free[: req.gpu_count]
    else:
        assigned = None  # fall back to --gpus all

    model_path = Path(req.model_path)
    lora_path = Path(req.lora_path) if req.lora_path else None
    max_lora_rank = req.max_lora_rank
    if lora_path and max_lora_rank is None:
        max_lora_rank = detect_lora_rank(lora_path)

    extra = list(req.extra_args)
    if req.gpu_memory_utilization != 0.9:
        extra += ["--gpu-memory-utilization", str(req.gpu_memory_utilization)]

    config = RunConfig(
        model_path=model_path,
        port=req.port,
        name=name,
        gpu=True,
        dtype=req.dtype,
        max_model_len=req.max_model_len,
        lora_path=lora_path,
        max_lora_rank=max_lora_rank,
        extra_args=extra,
        gpu_devices=assigned,
    )

    cmd = build_docker_run_cmd(config)
    # detach the container so the agent isn't blocked
    cmd.insert(2, "-d")
    subprocess.run(cmd, check=True)
    return {"name": name, "status": "started"}


@app.post("/models/{name}/stop")
def stop_model(name: str, _: Auth) -> dict:
    if not container_exists(name):
        raise HTTPException(status_code=404, detail=f"Container '{name}' not found.")
    stop(name)
    return {"name": name, "status": "stopped"}


@app.get("/models/{name}/logs")
def get_logs(name: str, tail: int = 100, _: Auth = None) -> dict:
    result = subprocess.run(
        [CONTAINER_RUNTIME, "logs", "--tail", str(tail), name],
        capture_output=True,
        text=True,
    )
    return {"name": name, "logs": result.stdout + result.stderr}


# ---------------------------------------------------------------------------
# Entry point helper
# ---------------------------------------------------------------------------


def create_app(api_key: str = "", container_runtime: str = "docker") -> FastAPI:
    global _api_key
    from ..runner import set_runtime

    _api_key = api_key
    set_runtime(container_runtime)
    return app
