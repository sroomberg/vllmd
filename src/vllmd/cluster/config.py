"""Cluster configuration: nodes, models, and service settings."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..vectordb.factory import _load_config


@dataclass
class NodeConfig:
    name: str
    host: str
    agent_port: int = 7861
    gpus: list[int] = field(default_factory=list)

    @property
    def agent_url(self) -> str:
        return f"http://{self.host}:{self.agent_port}"


@dataclass
class ModelSpec:
    name: str
    model_path: str
    nodes: list[str]
    port: int = 8001
    gpu_count: int = 1
    dtype: str = "auto"
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    lora_path: str | None = None
    extra_args: list[str] = field(default_factory=list)


@dataclass
class ClusterConfig:
    nodes: list[NodeConfig] = field(default_factory=list)
    models: list[ModelSpec] = field(default_factory=list)
    orchestrator_host: str = "0.0.0.0"
    orchestrator_port: int = 7860
    agent_host: str = "0.0.0.0"
    agent_port: int = 7861
    api_key: str = ""
    container_runtime: str = "docker"

    def node(self, name: str) -> NodeConfig | None:
        return next((n for n in self.nodes if n.name == name), None)

    def model(self, name: str) -> ModelSpec | None:
        return next((m for m in self.models if m.name == name), None)

    def models_for_node(self, node_name: str) -> list[ModelSpec]:
        return [m for m in self.models if node_name in m.nodes]


def load_cluster_config() -> ClusterConfig:
    """Load cluster config from the standard vllmd config files."""
    raw = _load_config()

    nodes = [
        NodeConfig(
            name=n["name"],
            host=n["host"],
            agent_port=n.get("agent_port", 7861),
            gpus=n.get("gpus", []),
        )
        for n in raw.get("nodes", [])
    ]

    models = [
        ModelSpec(
            name=m["name"],
            model_path=m["model_path"],
            nodes=m.get("nodes", []),
            port=m.get("port", 8001),
            gpu_count=m.get("gpu_count", 1),
            dtype=m.get("dtype", "auto"),
            max_model_len=m.get("max_model_len"),
            gpu_memory_utilization=m.get("gpu_memory_utilization", 0.9),
            lora_path=m.get("lora_path"),
            extra_args=m.get("extra_args", []),
        )
        for m in raw.get("models", [])
    ]

    orch = raw.get("orchestrator", {})
    agent_cfg = raw.get("agent", {})

    return ClusterConfig(
        nodes=nodes,
        models=models,
        orchestrator_host=orch.get("host", "0.0.0.0"),
        orchestrator_port=orch.get("port", 7860),
        agent_host=agent_cfg.get("host", "0.0.0.0"),
        agent_port=agent_cfg.get("port", 7861),
        api_key=orch.get("api_key", ""),
        container_runtime=raw.get("container_runtime", "docker"),
    )
