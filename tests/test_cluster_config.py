"""Tests for cluster config loading."""

from __future__ import annotations

from unittest.mock import patch

from vllmd.cluster.config import ClusterConfig, NodeConfig, load_cluster_config


def _raw_config() -> dict:
    return {
        "nodes": [
            {"name": "local", "host": "localhost", "agent_port": 7861, "gpus": [0, 1]},
            {"name": "gpu2", "host": "10.0.0.2"},
        ],
        "models": [
            {
                "name": "llama3",
                "model_path": "meta-llama/Llama-3",
                "nodes": ["local", "gpu2"],
                "port": 8001,
                "gpu_count": 1,
                "dtype": "bfloat16",
                "max_model_len": 4096,
            },
            {
                "name": "llama70b",
                "model_path": "meta-llama/Llama-70B",
                "nodes": ["local"],
                "gpu_count": 2,
            },
        ],
        "orchestrator": {"host": "0.0.0.0", "port": 7860, "api_key": "secret"},
        "agent": {"host": "0.0.0.0", "port": 7861},
    }


def test_load_cluster_config():
    with patch("vllmd.cluster.config._load_config", return_value=_raw_config()):
        cfg = load_cluster_config()

    assert isinstance(cfg, ClusterConfig)
    assert len(cfg.nodes) == 2
    assert len(cfg.models) == 2
    assert cfg.api_key == "secret"
    assert cfg.orchestrator_port == 7860
    assert not hasattr(cfg, "container_runtime")


def test_node_defaults():
    with patch("vllmd.cluster.config._load_config", return_value=_raw_config()):
        cfg = load_cluster_config()

    gpu2 = cfg.node("gpu2")
    assert gpu2 is not None
    assert gpu2.agent_port == 7861
    assert gpu2.gpus == []
    assert gpu2.container_runtime == "docker"


def test_node_agent_url():
    node = NodeConfig(name="local", host="localhost", agent_port=7861)
    assert node.agent_url == "http://localhost:7861"


def test_model_defaults():
    with patch("vllmd.cluster.config._load_config", return_value=_raw_config()):
        cfg = load_cluster_config()

    m = cfg.model("llama70b")
    assert m is not None
    assert m.dtype == "auto"
    assert m.gpu_memory_utilization == 0.9
    assert m.max_model_len is None


def test_models_for_node():
    with patch("vllmd.cluster.config._load_config", return_value=_raw_config()):
        cfg = load_cluster_config()

    local_models = cfg.models_for_node("local")
    assert len(local_models) == 2

    gpu2_models = cfg.models_for_node("gpu2")
    assert len(gpu2_models) == 1
    assert gpu2_models[0].name == "llama3"


def test_missing_node_returns_none():
    with patch("vllmd.cluster.config._load_config", return_value=_raw_config()):
        cfg = load_cluster_config()
    assert cfg.node("nonexistent") is None


def test_empty_config():
    with patch("vllmd.cluster.config._load_config", return_value={}):
        cfg = load_cluster_config()
    assert cfg.nodes == []
    assert cfg.models == []
    assert cfg.orchestrator_port == 7860
