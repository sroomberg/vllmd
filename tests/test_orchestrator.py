"""Tests for the orchestrator registry, router, and server."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from vllmd.orchestrator.registry import ModelRegistry
from vllmd.orchestrator.router import _last_index, pick_endpoint

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_register_and_get():
    reg = ModelRegistry()
    reg.register("llama3", "local", "http://localhost:8001")
    eps = reg.get_endpoints("llama3")
    assert len(eps) == 1
    assert eps[0].node == "local"
    assert eps[0].endpoint == "http://localhost:8001"
    assert eps[0].healthy is True


def test_register_updates_existing():
    reg = ModelRegistry()
    reg.register("llama3", "local", "http://localhost:8001")
    reg.register("llama3", "local", "http://localhost:8002")
    eps = reg.get_endpoints("llama3")
    assert len(eps) == 1
    assert eps[0].endpoint == "http://localhost:8002"


def test_deregister():
    reg = ModelRegistry()
    reg.register("llama3", "local", "http://localhost:8001")
    reg.register("llama3", "node2", "http://10.0.0.2:8001")
    reg.deregister("llama3", "local")
    eps = reg.get_endpoints("llama3")
    assert len(eps) == 1
    assert eps[0].node == "node2"


def test_mark_healthy():
    reg = ModelRegistry()
    reg.register("llama3", "local", "http://localhost:8001")
    reg.mark_healthy("llama3", "local", False)
    assert reg.get_endpoints("llama3")[0].healthy is False
    reg.mark_healthy("llama3", "local", True)
    assert reg.get_endpoints("llama3")[0].healthy is True


def test_all_models():
    reg = ModelRegistry()
    reg.register("llama3", "local", "http://localhost:8001")
    reg.register("llama70b", "local", "http://localhost:8002")
    assert set(reg.all_models()) == {"llama3", "llama70b"}


def test_dump_and_load_roundtrip():
    reg = ModelRegistry()
    reg.register("llama3", "local", "http://localhost:8001")
    reg.register("llama3", "node2", "http://10.0.0.2:8001")
    reg.mark_healthy("llama3", "node2", False)

    data = reg.dump()
    reg2 = ModelRegistry()
    reg2.load(data)

    eps = reg2.get_endpoints("llama3")
    assert len(eps) == 2
    local_ep = next(e for e in eps if e.node == "local")
    node2_ep = next(e for e in eps if e.node == "node2")
    assert local_ep.endpoint == "http://localhost:8001"
    assert local_ep.healthy is True
    assert node2_ep.healthy is False


def test_load_clears_existing():
    reg = ModelRegistry()
    reg.register("stale", "old-node", "http://old:8001")
    reg.load({"llama3": [{"node": "local", "endpoint": "http://localhost:8001"}]})
    assert reg.all_models() == ["llama3"]


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


def _reg_with_two_nodes() -> ModelRegistry:
    reg = ModelRegistry()
    reg.register("llama3", "node1", "http://node1:8001")
    reg.register("llama3", "node2", "http://node2:8001")
    return reg


def test_pick_endpoint_round_robin():
    reg = _reg_with_two_nodes()
    _last_index.pop("llama3", None)
    ep1 = pick_endpoint("llama3", reg)
    ep2 = pick_endpoint("llama3", reg)
    assert ep1.node != ep2.node


def test_pick_endpoint_node_pin():
    reg = _reg_with_two_nodes()
    ep = pick_endpoint("llama3", reg, node_pin="node2")
    assert ep.node == "node2"


def test_pick_endpoint_no_healthy():
    reg = ModelRegistry()
    reg.register("llama3", "node1", "http://node1:8001")
    reg.mark_healthy("llama3", "node1", False)
    with pytest.raises(HTTPException) as exc:
        pick_endpoint("llama3", reg)
    assert exc.value.status_code == 503


def test_pick_endpoint_pinned_unhealthy():
    reg = _reg_with_two_nodes()
    reg.mark_healthy("llama3", "node2", False)
    with pytest.raises(HTTPException) as exc:
        pick_endpoint("llama3", reg, node_pin="node2")
    assert exc.value.status_code == 503


def test_pick_endpoint_missing_model():
    reg = ModelRegistry()
    with pytest.raises(HTTPException) as exc:
        pick_endpoint("unknown", reg)
    assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# Orchestrator server tests
# ---------------------------------------------------------------------------


@pytest.fixture
def orch_client():

    from vllmd.cluster.config import ClusterConfig, ModelSpec, NodeConfig
    from vllmd.orchestrator import server as orch_server

    cfg = ClusterConfig(
        nodes=[NodeConfig(name="local", host="localhost", agent_port=7861, gpus=[0])],
        models=[
            ModelSpec(
                name="llama3",
                model_path="meta-llama/Llama-3",
                nodes=["local"],
                port=8001,
            )
        ],
    )
    orch_server._state.config = cfg
    orch_server._state.api_key = ""
    orch_server._state.registry = ModelRegistry()

    with TestClient(orch_server.app, raise_server_exceptions=True) as c:
        yield c

    orch_server._state.config = None
    orch_server._state.registry = ModelRegistry()


def test_orch_cluster_status(orch_client):
    from unittest.mock import AsyncMock, patch

    mock_client = AsyncMock()
    mock_client.health = AsyncMock(return_value={"status": "ok"})
    mock_client.list_models = AsyncMock(return_value=[])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("vllmd.orchestrator.server.AgentClient", return_value=mock_client):
        resp = orch_client.get("/cluster/status")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 1
    assert data["nodes"][0]["node"] == "local"
    assert data["nodes"][0]["healthy"] is True


def test_orch_v1_models(orch_client):
    from unittest.mock import AsyncMock, patch

    mock_client = AsyncMock()
    mock_client.list_models = AsyncMock(
        return_value=[{"model_id": "llama3", "port": 8001}]
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("vllmd.orchestrator.server.AgentClient", return_value=mock_client):
        resp = orch_client.get("/v1/models")

    assert resp.status_code == 200
    data = resp.json()
    assert any(m["id"] == "llama3" for m in data["data"])


def test_orch_proxy_no_healthy_endpoint(orch_client):
    resp = orch_client.post(
        "/v1/chat/completions",
        json={"model": "llama3", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 503


def test_registry_persistence(tmp_path):
    import json

    from vllmd.orchestrator import server as orch_server

    persist_file = tmp_path / "registry.json"
    orch_server._state.persist_path = persist_file
    orch_server._state.registry = ModelRegistry()
    orch_server._state.registry.register("llama3", "local", "http://localhost:8001")

    orch_server._save_registry()
    assert persist_file.exists()
    data = json.loads(persist_file.read_text())
    assert "llama3" in data

    orch_server._state.registry = ModelRegistry()
    assert orch_server._state.registry.get_endpoints("llama3") == []

    orch_server._load_registry()
    eps = orch_server._state.registry.get_endpoints("llama3")
    assert len(eps) == 1
    assert eps[0].endpoint == "http://localhost:8001"

    orch_server._state.persist_path = (
        __import__("pathlib").Path.home()
        / ".local"
        / "share"
        / "vllmd"
        / "registry.json"
    )
