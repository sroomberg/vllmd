"""Tests for the vllmd agent daemon."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vllmd.agent.server import app


@pytest.fixture
def client():
    return TestClient(app)


def _mock_containers(gpu_devices: list[int] | None = None) -> list[dict]:
    return [
        {
            "name": "vllmd-llama3",
            "model_id": "llama3",
            "model_path": "/model",
            "port": 8001,
            "endpoint": "http://localhost:8001",
            "status": "Up 2 hours",
            "gpu_devices": gpu_devices or [],
        }
    ]


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_get_gpus_no_env(client, monkeypatch):
    monkeypatch.delenv("VLLMD_NODE_GPUS", raising=False)
    with patch("vllmd.agent.server.list_containers", return_value=[]):
        resp = client.get("/gpus")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_gpus"] == []
    assert data["free"] == []


def test_get_gpus_with_env(client, monkeypatch):
    monkeypatch.setenv("VLLMD_NODE_GPUS", "0,1,2")
    with patch(
        "vllmd.agent.server.list_containers", return_value=_mock_containers([0])
    ):
        resp = client.get("/gpus")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_gpus"] == [0, 1, 2]
    assert data["allocated"] == [0]
    assert data["free"] == [1, 2]


def test_list_models(client):
    with patch("vllmd.agent.server.list_containers", return_value=_mock_containers()):
        resp = client.get("/models")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_get_model(client):
    state = {"running": True, "api_healthy": True, "container": {}}
    with patch("vllmd.agent.server.status", return_value=state):
        resp = client.get("/models/vllmd-llama3")
    assert resp.status_code == 200
    assert resp.json()["running"] is True


def test_start_model(client, monkeypatch):
    monkeypatch.delenv("VLLMD_NODE_GPUS", raising=False)
    with (
        patch("vllmd.agent.server.container_exists", return_value=False),
        patch(
            "vllmd.agent.server.build_docker_run_cmd", return_value=["docker", "run"]
        ),
        patch("vllmd.agent.server.subprocess.run") as mock_run,
    ):
        resp = client.post(
            "/models/llama3/start",
            json={"model_path": "meta-llama/Llama-3", "port": 8001},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"
    mock_run.assert_called_once()


def test_start_model_conflict(client):
    with patch("vllmd.agent.server.container_exists", return_value=True):
        resp = client.post(
            "/models/llama3/start",
            json={"model_path": "meta-llama/Llama-3", "port": 8001},
        )
    assert resp.status_code == 409


def test_start_model_insufficient_gpus(client, monkeypatch):
    monkeypatch.setenv("VLLMD_NODE_GPUS", "0")
    with (
        patch("vllmd.agent.server.container_exists", return_value=False),
        patch("vllmd.agent.server.list_containers", return_value=_mock_containers([0])),
    ):
        resp = client.post(
            "/models/llama3/start",
            json={"model_path": "meta-llama/Llama-3", "port": 8001, "gpu_count": 2},
        )
    assert resp.status_code == 409


def test_stop_model(client):
    with (
        patch("vllmd.agent.server.container_exists", return_value=True),
        patch("vllmd.agent.server.stop") as mock_stop,
    ):
        resp = client.post("/models/llama3/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"
    mock_stop.assert_called_once_with("llama3")


def test_stop_model_not_found(client):
    with patch("vllmd.agent.server.container_exists", return_value=False):
        resp = client.post("/models/llama3/stop")
    assert resp.status_code == 404


def test_get_logs(client):
    with patch(
        "vllmd.agent.server.subprocess.run",
        return_value=MagicMock(stdout="log line\n", stderr=""),
    ):
        resp = client.get("/models/llama3/logs?tail=50")
    assert resp.status_code == 200
    assert "log line" in resp.json()["logs"]
