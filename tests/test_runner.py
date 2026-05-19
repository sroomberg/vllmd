"""Unit tests for vllmd.runner (no Docker required)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from vllmd.runner import (
    MANAGED_LABEL,
    MODEL_LABEL,
    RunConfig,
    _parse_host_port,
    _parse_labels,
    _wait_ready,
    build_docker_run_cmd,
    container_exists,
    detect_lora_rank,
)


def test_run_config_model_id(tmp_path: Path) -> None:
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    cfg = RunConfig(model_path=model_dir, port=8000)
    assert cfg.model_id == "my-model"


def test_run_config_container_name_default(tmp_path: Path) -> None:
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    cfg = RunConfig(model_path=model_dir)
    assert cfg.container_name == "vllmd-my-model"


def test_run_config_container_name_explicit() -> None:
    cfg = RunConfig(model_path=Path("/models/foo"), name="custom-name")
    assert cfg.container_name == "custom-name"


def test_run_config_endpoint() -> None:
    cfg = RunConfig(model_path=Path("/models/foo"), port=9001)
    assert cfg.endpoint == "http://localhost:9001"


def test_is_hub_model_true() -> None:
    cfg = RunConfig(model_path=Path("meta-llama/Llama-3-8B"))
    assert cfg.is_hub_model is True


def test_is_hub_model_false_local(tmp_path: Path) -> None:
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    cfg = RunConfig(model_path=model_dir)
    assert cfg.is_hub_model is False


def test_hub_model_id() -> None:
    cfg = RunConfig(model_path=Path("meta-llama/Llama-3-8B"))
    assert cfg.model_id == "meta-llama/Llama-3-8B"


def test_hub_model_container_name() -> None:
    cfg = RunConfig(model_path=Path("meta-llama/Llama-3-8B"))
    assert cfg.container_name == "vllmd-meta-llama-Llama-3-8B"


def test_hub_model_container_name_explicit() -> None:
    cfg = RunConfig(model_path=Path("meta-llama/Llama-3-8B"), name="my-llama")
    assert cfg.container_name == "my-llama"


def test_build_docker_run_cmd_hub_model() -> None:
    cfg = RunConfig(model_path=Path("meta-llama/Llama-3-8B"), port=8000, gpu=False)
    cmd = build_docker_run_cmd(cfg)
    # Should NOT mount a local model directory
    assert "/model:ro" not in " ".join(cmd)
    # Should mount HF cache
    assert "/root/.cache/huggingface" in " ".join(cmd)
    # --model should be the hub ID, not /model
    model_idx = cmd.index("--model")
    assert cmd[model_idx + 1] == "meta-llama/Llama-3-8B"
    # --served-model-name should be the full ID
    served_idx = cmd.index("--served-model-name")
    assert cmd[served_idx + 1] == "meta-llama/Llama-3-8B"


def test_build_docker_run_cmd_local_model_unchanged(tmp_path: Path) -> None:
    model_dir = tmp_path / "llama3"
    model_dir.mkdir()
    cfg = RunConfig(model_path=model_dir, port=8000, gpu=False)
    cmd = build_docker_run_cmd(cfg)
    assert "/model:ro" in " ".join(cmd)
    model_idx = cmd.index("--model")
    assert cmd[model_idx + 1] == "/model"


def test_build_docker_run_cmd_includes_label(tmp_path: Path) -> None:
    model_dir = tmp_path / "llama3"
    model_dir.mkdir()
    cfg = RunConfig(model_path=model_dir, port=8000)
    cmd = build_docker_run_cmd(cfg)
    assert f"--label={MANAGED_LABEL}=true" in cmd
    assert any(MODEL_LABEL in arg for arg in cmd)


def test_build_docker_run_cmd_no_gpu(tmp_path: Path) -> None:
    model_dir = tmp_path / "llama3"
    model_dir.mkdir()
    cfg = RunConfig(model_path=model_dir, port=8000, gpu=False)
    cmd = build_docker_run_cmd(cfg)
    assert "--gpus" not in cmd


def test_parse_host_port() -> None:
    assert _parse_host_port("0.0.0.0:8001->8000/tcp") == 8001
    assert _parse_host_port("0.0.0.0:9999->8000/tcp, :::9999->8000/tcp") == 9999
    assert _parse_host_port("") is None


def test_parse_labels() -> None:
    labels = _parse_labels("com.vllmd.managed=true,com.vllmd.model=llama3")
    assert labels["com.vllmd.managed"] == "true"
    assert labels["com.vllmd.model"] == "llama3"


def testcontainer_exists_false() -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="other-container\n")
        assert not container_exists("vllmd-llama3")


def testcontainer_exists_true() -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="vllmd-llama3\n")
        assert container_exists("vllmd-llama3")


def test_wait_ready_success() -> None:
    with patch("urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        assert _wait_ready("http://localhost:8000", timeout=5)


def test_wait_ready_timeout() -> None:
    with (
        patch("urllib.request.urlopen", side_effect=OSError("refused")),
        patch("time.sleep"),
    ):
        assert not _wait_ready("http://localhost:8000", timeout=1)


def testdetect_lora_rank_from_config(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "my-adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"r": 16, "lora_alpha": 32}')
    assert detect_lora_rank(adapter_dir) == 16


def testdetect_lora_rank_missing_file(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "my-adapter"
    adapter_dir.mkdir()
    assert detect_lora_rank(adapter_dir) is None


def testdetect_lora_rank_invalid_json(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "my-adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("not json")
    assert detect_lora_rank(adapter_dir) is None


def test_build_docker_run_cmd_with_lora(tmp_path: Path) -> None:
    model_dir = tmp_path / "llama3"
    model_dir.mkdir()
    lora_dir = tmp_path / "my-adapter"
    lora_dir.mkdir()
    cfg = RunConfig(
        model_path=model_dir, port=8000, lora_path=lora_dir, max_lora_rank=16
    )
    cmd = build_docker_run_cmd(cfg)
    assert "-v" in cmd
    lora_mount = f"{lora_dir.resolve()}:/lora:ro"
    assert lora_mount in cmd
    assert "--enable-lora" in cmd
    assert "--lora-modules" in cmd
    assert "my-adapter=/lora" in cmd
    assert "--max-lora-rank" in cmd
    assert "16" in cmd


def test_build_docker_run_cmd_with_lora_no_rank(tmp_path: Path) -> None:
    model_dir = tmp_path / "llama3"
    model_dir.mkdir()
    lora_dir = tmp_path / "my-adapter"
    lora_dir.mkdir()
    cfg = RunConfig(model_path=model_dir, port=8000, lora_path=lora_dir)
    cmd = build_docker_run_cmd(cfg)
    assert "--enable-lora" in cmd
    assert "--max-lora-rank" not in cmd


def test_build_docker_run_cmd_without_lora(tmp_path: Path) -> None:
    model_dir = tmp_path / "llama3"
    model_dir.mkdir()
    cfg = RunConfig(model_path=model_dir, port=8000)
    cmd = build_docker_run_cmd(cfg)
    assert "--enable-lora" not in cmd
    assert "--lora-modules" not in cmd
