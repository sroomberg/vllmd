"""Tests for ToolExecutor."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vllmd.tools.executor import ToolExecutor


@pytest.fixture
def executor(tmp_path):
    return ToolExecutor(workdir=str(tmp_path))


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


def test_bash_stdout(executor):
    out = executor.execute("bash", {"command": "echo hello"})
    assert "hello" in out


def test_bash_stderr(executor):
    out = executor.execute("bash", {"command": "echo err >&2"})
    assert "err" in out


def test_bash_exit_nonzero(executor):
    out = executor.execute("bash", {"command": "exit 1"})
    assert out == "(no output)"


def test_bash_workdir_override(executor, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    out = executor.execute("bash", {"command": "pwd", "workdir": str(sub)})
    assert str(sub) in out


def test_bash_truncates_large_output(executor):
    out = executor.execute("bash", {"command": "python3 -c \"print('x' * 100000)\""})
    assert "omitted" in out
    assert len(out) <= 8 * 1024 + 200  # slight buffer for the omission message


# ---------------------------------------------------------------------------
# read_file / write_file
# ---------------------------------------------------------------------------


def test_write_and_read_file(executor, tmp_path):
    path = str(tmp_path / "hello.txt")
    result = executor.execute("write_file", {"path": path, "content": "hello world"})
    assert "hello.txt" in result

    content = executor.execute("read_file", {"path": path})
    assert content == "hello world"


def test_write_creates_parent_dirs(executor, tmp_path):
    path = str(tmp_path / "a" / "b" / "c.txt")
    executor.execute("write_file", {"path": path, "content": "deep"})
    assert Path(path).read_text() == "deep"


def test_read_missing_file(executor, tmp_path):
    result = executor.execute("read_file", {"path": str(tmp_path / "nope.txt")})
    assert result.startswith("Error:")


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------


def test_unknown_tool(executor):
    result = executor.execute("frobnicate", {"x": 1})
    assert "unknown tool" in result


# ---------------------------------------------------------------------------
# git_commit
# ---------------------------------------------------------------------------


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_git_commit(executor, tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "file.txt").write_text("content")
    result = executor.execute(
        "git_commit", {"message": "init", "workdir": str(tmp_path)}
    )
    assert "init" in result or "master" in result or "main" in result


def test_git_commit_nothing_to_commit(executor, tmp_path):
    _init_repo(tmp_path)
    # Nothing staged — git commit should fail gracefully
    result = executor.execute(
        "git_commit", {"message": "empty", "workdir": str(tmp_path)}
    )
    # Either "nothing to commit" in output or an error message
    assert result  # just ensure we get something back, not an exception


# ---------------------------------------------------------------------------
# JSON argument parsing
# ---------------------------------------------------------------------------


def test_execute_accepts_json_string(executor):
    import json

    out = executor.execute("bash", json.dumps({"command": "echo json"}))
    assert "json" in out


def test_execute_bad_json(executor):
    result = executor.execute("bash", "not-json")
    assert result.startswith("Error:")
