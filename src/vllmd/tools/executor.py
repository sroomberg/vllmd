"""Tool executor — runs tool calls issued by the agent loop."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

_MAX_OUTPUT = 8 * 1024  # 8 KB cap on tool output sent back to the model


class ToolExecutor:
    """Executes tool calls in a working directory.

    Optionally uses a PEM key for git SSH operations.
    """

    def __init__(
        self,
        workdir: str = ".",
        pem_path: str | None = None,
    ) -> None:
        self.workdir = str(Path(workdir).resolve())
        self.pem_path = pem_path

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def execute(self, name: str, arguments: str | dict) -> str:
        """Execute *name* with *arguments* (JSON string or dict)."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return f"Error: could not parse arguments: {exc}"

        method = getattr(self, f"_tool_{name}", None)
        if method is None:
            return f"Error: unknown tool '{name}'"
        try:
            return method(**arguments)
        except TypeError as exc:
            return f"Error: bad arguments for '{name}': {exc}"
        except Exception as exc:
            return f"Error: {exc}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _git_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.pem_path:
            key = Path(self.pem_path).expanduser().resolve()
            env["GIT_SSH_COMMAND"] = f"ssh -i {key} -o StrictHostKeyChecking=no"
        return env

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= _MAX_OUTPUT:
            return text or "(no output)"
        half = _MAX_OUTPUT // 2
        omitted = len(text) - _MAX_OUTPUT
        return text[:half] + f"\n... [{omitted} bytes omitted] ...\n" + text[-half:]

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_bash(self, command: str, workdir: str | None = None) -> str:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=workdir or self.workdir,
        )
        combined = result.stdout + result.stderr
        return self._truncate(combined)

    def _tool_read_file(self, path: str) -> str:
        content = Path(path).read_text()
        return self._truncate(content)

    def _tool_write_file(self, path: str, content: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} characters to {path}"

    def _tool_git_clone(
        self,
        url: str,
        dest: str | None = None,
        branch: str | None = None,
    ) -> str:
        cmd = ["git", "clone", url]
        if branch:
            cmd += ["--branch", branch]
        if dest:
            cmd.append(dest)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.workdir,
            env=self._git_env(),
        )
        combined = result.stdout + result.stderr
        if result.returncode != 0:
            return f"Error (exit {result.returncode}): {combined}"
        return self._truncate(combined)

    def _tool_git_commit(self, message: str, workdir: str | None = None) -> str:
        wd = workdir or self.workdir
        add = subprocess.run(
            ["git", "add", "-A"],
            capture_output=True,
            text=True,
            cwd=wd,
        )
        if add.returncode != 0:
            return f"Error during git add: {add.stderr}"
        commit = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True,
            cwd=wd,
        )
        combined = commit.stdout + commit.stderr
        if commit.returncode != 0:
            return f"Error (exit {commit.returncode}): {combined}"
        return self._truncate(combined)

    def _tool_git_push(
        self,
        branch: str,
        remote: str = "origin",
        workdir: str | None = None,
    ) -> str:
        wd = workdir or self.workdir
        result = subprocess.run(
            ["git", "push", remote, branch],
            capture_output=True,
            text=True,
            cwd=wd,
            env=self._git_env(),
        )
        combined = result.stdout + result.stderr
        if result.returncode != 0:
            return f"Error (exit {result.returncode}): {combined}"
        return self._truncate(combined)
