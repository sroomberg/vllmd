"""CLI entry point."""

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from .runner import (
    RunConfig,
    _detect_lora_rank,
    build_docker_run_cmd,
    list_containers,
    logs,
    set_runtime,
    status,
    stop,
    stop_all,
    stop_by_port,
    wait_ready,
)
from .sessions.cli import session_app
from .vectordb.cli import db_app

app = typer.Typer(
    name="vllmd",
    help="Run local models via vLLM in Docker containers.",
    no_args_is_help=True,
)
app.add_typer(db_app, name="db")
app.add_typer(session_app, name="session")
console = Console()

_AGENT_PID = Path.home() / ".local" / "share" / "vllmd" / "agent.pid"
_ORCHESTRATOR_PID = Path.home() / ".local" / "share" / "vllmd" / "orchestrator.pid"


# ---------------------------------------------------------------------------
# Daemon helpers
# ---------------------------------------------------------------------------


def _start_daemon(
    app_import: str,
    host: str,
    port: int,
    pid_file: Path,
    label: str,
    runtime: str | None = None,
) -> None:
    """Start a uvicorn daemon, writing its PID to *pid_file*."""
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
            console.print(f"[yellow]{label} is already running (pid {pid}).[/yellow]")
            return
        except ProcessLookupError:
            pid_file.unlink()

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ}
    if runtime:
        env["VLLMD_RUNTIME"] = runtime
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            app_import,
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    pid_file.write_text(str(proc.pid))
    console.print(f"[green]{label} started[/green] (pid {proc.pid}) on {host}:{port}")


def _stop_daemon(pid_file: Path, label: str) -> None:
    if not pid_file.exists():
        console.print(f"[yellow]{label} is not running (no PID file).[/yellow]")
        return
    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]{label} stopped[/green] (pid {pid})")
    except ProcessLookupError:
        console.print(f"[yellow]Removed stale PID for {label} (pid {pid}).[/yellow]")
    pid_file.unlink()


# ---------------------------------------------------------------------------
# Agent sub-app
# ---------------------------------------------------------------------------

agent_app = typer.Typer(
    name="agent",
    help="Manage the vllmd node agent daemon.",
    no_args_is_help=True,
)
app.add_typer(agent_app, name="agent")


@agent_app.command(name="start")
def agent_start(
    host: Annotated[str, typer.Option("--host", help="Bind host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", help="Bind port")] = 7861,
    runtime: Annotated[
        str,
        typer.Option("--runtime", help="Container runtime (docker, podman, …)"),
    ] = "docker",
) -> None:
    """Start the vllmd agent daemon on this node."""
    _start_daemon("vllmd.agent.server:app", host, port, _AGENT_PID, "Agent", runtime)


@agent_app.command(name="stop")
def agent_stop() -> None:
    """Stop the running vllmd agent daemon."""
    _stop_daemon(_AGENT_PID, "Agent")


# ---------------------------------------------------------------------------
# Orchestrator sub-app
# ---------------------------------------------------------------------------

orchestrator_app = typer.Typer(
    name="orchestrator",
    help="Manage the vllmd orchestrator service.",
    no_args_is_help=True,
)
app.add_typer(orchestrator_app, name="orchestrator")


@orchestrator_app.command(name="start")
def orchestrator_start(
    host: Annotated[str, typer.Option("--host", help="Bind host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", help="Bind port")] = 7860,
) -> None:
    """Start the vllmd orchestrator service."""
    _start_daemon(
        "vllmd.orchestrator.server:app",
        host,
        port,
        _ORCHESTRATOR_PID,
        "Orchestrator",
    )


@orchestrator_app.command(name="stop")
def orchestrator_stop() -> None:
    """Stop the running vllmd orchestrator service."""
    _stop_daemon(_ORCHESTRATOR_PID, "Orchestrator")


# ---------------------------------------------------------------------------
# Cluster commands (up / down / nodes)
# ---------------------------------------------------------------------------


def _orchestrator_url() -> str:
    """Return the orchestrator base URL from config or default."""
    try:
        from .cluster.config import load_cluster_config

        cfg = load_cluster_config()
        h = cfg.orchestrator_host
        host = h if h != "0.0.0.0" else "localhost"
        return f"http://{host}:{cfg.orchestrator_port}"
    except Exception:
        return "http://localhost:7860"


@app.command(name="up")
def up_cmd(
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Start only this model (default: all)"),
    ] = None,
    wait: Annotated[
        bool,
        typer.Option("--wait/--no-wait", help="Wait for models to start"),
    ] = True,
) -> None:
    """Start all (or one) configured models via the orchestrator."""
    import urllib.request

    base = _orchestrator_url()
    path = f"/cluster/up/{model}" if model else "/cluster/up"
    url = f"{base}{path}"
    req = urllib.request.Request(
        url, method="POST", data=b"", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=300 if wait else 10) as resp:
            import json

            result = json.loads(resp.read())
        for r in result.get("results", []):
            if "error" in r:
                console.print(f"[red]{r['node']}/{r['model']}: {r['error']}[/red]")
            else:
                status_str = r.get("status", "ok")
                console.print(f"[green]{r['node']}/{r['model']}: {status_str}[/green]")
    except Exception as exc:
        console.print(f"[red]Failed to reach orchestrator at {base}: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command(name="down")
def down_cmd(
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Stop only this model (default: all)"),
    ] = None,
) -> None:
    """Stop all (or one) configured models via the orchestrator."""
    import urllib.request

    base = _orchestrator_url()
    path = f"/cluster/down/{model}" if model else "/cluster/down"
    url = f"{base}{path}"
    req = urllib.request.Request(
        url, method="POST", data=b"", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            import json

            result = json.loads(resp.read())
        for r in result.get("results", []):
            if "error" in r:
                console.print(f"[red]{r['node']}/{r['model']}: {r['error']}[/red]")
            else:
                status_str = r.get("status", "ok")
                console.print(f"[green]{r['node']}/{r['model']}: {status_str}[/green]")
    except Exception as exc:
        console.print(f"[red]Failed to reach orchestrator at {base}: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command(name="nodes")
def nodes_cmd() -> None:
    """List configured nodes and their agent health."""
    import json
    import urllib.request

    base = _orchestrator_url()
    try:
        with urllib.request.urlopen(f"{base}/cluster/status", timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        console.print(f"[red]Failed to reach orchestrator at {base}: {exc}[/red]")
        raise typer.Exit(1) from exc

    table = Table("Node", "Host", "Health", "Models", show_header=True)
    for node in data.get("nodes", []):
        health = (
            "[green]healthy[/green]" if node["healthy"] else "[red]unreachable[/red]"
        )
        model_names = (
            ", ".join(m.get("model_id", "?") for m in node.get("models", [])) or "—"
        )
        table.add_row(node["node"], node["host"], health, model_names)
    console.print(table)


_NAME_HELP = "Container name (default: vllmd-<model-dir-name>)"


@app.command()
def run(
    model: Annotated[
        Path,
        typer.Option("--model", "-m", help="Path to the model directory on disk"),
    ],
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Host port to expose the vLLM API on"),
    ] = 8000,
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help=_NAME_HELP),
    ] = None,
    gpu: Annotated[
        bool,
        typer.Option("--gpu/--no-gpu", help="Pass --gpus all to the container"),
    ] = True,
    dtype: Annotated[
        str,
        typer.Option("--dtype", help="Model dtype (auto, float16, bfloat16, float32)"),
    ] = "auto",
    max_model_len: Annotated[
        Optional[int],
        typer.Option("--max-model-len", help="Override max context length"),
    ] = None,
    lora: Annotated[
        Optional[Path],
        typer.Option("--lora", "-l", help="Path to LoRA adapter directory"),
    ] = None,
    max_lora_rank: Annotated[
        Optional[int],
        typer.Option(
            "--max-lora-rank",
            help="Max LoRA rank (auto-detected from adapter_config.json if omitted)",
        ),
    ] = None,
    detach: Annotated[
        bool,
        typer.Option(
            "--detach",
            "-d",
            is_flag=True,
            help="Start container in background; wait for API ready, then return.",
        ),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait/--no-wait",
            help="With --detach: wait for the API to be ready before returning.",
        ),
    ] = True,
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            help="Container runtime executable (default: docker)",
        ),
    ] = "docker",
    extra: Annotated[
        Optional[list[str]],
        typer.Argument(help="Extra args forwarded verbatim to vLLM"),
    ] = None,
) -> None:
    """Start a vLLM container serving MODEL on PORT."""
    set_runtime(runtime)
    if lora is not None and max_lora_rank is None:
        max_lora_rank = _detect_lora_rank(lora)

    config = RunConfig(
        model_path=model,
        port=port,
        name=name,
        gpu=gpu,
        dtype=dtype,
        max_model_len=max_model_len,
        lora_path=lora,
        max_lora_rank=max_lora_rank,
        extra_args=extra or [],
    )

    console.print(f"[bold]Starting vLLM container[/bold] '{config.container_name}'")
    if config.is_hub_model:
        console.print(f"  Model:    {config.model_id} [dim](HuggingFace Hub)[/dim]")
    else:
        console.print(f"  Model:    {model.resolve()}")
        console.print(f"  Model ID: {config.model_id}")
    console.print(f"  Port:     {port}")
    console.print(f"  GPU:      {'yes' if gpu else 'no'}")
    if lora is not None:
        console.print(f"  LoRA:     {lora.resolve()}")
        if max_lora_rank is not None:
            console.print(f"  LoRA rank: {max_lora_rank}")
    console.print()

    docker_cmd = build_docker_run_cmd(config)

    if detach:
        subprocess.Popen(
            docker_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if wait:
            console.print("[dim]Waiting for API to become ready…[/dim]")
            if wait_ready(config):
                console.print(f"[green]Ready.[/green] Endpoint: {config.endpoint}")
                console.print(f"  Model ID: [bold]{config.model_id}[/bold]")
            else:
                console.print(
                    "[yellow]Timed out waiting for API. "
                    "Container may still be loading.[/yellow]"
                )
                console.print(f"  Endpoint: {config.endpoint}")
        else:
            console.print(f"Container started. Endpoint: {config.endpoint}")
    else:
        try:
            subprocess.run(docker_cmd, check=True)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Docker exited with code {e.returncode}[/red]")
            raise typer.Exit(e.returncode) from e


@app.command(name="stop")
def stop_cmd(
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Container name to stop"),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option("--port", "-p", help="Port of the container to stop"),
    ] = None,
    all_containers: Annotated[
        bool,
        typer.Option(
            "--all", "-a", is_flag=True, help="Stop all vllmd-managed containers"
        ),
    ] = False,
) -> None:
    """Stop one or all vllmd containers."""
    if all_containers:
        stopped = stop_all()
        if stopped:
            for n in stopped:
                console.print(f"[green]Stopped '{n}'.[/green]")
        else:
            console.print("[yellow]No running vllmd containers found.[/yellow]")
        return

    if name is None and port is None:
        console.print("[red]Specify --name, --port, or --all.[/red]")
        raise typer.Exit(1)

    try:
        if port is not None:
            stopped_name = stop_by_port(port)
            console.print(f"[green]Stopped '{stopped_name}'.[/green]")
        else:
            stop(name)  # type: ignore[arg-type]
            console.print(f"[green]Stopped '{name}'.[/green]")
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Docker exited with code {e.returncode}[/red]")
        raise typer.Exit(e.returncode) from e


@app.command(name="ps")
def ps_cmd() -> None:
    """List all running vllmd-managed containers."""
    containers = list_containers()
    if not containers:
        console.print("[dim]No vllmd containers running.[/dim]")
        return

    table = Table("Name", "Model", "Port", "Endpoint", "Status", show_header=True)
    for c in containers:
        table.add_row(
            c["name"],
            c["model_id"],
            str(c["port"] or "?"),
            c["endpoint"],
            c["status"],
        )
    console.print(table)


@app.command(name="status")
def status_cmd(
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Container name (omit to show all)"),
    ] = None,
) -> None:
    """Show container and API status. Omit --name to show all containers."""
    if name is None:
        # Show summary table for all managed containers
        containers = list_containers()
        if not containers:
            console.print("[dim]No vllmd containers running.[/dim]")
            raise typer.Exit(1)

        table = Table("Name", "Model", "Port", "API", "Status", show_header=True)
        all_healthy = True
        for c in containers:
            info = status(c["name"])
            api_str = (
                "[green]healthy[/green]"
                if info["api_healthy"]
                else "[yellow]unreachable[/yellow]"
            )
            if not info["api_healthy"]:
                all_healthy = False
            table.add_row(
                c["name"],
                c["model_id"],
                str(c["port"] or "?"),
                api_str,
                c["status"],
            )
        console.print(table)
        if not all_healthy:
            raise typer.Exit(1)
        return

    info = status(name)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    running_str = "[green]running[/green]" if info["running"] else "[red]stopped[/red]"
    api_str = (
        "[green]healthy[/green]"
        if info["api_healthy"]
        else "[yellow]unreachable[/yellow]"
    )

    table.add_row("Container", running_str)
    table.add_row("API", api_str)
    if info["container"]:
        table.add_row("Started", info["container"].get("StartedAt", "—"))

    console.print(table)
    if not info["running"]:
        raise typer.Exit(1)


@app.command(name="logs")
def logs_cmd(
    name: Annotated[
        Optional[str],
        typer.Option(
            "--name", "-n", help="Container name (auto-resolved if only one is running)"
        ),
    ] = None,
    follow: Annotated[
        bool,
        typer.Option("--follow", "-f", is_flag=True, help="Follow log output"),
    ] = False,
) -> None:
    """Print logs from a vllmd container."""
    if name is None:
        running = list_containers()
        if len(running) == 1:
            name = running[0]["name"]
        elif len(running) == 0:
            console.print("[yellow]No running vllmd containers found.[/yellow]")
            raise typer.Exit(1)
        else:
            console.print("[red]Multiple containers running — specify --name:[/red]")
            for c in running:
                console.print(f"  {c['name']}")
            raise typer.Exit(1)

    try:
        logs(name, follow=follow)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Docker exited with code {e.returncode}[/red]")
        raise typer.Exit(e.returncode) from e
