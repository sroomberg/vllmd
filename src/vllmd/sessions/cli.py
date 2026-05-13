"""CLI subcommands for session management."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ..runner import list_containers
from .chat import chat, retrieve_context
from .factory import get_session_store
from .providers.base import BaseSessionStore
from .providers.local import LocalSessionStore
from .session import Session

session_app = typer.Typer(
    name="session",
    help="Create and manage persistent model chat sessions.",
    no_args_is_help=True,
)
console = Console()


def _resolve_store(sessions_dir_override: Optional[Path]) -> BaseSessionStore:
    if sessions_dir_override:
        return LocalSessionStore(sessions_dir_override)
    return get_session_store()


def _load_session_or_exit(session_id: str, store: BaseSessionStore) -> Session:
    try:
        return Session.load(session_id, store)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e


def _resolve_endpoint_and_model(
    endpoint: Optional[str],
    model_id: Optional[str],
    name: Optional[str],
) -> tuple[str, str]:
    """Return (endpoint, model_id), auto-resolving from running containers if needed."""
    if endpoint and model_id:
        return endpoint, model_id

    containers = list_containers()

    if name:
        match = next((c for c in containers if c["name"] == name), None)
        if not match:
            raise RuntimeError(f"No running container named '{name}'.")
        return match["endpoint"], match["model_id"]

    if len(containers) == 1:
        c = containers[0]
        return c["endpoint"], c["model_id"]

    if not containers:
        raise RuntimeError(
            "No running vllmd containers found. "
            "Start one with `vllmd run`, or pass --endpoint and --model."
        )

    names = ", ".join(c["name"] for c in containers)
    raise RuntimeError(
        f"Multiple containers running ({names}). "
        "Specify --container <name>, or pass --endpoint and --model explicitly."
    )


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------


@session_app.command()
def create(
    session_id: Annotated[str, typer.Argument(help="Unique session name/ID")],
    container: Annotated[
        Optional[str],
        typer.Option(
            "--container",
            "-c",
            help="Container name to bind to (auto-resolved if one is running)",
        ),
    ] = None,
    endpoint: Annotated[
        Optional[str],
        typer.Option(
            "--endpoint",
            "-e",
            help="vLLM endpoint override (e.g. http://localhost:8001)",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Served model ID override"),
    ] = None,
    system_prompt: Annotated[
        str,
        typer.Option(
            "--system-prompt", "-s", help="System prompt prepended to every request"
        ),
    ] = "",
    embedding_model: Annotated[
        str,
        typer.Option(
            "--embedding-model",
            help="Model ID for context retrieval embeddings (leave empty to disable)",
        ),
    ] = "",
    db_path: Annotated[
        Optional[Path],
        typer.Option(
            "--db-path",
            help="Context vector store path (default: <sessions-dir>/<id>/vectordb)",
        ),
    ] = None,
    sessions_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--sessions-dir",
            help="Sessions directory (default: ~/.vllmd/sessions)",
        ),
    ] = None,
) -> None:
    """Create a new chat session bound to a running model."""
    store = _resolve_store(sessions_dir)

    try:
        ep, mid = _resolve_endpoint_and_model(endpoint, model, container)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    if isinstance(store, LocalSessionStore):
        resolved_db = db_path or (store._dir / session_id / "vectordb")
    else:
        resolved_db = db_path or Path(f"./{session_id}/vectordb")

    session = Session.create(
        session_id,
        endpoint=ep,
        model_id=mid,
        db_path=resolved_db,
        system_prompt=system_prompt,
        embedding_model=embedding_model,
    )
    session.save(store)

    console.print(f"[green]Session '[bold]{session_id}[/bold]' created.[/green]")
    console.print(f"  Endpoint:  {ep}")
    console.print(f"  Model:     {mid}")
    console.print(f"  DB path:   {resolved_db}")
    if embedding_model:
        console.print(f"  Embeddings: {embedding_model}")
    if system_prompt:
        truncated = system_prompt[:60] + ("…" if len(system_prompt) > 60 else "")
        console.print(f"  System:    {truncated}")


@session_app.command(name="list")
def list_cmd(
    sessions_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--sessions-dir",
            help="Sessions directory (default: ~/.vllmd/sessions)",
        ),
    ] = None,
) -> None:
    """List all sessions."""
    store = _resolve_store(sessions_dir)
    sessions = Session.list_all(store)

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table("ID", "Model", "Endpoint", "Messages", "Created", show_header=True)
    for s in sessions:
        table.add_row(
            s.id,
            s.model_id,
            s.endpoint,
            str(s.message_count()),
            s.created_at[:19].replace("T", " "),
        )
    console.print(table)


@session_app.command(name="chat")
def chat_cmd(
    session_id: Annotated[str, typer.Argument(help="Session ID")],
    message: Annotated[str, typer.Argument(help="Message to send")],
    n_context: Annotated[
        int,
        typer.Option("--context", help="Number of context chunks to retrieve"),
    ] = 3,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Max tokens for the response"),
    ] = 2048,
    sessions_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--sessions-dir",
            help="Sessions directory (default: ~/.vllmd/sessions)",
        ),
    ] = None,
) -> None:
    """Send a single message in a session and print the response."""
    store = _resolve_store(sessions_dir)
    session = _load_session_or_exit(session_id, store)

    try:
        response = chat(session, message, n_context=n_context, max_tokens=max_tokens)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    session.save(store)
    console.print(Markdown(response))


@session_app.command()
def attach(
    session_id: Annotated[str, typer.Argument(help="Session ID")],
    n_context: Annotated[
        int,
        typer.Option("--context", help="Number of context chunks to retrieve"),
    ] = 3,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Max tokens per response"),
    ] = 2048,
    sessions_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--sessions-dir",
            help="Sessions directory (default: ~/.vllmd/sessions)",
        ),
    ] = None,
) -> None:
    """Open an interactive REPL for a session.

    Commands: /history, /context <query>, /reset, /exit
    """
    store = _resolve_store(sessions_dir)
    session = _load_session_or_exit(session_id, store)

    ctx_status = "on" if session.embedding_model else "off"
    console.print(
        Panel(
            f"[bold]{session.id}[/bold]  ·  "
            f"model [cyan]{session.model_id}[/cyan]  ·  {session.endpoint}\n"
            f"[dim]{session.message_count()} messages in history  ·  "
            f"context retrieval {ctx_status}[/dim]\n"
            "[dim]/history  /context <q>  /reset  /exit[/dim]",
            title="vllmd Session",
        )
    )

    while True:
        try:
            user_input = console.input("[bold cyan]>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting.[/dim]")
            break

        if not user_input:
            continue

        if user_input in ("/exit", "exit", "quit", "/quit"):
            break

        if user_input == "/history":
            _print_history(session)
            continue

        if user_input == "/reset":
            session.clear_history()
            session.save(store)
            console.print("[dim]History cleared.[/dim]")
            continue

        if user_input.startswith("/context "):
            query = user_input[len("/context ") :].strip()
            ctx = retrieve_context(session, query, n_context)
            if ctx:
                console.print(Panel(ctx, title="Retrieved context"))
            else:
                console.print("[dim]No context retrieved.[/dim]")
            continue

        if user_input.startswith("/"):
            console.print(f"[yellow]Unknown command: {user_input}[/yellow]")
            continue

        try:
            response = chat(
                session, user_input, n_context=n_context, max_tokens=max_tokens
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

        session.save(store)
        console.print()
        console.print(Markdown(response))
        console.print()


@session_app.command()
def history(
    session_id: Annotated[str, typer.Argument(help="Session ID")],
    last: Annotated[
        int,
        typer.Option("--last", "-n", help="Show only the last N messages"),
    ] = 0,
    sessions_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--sessions-dir",
            help="Sessions directory (default: ~/.vllmd/sessions)",
        ),
    ] = None,
) -> None:
    """Print the conversation history for a session."""
    store = _resolve_store(sessions_dir)
    session = _load_session_or_exit(session_id, store)
    _print_history(session, last=last or None)


@session_app.command()
def clear(
    session_id: Annotated[str, typer.Argument(help="Session ID")],
    sessions_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--sessions-dir",
            help="Sessions directory (default: ~/.vllmd/sessions)",
        ),
    ] = None,
) -> None:
    """Clear the conversation history for a session (keeps session config)."""
    store = _resolve_store(sessions_dir)
    session = _load_session_or_exit(session_id, store)
    count = session.message_count()
    session.clear_history()
    session.save(store)
    console.print(
        f"[green]Cleared {count} messages from session '{session_id}'.[/green]"
    )


@session_app.command()
def delete(
    session_id: Annotated[str, typer.Argument(help="Session ID")],
    sessions_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--sessions-dir",
            help="Sessions directory (default: ~/.vllmd/sessions)",
        ),
    ] = None,
) -> None:
    """Delete a session and its metadata."""
    store = _resolve_store(sessions_dir)
    session = _load_session_or_exit(session_id, store)
    session.delete(store)
    console.print(f"[green]Session '{session_id}' deleted.[/green]")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _print_history(session: Session, last: Optional[int] = None) -> None:
    messages = session.messages
    if last:
        messages = messages[-last:]
    if not messages:
        console.print("[dim]No messages.[/dim]")
        return
    for msg in messages:
        role_style = "cyan" if msg.role == "user" else "green"
        console.print(f"[bold {role_style}]{msg.role}[/bold {role_style}]")
        console.print(Markdown(msg.content))
        console.print()
