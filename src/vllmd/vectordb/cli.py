"""CLI subcommands for the vector context database."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from .embeddings import embed, make_embedder
from .store import (
    COLLECTION_CODE,
    COLLECTION_CONVERSATIONS,
    COLLECTION_DOCUMENTS,
    VectorStore,
)
from .sync import sync_pull, sync_push

db_app = typer.Typer(
    name="db",
    help="Manage the local context vector database.",
    no_args_is_help=True,
)
console = Console()


@db_app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File or directory to ingest")],
    type: Annotated[
        str,
        typer.Option("--type", "-t", help="Collection type: documents or code"),
    ] = "documents",
    endpoint: Annotated[
        str, typer.Option("--endpoint", "-e", help="vLLM server endpoint")
    ] = "http://localhost:8000",
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model ID for embeddings")
    ] = ...,
    db_path: Annotated[
        Path, typer.Option("--db-path", help="Path to the ChromaDB directory")
    ] = Path("./vectordb"),
) -> None:
    """Ingest documents or code files into the vector database."""
    embedder = make_embedder(endpoint, model)
    store = VectorStore(db_path)

    if type == "code":
        if path.is_dir():
            console.print(f"[dim]Scanning {path} for code files…[/dim]")
            results = store.ingest_code_dir(path, embedder)
            table = Table("File", "Chunks", show_header=True)
            for filepath, n in results.items():
                table.add_row(filepath, str(n))
            console.print(table)
            console.print(f"[green]Ingested {len(results)} files.[/green]")
        else:
            n = store.ingest_code_file(path, embedder)
            console.print(f"[green]Ingested {n} chunks from {path}.[/green]")
    else:
        if path.is_dir():
            total = 0
            files = [f for f in sorted(path.rglob("*")) if f.is_file()]
            for f in files:
                n = store.ingest_document(f, embedder)
                total += n
                console.print(f"  {f.name}: {n} chunks")
            console.print(
                f"[green]Ingested {total} chunks from {len(files)} files.[/green]"
            )
        else:
            n = store.ingest_document(path, embedder)
            console.print(f"[green]Ingested {n} chunks from {path}.[/green]")


@db_app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    collection: Annotated[
        str,
        typer.Option(
            "--collection",
            "-c",
            help="Collection to search: documents, code, conversations",
        ),
    ] = COLLECTION_DOCUMENTS,
    n: Annotated[int, typer.Option("--n", help="Number of results")] = 5,
    session: Annotated[
        Optional[str],
        typer.Option("--session", "-s", help="Filter conversations by session ID"),
    ] = None,
    endpoint: Annotated[
        str, typer.Option("--endpoint", "-e", help="vLLM server endpoint")
    ] = "http://localhost:8000",
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model ID for embeddings")
    ] = ...,
    db_path: Annotated[
        Path, typer.Option("--db-path", help="Path to the ChromaDB directory")
    ] = Path("./vectordb"),
) -> None:
    """Search the vector database for relevant context."""
    store = VectorStore(db_path)

    query_vec = embed(endpoint, model, [query])[0]
    where = (
        {"session_id": session}
        if session and collection == COLLECTION_CONVERSATIONS
        else None
    )

    results = store.search(query_vec, collection, n_results=n, where=where)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, r in enumerate(results, 1):
        console.rule(f"[bold]Result {i}[/bold] (distance: {r['distance']:.4f})")
        meta = r["metadata"]
        if collection == COLLECTION_CODE:
            console.print(
                f"[dim]{meta.get('filepath', '?')} ({meta.get('language', '?')})[/dim]"
            )
        elif collection == COLLECTION_DOCUMENTS:
            console.print(
                f"[dim]{meta.get('source', '?')} chunk {meta.get('chunk', '?')}[/dim]"
            )
        elif collection == COLLECTION_CONVERSATIONS:
            role = meta.get("role", "?")
            sid = meta.get("session_id", "?")
            console.print(f"[dim]{role} @ session {sid}[/dim]")
        console.print(r["content"][:500] + ("…" if len(r["content"]) > 500 else ""))


@db_app.command()
def history(
    content: Annotated[str, typer.Argument(help="Message content to store")],
    role: Annotated[
        str,
        typer.Option("--role", "-r", help="Message role: user or assistant"),
    ] = "user",
    session: Annotated[
        str,
        typer.Option("--session", "-s", help="Session ID (default: 'default')"),
    ] = "default",
    endpoint: Annotated[
        str, typer.Option("--endpoint", "-e", help="vLLM server endpoint")
    ] = "http://localhost:8000",
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model ID for embeddings")
    ] = ...,
    db_path: Annotated[
        Path, typer.Option("--db-path", help="Path to the ChromaDB directory")
    ] = Path("./vectordb"),
) -> None:
    """Store a conversation message in the history collection."""
    embedder = make_embedder(endpoint, model)
    store = VectorStore(db_path)
    msg_id = store.add_history(session, role, content, embedder)
    console.print(
        f"[green]Stored message {msg_id[:8]}… in session '{session}'.[/green]"
    )


@db_app.command()
def summarize(
    session: Annotated[
        str,
        typer.Option("--session", "-s", help="Session ID to summarize"),
    ] = "default",
    summary: Annotated[
        str,
        typer.Argument(help="Summary text to replace the session history"),
    ] = ...,
    endpoint: Annotated[
        str, typer.Option("--endpoint", "-e", help="vLLM server endpoint")
    ] = "http://localhost:8000",
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model ID for embeddings")
    ] = ...,
    db_path: Annotated[
        Path, typer.Option("--db-path", help="Path to the ChromaDB directory")
    ] = Path("./vectordb"),
) -> None:
    """Replace a session's conversation history with an abridged summary."""
    embedder = make_embedder(endpoint, model)
    store = VectorStore(db_path)
    store.replace_history_with_summary(session, summary, embedder)
    console.print(f"[green]Session '{session}' history replaced with summary.[/green]")


@db_app.command()
def sync(
    s3_uri: Annotated[
        str, typer.Argument(help="S3 URI (e.g. s3://my-bucket/vectordb)")
    ],
    direction: Annotated[
        str,
        typer.Option("--direction", help="push (local → S3) or pull (S3 → local)"),
    ] = "push",
    db_path: Annotated[
        Path, typer.Option("--db-path", help="Path to the ChromaDB directory")
    ] = Path("./vectordb"),
) -> None:
    """Sync the vector database to or from an S3 bucket."""
    if direction == "push":
        console.print(f"[dim]Syncing {db_path} → {s3_uri}…[/dim]")
        sync_push(db_path, s3_uri)
        console.print("[green]Push complete.[/green]")
    elif direction == "pull":
        console.print(f"[dim]Syncing {s3_uri} → {db_path}…[/dim]")
        sync_pull(s3_uri, db_path)
        console.print("[green]Pull complete.[/green]")
    else:
        console.print(
            f"[red]Unknown direction '{direction}'. Use 'push' or 'pull'.[/red]"
        )
        raise typer.Exit(1)


@db_app.command()
def stats(
    db_path: Annotated[
        Path, typer.Option("--db-path", help="Path to the ChromaDB directory")
    ] = Path("./vectordb"),
) -> None:
    """Show collection sizes in the vector database."""
    store = VectorStore(db_path)
    counts = store.stats()
    table = Table("Collection", "Entries", show_header=True)
    for name, count in counts.items():
        table.add_row(name, str(count))
    console.print(table)
