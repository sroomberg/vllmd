"""Chat function: assembles context and calls the vLLM completions endpoint."""

import logging
from pathlib import Path

from ..vectordb.embeddings import _post_json, embed, make_embedder
from ..vectordb.store import COLLECTION_CODE, COLLECTION_DOCUMENTS, VectorStore
from .session import Session

log = logging.getLogger(__name__)

MAX_HISTORY = 20
N_CONTEXT_CHUNKS = 3
MAX_TOKENS = 2048
TIMEOUT = 120


def chat(
    session: Session,
    user_message: str,
    *,
    max_history: int = MAX_HISTORY,
    n_context: int = N_CONTEXT_CHUNKS,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Send *user_message* within *session*, returning the assistant reply.

    Retrieves semantic context from the session's vector store if an
    embedding_model is configured. Falls back silently to history-only context
    if the embedding endpoint is unavailable.
    """
    messages: list[dict] = []

    if session.system_prompt:
        messages.append({"role": "system", "content": session.system_prompt})

    # Semantic context retrieval (skip entirely if no embedding model configured)
    context_text = (
        _retrieve_context(session, user_message, n_context)
        if session.embedding_model
        else ""
    )
    if context_text:
        messages.append(
            {"role": "system", "content": f"Relevant context:\n\n{context_text}"}
        )

    # Recent sequential history
    for msg in session.messages[-max_history:]:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": user_message})

    response = _complete(session.endpoint, session.model_id, messages, max_tokens)

    # Persist the exchange
    from .session import Message

    session.messages.append(Message(role="user", content=user_message))
    session.messages.append(Message(role="assistant", content=response))

    # Store embeddings in vector DB for future retrieval
    if session.embedding_model:
        _store_history(session, user_message, response)

    return response


def retrieve_context(session: Session, query: str, n: int = N_CONTEXT_CHUNKS) -> str:
    """Public helper to show what context would be injected for *query*."""
    return _retrieve_context(session, query, n)


# ------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------


def _retrieve_context(session: Session, query: str, n: int) -> str:
    if not session.embedding_model:
        return ""
    try:
        store = VectorStore(Path(session.db_path))
        query_vec = embed(session.endpoint, session.embedding_model, [query])[0]
        chunks: list[str] = []
        for collection in (COLLECTION_DOCUMENTS, COLLECTION_CODE):
            results = store.search(query_vec, collection, n_results=n)
            chunks.extend(r["content"] for r in results)
        if not chunks:
            return ""
        return "\n\n---\n\n".join(chunks[:n])
    except Exception:
        log.debug("Context retrieval failed", exc_info=True)
        return ""


def _complete(
    endpoint: str, model_id: str, messages: list[dict], max_tokens: int
) -> str:
    data = _post_json(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        {"model": model_id, "messages": messages, "max_tokens": max_tokens},
        TIMEOUT,
    )
    return data["choices"][0]["message"]["content"]


def _store_history(session: Session, user_message: str, response: str) -> None:
    try:
        store = VectorStore(Path(session.db_path))
        embedder = make_embedder(session.endpoint, session.embedding_model)
        store.add_history(session.id, "user", user_message, embedder)
        store.add_history(session.id, "assistant", response, embedder)
    except Exception:
        log.debug("Failed to store conversation embedding", exc_info=True)
