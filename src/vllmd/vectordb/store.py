"""ChromaDB-backed vector store with collections for docs, code, and history."""

import contextlib
import hashlib
import time
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rb",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".md",
    ".txt",
}

COLLECTION_DOCUMENTS = "documents"
COLLECTION_CODE = "code"
COLLECTION_CONVERSATIONS = "conversations"


def _chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def _file_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]


def _ingest_chunks(
    col: chromadb.Collection,
    chunks: list[str],
    file_id: str,
    metadatas: list[dict],
    embedder: Any,
) -> int:
    embeddings = embedder(chunks)
    ids = [f"{file_id}:{i}" for i in range(len(chunks))]
    with contextlib.suppress(Exception):
        col.delete(where={"file_id": file_id})
    col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


class VectorStore:
    def __init__(self, db_path: Path) -> None:
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )

    def _collection(self, name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(name)

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        path: Path,
        embedder: Any,
        *,
        source_label: str | None = None,
    ) -> int:
        text = path.read_text(errors="replace")
        chunks = _chunk_text(text)
        if not chunks:
            return 0
        file_id = _file_id(path)
        source = source_label or str(path)
        metadatas = [{"source": source, "chunk": i, "file_id": file_id} for i in range(len(chunks))]
        return _ingest_chunks(self._collection(COLLECTION_DOCUMENTS), chunks, file_id, metadatas, embedder)

    # ------------------------------------------------------------------
    # Code
    # ------------------------------------------------------------------

    def ingest_code_file(
        self, path: Path, embedder: Any, *, root: Path | None = None
    ) -> int:
        text = path.read_text(errors="replace")
        chunks = _chunk_text(text)
        if not chunks:
            return 0
        file_id = _file_id(path)
        rel_path = str(path.relative_to(root)) if root else str(path)
        language = path.suffix.lstrip(".") or "unknown"
        metadatas = [{"filepath": rel_path, "language": language, "chunk": i, "file_id": file_id} for i in range(len(chunks))]
        return _ingest_chunks(self._collection(COLLECTION_CODE), chunks, file_id, metadatas, embedder)

    def ingest_code_dir(
        self,
        directory: Path,
        embedder: Any,
        *,
        extensions: set[str] | None = None,
    ) -> dict[str, int]:
        exts = extensions or CODE_EXTENSIONS
        results: dict[str, int] = {}
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in exts:
                n = self.ingest_code_file(path, embedder, root=directory)
                if n:
                    results[str(path.relative_to(directory))] = n
        return results

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def add_history(
        self,
        session_id: str,
        role: str,
        content: str,
        embedder: Any,
        *,
        summarized: bool = False,
    ) -> str:
        col = self._collection(COLLECTION_CONVERSATIONS)
        msg_id = str(uuid.uuid4())
        metadata = {
            "session_id": session_id,
            "role": role,
            "timestamp": time.time(),
            "summarized": summarized,
        }
        embedding = embedder([content])[0]
        col.add(
            ids=[msg_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )
        return msg_id

    def get_history(self, session_id: str, *, limit: int = 50) -> list[dict]:
        col = self._collection(COLLECTION_CONVERSATIONS)
        result = col.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return []
        entries = [
            {"id": i, "content": d, **m}
            for i, d, m in zip(
                result["ids"], result["documents"], result["metadatas"], strict=False
            )
        ]
        entries.sort(key=lambda x: x.get("timestamp", 0))
        return entries[-limit:]

    def replace_history_with_summary(
        self,
        session_id: str,
        summary: str,
        embedder: Any,
    ) -> None:
        col = self._collection(COLLECTION_CONVERSATIONS)
        result = col.get(where={"session_id": session_id})
        if result["ids"]:
            col.delete(ids=result["ids"])
        self.add_history(session_id, "assistant", summary, embedder, summarized=True)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        collection: str,
        *,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        col = self._collection(collection)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = col.query(**kwargs)
        if not result["ids"] or not result["ids"][0]:
            return []

        return [
            {
                "id": i,
                "content": d,
                "metadata": m,
                "distance": dist,
            }
            for i, d, m, dist in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=False,
            )
        ]

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name in (COLLECTION_DOCUMENTS, COLLECTION_CODE, COLLECTION_CONVERSATIONS):
            try:
                col = self._client.get_collection(name)
                out[name] = col.count()
            except Exception:
                out[name] = 0
        return out
