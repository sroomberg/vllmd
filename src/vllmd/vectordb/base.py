"""Abstract base class and shared helpers for vllmd vector stores."""

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

COLLECTION_DOCUMENTS = "documents"
COLLECTION_CODE = "code"
COLLECTION_CONVERSATIONS = "conversations"

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


class BaseVectorStore(ABC):
    @abstractmethod
    def ingest_document(
        self,
        path: Path,
        embedder: Any,
        *,
        source_label: str | None = None,
    ) -> int:
        """Ingest a document file. Returns number of chunks stored."""

    @abstractmethod
    def ingest_code_file(
        self, path: Path, embedder: Any, *, root: Path | None = None
    ) -> int:
        """Ingest a source code file. Returns number of chunks stored."""

    def ingest_code_dir(
        self,
        directory: Path,
        embedder: Any,
        *,
        extensions: set[str] | None = None,
    ) -> dict[str, int]:
        """Ingest all code files under *directory*.

        Returns {relative_path: chunk_count}.
        """
        exts = extensions or CODE_EXTENSIONS
        results: dict[str, int] = {}
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in exts:
                n = self.ingest_code_file(path, embedder, root=directory)
                if n:
                    results[str(path.relative_to(directory))] = n
        return results

    @abstractmethod
    def add_history(
        self,
        session_id: str,
        role: str,
        content: str,
        embedder: Any,
        *,
        summarized: bool = False,
    ) -> str:
        """Store a conversation message. Returns the message ID."""

    @abstractmethod
    def get_history(self, session_id: str, *, limit: int = 50) -> list[dict]:
        """Return up to *limit* recent messages for *session_id*, ordered by time."""

    @abstractmethod
    def replace_history_with_summary(
        self, session_id: str, summary: str, embedder: Any
    ) -> None:
        """Replace all session history with a single summary message."""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        collection: str,
        *,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Return up to *n_results* nearest neighbours from *collection*."""

    @abstractmethod
    def stats(self) -> dict[str, int]:
        """Return document counts per collection."""
