"""Amazon OpenSearch Service-backed vector store."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from .base import (
    COLLECTION_CODE,
    COLLECTION_CONVERSATIONS,
    COLLECTION_DOCUMENTS,
    BaseVectorStore,
    _chunk_text,
    _file_id,
)

_COLLECTIONS = (COLLECTION_DOCUMENTS, COLLECTION_CODE, COLLECTION_CONVERSATIONS)


class AWSVectorStore(BaseVectorStore):
    """Vector store backed by Amazon OpenSearch Service.

    Requires the ``vllmd[aws]`` extras (``opensearch-py``, ``boto3``).

    For OpenSearch Serverless, pass ``service="aoss"`` instead of the default
    ``"es"``.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        region: str = "us-east-1",
        index_prefix: str = "vllmd",
        embedding_dim: int = 1536,
        service: str = "es",
    ) -> None:
        try:
            import boto3
            from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
        except ImportError as exc:
            raise ImportError(
                "Install vllmd[aws] for AWS vector store support: "
                "pip install 'vllmd[aws]'"
            ) from exc

        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, region, service)
        self._client = OpenSearch(
            hosts=[{"host": endpoint, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
        self._prefix = index_prefix
        self._dim = embedding_dim
        self._ensure_indices()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index(self, collection: str) -> str:
        return f"{self._prefix}-{collection}"

    def _ensure_indices(self) -> None:
        mapping = {
            "settings": {"index.knn": True},
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self._dim,
                    },
                    "content": {"type": "text"},
                    "metadata": {"type": "object", "dynamic": True},
                }
            },
        }
        for collection in _COLLECTIONS:
            idx = self._index(collection)
            if not self._client.indices.exists(idx):
                self._client.indices.create(idx, body=mapping)

    def _ingest_file(
        self,
        collection: str,
        chunks: list[str],
        file_id: str,
        metadatas: list[dict],
        embedder: Any,
    ) -> int:
        embeddings = embedder(chunks)
        self._client.delete_by_query(
            index=self._index(collection),
            body={"query": {"term": {"metadata.file_id": file_id}}},
        )
        for i, (chunk, embedding, metadata) in enumerate(
            zip(chunks, embeddings, metadatas, strict=False)
        ):
            self._client.index(
                index=self._index(collection),
                id=f"{file_id}:{i}",
                body={"content": chunk, "embedding": embedding, "metadata": metadata},
            )
        return len(chunks)

    @staticmethod
    def _build_filter(where: dict) -> dict:
        """Translate a {key: value} where-clause to an OpenSearch bool filter."""
        must = [{"term": {f"metadata.{k}": v}} for k, v in where.items()]
        return {"bool": {"must": must}} if len(must) > 1 else must[0]

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
        metadatas = [
            {"source": source, "chunk": i, "file_id": file_id}
            for i in range(len(chunks))
        ]
        return self._ingest_file(
            COLLECTION_DOCUMENTS, chunks, file_id, metadatas, embedder
        )

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
        metadatas = [
            {"filepath": rel_path, "language": language, "chunk": i, "file_id": file_id}
            for i in range(len(chunks))
        ]
        return self._ingest_file(COLLECTION_CODE, chunks, file_id, metadatas, embedder)

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
        msg_id = str(uuid.uuid4())
        metadata = {
            "session_id": session_id,
            "role": role,
            "timestamp": time.time(),
            "summarized": summarized,
        }
        embedding = embedder([content])[0]
        self._client.index(
            index=self._index(COLLECTION_CONVERSATIONS),
            id=msg_id,
            body={"content": content, "embedding": embedding, "metadata": metadata},
        )
        return msg_id

    def get_history(self, session_id: str, *, limit: int = 50) -> list[dict]:
        result = self._client.search(
            index=self._index(COLLECTION_CONVERSATIONS),
            body={
                "size": limit,
                "query": {"term": {"metadata.session_id": session_id}},
                "sort": [{"metadata.timestamp": {"order": "asc"}}],
            },
        )
        hits = result.get("hits", {}).get("hits", [])
        return [
            {
                "id": h["_id"],
                "content": h["_source"]["content"],
                **h["_source"]["metadata"],
            }
            for h in hits
        ]

    def replace_history_with_summary(
        self, session_id: str, summary: str, embedder: Any
    ) -> None:
        self._client.delete_by_query(
            index=self._index(COLLECTION_CONVERSATIONS),
            body={"query": {"term": {"metadata.session_id": session_id}}},
        )
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
        knn_clause: dict[str, Any] = {
            "knn": {"embedding": {"vector": query_embedding, "k": n_results}}
        }
        body: dict[str, Any] = {"size": n_results, "query": knn_clause}
        if where:
            body["query"] = {
                "bool": {
                    "must": knn_clause,
                    "filter": self._build_filter(where),
                }
            }
        result = self._client.search(index=self._index(collection), body=body)
        hits = result.get("hits", {}).get("hits", [])
        return [
            {
                "id": h["_id"],
                "content": h["_source"]["content"],
                "metadata": h["_source"]["metadata"],
                "distance": 1.0 - h.get("_score", 0.0),
            }
            for h in hits
        ]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for collection in _COLLECTIONS:
            try:
                result = self._client.count(index=self._index(collection))
                out[collection] = result.get("count", 0)
            except Exception:
                out[collection] = 0
        return out
