"""S3-backed vector store: ChromaDB with automatic S3 sync."""

from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .base import BaseVectorStore
from .local import LocalVectorStore


class S3VectorStore(BaseVectorStore):
    """Wraps LocalVectorStore (ChromaDB) and syncs the database to S3.

    On construction the archive is downloaded from S3 if it exists.
    After each write operation the local ChromaDB directory is re-uploaded.
    Read-only operations (search, get_history, stats) do not trigger a sync.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "vectordb/",
        region: str = "us-east-1",
        local_cache: Path | None = None,
    ) -> None:
        import boto3

        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._archive_key = f"{self._prefix}chromadb.tar.gz"
        self._client = boto3.client("s3", region_name=region)
        self._cache_dir = local_cache or Path(
            tempfile.mkdtemp(prefix="vllmd-vectordb-")
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._sync_from_s3()
        self._local = LocalVectorStore(self._cache_dir)

    # ------------------------------------------------------------------
    # S3 sync helpers
    # ------------------------------------------------------------------

    def _sync_from_s3(self) -> None:
        from botocore.exceptions import ClientError

        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=self._archive_key)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return
            raise
        buf = io.BytesIO(resp["Body"].read())
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            try:
                tar.extractall(self._cache_dir, filter="data")
            except TypeError:
                tar.extractall(self._cache_dir)

    def _sync_to_s3(self) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(self._cache_dir, arcname=".")
        self._client.put_object(
            Bucket=self._bucket,
            Key=self._archive_key,
            Body=buf.getvalue(),
            ContentType="application/gzip",
        )

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        path: Path,
        embedder: Any,
        *,
        source_label: str | None = None,
    ) -> int:
        n = self._local.ingest_document(path, embedder, source_label=source_label)
        self._sync_to_s3()
        return n

    def ingest_code_file(
        self, path: Path, embedder: Any, *, root: Path | None = None
    ) -> int:
        n = self._local.ingest_code_file(path, embedder, root=root)
        self._sync_to_s3()
        return n

    def add_history(
        self,
        session_id: str,
        role: str,
        content: str,
        embedder: Any,
        *,
        summarized: bool = False,
    ) -> str:
        msg_id = self._local.add_history(
            session_id, role, content, embedder, summarized=summarized
        )
        self._sync_to_s3()
        return msg_id

    def get_history(self, session_id: str, *, limit: int = 50) -> list[dict]:
        return self._local.get_history(session_id, limit=limit)

    def replace_history_with_summary(
        self, session_id: str, summary: str, embedder: Any
    ) -> None:
        self._local.replace_history_with_summary(session_id, summary, embedder)
        self._sync_to_s3()

    def search(
        self,
        query_embedding: list[float],
        collection: str,
        *,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        return self._local.search(
            query_embedding, collection, n_results=n_results, where=where
        )

    def stats(self) -> dict[str, int]:
        return self._local.stats()
