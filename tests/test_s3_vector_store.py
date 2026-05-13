"""Unit tests for S3VectorStore (mocked boto3, no real AWS required)."""

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from vllmd.vectordb.providers.s3 import S3VectorStore


def _empty_tar_gz() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz"):
        pass
    return buf.getvalue()


def _make_store(tmp_path: Path) -> tuple[S3VectorStore, MagicMock]:
    """Return an S3VectorStore with a mocked boto3 client (no existing archive)."""
    from botocore.exceptions import ClientError

    mock_client = MagicMock()
    mock_client.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": ""}}, "GetObject"
    )
    with patch("boto3.client", return_value=mock_client):
        store = S3VectorStore(
            "my-bucket", prefix="vectordb/", region="us-east-1", local_cache=tmp_path
        )
    return store, mock_client


def test_init_attempts_download(tmp_path: Path) -> None:
    _, mock_client = _make_store(tmp_path)
    mock_client.get_object.assert_called_once_with(
        Bucket="my-bucket", Key="vectordb/chromadb.tar.gz"
    )


def test_init_downloads_existing_archive(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_client.get_object.return_value = {
        "Body": MagicMock(read=lambda: _empty_tar_gz())
    }
    with patch("boto3.client", return_value=mock_client):
        S3VectorStore("my-bucket", local_cache=tmp_path)
    mock_client.get_object.assert_called_once()


def test_ingest_document_syncs_to_s3(tmp_path: Path) -> None:
    store, mock_client = _make_store(tmp_path)
    mock_client.reset_mock()

    doc = tmp_path / "doc.txt"
    doc.write_text("some content")

    with patch.object(store._local, "ingest_document", return_value=1):
        store.ingest_document(doc, MagicMock())

    mock_client.put_object.assert_called_once()
    assert mock_client.put_object.call_args.kwargs["Key"] == "vectordb/chromadb.tar.gz"


def test_ingest_code_file_syncs_to_s3(tmp_path: Path) -> None:
    store, mock_client = _make_store(tmp_path)
    mock_client.reset_mock()

    with patch.object(store._local, "ingest_code_file", return_value=2):
        store.ingest_code_file(tmp_path / "foo.py", MagicMock())

    mock_client.put_object.assert_called_once()


def test_add_history_syncs_to_s3(tmp_path: Path) -> None:
    store, mock_client = _make_store(tmp_path)
    mock_client.reset_mock()

    with patch.object(store._local, "add_history", return_value="msg-id"):
        result = store.add_history("sess", "user", "hello", MagicMock())

    assert result == "msg-id"
    mock_client.put_object.assert_called_once()


def test_replace_history_with_summary_syncs(tmp_path: Path) -> None:
    store, mock_client = _make_store(tmp_path)
    mock_client.reset_mock()

    with patch.object(store._local, "replace_history_with_summary"):
        store.replace_history_with_summary("sess", "summary", MagicMock())

    mock_client.put_object.assert_called_once()


def test_get_history_does_not_sync(tmp_path: Path) -> None:
    store, mock_client = _make_store(tmp_path)
    mock_client.reset_mock()

    with patch.object(store._local, "get_history", return_value=[]):
        store.get_history("sess")

    mock_client.put_object.assert_not_called()


def test_search_does_not_sync(tmp_path: Path) -> None:
    store, mock_client = _make_store(tmp_path)
    mock_client.reset_mock()

    with patch.object(store._local, "search", return_value=[]):
        store.search([0.1, 0.2], "documents")

    mock_client.put_object.assert_not_called()


def test_stats_does_not_sync(tmp_path: Path) -> None:
    store, mock_client = _make_store(tmp_path)
    mock_client.reset_mock()

    with patch.object(store._local, "stats", return_value={}):
        store.stats()

    mock_client.put_object.assert_not_called()


def test_cache_dir_is_set(tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path)
    assert store._cache_dir == tmp_path
