"""Unit tests for session store providers (no network required)."""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllmd.sessions.providers.local import LocalSessionStore
from vllmd.sessions.providers.s3 import S3SessionStore
from vllmd.sessions.session import Message, Session


def _make_session(session_id: str, db_path: Path) -> Session:
    return Session.create(
        session_id,
        endpoint="http://localhost:8001",
        model_id="llama3",
        db_path=db_path,
    )


# ------------------------------------------------------------------
# LocalSessionStore
# ------------------------------------------------------------------


def test_local_store_save_creates_file(tmp_path: Path) -> None:
    store = LocalSessionStore(tmp_path)
    store.save(_make_session("local-save", tmp_path))
    assert (tmp_path / "local-save.json").exists()


def test_local_store_roundtrip(tmp_path: Path) -> None:
    store = LocalSessionStore(tmp_path)
    session = _make_session("local-rt", tmp_path)
    session.messages.append(Message(role="user", content="hi"))
    store.save(session)
    loaded = store.load("local-rt")
    assert loaded.id == "local-rt"
    assert loaded.model_id == "llama3"
    assert len(loaded.messages) == 1
    assert loaded.messages[0].content == "hi"


def test_local_store_load_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LocalSessionStore(tmp_path).load("ghost")


def test_local_store_list_all(tmp_path: Path) -> None:
    store = LocalSessionStore(tmp_path)
    for name in ("a", "b", "c"):
        store.save(_make_session(name, tmp_path))
    assert {s.id for s in store.list_all()} == {"a", "b", "c"}


def test_local_store_list_empty(tmp_path: Path) -> None:
    assert LocalSessionStore(tmp_path).list_all() == []


def test_local_store_list_missing_dir(tmp_path: Path) -> None:
    assert LocalSessionStore(tmp_path / "nonexistent").list_all() == []


def test_local_store_delete(tmp_path: Path) -> None:
    store = LocalSessionStore(tmp_path)
    store.save(_make_session("del-me", tmp_path))
    store.delete("del-me")
    assert not (tmp_path / "del-me.json").exists()


def test_local_store_delete_noop_if_missing(tmp_path: Path) -> None:
    LocalSessionStore(tmp_path).delete("ghost")  # must not raise


# ------------------------------------------------------------------
# S3SessionStore (mocked boto3)
# ------------------------------------------------------------------


def _make_s3_store() -> tuple[S3SessionStore, MagicMock]:
    mock_client = MagicMock()
    with patch("boto3.client", return_value=mock_client):
        store = S3SessionStore("my-bucket", prefix="sessions/", region="us-east-1")
    return store, mock_client


def test_s3_store_save(tmp_path: Path) -> None:
    store, mock_client = _make_s3_store()
    store.save(_make_session("s3-save", tmp_path))
    mock_client.put_object.assert_called_once()
    kwargs = mock_client.put_object.call_args.kwargs
    assert kwargs["Bucket"] == "my-bucket"
    assert kwargs["Key"] == "sessions/s3-save.json"
    body = json.loads(kwargs["Body"])
    assert body["id"] == "s3-save"


def test_s3_store_load(tmp_path: Path) -> None:
    store, mock_client = _make_s3_store()
    session = _make_session("s3-load", tmp_path)
    mock_client.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps(asdict(session)).encode())
    }
    loaded = store.load("s3-load")
    assert loaded.id == "s3-load"
    mock_client.get_object.assert_called_once_with(
        Bucket="my-bucket", Key="sessions/s3-load.json"
    )


def test_s3_store_load_missing(tmp_path: Path) -> None:
    from botocore.exceptions import ClientError

    store, mock_client = _make_s3_store()
    mock_client.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject"
    )
    with pytest.raises(FileNotFoundError):
        store.load("ghost")


def test_s3_store_delete(tmp_path: Path) -> None:
    store, mock_client = _make_s3_store()
    store.delete("old")
    mock_client.delete_object.assert_called_once_with(
        Bucket="my-bucket", Key="sessions/old.json"
    )


def test_s3_store_list_all(tmp_path: Path) -> None:
    store, mock_client = _make_s3_store()
    sessions = [_make_session(name, tmp_path) for name in ("alpha", "beta")]

    mock_client.get_paginator.return_value.paginate.return_value = [
        {
            "Contents": [
                {"Key": "sessions/alpha.json"},
                {"Key": "sessions/beta.json"},
            ]
        }
    ]
    mock_client.get_object.side_effect = [
        {"Body": MagicMock(read=lambda b=s: json.dumps(asdict(b)).encode())}
        for s in sessions
    ]

    result = store.list_all()
    assert {s.id for s in result} == {"alpha", "beta"}


def test_s3_store_list_all_skips_non_json(tmp_path: Path) -> None:
    store, mock_client = _make_s3_store()
    mock_client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "sessions/archive.tar.gz"}]}
    ]
    assert store.list_all() == []
    mock_client.get_object.assert_not_called()
