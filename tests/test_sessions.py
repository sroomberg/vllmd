"""Unit tests for vllmd.sessions (no network required)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllmd.sessions.providers.local import LocalSessionStore
from vllmd.sessions.session import Message, Session

# ------------------------------------------------------------------
# Session persistence
# ------------------------------------------------------------------


def test_session_create_and_save(tmp_path: Path) -> None:
    store = LocalSessionStore(tmp_path)
    session = Session.create(
        "test-session",
        endpoint="http://localhost:8001",
        model_id="llama3",
        db_path=tmp_path / "vectordb",
    )
    session.save(store)
    assert (tmp_path / "test-session.json").exists()


def test_session_roundtrip(tmp_path: Path) -> None:
    store = LocalSessionStore(tmp_path)
    session = Session.create(
        "roundtrip",
        endpoint="http://localhost:8001",
        model_id="mistral",
        db_path=tmp_path / "vectordb",
        system_prompt="You are helpful.",
        embedding_model="mistral",
    )
    session.messages.append(Message(role="user", content="hello"))
    session.messages.append(Message(role="assistant", content="hi there"))
    session.save(store)

    loaded = Session.load("roundtrip", store)
    assert loaded.id == "roundtrip"
    assert loaded.model_id == "mistral"
    assert loaded.system_prompt == "You are helpful."
    assert loaded.embedding_model == "mistral"
    assert len(loaded.messages) == 2
    assert loaded.messages[0].role == "user"
    assert loaded.messages[0].content == "hello"
    assert loaded.messages[1].role == "assistant"


def test_session_list(tmp_path: Path) -> None:
    store = LocalSessionStore(tmp_path)
    for name in ("alpha", "beta", "gamma"):
        s = Session.create(
            name, endpoint="http://localhost:8000", model_id="m", db_path=tmp_path
        )
        s.save(store)
    sessions = Session.list_all(store)
    assert len(sessions) == 3
    assert {s.id for s in sessions} == {"alpha", "beta", "gamma"}


def test_session_list_empty_dir(tmp_path: Path) -> None:
    store = LocalSessionStore(tmp_path)
    assert Session.list_all(store) == []


def test_session_list_missing_dir(tmp_path: Path) -> None:
    store = LocalSessionStore(tmp_path / "nonexistent")
    assert Session.list_all(store) == []


def test_session_load_missing(tmp_path: Path) -> None:
    store = LocalSessionStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        Session.load("nonexistent", store)


def test_session_clear_history(tmp_path: Path) -> None:
    session = Session.create(
        "s", endpoint="http://localhost:8000", model_id="m", db_path=tmp_path
    )
    session.messages.append(Message(role="user", content="x"))
    session.messages.append(Message(role="assistant", content="y"))
    assert session.message_count() == 2
    session.clear_history()
    assert session.message_count() == 0


def test_session_delete(tmp_path: Path) -> None:
    store = LocalSessionStore(tmp_path)
    session = Session.create(
        "bye", endpoint="http://localhost:8000", model_id="m", db_path=tmp_path
    )
    session.save(store)
    assert (tmp_path / "bye.json").exists()
    session.delete(store)
    assert not (tmp_path / "bye.json").exists()


# ------------------------------------------------------------------
# chat() — message assembly
# ------------------------------------------------------------------


def _fake_urlopen(response_text: str) -> MagicMock:
    """Return a context-manager mock that yields *response_text* on .read()."""
    mock_cm = MagicMock()
    mock_cm.__enter__ = lambda s: s
    mock_cm.__exit__ = MagicMock(return_value=False)
    mock_cm.read.return_value = response_text.encode()
    return mock_cm


def _completions_response(content: str) -> str:
    return json.dumps({"choices": [{"message": {"content": content}}]})


def test_chat_returns_response(tmp_path: Path) -> None:
    from vllmd.sessions.chat import chat

    session = Session.create(
        "chat-test",
        endpoint="http://localhost:8001",
        model_id="llama3",
        db_path=tmp_path,
    )
    fake = _fake_urlopen(_completions_response("Hello!"))
    with patch("urllib.request.urlopen", return_value=fake):
        reply = chat(session, "Hi model")
    assert reply == "Hello!"


def test_chat_persists_exchange(tmp_path: Path) -> None:
    from vllmd.sessions.chat import chat

    session = Session.create(
        "persist", endpoint="http://localhost:8001", model_id="llama3", db_path=tmp_path
    )
    fake = _fake_urlopen(_completions_response("Pong"))
    with patch("urllib.request.urlopen", return_value=fake):
        chat(session, "Ping")
    assert len(session.messages) == 2
    assert session.messages[0] == Message(
        role="user", content="Ping", timestamp=session.messages[0].timestamp
    )
    assert session.messages[1].role == "assistant"
    assert session.messages[1].content == "Pong"


def test_chat_includes_system_prompt(tmp_path: Path) -> None:
    """System prompt must appear as the first message sent to the API."""
    from vllmd.sessions.chat import chat

    session = Session.create(
        "sysprompt",
        endpoint="http://localhost:8001",
        model_id="llama3",
        db_path=tmp_path,
        system_prompt="You are a pirate.",
    )
    captured: list[dict] = []

    def fake_urlopen(req, timeout=None):
        captured.extend(json.loads(req.data)["messages"])
        return _fake_urlopen(_completions_response("Arrr"))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        chat(session, "Hello")

    assert captured[0] == {"role": "system", "content": "You are a pirate."}
    assert captured[-1] == {"role": "user", "content": "Hello"}


def test_chat_includes_prior_history(tmp_path: Path) -> None:
    """Prior conversation history must appear in the request before the new message."""
    from vllmd.sessions.chat import chat

    session = Session.create(
        "history", endpoint="http://localhost:8001", model_id="llama3", db_path=tmp_path
    )
    session.messages.append(Message(role="user", content="first question"))
    session.messages.append(Message(role="assistant", content="first answer"))

    captured: list[dict] = []

    def fake_urlopen(req, timeout=None):
        captured.extend(json.loads(req.data)["messages"])
        return _fake_urlopen(_completions_response("second answer"))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        chat(session, "second question")

    contents = [m["content"] for m in captured]
    assert "first question" in contents
    assert "first answer" in contents
    assert contents[-1] == "second question"
    assert contents.index("first question") < contents.index("second question")


def test_chat_context_retrieval_injected(tmp_path: Path) -> None:
    """Retrieved context must appear as a system message when embedding_model is set."""
    from vllmd.sessions.chat import chat as _chat

    session = Session.create(
        "ctx",
        endpoint="http://localhost:8001",
        model_id="llama3",
        db_path=tmp_path / "vectordb",
        embedding_model="llama3",
    )
    captured: list[dict] = []

    def fake_urlopen(req, timeout=None):
        captured.extend(json.loads(req.data)["messages"])
        return _fake_urlopen(_completions_response("answer"))

    fake_context = "Relevant doc chunk"

    with (
        patch("vllmd.sessions.chat._retrieve_context", return_value=fake_context),
        patch("urllib.request.urlopen", side_effect=fake_urlopen),
    ):
        _chat(session, "question")

    system_messages = [m["content"] for m in captured if m["role"] == "system"]
    assert any(fake_context in msg for msg in system_messages)


def test_chat_context_retrieval_skipped_without_embedding_model(tmp_path: Path) -> None:
    """_retrieve_context must not be called when embedding_model is empty."""
    from vllmd.sessions.chat import chat

    session = Session.create(
        "noctx", endpoint="http://localhost:8001", model_id="llama3", db_path=tmp_path
    )
    fake = _fake_urlopen(_completions_response("ok"))
    with (
        patch("vllmd.sessions.chat._retrieve_context") as mock_retrieve,
        patch("urllib.request.urlopen", return_value=fake),
    ):
        chat(session, "hi")
    mock_retrieve.assert_not_called()


def test_chat_graceful_on_embedding_failure(tmp_path: Path) -> None:
    """A failing embedding endpoint must not prevent the chat call from succeeding."""
    from vllmd.sessions.chat import chat

    session = Session.create(
        "fail-embed",
        endpoint="http://localhost:8001",
        model_id="llama3",
        db_path=tmp_path / "vectordb",
        embedding_model="llama3",
    )
    fake = _fake_urlopen(_completions_response("fine"))
    with (
        patch("vllmd.sessions.chat._retrieve_context", return_value=""),
        patch("urllib.request.urlopen", return_value=fake),
    ):
        reply = chat(session, "hi")
    assert reply == "fine"
