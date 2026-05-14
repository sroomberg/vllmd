"""Tests for AgentLoop."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllmd.loop import AgentLoop


def _make_response(content: str | None = None, tool_calls: list | None = None) -> dict:
    finish = "tool_calls" if tool_calls else "stop"
    msg: dict = {"role": "assistant"}
    if content:
        msg["content"] = content
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"finish_reason": finish, "message": msg}]}


@pytest.fixture
def loop(tmp_path):
    return AgentLoop(
        endpoint="http://localhost:8001",
        model="llama3",
        workdir=str(tmp_path),
    )


# ---------------------------------------------------------------------------
# Basic completion (no tool calls)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_plain_response(loop):
    response = _make_response(content="Done!")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=response)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        result = await loop.run("say done")

    assert result == "Done!"


# ---------------------------------------------------------------------------
# Single tool call then done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_one_tool_call(loop, tmp_path):
    tool_response = _make_response(
        tool_calls=[
            {
                "id": "tc1",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps(
                        {
                            "path": str(tmp_path / "out.txt"),
                            "content": "hello from agent",
                        }
                    ),
                },
            }
        ]
    )
    final_response = _make_response(content="File written.")

    responses = [tool_response, final_response]
    idx = 0

    async def _post(*args, **kwargs):
        nonlocal idx
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=responses[idx])
        idx += 1
        return mock_resp

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = _post
        mock_client_cls.return_value = mock_client

        result = await loop.run("write a file")

    assert result == "File written."
    assert (tmp_path / "out.txt").read_text() == "hello from agent"


# ---------------------------------------------------------------------------
# Max turns guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_max_turns(loop):
    tool_call_response = _make_response(
        tool_calls=[
            {
                "id": "tc1",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": "echo x"}),
                },
            }
        ]
    )

    async def _post(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=tool_call_response)
        return mock_resp

    small_loop = AgentLoop(
        endpoint="http://localhost:8001",
        model="llama3",
        max_turns=3,
    )

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = _post
        mock_client_cls.return_value = mock_client

        result = await small_loop.run("spin forever")

    assert "max turns" in result


# ---------------------------------------------------------------------------
# on_message callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_on_message_callback(loop):
    events: list[tuple[str, str]] = []

    def _cb(role: str, content: str) -> None:
        events.append((role, content))

    loop.on_message = _cb

    response = _make_response(content="hello")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=response)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        await loop.run("hi")

    roles = [e[0] for e in events]
    assert "user" in roles
    assert "assistant" in roles


# ---------------------------------------------------------------------------
# Auth header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_sends_auth_header():
    captured_headers: dict = {}

    async def _post(url, json, headers):
        captured_headers.update(headers)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=_make_response(content="ok"))
        return mock_resp

    auth_loop = AgentLoop(
        endpoint="http://localhost:8001",
        model="llama3",
        api_key="secret",
    )

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = _post
        mock_client_cls.return_value = mock_client

        await auth_loop.run("test")

    assert captured_headers.get("Authorization") == "Bearer secret"
