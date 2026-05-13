"""Local filesystem session store."""

import json
from dataclasses import asdict
from pathlib import Path

from ..session import DEFAULT_SESSIONS_DIR, Session, _parse_messages
from .base import BaseSessionStore


class LocalSessionStore(BaseSessionStore):
    def __init__(self, sessions_dir: Path = DEFAULT_SESSIONS_DIR) -> None:
        self._dir = sessions_dir

    def save(self, session: Session) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / f"{session.id}.json").write_text(
            json.dumps(asdict(session), indent=2)
        )

    def load(self, session_id: str) -> Session:
        path = self._dir / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Session '{session_id}' not found.")
        data = json.loads(path.read_text())
        messages = _parse_messages(data.pop("messages", []))
        return Session(**data, messages=messages)

    def list_all(self) -> list[Session]:
        if not self._dir.exists():
            return []
        sessions = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                messages = _parse_messages(data.pop("messages", []))
                sessions.append(Session(**data, messages=messages))
            except Exception:
                continue
        return sessions

    def delete(self, session_id: str) -> None:
        path = self._dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
