"""Session dataclass with pluggable persistence via BaseSessionStore."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .providers.base import BaseSessionStore

DEFAULT_SESSIONS_DIR = Path.home() / ".vllmd" / "sessions"


def _parse_messages(raw: list[dict]) -> list[Message]:
    return [Message(**m) for m in raw]


@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str
    endpoint: str
    model_id: str
    db_path: str
    created_at: str
    system_prompt: str = ""
    embedding_model: str = ""
    messages: list[Message] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, store: BaseSessionStore) -> None:
        store.save(self)

    @classmethod
    def load(cls, session_id: str, store: BaseSessionStore) -> Session:
        return store.load(session_id)

    @classmethod
    def list_all(cls, store: BaseSessionStore) -> list[Session]:
        return store.list_all()

    def delete(self, store: BaseSessionStore) -> None:
        store.delete(self.id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        session_id: str,
        endpoint: str,
        model_id: str,
        db_path: Path,
        *,
        system_prompt: str = "",
        embedding_model: str = "",
    ) -> Session:
        return cls(
            id=session_id,
            endpoint=endpoint,
            model_id=model_id,
            db_path=str(db_path),
            created_at=datetime.now(timezone.utc).isoformat(),
            system_prompt=system_prompt,
            embedding_model=embedding_model,
        )

    def message_count(self) -> int:
        return len(self.messages)

    def clear_history(self) -> None:
        self.messages.clear()
