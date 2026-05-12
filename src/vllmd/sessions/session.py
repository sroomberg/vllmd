"""Session dataclass with JSON persistence."""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SESSIONS_DIR = Path.home() / ".vllmd" / "sessions"


def _parse_messages(raw: list[dict]) -> "list[Message]":
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

    def save(self, sessions_dir: Path = DEFAULT_SESSIONS_DIR) -> None:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        path = sessions_dir / f"{self.id}.json"
        data = asdict(self)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(
        cls, session_id: str, sessions_dir: Path = DEFAULT_SESSIONS_DIR
    ) -> "Session":
        path = sessions_dir / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Session '{session_id}' not found.")
        data = json.loads(path.read_text())
        messages = _parse_messages(data.pop("messages", []))
        return cls(**data, messages=messages)

    @classmethod
    def list_all(cls, sessions_dir: Path = DEFAULT_SESSIONS_DIR) -> "list[Session]":
        if not sessions_dir.exists():
            return []
        sessions = []
        for path in sorted(sessions_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                messages = _parse_messages(data.pop("messages", []))
                sessions.append(cls(**data, messages=messages))
            except Exception:
                continue
        return sessions

    def delete(self, sessions_dir: Path = DEFAULT_SESSIONS_DIR) -> None:
        path = sessions_dir / f"{self.id}.json"
        if path.exists():
            path.unlink()

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
    ) -> "Session":
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
