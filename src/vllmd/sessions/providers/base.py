"""Abstract base class for session storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..session import Session


class BaseSessionStore(ABC):
    @abstractmethod
    def save(self, session: "Session") -> None:
        """Persist session data."""

    @abstractmethod
    def load(self, session_id: str) -> "Session":
        """Load a session by ID. Raises FileNotFoundError if not found."""

    @abstractmethod
    def list_all(self) -> "list[Session]":
        """Return all stored sessions, sorted by ID."""

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Delete a session by ID."""
