"""Factory for creating session store instances from config files."""

from __future__ import annotations

from pathlib import Path

from ..vectordb.factory import _load_config
from .providers.base import BaseSessionStore
from .providers.local import LocalSessionStore
from .session import DEFAULT_SESSIONS_DIR


def get_session_store() -> BaseSessionStore:
    """Return a session store configured by the local or global config file.

    Reads ``sessions.store`` from the merged config (same files as vectordb config).
    Defaults to ``LocalSessionStore(DEFAULT_SESSIONS_DIR)``.

    Example config (``vllmd.yaml``)::

        sessions:
          store: local
          local:
            sessions_dir: ~/.vllmd/sessions

    Or for S3::

        sessions:
          store: s3
          s3:
            bucket: my-sessions-bucket
            prefix: vllmd/sessions/
            region: us-east-1
    """
    config = _load_config()
    cfg = config.get("sessions", {})
    backend = cfg.get("store", "local")

    if backend == "local":
        local_cfg = cfg.get("local", {})
        sessions_dir_str = local_cfg.get("sessions_dir")
        sessions_dir = (
            Path(sessions_dir_str).expanduser()
            if sessions_dir_str
            else DEFAULT_SESSIONS_DIR
        )
        return LocalSessionStore(sessions_dir)

    if backend == "s3":
        from .providers.s3 import S3SessionStore

        s3_cfg = cfg.get("s3", {})
        if "bucket" not in s3_cfg:
            raise ValueError(
                "sessions.s3.bucket is required when sessions.store = 's3'"
            )
        return S3SessionStore(
            s3_cfg["bucket"],
            prefix=s3_cfg.get("prefix", "sessions/"),
            region=s3_cfg.get("region", "us-east-1"),
        )

    raise ValueError(
        f"Unknown sessions store {backend!r}. Expected 'local' or 's3'."
    )
