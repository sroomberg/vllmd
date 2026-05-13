"""Factory for creating vector store instances from config files."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

from .providers.aws import AWSVectorStore
from .providers.base import BaseVectorStore
from .providers.local import LocalVectorStore

_LOCAL_CONFIG = Path(".vllmd.toml")
_GLOBAL_CONFIG = Path.home() / ".vllmd" / "config.toml"


def get_vector_store(*, db_path: Path | None = None) -> BaseVectorStore:
    """Return a vector store instance configured by the local or global config file.

    Configuration is read from (highest priority first):
      1. ``.vllmd.toml`` in the current working directory
      2. ``~/.vllmd/config.toml`` (global)

    If neither file exists, returns a ``LocalVectorStore`` at ``./vectordb``
    (or *db_path* if given).

    *db_path* overrides the configured path for ``local`` backends only.

    Example config::

        [vectordb]
        backend = "local"

        [vectordb.local]
        db_path = "./vectordb"

    Or for AWS::

        [vectordb]
        backend = "aws"

        [vectordb.aws]
        endpoint = "my-domain.us-east-1.es.amazonaws.com"
        region = "us-east-1"
        index_prefix = "vllmd"
        embedding_dim = 1536
        service = "es"
    """
    config = _load_config()
    vdb = config.get("vectordb", {})
    backend = vdb.get("backend", "local")

    if backend == "local":
        cfg = vdb.get("local", {})
        path = db_path or Path(cfg.get("db_path", "./vectordb"))
        return LocalVectorStore(path)

    if backend == "aws":
        cfg = vdb.get("aws", {})
        if "endpoint" not in cfg:
            raise ValueError("vectordb.aws.endpoint is required when backend = 'aws'")
        return AWSVectorStore(
            cfg["endpoint"],
            region=cfg.get("region", "us-east-1"),
            index_prefix=cfg.get("index_prefix", "vllmd"),
            embedding_dim=cfg.get("embedding_dim", 1536),
            service=cfg.get("service", "es"),
        )

    raise ValueError(
        f"Unknown vectordb backend {backend!r}. Expected 'local' or 'aws'."
    )


def _load_config() -> dict:
    """Load and merge global then local config, with local taking precedence."""
    config: dict = {}
    for path in (_GLOBAL_CONFIG, _LOCAL_CONFIG):
        if path.exists():
            with open(path, "rb") as fh:
                _deep_merge(config, tomllib.load(fh))
    return config


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
