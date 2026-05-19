"""Factory for creating vector store instances from config files."""

from __future__ import annotations

from pathlib import Path

import yaml

from .providers.aws import AWSVectorStore
from .providers.base import BaseVectorStore
from .providers.local import LocalVectorStore
from .providers.s3 import S3VectorStore

GLOBAL_CONFIG_DIR = Path.home() / ".config" / "vllmd"

LOCAL_CONFIG_CANDIDATES = [
    "vllmd.yaml",
    "vllmd.yml",
    ".vllmd.yaml",
    ".vllmd.yml",
]


def get_vector_store(*, db_path: Path | None = None) -> BaseVectorStore:
    """Return a vector store instance configured by the local or global config file.

    Configuration is read from (highest priority first):
      1. First of ``vllmd.yaml``, ``vllmd.yml``, ``.vllmd.yaml``, ``.vllmd.yml``
         found in the current working directory
      2. ``~/.config/vllmd/config.yml`` (global)

    If neither file exists, returns a ``LocalVectorStore`` at ``./vectordb``
    (or *db_path* if given).

    *db_path* overrides the configured path for ``local`` backends only.

    Example config (``vllmd.yaml``)::

        vectordb:
          backend: local
          local:
            db_path: ./vectordb

    Or for AWS::

        vectordb:
          backend: aws
          aws:
            endpoint: my-domain.us-east-1.es.amazonaws.com
            region: us-east-1
            index_prefix: vllmd
            embedding_dim: 1536
            service: es
    """
    config = load_config()
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

    if backend == "s3":
        cfg = vdb.get("s3", {})
        if "bucket" not in cfg:
            raise ValueError("vectordb.s3.bucket is required when backend = 's3'")
        return S3VectorStore(
            cfg["bucket"],
            prefix=cfg.get("prefix", "vectordb/"),
            region=cfg.get("region", "us-east-1"),
            local_cache=Path(cfg["local_cache"]) if "local_cache" in cfg else None,
        )

    raise ValueError(
        f"Unknown vectordb backend {backend!r}. Expected 'local', 'aws', or 's3'."
    )


def _find_local_config() -> Path | None:
    for candidate in LOCAL_CONFIG_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return p
    return None


def _get_global_config() -> Path | None:
    for name in ("config.yml", "config.yaml"):
        p = GLOBAL_CONFIG_DIR / name
        if p.exists():
            return p
    return None


def _load_yaml(path: Path) -> dict:
    with open(path) as fh:
        try:
            return yaml.safe_load(fh) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e


def load_config() -> dict:
    """Load and merge global then local config, with local taking precedence."""
    config: dict = {}
    for path in filter(None, [_get_global_config(), _find_local_config()]):
        _deep_merge(config, _load_yaml(path))
    return config


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
