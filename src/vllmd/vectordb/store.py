"""Backward-compatible re-exports from the vectordb package."""

from .providers.base import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CODE_EXTENSIONS,
    COLLECTION_CODE,
    COLLECTION_CONVERSATIONS,
    COLLECTION_DOCUMENTS,
    BaseVectorStore,
)
from .providers.local import LocalVectorStore

# LocalVectorStore is the default; kept as VectorStore for existing callers.
VectorStore = LocalVectorStore

__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "CODE_EXTENSIONS",
    "COLLECTION_CODE",
    "COLLECTION_CONVERSATIONS",
    "COLLECTION_DOCUMENTS",
    "BaseVectorStore",
    "LocalVectorStore",
    "VectorStore",
]
