"""Vector store backends and factory for vllmd."""

from .factory import get_vector_store
from .providers.aws import AWSVectorStore
from .providers.base import BaseVectorStore
from .providers.local import LocalVectorStore

# LocalVectorStore is the default backend.
VectorStore = LocalVectorStore

__all__ = [
    "AWSVectorStore",
    "BaseVectorStore",
    "LocalVectorStore",
    "VectorStore",
    "get_vector_store",
]
