"""Vector store backends for vllmd."""

from .aws import AWSVectorStore
from .base import BaseVectorStore
from .local import LocalVectorStore

# LocalVectorStore is the default backend.
VectorStore = LocalVectorStore

__all__ = ["AWSVectorStore", "BaseVectorStore", "LocalVectorStore", "VectorStore"]
