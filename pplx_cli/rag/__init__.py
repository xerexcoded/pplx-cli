"""
Fast RAG (Retrieval Augmented Generation) module for pplx-cli.

This module provides enhanced retrieval capabilities using sqlite-vec for fast vector search,
BGE embeddings for better quality, and hybrid search combining semantic and keyword search.
"""

from .database import RagDB
from .embeddings import EmbeddingModel
from .search import HybridSearchEngine
from .indexer import BatchIndexer

__all__ = [
    "RagDB",
    "EmbeddingModel", 
    "HybridSearchEngine",
    "BatchIndexer"
]