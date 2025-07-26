"""
Fast RAG (Retrieval Augmented Generation) module for pplx-cli.

This module provides enhanced retrieval capabilities using sqlite-vec for fast vector search,
BGE embeddings for better quality, and hybrid search combining semantic and keyword search.
"""

from .database import RagDB, ContentType
from .embeddings import EmbeddingModel, get_embedding_model
from .search import HybridSearchEngine, SearchMode
from .indexer import BatchIndexer

__all__ = [
    "RagDB",
    "ContentType", 
    "EmbeddingModel",
    "get_embedding_model",
    "HybridSearchEngine",
    "SearchMode",
    "BatchIndexer"
]