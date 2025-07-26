"""
Hybrid search engine combining vector similarity and keyword search.

This module implements Reciprocal Rank Fusion (RRF) to combine results from
vector search and full-text search for improved retrieval quality.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple, Set
from enum import Enum
import math

from .database import RagDB, ContentType

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Search modes available in the hybrid search engine."""
    VECTOR_ONLY = "vector"
    KEYWORD_ONLY = "keyword"
    HYBRID = "hybrid"


class SearchResult:
    """Represents a search result with content and metadata."""
    
    def __init__(
        self,
        content: str,
        score: float,
        source_id: int,
        content_type: str,
        metadata: Dict[str, Any],
        chunk_index: int = 0,
        rank: int = 0
    ):
        self.content = content
        self.score = score
        self.source_id = source_id
        self.content_type = content_type
        self.metadata = metadata
        self.chunk_index = chunk_index
        self.rank = rank
    
    def __repr__(self):
        return f"SearchResult(score={self.score:.3f}, type={self.content_type}, source={self.source_id})"


class HybridSearchEngine:
    """
    Advanced search engine combining vector similarity and keyword search.
    
    Features:
    - Vector similarity search using embeddings
    - Full-text keyword search with FTS5
    - Reciprocal Rank Fusion (RRF) for result combination
    - Content type filtering
    - Query expansion and preprocessing
    - Result deduplication and reranking
    """
    
    def __init__(
        self,
        rag_db: RagDB,
        default_mode: SearchMode = SearchMode.HYBRID,
        rrf_k: int = 60,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize the hybrid search engine.
        
        Args:
            rag_db: RAG database instance
            default_mode: Default search mode
            rrf_k: RRF constant (higher = less aggressive fusion)
            vector_weight: Weight for vector search results
            keyword_weight: Weight for keyword search results
        """
        self.rag_db = rag_db
        self.default_mode = default_mode
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        
        # Ensure weights sum to 1.0
        total_weight = vector_weight + keyword_weight
        self.vector_weight = vector_weight / total_weight
        self.keyword_weight = keyword_weight / total_weight
        
        logger.info(f"Initialized HybridSearchEngine with mode={default_mode.value}")
    
    def search(
        self,
        query: str,
        mode: Optional[SearchMode] = None,
        content_types: Optional[List[ContentType]] = None,
        limit: int = 10,
        vector_limit: int = 20,
        keyword_limit: int = 20,
        similarity_threshold: float = 0.0,
        deduplicate: bool = True,
        include_metadata: bool = True
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Search query
            mode: Search mode (vector, keyword, or hybrid)
            content_types: Filter by content types
            limit: Maximum number of results to return
            vector_limit: Maximum results from vector search
            keyword_limit: Maximum results from keyword search
            similarity_threshold: Minimum similarity score for vector results
            deduplicate: Remove duplicate results
            include_metadata: Include full metadata in results
            
        Returns:
            List of SearchResult objects ranked by relevance
        """
        mode = mode or self.default_mode
        
        # Preprocess query
        processed_query = self._preprocess_query(query)
        
        if mode == SearchMode.VECTOR_ONLY:
            return self._vector_search(
                processed_query, content_types, limit, similarity_threshold, include_metadata
            )
        elif mode == SearchMode.KEYWORD_ONLY:
            return self._keyword_search(
                processed_query, content_types, limit, include_metadata
            )
        else:  # HYBRID
            return self._hybrid_search(
                processed_query, content_types, limit, vector_limit, keyword_limit,
                similarity_threshold, deduplicate, include_metadata
            )
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the search query for better results.
        
        Args:
            query: Raw search query
            
        Returns:
            Processed query
        """
        # Basic preprocessing
        query = query.strip()
        
        # Add query expansion logic here if needed
        # For now, just return the cleaned query
        return query
    
    def _vector_search(
        self,
        query: str,
        content_types: Optional[List[ContentType]],
        limit: int,
        similarity_threshold: float,
        include_metadata: bool
    ) -> List[SearchResult]:
        """Perform vector similarity search."""
        try:
            results = self.rag_db.vector_search(
                query=query,
                content_types=content_types,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            search_results = []
            for i, (doc, score) in enumerate(results):
                metadata = doc["metadata"] if include_metadata else {}
                
                search_result = SearchResult(
                    content=doc["content"],
                    score=score,
                    source_id=doc["source_id"],
                    content_type=doc["content_type"],
                    metadata=metadata,
                    chunk_index=doc.get("chunk_index", 0),
                    rank=i + 1
                )
                search_results.append(search_result)
            
            logger.debug(f"Vector search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _keyword_search(
        self,
        query: str,
        content_types: Optional[List[ContentType]],
        limit: int,
        include_metadata: bool
    ) -> List[SearchResult]:
        """Perform keyword search."""
        try:
            results = self.rag_db.keyword_search(
                query=query,
                content_types=content_types,
                limit=limit
            )
            
            search_results = []
            for i, (doc, score) in enumerate(results):
                metadata = doc["metadata"] if include_metadata else {}
                
                search_result = SearchResult(
                    content=doc["content"],
                    score=score,
                    source_id=doc["source_id"],
                    content_type=doc["content_type"],
                    metadata=metadata,
                    chunk_index=doc.get("chunk_index", 0),
                    rank=i + 1
                )
                search_results.append(search_result)
            
            logger.debug(f"Keyword search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _hybrid_search(
        self,
        query: str,
        content_types: Optional[List[ContentType]],
        limit: int,
        vector_limit: int,
        keyword_limit: int,
        similarity_threshold: float,
        deduplicate: bool,
        include_metadata: bool
    ) -> List[SearchResult]:
        """Perform hybrid search using Reciprocal Rank Fusion."""
        try:
            # Get results from both search methods
            vector_results = self._vector_search(
                query, content_types, vector_limit, similarity_threshold, include_metadata
            )
            
            keyword_results = self._keyword_search(
                query, content_types, keyword_limit, include_metadata
            )
            
            # Apply Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(
                vector_results, keyword_results, limit
            )
            
            # Deduplicate if requested
            if deduplicate:
                fused_results = self._deduplicate_results(fused_results)
            
            # Limit final results
            return fused_results[:limit]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to vector search only
            return self._vector_search(
                query, content_types, limit, similarity_threshold, include_metadata
            )
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        limit: int
    ) -> List[SearchResult]:
        """
        Apply Reciprocal Rank Fusion to combine search results.
        
        RRF formula: score = Σ(weight / (k + rank))
        where k is a constant (typically 60) and rank is the result position.
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            limit: Maximum number of results
            
        Returns:
            Fused and ranked results
        """
        # Create a mapping from document identifier to fused score
        document_scores: Dict[str, float] = {}
        document_results: Dict[str, SearchResult] = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            doc_id = self._get_document_id(result)
            rrf_score = self.vector_weight / (self.rrf_k + rank)
            
            if doc_id in document_scores:
                document_scores[doc_id] += rrf_score
            else:
                document_scores[doc_id] = rrf_score
                document_results[doc_id] = result
        
        # Process keyword results
        for rank, result in enumerate(keyword_results, 1):
            doc_id = self._get_document_id(result)
            rrf_score = self.keyword_weight / (self.rrf_k + rank)
            
            if doc_id in document_scores:
                document_scores[doc_id] += rrf_score
            else:
                document_scores[doc_id] = rrf_score
                document_results[doc_id] = result
        
        # Sort by fused score and create final results
        sorted_docs = sorted(
            document_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        fused_results = []
        for rank, (doc_id, fused_score) in enumerate(sorted_docs, 1):
            result = document_results[doc_id]
            result.score = fused_score  # Update with fused score
            result.rank = rank
            fused_results.append(result)
        
        logger.debug(f"RRF fusion: {len(vector_results)} vector + {len(keyword_results)} keyword → {len(fused_results)} fused")
        
        return fused_results
    
    def _get_document_id(self, result: SearchResult) -> str:
        """
        Generate a unique identifier for a document.
        
        Args:
            result: Search result
            
        Returns:
            Unique document identifier
        """
        return f"{result.content_type}:{result.source_id}:{result.chunk_index}"
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate results based on content similarity.
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated results
        """
        if not results:
            return results
        
        deduplicated = []
        seen_content: Set[str] = set()
        
        for result in results:
            # Use content hash for deduplication
            content_hash = hash(result.content.strip().lower())
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                deduplicated.append(result)
        
        logger.debug(f"Deduplication: {len(results)} → {len(deduplicated)} results")
        return deduplicated
    
    def search_similar_sources(
        self,
        source_id: int,
        content_type: ContentType,
        limit: int = 5,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """
        Find documents similar to a specific source document.
        
        Args:
            source_id: ID of the source document
            content_type: Type of the source document
            limit: Maximum number of similar documents
            exclude_self: Exclude the source document from results
            
        Returns:
            Similar documents
        """
        try:
            # Get the source document content
            import sqlite3
            with sqlite3.connect(self.rag_db.db_path) as conn:
                cursor = conn.execute(f"""
                    SELECT content FROM {self.rag_db.table_name}
                    WHERE source_id = ? AND content_type = ?
                    LIMIT 1
                """, (source_id, content_type.value))
                
                row = cursor.fetchone()
                if not row:
                    return []
                
                source_content = row[0]
            
            # Search for similar content
            results = self.search(
                query=source_content,
                mode=SearchMode.VECTOR_ONLY,
                limit=limit + (1 if exclude_self else 0)
            )
            
            # Filter out the source document if requested
            if exclude_self:
                results = [
                    r for r in results
                    if not (r.source_id == source_id and r.content_type == content_type.value)
                ]
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Similar sources search failed: {e}")
            return []
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        db_stats = self.rag_db.get_stats()
        
        return {
            "database_stats": db_stats,
            "search_config": {
                "default_mode": self.default_mode.value,
                "rrf_k": self.rrf_k,
                "vector_weight": self.vector_weight,
                "keyword_weight": self.keyword_weight
            },
            "embedding_model": self.rag_db.embedding_model.get_model_info()
        }
    
    def explain_search(
        self,
        query: str,
        mode: Optional[SearchMode] = None,
        content_types: Optional[List[ContentType]] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Explain how a search query would be processed (for debugging).
        
        Args:
            query: Search query
            mode: Search mode
            content_types: Content type filters
            limit: Result limit
            
        Returns:
            Explanation of search process
        """
        mode = mode or self.default_mode
        processed_query = self._preprocess_query(query)
        
        explanation = {
            "original_query": query,
            "processed_query": processed_query,
            "search_mode": mode.value,
            "content_type_filters": [ct.value for ct in content_types] if content_types else None,
            "limit": limit
        }
        
        if mode == SearchMode.HYBRID:
            explanation.update({
                "rrf_config": {
                    "k": self.rrf_k,
                    "vector_weight": self.vector_weight,
                    "keyword_weight": self.keyword_weight
                }
            })
        
        return explanation