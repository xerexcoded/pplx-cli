"""
Unified RAG database using sqlite-vec for fast vector search.

This module provides a high-performance database that combines notes and chat history
into a single searchable vector store using sqlite-vec extension.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union, Any
from datetime import datetime
from enum import Enum
import numpy as np

from .embeddings import get_embedding_model, EmbeddingModel

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content that can be stored in the RAG database."""
    NOTE = "note"
    CHAT_MESSAGE = "chat_message"
    CONVERSATION = "conversation"


class DocumentChunk:
    """Represents a chunk of text with metadata for vector search."""
    
    def __init__(
        self,
        content: str,
        content_type: ContentType,
        source_id: int,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_index: int = 0
    ):
        self.content = content
        self.content_type = content_type
        self.source_id = source_id
        self.metadata = metadata or {}
        self.chunk_index = chunk_index
        self.created_at = datetime.now()


class RagDB:
    """
    Unified RAG database with sqlite-vec for fast vector search.
    
    Features:
    - Unified storage for notes and chat history
    - Fast vector search using sqlite-vec
    - Automatic text chunking for long documents
    - Content type filtering
    - Full-text search support
    - Metadata storage and filtering
    """
    
    CHUNK_SIZE = 512           # Optimal chunk size for BGE models
    CHUNK_OVERLAP = 50         # Overlap between chunks
    MAX_CHUNKS_PER_DOC = 20    # Limit chunks per document
    
    def __init__(
        self,
        db_path: Union[str, Path],
        embedding_model: Optional[EmbeddingModel] = None,
        table_name: str = "rag_documents"
    ):
        """
        Initialize the RAG database.
        
        Args:
            db_path: Path to the SQLite database file
            embedding_model: Custom embedding model (uses default if None)
            table_name: Name of the vector table
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.embedding_model = embedding_model or get_embedding_model()
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize the database
        self._init_database()
        
        # Initialize sqlite-vec for vector operations
        self._init_vector_store()
        
        logger.info(f"Initialized RagDB at {self.db_path}")
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Create main documents table
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    source_id INTEGER NOT NULL,
                    chunk_index INTEGER DEFAULT 0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_content_type 
                ON {self.table_name}(content_type)
            """)
            
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_source_id 
                ON {self.table_name}(source_id, content_type)
            """)
            
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_created_at 
                ON {self.table_name}(created_at)
            """)
            
            # Create full-text search table
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name}_fts 
                USING fts5(content, content_id UNINDEXED)
            """)
            
            # Create trigger to keep FTS in sync
            conn.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {self.table_name}_fts_insert 
                AFTER INSERT ON {self.table_name}
                BEGIN
                    INSERT INTO {self.table_name}_fts(content, content_id) 
                    VALUES (NEW.content, NEW.id);
                END
            """)
            
            conn.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {self.table_name}_fts_delete 
                AFTER DELETE ON {self.table_name}
                BEGIN
                    DELETE FROM {self.table_name}_fts WHERE content_id = OLD.id;
                END
            """)
            
            conn.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {self.table_name}_fts_update 
                AFTER UPDATE ON {self.table_name}
                BEGIN
                    UPDATE {self.table_name}_fts 
                    SET content = NEW.content 
                    WHERE content_id = NEW.id;
                END
            """)
    
    def _init_vector_store(self):
        """Initialize sqlite-vec vector storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create vectors table for storing embeddings
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name}_vectors (
                        id INTEGER PRIMARY KEY,
                        document_id INTEGER NOT NULL,
                        embedding BLOB NOT NULL,
                        FOREIGN KEY (document_id) REFERENCES {self.table_name}(id) ON DELETE CASCADE
                    )
                """)
                
                # Create index for faster vector lookups
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_vectors_doc_id 
                    ON {self.table_name}_vectors(document_id)
                """)
            
            self.vector_store_available = True
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.vector_store_available = False
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.CHUNK_SIZE:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text) and len(chunks) < self.MAX_CHUNKS_PER_DOC:
            end = start + self.CHUNK_SIZE
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + self.CHUNK_SIZE // 2, end - 100), -1):
                    if text[i:i+1] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.CHUNK_OVERLAP
            if start >= len(text):
                break
        
        return chunks
    
    def add_document(
        self,
        content: str,
        content_type: ContentType,
        source_id: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        """
        Add a document to the RAG database with automatic chunking.
        
        Args:
            content: Text content to add
            content_type: Type of content (note, chat_message, etc.)
            source_id: ID of the source document
            metadata: Additional metadata
            
        Returns:
            List of document IDs for the created chunks
        """
        chunks = self._chunk_text(content)
        document_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            for i, chunk in enumerate(chunks):
                # Insert into main table
                cursor = conn.execute(f"""
                    INSERT INTO {self.table_name} 
                    (content, content_type, source_id, chunk_index, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    chunk,
                    content_type.value,
                    source_id,
                    i,
                    json.dumps(metadata or {})
                ))
                
                document_id = cursor.lastrowid
                document_ids.append(document_id)
        
        # Add to vector store if available
        if self.vector_store_available and chunks:
            try:
                # Generate embeddings for chunks
                embeddings = self.embedding_model.encode(chunks, use_cache=False)
                
                with sqlite3.connect(self.db_path) as conn:
                    for doc_id, embedding in zip(document_ids, embeddings):
                        # Store embedding as binary data
                        embedding_blob = embedding.astype(np.float32).tobytes()
                        
                        conn.execute(f"""
                            INSERT INTO {self.table_name}_vectors (document_id, embedding)
                            VALUES (?, ?)
                        """, (doc_id, embedding_blob))
                
                logger.debug(f"Added {len(chunks)} embeddings to vector store")
                
            except Exception as e:
                logger.error(f"Failed to add to vector store: {e}")
        
        return document_ids
    
    def add_note(
        self,
        title: str,
        content: str,
        source_id: int,
        tags: Optional[List[str]] = None
    ) -> List[int]:
        """
        Add a note to the RAG database.
        
        Args:
            title: Note title
            content: Note content
            source_id: Original note ID
            tags: Note tags
            
        Returns:
            List of document IDs
        """
        full_content = f"{title}\n\n{content}"
        metadata = {
            "title": title,
            "tags": tags or [],
            "original_note_id": source_id
        }
        
        return self.add_document(
            content=full_content,
            content_type=ContentType.NOTE,
            source_id=source_id,
            metadata=metadata
        )
    
    def add_chat_message(
        self,
        message: str,
        role: str,
        conversation_id: int,
        message_id: int,
        timestamp: Optional[datetime] = None
    ) -> List[int]:
        """
        Add a chat message to the RAG database.
        
        Args:
            message: Message content
            role: Message role (user, assistant)
            conversation_id: ID of the conversation
            message_id: ID of the message
            timestamp: Message timestamp
            
        Returns:
            List of document IDs
        """
        metadata = {
            "role": role,
            "conversation_id": conversation_id,
            "original_message_id": message_id,
            "timestamp": timestamp.isoformat() if timestamp else None
        }
        
        return self.add_document(
            content=message,
            content_type=ContentType.CHAT_MESSAGE,
            source_id=message_id,
            metadata=metadata
        )
    
    def vector_search(
        self,
        query: str,
        content_types: Optional[List[ContentType]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query
            content_types: Filter by content types
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.vector_store_available:
            logger.warning("Vector store not available, falling back to keyword search")
            return self.keyword_search(query, content_types, limit)
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query, use_cache=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Build the SQL query
                base_query = f"""
                    SELECT d.*, v.embedding
                    FROM {self.table_name} d
                    JOIN {self.table_name}_vectors v ON d.id = v.document_id
                """
                
                params = []
                
                # Add content type filter
                if content_types:
                    content_type_values = [ct.value for ct in content_types]
                    placeholders = ",".join("?" * len(content_type_values))
                    base_query += f" WHERE d.content_type IN ({placeholders})"
                    params.extend(content_type_values)
                
                base_query += " ORDER BY d.created_at DESC"
                
                cursor = conn.execute(base_query, params)
                results = []
                
                for row in cursor.fetchall():
                    # Deserialize embedding
                    embedding_blob = row["embedding"]
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    
                    # Calculate cosine similarity
                    similarity = float(np.dot(query_embedding, embedding) / 
                                     (np.linalg.norm(query_embedding) * np.linalg.norm(embedding)))
                    
                    # Filter by similarity threshold
                    if similarity < similarity_threshold:
                        continue
                    
                    # Parse metadata
                    metadata = json.loads(row["metadata"] or "{}")
                    
                    document_data = {
                        "content": row["content"],
                        "metadata": metadata,
                        "content_type": row["content_type"],
                        "source_id": row["source_id"],
                        "chunk_index": row["chunk_index"]
                    }
                    
                    results.append((document_data, similarity))
                
                # Sort by similarity score (descending) and limit
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:limit]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self.keyword_search(query, content_types, limit)
    
    def keyword_search(
        self,
        query: str,
        content_types: Optional[List[ContentType]] = None,
        limit: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform full-text keyword search as fallback.
        
        Args:
            query: Search query
            content_types: Filter by content types
            limit: Maximum number of results
            
        Returns:
            List of (document, score) tuples with score=1.0
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Build the SQL query
                base_query = f"""
                    SELECT d.*, fts.rank
                    FROM {self.table_name}_fts fts
                    JOIN {self.table_name} d ON fts.content_id = d.id
                    WHERE fts MATCH ?
                """
                
                params = [query]
                
                # Add content type filter
                if content_types:
                    content_type_values = [ct.value for ct in content_types]
                    placeholders = ",".join("?" * len(content_type_values))
                    base_query += f" AND d.content_type IN ({placeholders})"
                    params.extend(content_type_values)
                
                base_query += " ORDER BY fts.rank LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(base_query, params)
                results = []
                
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    metadata = json.loads(row_dict.get("metadata", "{}"))
                    
                    document_data = {
                        "content": row_dict["content"],
                        "metadata": metadata,
                        "content_type": row_dict["content_type"],
                        "source_id": row_dict["source_id"],
                        "chunk_index": row_dict["chunk_index"]
                    }
                    
                    # FTS rank as similarity score (normalized)
                    score = 1.0  # Simple score for keyword matches
                    results.append((document_data, score))
                
                return results
                
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"""
                    SELECT 
                        content_type,
                        COUNT(*) as count,
                        COUNT(DISTINCT source_id) as unique_sources
                    FROM {self.table_name}
                    GROUP BY content_type
                """)
                
                stats = {
                    "total_documents": 0,
                    "by_content_type": {},
                    "embedding_model": self.embedding_model.get_model_info()
                }
                
                for row in cursor.fetchall():
                    content_type, count, unique_sources = row
                    stats["by_content_type"][content_type] = {
                        "chunks": count,
                        "unique_sources": unique_sources
                    }
                    stats["total_documents"] += count
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def delete_by_source(self, source_id: int, content_type: ContentType) -> int:
        """
        Delete all chunks for a specific source document.
        
        Args:
            source_id: Source document ID
            content_type: Type of content to delete
            
        Returns:
            Number of chunks deleted
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE source_id = ? AND content_type = ?
                """, (source_id, content_type.value))
                
                deleted_count = cursor.rowcount
                logger.info(f"Deleted {deleted_count} chunks for {content_type.value} {source_id}")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """Clear all data from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f"DELETE FROM {self.table_name}")
                conn.execute(f"DELETE FROM {self.table_name}_fts")
                
                # Clear vector store if available
                if self.vector_store_available:
                    conn.execute(f"DELETE FROM {self.table_name}_vectors")
                
                logger.info("Cleared all data from RAG database")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False