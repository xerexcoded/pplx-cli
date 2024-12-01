import sqlite3
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class NotesDB:
    # Even smaller model, only 22MB
    DEFAULT_MODEL = 'paraphrase-MiniLM-L3-v2'
    
    def __init__(self, directory: Path):
        self.directory = directory
        self.db_path = directory / "notes.db"
        self._model = None
        self._init_db()

    @property
    def model(self):
        """Lazy load the model only when needed."""
        if self._model is None:
            self._model = SentenceTransformer(self.DEFAULT_MODEL)
        return self._model

    def _init_db(self):
        """Initialize the database with required tables."""
        self.directory.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    note_id INTEGER PRIMARY KEY,
                    embedding BLOB,
                    FOREIGN KEY(note_id) REFERENCES notes(id) ON DELETE CASCADE
                )
            """)

    def _generate_embedding(self, text: str) -> bytes:
        """Generate embedding for text using sentence-transformers."""
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            return embedding.astype(np.float32).tobytes()
        except Exception as e:
            print(f"Warning: Could not generate embedding: {str(e)}")
            # Return empty embedding as fallback
            return np.zeros(384, dtype=np.float32).tobytes()

    def _load_embedding(self, blob: bytes) -> np.ndarray:
        """Load embedding from bytes."""
        return np.frombuffer(blob, dtype=np.float32).copy()

    def add_note(self, title: str, content: str, tags: Optional[List[str]] = None) -> int:
        """Add a new note to the database and generate its embedding."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO notes (title, content, tags) VALUES (?, ?, ?)",
                (title, content, json.dumps(tags or []))
            )
            note_id = cursor.lastrowid
            
            # Generate and store embedding for the note
            full_text = f"{title}\n{content}"
            embedding = self._generate_embedding(full_text)
            conn.execute(
                "INSERT INTO embeddings (note_id, embedding) VALUES (?, ?)",
                (note_id, embedding)
            )
            
            return note_id

    def get_note(self, note_id: int) -> Optional[dict]:
        """Retrieve a note by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def list_notes(self, tag: Optional[str] = None) -> List[dict]:
        """List all notes, optionally filtered by tag."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if tag:
                cursor = conn.execute(
                    "SELECT * FROM notes WHERE tags LIKE ? ORDER BY created_at DESC",
                    (f'%"{tag}"%',)
                )
            else:
                cursor = conn.execute("SELECT * FROM notes ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def update_note(self, note_id: int, title: Optional[str] = None, 
                   content: Optional[str] = None, tags: Optional[List[str]] = None) -> bool:
        """Update an existing note."""
        updates = []
        values = []
        if title is not None:
            updates.append("title = ?")
            values.append(title)
        if content is not None:
            updates.append("content = ?")
            values.append(content)
        if tags is not None:
            updates.append("tags = ?")
            values.append(json.dumps(tags))
        
        if not updates:
            return False

        updates.append("updated_at = CURRENT_TIMESTAMP")
        query = f"UPDATE notes SET {', '.join(updates)} WHERE id = ?"
        values.append(note_id)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, values)
            return cursor.rowcount > 0

    def delete_note(self, note_id: int) -> bool:
        """Delete a note by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            return cursor.rowcount > 0

    def search_similar_notes(self, query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        """Search for notes similar to the query using vector similarity."""
        try:
            query_embedding = self.model.encode(query, show_progress_bar=False)
            query_embedding = query_embedding.astype(np.float32)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT notes.*, embeddings.embedding 
                    FROM notes 
                    LEFT JOIN embeddings ON notes.id = embeddings.note_id
                    WHERE embeddings.embedding IS NOT NULL
                """)
                
                results = []
                for row in cursor:
                    note_dict = dict(row)
                    note_embedding = self._load_embedding(note_dict.pop('embedding'))
                    
                    # Ensure embeddings are float32 and properly shaped
                    note_embedding = note_embedding.astype(np.float32)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, note_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(note_embedding)
                    )
                    
                    results.append((note_dict, float(similarity)))
                
                # Sort by similarity and return top_k results
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]
        except Exception as e:
            print(f"Warning: Search failed: {str(e)}")
            # Fallback to returning most recent notes
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM notes ORDER BY created_at DESC LIMIT ?", (top_k,))
                return [(dict(row), 0.0) for row in cursor.fetchall()]