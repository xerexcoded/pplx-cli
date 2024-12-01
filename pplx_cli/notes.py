import sqlite3
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import json

class NotesDB:
    def __init__(self, directory: Path):
        self.directory = directory
        self.db_path = directory / "notes.db"
        self._init_db()

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
                    note_id INTEGER,
                    embedding BLOB,
                    FOREIGN KEY(note_id) REFERENCES notes(id)
                )
            """)

    def add_note(self, title: str, content: str, tags: Optional[List[str]] = None) -> int:
        """Add a new note to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO notes (title, content, tags) VALUES (?, ?, ?)",
                (title, content, json.dumps(tags or []))
            )
            return cursor.lastrowid

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