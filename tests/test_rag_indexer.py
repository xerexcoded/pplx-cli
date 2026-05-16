import pytest
import numpy as np
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
from pplx_cli.rag.indexer import BatchIndexer, IndexingProgress
from pplx_cli.rag.database import RagDB, ContentType


@pytest.fixture
def mock_rag_db():
    db = MagicMock()
    db.db_path = Path(tempfile.mkdtemp()) / "test.db"
    db.table_name = "rag_documents"
    db.add_note.return_value = [1]
    db.add_chat_message.return_value = [1]
    db.clear_all.return_value = True
    db.embedding_model = MagicMock()
    db.embedding_model.encode.return_value = np.array([np.zeros(768, dtype=np.float32)])
    return db


@pytest.fixture
def indexer(mock_rag_db):
    return BatchIndexer(mock_rag_db, batch_size=10, max_retries=1, show_progress=False)


def test_indexing_progress():
    progress = IndexingProgress(100, "Test Operation")
    assert progress.total_items == 100
    assert progress.processed_items == 0

    progress.update(processed=50)
    assert progress.processed_items == 50

    progress.update(processed=50, failed=5)
    assert progress.processed_items == 100
    assert progress.failed_items == 5
    assert progress.is_complete()


def test_indexing_progress_zero_items():
    progress = IndexingProgress(0, "Empty")
    progress.update()
    assert progress.is_complete()


def test_migrate_notes_database_not_found(indexer):
    migrated, failed = indexer.migrate_notes_database(Path("/nonexistent/notes.db"))
    assert migrated == 0
    assert failed == 0


def test_migrate_notes_database_empty(indexer, tmp_path):
    notes_path = tmp_path / "empty_notes.db"
    with sqlite3.connect(notes_path) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, title TEXT, content TEXT, tags TEXT)")
    migrated, failed = indexer.migrate_notes_database(notes_path)
    assert migrated == 0


def test_migrate_notes_database_with_data(indexer, tmp_path):
    notes_path = tmp_path / "notes.db"
    with sqlite3.connect(notes_path) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, title TEXT, content TEXT, tags TEXT)")
        conn.execute("INSERT INTO notes (id, title, content, tags) VALUES (1, 'Note 1', 'Content 1', '[]')")
        conn.execute("INSERT INTO notes (id, title, content, tags) VALUES (2, 'Note 2', 'Content 2', '[]')")

    indexer.rag_db.add_note.return_value = [1]
    migrated, failed = indexer.migrate_notes_database(notes_path)
    assert migrated == 2
    assert failed == 0
    assert indexer.rag_db.add_note.call_count == 2


def test_migrate_chat_history_not_found(indexer):
    migrated, failed = indexer.migrate_chat_history_database(Path("/nonexistent/chat.db"))
    assert migrated == 0
    assert failed == 0


def test_migrate_chat_history_with_data(indexer, tmp_path):
    chat_path = tmp_path / "chat.db"
    with sqlite3.connect(chat_path) as conn:
        conn.execute("CREATE TABLE conversations (id INTEGER PRIMARY KEY, title TEXT, topic TEXT)")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id INTEGER, role TEXT, content TEXT, timestamp TEXT)")
        conn.execute("INSERT INTO conversations (id, title) VALUES (1, 'Conv 1')")
        conn.execute("INSERT INTO messages (id, conversation_id, role, content) VALUES (1, 1, 'user', 'Hello')")

    migrated, failed = indexer.migrate_chat_history_database(chat_path)
    assert migrated == 1
    assert failed == 0


def test_migrate_notes_with_clear(indexer, tmp_path):
    notes_path = tmp_path / "notes.db"
    with sqlite3.connect(notes_path) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, title TEXT, content TEXT, tags TEXT)")
        conn.execute("INSERT INTO notes (id, title, content, tags) VALUES (1, 'Note 1', 'Content 1', '[]')")

    migrated, failed = indexer.migrate_notes_database(notes_path, clear_existing=True)
    assert indexer.rag_db.clear_all.called


def test_estimate_migration_time(indexer, tmp_path):
    notes_path = tmp_path / "notes.db"
    with sqlite3.connect(notes_path) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, title TEXT, content TEXT, tags TEXT)")
        conn.execute("INSERT INTO notes (id, title, content, tags) VALUES (1, 'N', 'C', '[]')")

    estimates = indexer.estimate_migration_time(notes_db_path=notes_path)
    assert "notes" in estimates
    assert estimates["notes"]["count"] == 1
    assert "total" in estimates


def test_estimate_migration_time_with_chats(indexer, tmp_path):
    chat_path = tmp_path / "chat.db"
    with sqlite3.connect(chat_path) as conn:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT)")
        conn.execute("INSERT INTO messages (id, content) VALUES (1, 'Hello')")

    estimates = indexer.estimate_migration_time(chat_db_path=chat_path)
    assert "chat_history" in estimates
    assert estimates["chat_history"]["count"] == 1


def test_get_migration_stats(indexer):
    stats = indexer.get_migration_stats()
    assert "rag_database" in stats
    assert "indexer_config" in stats
    assert stats["indexer_config"]["batch_size"] == 10


def test_process_notes_batch_with_retry(indexer):
    batch = [{"id": 1, "title": "Test", "content": "Content", "tags": '["tag1"]'}]
    indexer.max_retries = 2
    indexer.rag_db.add_note.side_effect = [Exception("fail"), [1]]
    results = indexer._process_notes_batch(batch)
    assert results == [True]


def test_process_notes_batch_all_fail(indexer):
    batch = [{"id": 1, "title": "Test", "content": "Content", "tags": '["tag1"]'}]
    indexer.rag_db.add_note.side_effect = Exception("fail")
    results = indexer._process_notes_batch(batch)
    assert results == [False]


def test_process_chat_batch(indexer):
    batch = [{"id": 1, "role": "user", "content": "Hello", "conversation_id": 1, "timestamp": "2024-01-01T00:00:00"}]
    results = indexer._process_chat_batch(batch)
    assert results == [True]
    indexer.rag_db.add_chat_message.assert_called_once()


def test_reindex_all_empty_db(indexer):
    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = indexer.reindex_all()
        assert result is True


def test_stream_rag_documents(indexer):
    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.side_effect = [
            [{"id": 1, "content": "Doc 1"}, {"id": 2, "content": "Doc 2"}],
            []
        ]
        mock_conn.execute.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        batches = list(indexer._stream_rag_documents())
        assert len(batches) == 1
        assert len(batches[0]) == 2
