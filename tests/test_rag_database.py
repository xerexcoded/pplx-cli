import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from pplx_cli.rag.database import RagDB, ContentType, DocumentChunk


@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    model.encode.return_value = np.zeros(768, dtype=np.float32)
    model.get_model_info.return_value = {
        "model_name": "BAAI/bge-base-en-v1.5",
        "device": "cpu",
        "embedding_dim": 768,
        "quantized": False,
        "cache_size": 1000,
        "max_sequence_length": 512
    }
    return model


@pytest.fixture
def rag_db(tmp_path, mock_embedding_model):
    db_path = tmp_path / "rag_test.db"
    return RagDB(db_path, embedding_model=mock_embedding_model, table_name="test_docs")


def test_init_creates_db(rag_db):
    assert rag_db.db_path.exists()


def test_add_document(rag_db, mock_embedding_model):
    mock_embedding_model.encode.return_value = np.array([np.zeros(768, dtype=np.float32)])
    doc_ids = rag_db.add_document(
        "Test content for testing.",
        ContentType.NOTE,
        source_id=1,
        metadata={"title": "Test"}
    )
    assert len(doc_ids) > 0
    assert all(isinstance(did, int) for did in doc_ids)


def test_add_note(rag_db, mock_embedding_model):
    mock_embedding_model.encode.return_value = np.array([np.zeros(768, dtype=np.float32)])
    doc_ids = rag_db.add_note(
        title="My Note",
        content="Note content here.",
        source_id=1,
        tags=["tag1", "tag2"]
    )
    assert len(doc_ids) > 0


def test_add_chat_message(rag_db, mock_embedding_model):
    from datetime import datetime
    mock_embedding_model.encode.return_value = np.array([np.zeros(768, dtype=np.float32)])
    doc_ids = rag_db.add_chat_message(
        message="Chat message content",
        role="user",
        conversation_id=1,
        message_id=1,
        timestamp=datetime.now()
    )
    assert len(doc_ids) > 0


def test_text_chunking(rag_db):
    short_text = "Short text."
    chunks = rag_db._chunk_text(short_text)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_text_chunking_long(rag_db):
    long_text = "A. " * 300
    chunks = rag_db._chunk_text(long_text)
    assert len(chunks) > 1


def test_vector_search(rag_db, mock_embedding_model):
    mock_embedding_model.encode.return_value = np.array([np.ones(768, dtype=np.float32)])
    rag_db.add_document("Python programming guide", ContentType.NOTE, source_id=1, metadata={"title": "Python"})
    mock_embedding_model.encode.return_value = np.array([np.ones(768, dtype=np.float32)])
    results = rag_db.vector_search("python", limit=5)
    assert len(results) > 0


def test_vector_search_empty(rag_db, mock_embedding_model):
    mock_embedding_model.encode.return_value = np.ones(768, dtype=np.float32)
    results = rag_db.vector_search("nothing", limit=5)
    assert len(results) == 0


def test_keyword_search(rag_db, mock_embedding_model):
    mock_embedding_model.encode.return_value = np.array([np.zeros(768, dtype=np.float32)])
    rag_db.add_document("Unique searchable content xyz", ContentType.NOTE, source_id=1)
    results = rag_db.keyword_search("Unique", limit=5)
    assert len(results) > 0


def test_keyword_search_no_match(rag_db):
    results = rag_db.keyword_search("nonexistent", limit=5)
    assert len(results) == 0


def test_keyword_search_returns_fst_rank(rag_db, mock_embedding_model):
    mock_embedding_model.encode.return_value = np.array([np.zeros(768, dtype=np.float32)])
    rag_db.add_document("Matchable content here", ContentType.NOTE, source_id=1)
    results = rag_db.keyword_search("Matchable", limit=5)
    assert len(results) > 0
    _, score = results[0]
    assert isinstance(score, float)


def test_get_stats(rag_db, mock_embedding_model):
    mock_embedding_model.encode.return_value = np.array([np.zeros(768, dtype=np.float32)])
    rag_db.add_document("Note content", ContentType.NOTE, source_id=1)
    rag_db.add_document("Chat content", ContentType.CHAT_MESSAGE, source_id=1)
    stats = rag_db.get_stats()
    assert stats["total_documents"] == 2
    assert "note" in stats["by_content_type"]
    assert "chat_message" in stats["by_content_type"]


def test_delete_by_source(rag_db, mock_embedding_model):
    mock_embedding_model.encode.return_value = np.array([np.zeros(768, dtype=np.float32)])
    rag_db.add_document("To delete", ContentType.NOTE, source_id=42)
    rag_db.add_document("To keep", ContentType.NOTE, source_id=43)
    deleted = rag_db.delete_by_source(42, ContentType.NOTE)
    assert deleted > 0
    results = rag_db.keyword_search("delete", limit=5)
    assert len(results) == 0


def test_clear_all(rag_db, mock_embedding_model):
    mock_embedding_model.encode.return_value = np.array([np.zeros(768, dtype=np.float32)])
    rag_db.add_document("Clear me", ContentType.NOTE, source_id=1)
    rag_db.clear_all()
    stats = rag_db.get_stats()
    assert stats["total_documents"] == 0


def test_content_type_filter(rag_db, mock_embedding_model):
    mock_embedding_model.encode.return_value = np.array([np.zeros(768, dtype=np.float32)])
    rag_db.add_document("Note abc", ContentType.NOTE, source_id=1)
    rag_db.add_document("Chat xyz", ContentType.CHAT_MESSAGE, source_id=1)
    mock_embedding_model.encode.return_value = np.array([np.ones(768, dtype=np.float32)])
    results = rag_db.vector_search("abc", content_types=[ContentType.NOTE], limit=5)
    assert len(results) == 1
    assert results[0][0]["content_type"] == "note"


@pytest.mark.parametrize(
    "similarity_threshold, fail_native_query, expected_source_ids, expected_limits",
    [
        (0.0, False, [101, 102], [10, 20]),
        (0.75, False, [101], [10, 20]),
        (0.0, True, [101, 102], [10]),
    ],
)
def test_native_vector_search_expands_candidates_for_content_filter(
    rag_db,
    mock_embedding_model,
    monkeypatch,
    similarity_threshold,
    fail_native_query,
    expected_source_ids,
    expected_limits,
):
    import sqlite3

    query_embedding = np.array([1.0, 0.0], dtype=np.float32)
    vectors = {}
    documents = []

    def encode(texts, **kwargs):
        if isinstance(texts, str):
            return query_embedding
        return np.vstack([vectors[text] for text in texts])

    mock_embedding_model.encode.side_effect = encode
    items = [
        (ContentType.CHAT_MESSAGE, index + 1, index * 0.03)
        for index in range(10)
    ] + [
        (ContentType.NOTE, 101, 0.5),
        (ContentType.NOTE, 102, 0.8),
    ]
    for content_type, source_id, angle in items:
        content = f"document {source_id}"
        vectors[content] = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        document_id = rag_db.add_document(content, content_type, source_id)[0]
        documents.append((document_id, vectors[content]))

    candidate_rows = sorted(
        [
            {
                "rowid": document_id,
                "distance": 1.0 - float(np.dot(query_embedding, vector)),
            }
            for document_id, vector in documents
        ],
        key=lambda row: row["distance"],
    )
    candidate_limits = []
    connect = sqlite3.connect

    class Cursor:
        def __init__(self, rows):
            self.rows = rows

        def fetchone(self):
            return self.rows[0] if self.rows else None

        def fetchall(self):
            return self.rows

    class NativeCandidateConnection:
        def __init__(self, path):
            self.connection = connect(path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.connection.close()

        @property
        def row_factory(self):
            return self.connection.row_factory

        @row_factory.setter
        def row_factory(self, value):
            self.connection.row_factory = value

        def execute(self, sql, params=()):
            if "FROM sqlite_master" in sql:
                return Cursor([(1,)])
            if "embedding MATCH ?" in sql:
                candidate_limit = params[-1]
                candidate_limits.append(candidate_limit)
                if fail_native_query:
                    raise sqlite3.OperationalError("native query failed")
                return Cursor(candidate_rows[:candidate_limit])
            return self.connection.execute(sql, params)

    monkeypatch.setattr(rag_db, "_load_sqlite_vec", lambda conn: True)
    monkeypatch.setattr(
        "pplx_cli.rag.database.sqlite3.connect", NativeCandidateConnection
    )
    native_results = rag_db.vector_search(
        "query",
        content_types=[ContentType.NOTE],
        limit=2,
        similarity_threshold=similarity_threshold,
    )

    assert candidate_limits == expected_limits
    assert [result[0]["source_id"] for result in native_results] == expected_source_ids

    monkeypatch.setattr(rag_db, "_load_sqlite_vec", lambda conn: False)
    exact_results = rag_db.vector_search(
        "query",
        content_types=[ContentType.NOTE],
        limit=2,
        similarity_threshold=similarity_threshold,
    )
    assert [result[0]["source_id"] for result in native_results] == [
        result[0]["source_id"] for result in exact_results
    ]
    assert [result[1] for result in native_results] == pytest.approx(
        [result[1] for result in exact_results]
    )


def test_vector_search_non_positive_limit_skips_embedding(rag_db, mock_embedding_model):
    mock_embedding_model.encode.side_effect = AssertionError("must not encode")

    assert rag_db.vector_search("query", limit=0) == []
    assert rag_db.vector_search("query", limit=-1) == []
    mock_embedding_model.encode.assert_not_called()


def test_document_chunk_class():
    chunk = DocumentChunk(
        content="Test",
        content_type=ContentType.NOTE,
        source_id=1,
        metadata={"key": "value"},
        chunk_index=0
    )
    assert chunk.content == "Test"
    assert chunk.content_type == ContentType.NOTE
    assert chunk.source_id == 1
    assert chunk.metadata == {"key": "value"}
    assert chunk.chunk_index == 0


def test_content_type_enum():
    assert ContentType.NOTE.value == "note"
    assert ContentType.CHAT_MESSAGE.value == "chat_message"
    assert ContentType.CONVERSATION.value == "conversation"
