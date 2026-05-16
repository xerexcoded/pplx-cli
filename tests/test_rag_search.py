import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from pplx_cli.rag.search import HybridSearchEngine, SearchMode, SearchResult


@pytest.fixture
def mock_rag_db():
    db = MagicMock()
    db.vector_search.return_value = []
    db.keyword_search.return_value = []
    db.get_stats.return_value = {"total_documents": 0, "by_content_type": {}}
    db.embedding_model = MagicMock()
    db.embedding_model.get_model_info.return_value = {
        "model_name": "BAAI/bge-base-en-v1.5",
        "device": "cpu",
        "embedding_dim": 768,
        "quantized": False,
        "cache_size": 1000,
        "max_sequence_length": 512
    }
    return db


@pytest.fixture
def engine(mock_rag_db):
    return HybridSearchEngine(mock_rag_db)


def test_search_mode_enum():
    assert SearchMode.VECTOR_ONLY.value == "vector"
    assert SearchMode.KEYWORD_ONLY.value == "keyword"
    assert SearchMode.HYBRID.value == "hybrid"


def test_search_result_creation():
    result = SearchResult(
        content="Test content",
        score=0.95,
        source_id=1,
        content_type="note",
        metadata={"title": "Test"},
        chunk_index=0,
        rank=1
    )
    assert result.content == "Test content"
    assert result.score == 0.95
    assert result.source_id == 1
    assert "0.950" in repr(result)


def test_vector_search_mode(engine, mock_rag_db):
    mock_rag_db.vector_search.return_value = [
        ({"content": "Result", "metadata": {}, "content_type": "note", "source_id": 1, "chunk_index": 0}, 0.9)
    ]
    results = engine.search("query", mode=SearchMode.VECTOR_ONLY, limit=5)
    assert len(results) == 1
    assert results[0].score == 0.9
    mock_rag_db.vector_search.assert_called_once()
    mock_rag_db.keyword_search.assert_not_called()


def test_keyword_search_mode(engine, mock_rag_db):
    mock_rag_db.keyword_search.return_value = [
        ({"content": "KW Result", "metadata": {}, "content_type": "note", "source_id": 1, "chunk_index": 0}, 1.0)
    ]
    results = engine.search("query", mode=SearchMode.KEYWORD_ONLY, limit=5)
    assert len(results) == 1
    assert results[0].content == "KW Result"
    mock_rag_db.keyword_search.assert_called_once()
    mock_rag_db.vector_search.assert_not_called()


def test_hybrid_search(engine, mock_rag_db):
    mock_rag_db.vector_search.return_value = [
        ({"content": "Vec Result", "metadata": {}, "content_type": "note", "source_id": 1, "chunk_index": 0}, 0.9)
    ]
    mock_rag_db.keyword_search.return_value = [
        ({"content": "KW Result", "metadata": {}, "content_type": "chat_message", "source_id": 2, "chunk_index": 0}, 1.0)
    ]
    results = engine.search("query", mode=SearchMode.HYBRID, limit=5)
    assert len(results) >= 1
    mock_rag_db.vector_search.assert_called()
    mock_rag_db.keyword_search.assert_called()


def test_hybrid_deduplication(engine, mock_rag_db):
    mock_rag_db.vector_search.return_value = [
        ({"content": "Same Content", "metadata": {}, "content_type": "note", "source_id": 1, "chunk_index": 0}, 0.9)
    ]
    mock_rag_db.keyword_search.return_value = [
        ({"content": "Same Content", "metadata": {}, "content_type": "note", "source_id": 1, "chunk_index": 0}, 1.0)
    ]
    results = engine.search("query", mode=SearchMode.HYBRID, limit=5, deduplicate=True)
    assert len(results) == 1


def test_search_preprocessing(engine):
    result = engine._preprocess_query("  test query  ")
    assert result == "test query"


def test_get_search_stats(engine):
    stats = engine.get_search_stats()
    assert "database_stats" in stats
    assert "search_config" in stats
    assert "embedding_model" in stats
    assert stats["search_config"]["default_mode"] == SearchMode.HYBRID.value


def test_explain_search(engine):
    explanation = engine.explain_search("test query", mode=SearchMode.VECTOR_ONLY)
    assert explanation["original_query"] == "test query"
    assert explanation["search_mode"] == "vector"


def test_explain_search_hybrid(engine):
    explanation = engine.explain_search("test", mode=SearchMode.HYBRID)
    assert "rrf_config" in explanation
    assert explanation["rrf_config"]["k"] == 60


def test_rrf_combines_scores(engine):
    from pplx_cli.rag.search import SearchResult
    vr = [
        SearchResult("Vec 1", 0.9, 1, "note", {}, 0, 1),
        SearchResult("Vec 2", 0.8, 2, "note", {}, 0, 2),
    ]
    kw = [
        SearchResult("KW 1", 1.0, 3, "chat_message", {}, 0, 1),
    ]
    fused = engine._reciprocal_rank_fusion(vr, kw, limit=10)
    assert len(fused) == 3


def test_get_document_id(engine):
    result = SearchResult("content", 1.0, 42, "note", {}, 3)
    doc_id = engine._get_document_id(result)
    assert doc_id == "note:42:3"


def test_search_similar_sources(engine, mock_rag_db):
    import sqlite3
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = ("Source content",)
    mock_conn.execute.return_value = mock_cursor
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with patch("sqlite3.connect", return_value=mock_conn):
        mock_rag_db.table_name = "rag_documents"
        mock_rag_db.vector_search.return_value = []
        results = engine.search_similar_sources(1, "note")
        assert results == []


def test_deduplicate_results(engine):
    from pplx_cli.rag.search import SearchResult
    results = [
        SearchResult("Duplicate", 0.9, 1, "note", {}),
        SearchResult("Duplicate", 0.8, 1, "note", {}),
        SearchResult("Unique", 0.7, 2, "chat_message", {}),
    ]
    deduped = engine._deduplicate_results(results)
    assert len(deduped) == 2
