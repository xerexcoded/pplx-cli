import pytest
import os
import tempfile
from pathlib import Path
from pplx_cli.notes import NotesDB


@pytest.fixture
def notes_db(tmp_path):
    return NotesDB(tmp_path)


def test_init_creates_db(notes_db):
    assert notes_db.db_path.exists()


def test_add_note(notes_db):
    note_id = notes_db.add_note("Test Title", "Test Content", ["tag1", "tag2"])
    assert note_id is not None
    assert note_id > 0


def test_get_note(notes_db):
    note_id = notes_db.add_note("Title", "Content", ["test"])
    note = notes_db.get_note(note_id)
    assert note is not None
    assert note["title"] == "Title"
    assert note["content"] == "Content"
    assert "test" in note["tags"]


def test_get_nonexistent_note(notes_db):
    assert notes_db.get_note(9999) is None


def test_list_notes(notes_db):
    notes_db.add_note("A", "Content A")
    notes_db.add_note("B", "Content B")
    notes = notes_db.list_notes()
    assert len(notes) == 2
    assert notes[0]["title"] == "B"


def test_list_notes_by_tag(notes_db):
    notes_db.add_note("Tagged", "Content", ["python"])
    notes_db.add_note("Untagged", "Content", [])
    tagged = notes_db.list_notes(tag="python")
    assert len(tagged) == 1
    assert tagged[0]["title"] == "Tagged"


def test_update_note(notes_db):
    note_id = notes_db.add_note("Old", "Old Content", ["old"])
    result = notes_db.update_note(note_id, title="New", content="New Content")
    assert result is True
    note = notes_db.get_note(note_id)
    assert note["title"] == "New"
    assert note["content"] == "New Content"


def test_update_note_no_changes(notes_db):
    note_id = notes_db.add_note("Stay", "Same")
    result = notes_db.update_note(note_id)
    assert result is False


def test_delete_note(notes_db):
    note_id = notes_db.add_note("Delete", "Me")
    result = notes_db.delete_note(note_id)
    assert result is True
    assert notes_db.get_note(note_id) is None


def test_delete_nonexistent_note(notes_db):
    assert notes_db.delete_note(9999) is False


def test_search_similar_notes(notes_db):
    notes_db.add_note("Python basics", "Python is a great programming language for beginners.")
    notes_db.add_note("Cooking tips", "How to make the perfect pasta with tomato sauce.")
    results = notes_db.search_similar_notes("programming language", top_k=2)
    assert len(results) > 0
    assert any("Python" in r[0]["title"] for r in results)


def test_search_similar_notes_empty_db(notes_db):
    results = notes_db.search_similar_notes("anything")
    assert len(results) == 0


def test_embeddings_persisted(notes_db):
    note_id = notes_db.add_note("Embedding Test", "Some content for embedding")
    import sqlite3
    with sqlite3.connect(notes_db.db_path) as conn:
        cursor = conn.execute("SELECT embedding FROM embeddings WHERE note_id = ?", (note_id,))
        row = cursor.fetchone()
    assert row is not None
    assert row[0] is not None
