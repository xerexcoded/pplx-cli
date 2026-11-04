import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from pplx_cli.migrations import migrate_to_rag as migration


def test_find_existing_databases_discovers_notes_and_chat_history(
    tmp_path, monkeypatch
):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "notes.db").touch()
    home = tmp_path / "home"
    chat_path = home / ".local/share/perplexity/chat_history/chat_history.db"
    chat_path.parent.mkdir(parents=True)
    chat_path.touch()
    monkeypatch.setattr(
        migration.Config,
        "get_instance",
        lambda: SimpleNamespace(notes_dir=notes_dir),
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    databases = migration.find_existing_databases()

    assert databases == {"notes": notes_dir / "notes.db", "chat_history": chat_path}


def test_analyze_existing_data_reports_note_and_chat_counts(tmp_path):
    notes_path = tmp_path / "notes.db"
    with sqlite3.connect(notes_path) as conn:
        conn.execute("CREATE TABLE notes (title TEXT, content TEXT)")
        conn.executemany(
            "INSERT INTO notes (title, content) VALUES (?, ?)",
            [("First", "hello"), ("Second", "world!")],
        )
    chat_path = tmp_path / "chat.db"
    with sqlite3.connect(chat_path) as conn:
        conn.execute(
            "CREATE TABLE messages (conversation_id INTEGER, role TEXT, content TEXT)"
        )
        conn.executemany(
            "INSERT INTO messages VALUES (?, ?, ?)",
            [(1, "user", "question"), (1, "assistant", "answer"), (2, "user", "again")],
        )

    result = migration.analyze_existing_data(
        {"notes": notes_path, "chat_history": chat_path}
    )

    assert result["notes"]["count"] == 2
    assert result["notes"]["avg_content_length"] == 5
    assert len(result["notes"]["samples"]) == 2
    assert result["chat_history"]["total_messages"] == 3
    assert result["chat_history"]["total_conversations"] == 2
    assert result["chat_history"]["role_counts"] == {"assistant": 1, "user": 2}
    assert result["chat_history"]["avg_content_length"] == 6


def test_analyze_existing_data_reports_invalid_database(tmp_path):
    result = migration.analyze_existing_data({"notes": tmp_path / "missing.db"})

    assert "error" in result["notes"]


def test_migrate_data_clears_only_before_first_source(monkeypatch, tmp_path):
    calls = []

    class FakeIndexer:
        def __init__(self, rag_db, show_progress):
            assert rag_db == "rag-db"
            assert show_progress is True

        def migrate_notes_database(self, path, clear_existing=False):
            calls.append(("notes", path, clear_existing))
            return 4, 1

        def migrate_chat_history_database(
            self,
            path,
            clear_existing=False,
            include_user_messages=True,
            include_assistant_messages=True,
        ):
            calls.append(
                (
                    "chat_history",
                    path,
                    clear_existing,
                    include_user_messages,
                    include_assistant_messages,
                )
            )
            return 3, 0

    monkeypatch.setattr(migration, "create_rag_database", lambda: "rag-db")
    monkeypatch.setattr(migration, "BatchIndexer", FakeIndexer)
    notes = tmp_path / "notes.db"
    chat = tmp_path / "chat.db"

    results = migration.migrate_data(
        {"notes": notes, "chat_history": chat},
        clear_existing=True,
        include_user_messages=False,
        include_assistant_messages=True,
    )

    assert calls == [
        ("notes", notes, True),
        ("chat_history", chat, False, False, True),
    ]
    assert results["total_migrated"] == 7
    assert results["total_failed"] == 1
    assert results["by_source"]["notes"] == {"migrated": 4, "failed": 1}


def test_migrate_data_dry_run_returns_estimates(monkeypatch):
    estimate = {"notes": {"count": 2}, "total": {"estimated_time_formatted": "1s"}}

    class FakeIndexer:
        def __init__(self, rag_db, show_progress):
            assert show_progress is False

        def estimate_migration_time(self, **kwargs):
            assert kwargs == {"notes_db_path": "notes.db", "chat_db_path": None}
            return estimate

    monkeypatch.setattr(migration, "create_rag_database", lambda: "temporary-db")
    monkeypatch.setattr(migration, "BatchIndexer", FakeIndexer)

    result = migration.migrate_data({"notes": "notes.db"}, dry_run=True)

    assert result == {"dry_run": True, "estimates": estimate}


def test_print_migration_summary_handles_dry_run_and_empty_sources(capsys):
    migration.print_migration_summary({}, {"dry_run": True})
    assert "DRY RUN" in capsys.readouterr().out

    migration.print_migration_summary(
        {"notes": {"count": 0}},
        {
            "total_migrated": 0,
            "total_failed": 1,
            "by_source": {"notes": {"migrated": 0, "failed": 1}},
        },
    )
    output = capsys.readouterr().out
    assert "Total items failed: 1" in output
    assert "Success rate: 0.0%" in output


def test_main_returns_when_no_databases_match_filter(monkeypatch, capsys):
    monkeypatch.setattr(migration, "find_existing_databases", lambda: {"notes": "notes.db"})
    monkeypatch.setattr("sys.argv", ["migrate", "--chat-only"])

    assert migration.main() == 1
    assert "matching the specified filters" in capsys.readouterr().out


def test_main_cancels_after_user_declines(monkeypatch, capsys):
    monkeypatch.setattr(migration, "find_existing_databases", lambda: {"notes": "notes.db"})
    monkeypatch.setattr(migration, "analyze_existing_data", lambda databases: {})
    monkeypatch.setattr(migration, "input", lambda prompt: "n", raising=False)
    monkeypatch.setattr("sys.argv", ["migrate"])
    monkeypatch.setattr(
        migration,
        "migrate_data",
        lambda **kwargs: pytest.fail("migration must not run after cancellation"),
    )

    assert migration.main() == 0
    assert "Migration cancelled" in capsys.readouterr().out


def test_main_dry_run_skips_confirmation_and_summarizes(monkeypatch):
    calls = []
    monkeypatch.setattr(migration, "find_existing_databases", lambda: {"notes": "notes.db"})
    monkeypatch.setattr(
        migration,
        "analyze_existing_data",
        lambda databases: {"notes": {"count": 2, "avg_content_length": 8}},
    )
    monkeypatch.setattr("sys.argv", ["migrate", "--dry-run", "--verbose"])
    monkeypatch.setattr(
        migration,
        "migrate_data",
        lambda **kwargs: calls.append(kwargs) or {"dry_run": True},
    )
    monkeypatch.setattr(migration, "print_migration_summary", lambda analysis, results: None)

    assert migration.main() == 0
    assert calls == [
        {
            "databases": {"notes": "notes.db"},
            "clear_existing": False,
            "include_user_messages": True,
            "include_assistant_messages": True,
            "dry_run": True,
        }
    ]


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("broken")])
def test_main_converts_interruption_and_errors_to_failure(monkeypatch, error):
    def fail():
        raise error

    monkeypatch.setattr(migration, "find_existing_databases", fail)
    monkeypatch.setattr("sys.argv", ["migrate"])

    assert migration.main() == 1
