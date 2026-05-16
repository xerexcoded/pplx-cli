import pytest
import tempfile
from pathlib import Path
from pplx_cli.chat_history import ChatHistoryDB


@pytest.fixture
def chat_db(tmp_path):
    return ChatHistoryDB(tmp_path)


def test_init_creates_db(chat_db):
    assert chat_db.db_path.exists()


def test_create_conversation(chat_db):
    conv_id = chat_db.create_conversation(title="Test Conv", topic="testing")
    assert conv_id is not None
    assert conv_id > 0


def test_create_conversation_no_topic(chat_db):
    conv_id = chat_db.create_conversation(title="No Topic")
    assert conv_id > 0


def test_add_message(chat_db):
    conv_id = chat_db.create_conversation(title="Chat")
    msg_id = chat_db.add_message(conv_id, "user", "Hello world")
    assert msg_id > 0


def test_get_conversation(chat_db):
    conv_id = chat_db.create_conversation(title="Full Chat", topic="demo")
    chat_db.add_message(conv_id, "user", "Question")
    chat_db.add_message(conv_id, "assistant", "Answer")
    conv = chat_db.get_conversation(conv_id)
    assert conv is not None
    assert conv["title"] == "Full Chat"
    assert conv["topic"] == "demo"
    assert len(conv["messages"]) == 2
    assert conv["messages"][0]["role"] == "user"
    assert conv["messages"][1]["role"] == "assistant"


def test_get_nonexistent_conversation(chat_db):
    assert chat_db.get_conversation(9999) is None


def test_list_conversations(chat_db):
    chat_db.create_conversation(title="A")
    chat_db.create_conversation(title="B")
    convs = chat_db.list_conversations()
    assert len(convs) == 2


def test_list_conversations_by_topic(chat_db):
    chat_db.create_conversation(title="Topic Conv", topic="machine-learning")
    chat_db.create_conversation(title="Other", topic="cooking")
    filtered = chat_db.list_conversations(topic="learning")
    assert len(filtered) == 1
    assert filtered[0]["title"] == "Topic Conv"


def test_search_conversations(chat_db):
    conv_id = chat_db.create_conversation(title="Searchable", topic="search")
    chat_db.add_message(conv_id, "user", "unique search term")
    results = chat_db.search_conversations("unique")
    assert len(results) >= 1


def test_search_conversations_no_match(chat_db):
    results = chat_db.search_conversations("nonexistent_term_xyz")
    assert len(results) == 0


def test_get_messages(chat_db):
    conv_id = chat_db.create_conversation(title="Msg Test")
    chat_db.add_message(conv_id, "user", "First")
    chat_db.add_message(conv_id, "assistant", "Second")
    messages = chat_db.get_messages(conv_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_export_markdown(chat_db):
    conv_id = chat_db.create_conversation(title="Export Test")
    chat_db.add_message(conv_id, "user", "Hello")
    chat_db.add_message(conv_id, "assistant", "Hi there")
    result = chat_db.export_conversation(conv_id, "markdown")
    assert isinstance(result, str)
    assert "Export Test" in result
    assert "Hello" in result
    assert "Hi there" in result


def test_export_json(chat_db):
    conv_id = chat_db.create_conversation(title="JSON Export")
    chat_db.add_message(conv_id, "user", "Test message")
    result = chat_db.export_conversation(conv_id, "json")
    assert isinstance(result, str)
    assert "JSON Export" in result
    import json
    parsed = json.loads(result)
    assert parsed["title"] == "JSON Export"


def test_export_text(chat_db):
    conv_id = chat_db.create_conversation(title="Text Export")
    chat_db.add_message(conv_id, "user", "Hello")
    result = chat_db.export_conversation(conv_id, "txt")
    assert isinstance(result, str)
    assert "Text Export" in result
    assert "Hello" in result


def test_export_csv(chat_db):
    conv_id = chat_db.create_conversation(title="CSV Export")
    chat_db.add_message(conv_id, "user", "Hello")
    result = chat_db.export_conversation(conv_id, "csv")
    assert isinstance(result, str)
    assert "CSV Export" in result


def test_export_excel(chat_db):
    conv_id = chat_db.create_conversation(title="Excel Export")
    chat_db.add_message(conv_id, "user", "Hello")
    result = chat_db.export_conversation(conv_id, "excel")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_export_nonexistent_conversation(chat_db):
    with pytest.raises(ValueError, match="not found"):
        chat_db.export_conversation(9999, "markdown")


def test_export_unsupported_format(chat_db):
    conv_id = chat_db.create_conversation(title="Bad Format")
    with pytest.raises(ValueError, match="Unsupported format"):
        chat_db.export_conversation(conv_id, "pdf")


def test_to_dataframe(chat_db):
    conv_id = chat_db.create_conversation(title="DF Test")
    chat_db.add_message(conv_id, "user", "Message content")
    df = chat_db.to_dataframe()
    assert len(df) > 0
    assert "role" in df.columns
    assert "content" in df.columns


def test_to_dataframe_specific_conversation(chat_db):
    conv_id = chat_db.create_conversation(title="Specific")
    chat_db.add_message(conv_id, "user", "Hello")
    chat_db.create_conversation(title="Other")
    df = chat_db.to_dataframe(conversation_id=conv_id)
    assert len(df) == 1


def test_get_conversation_stats_empty(chat_db):
    stats = chat_db.get_conversation_stats()
    assert len(stats) == 0


def test_get_conversation_stats_with_data(chat_db):
    conv_id = chat_db.create_conversation(title="Stats Test")
    chat_db.add_message(conv_id, "user", "Hello world")
    chat_db.add_message(conv_id, "assistant", "Hi there")
    stats = chat_db.get_conversation_stats()
    assert len(stats) == 1
    assert stats.iloc[0]["message_count"] == 2


def test_export_all_conversations_excel(chat_db):
    conv_id = chat_db.create_conversation(title="All Export")
    chat_db.add_message(conv_id, "user", "Test")
    result = chat_db.export_all_conversations("excel")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_export_all_conversations_csv(chat_db):
    conv_id = chat_db.create_conversation(title="CSV All")
    chat_db.add_message(conv_id, "user", "Test")
    result = chat_db.export_all_conversations("csv")
    assert isinstance(result, str)


def test_export_all_conversations_json(chat_db):
    conv_id = chat_db.create_conversation(title="JSON All")
    chat_db.add_message(conv_id, "user", "Test")
    result = chat_db.export_all_conversations("json")
    assert isinstance(result, str)
    assert "JSON All" in result
    import json
    parsed = json.loads(result)
    assert "conversations" in parsed


def test_export_all_to_file(tmp_path, chat_db):
    conv_id = chat_db.create_conversation(title="File Export")
    chat_db.add_message(conv_id, "user", "Test")
    out_path = tmp_path / "export.json"
    chat_db.export_all(str(out_path), "json")
    assert out_path.exists()
    content = out_path.read_text()
    assert "File Export" in content
