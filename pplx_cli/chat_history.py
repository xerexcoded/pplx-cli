import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Union, Any
from datetime import datetime
import json
from dataclasses import dataclass
import pandas as pd
import csv
import io

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: str

class ChatHistoryDB:
    DEFAULT_HISTORY_DIR = Path.home() / ".local" / "share" / "perplexity" / "chat_history"
    
    def __init__(self, directory: Path = DEFAULT_HISTORY_DIR):
        self.directory = directory
        self.db_path = directory / "chat_history.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize the database with required tables."""
        self.directory.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Create conversations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    topic TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
    
    def create_conversation(self, title: Optional[str] = None, topic: Optional[str] = None) -> int:
        """Create a new conversation and return its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (title, topic) VALUES (?, ?)",
                (title, topic)
            )
            return cursor.lastrowid
    
    def add_message(self, conversation_id: int, role: str, content: str) -> int:
        """Add a message to a conversation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content)
            )
            return cursor.lastrowid
    
    def get_conversation(self, conversation_id: int) -> Optional[Dict]:
        """Get a conversation by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            conversation = cursor.fetchone()
            if not conversation:
                return None
            
            cursor = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
                (conversation_id,)
            )
            messages = [dict(row) for row in cursor.fetchall()]
            
            return {
                **dict(conversation),
                "messages": messages
            }
    
    def list_conversations(self, topic: Optional[str] = None) -> List[Dict]:
        """List all conversations, optionally filtered by topic."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if topic:
                cursor = conn.execute(
                    "SELECT * FROM conversations WHERE topic LIKE ? ORDER BY created_at DESC",
                    (f"%{topic}%",)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM conversations ORDER BY created_at DESC"
                )
            return [dict(row) for row in cursor.fetchall()]
    
    def search_conversations(self, query: str) -> List[Dict]:
        """Search conversations and messages for a query string."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT DISTINCT c.* 
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.title LIKE ? 
                OR c.topic LIKE ?
                OR m.content LIKE ?
                ORDER BY c.created_at DESC
            """, (f"%{query}%", f"%{query}%", f"%{query}%"))
            return [dict(row) for row in cursor.fetchall()]
    
    def export_conversation(self, conversation_id: int, format: str = "markdown") -> Union[str, bytes]:
        """Export a conversation in the specified format."""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        if format == "markdown":
            return self._export_markdown(conversation)
        elif format == "json":
            return json.dumps(conversation, indent=2)
        elif format == "txt":
            return self._export_text(conversation)
        elif format == "csv":
            return self._export_csv(conversation)
        elif format == "excel":
            return self._export_excel(conversation)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_markdown(self, conversation: Dict) -> str:
        """Export conversation as markdown."""
        lines = [
            f"# {conversation['title'] or f'Conversation {conversation['id']}'}",
            f"Topic: {conversation['topic'] or 'No topic'}",
            f"Created: {conversation['created_at']}",
            "\n## Messages\n"
        ]
        
        for msg in conversation['messages']:
            lines.append(f"### {msg['role'].title()} ({msg['timestamp']})")
            lines.append(msg['content'])
            lines.append("")
        
        return "\n".join(lines)
    
    def _export_text(self, conversation: Dict) -> str:
        """Export conversation as plain text."""
        lines = [
            f"Conversation: {conversation['title'] or f'#{conversation['id']}'}",
            f"Topic: {conversation['topic'] or 'No topic'}",
            f"Created: {conversation['created_at']}",
            "\nMessages:\n"
        ]
        
        for msg in conversation['messages']:
            lines.append(f"{msg['role'].title()} ({msg['timestamp']}):")
            lines.append(msg['content'])
            lines.append("")
        
        return "\n".join(lines)

    def _export_csv(self, conversation: Dict) -> str:
        """Export conversation as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write conversation metadata
        writer.writerow(["Conversation ID", conversation['id']])
        writer.writerow(["Title", conversation['title']])
        writer.writerow(["Topic", conversation['topic']])
        writer.writerow(["Created", conversation['created_at']])
        writer.writerow([])  # Empty row for separation
        
        # Write messages
        writer.writerow(["Role", "Content", "Timestamp"])
        for msg in conversation['messages']:
            writer.writerow([msg['role'], msg['content'], msg['timestamp']])
        
        return output.getvalue()

    def _export_excel(self, conversation: Dict) -> bytes:
        """Export conversation as Excel file."""
        # Create a Pandas Excel writer
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Metadata sheet
            metadata = pd.DataFrame([
                ["Conversation ID", conversation['id']],
                ["Title", conversation['title']],
                ["Topic", conversation['topic']],
                ["Created", conversation['created_at']]
            ], columns=["Field", "Value"])
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Messages sheet
            messages = pd.DataFrame([
                {
                    "Role": msg['role'],
                    "Content": msg['content'],
                    "Timestamp": msg['timestamp']
                }
                for msg in conversation['messages']
            ])
            messages.to_excel(writer, sheet_name='Messages', index=False)
        
        return output.getvalue()

    def to_dataframe(self, conversation_id: Optional[int] = None) -> pd.DataFrame:
        """Convert conversation(s) to a pandas DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            if conversation_id is not None:
                query = """
                    SELECT c.id as conversation_id, c.title, c.topic, c.created_at,
                           m.role, m.content, m.timestamp
                    FROM conversations c
                    JOIN messages m ON c.id = m.conversation_id
                    WHERE c.id = ?
                    ORDER BY m.timestamp
                """
                df = pd.read_sql_query(query, conn, params=(conversation_id,))
            else:
                query = """
                    SELECT c.id as conversation_id, c.title, c.topic, c.created_at,
                           m.role, m.content, m.timestamp
                    FROM conversations c
                    JOIN messages m ON c.id = m.conversation_id
                    ORDER BY c.created_at DESC, m.timestamp
                """
                df = pd.read_sql_query(query, conn)
            
            return df

    def get_conversation_stats(self) -> pd.DataFrame:
        """Get statistics about conversations."""
        with sqlite3.connect(self.db_path) as conn:
            # First check if we have any conversations
            cursor = conn.execute("SELECT COUNT(*) FROM conversations")
            count = cursor.fetchone()[0]
            
            if count == 0:
                # Return empty DataFrame with correct columns
                return pd.DataFrame(columns=[
                    'conversation_id', 'title', 'topic', 'created_at',
                    'message_count', 'user_messages', 'assistant_messages',
                    'avg_message_length'
                ])
            
            # Query for conversation statistics
            query = """
                SELECT 
                    c.id as conversation_id,
                    c.title,
                    c.topic,
                    c.created_at,
                    COUNT(m.id) as message_count,
                    SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_messages,
                    SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages,
                    AVG(LENGTH(m.content)) as avg_message_length
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.created_at DESC
            """
            return pd.read_sql_query(query, conn)

    def get_messages(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Get all messages for a conversation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT role, content, timestamp
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                """,
                (conversation_id,)
            )
            messages = []
            for row in cursor:
                messages.append({
                    'role': row[0],
                    'content': row[1],
                    'timestamp': row[2]
                })
            return messages

    def export_all_conversations(self, format: str = "excel") -> Union[str, bytes]:
        """Export all conversations in the specified format."""
        df = self.to_dataframe()
        
        if format == "excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Conversations sheet
                conversations = df.groupby('conversation_id').first().reset_index()
                conversations[['conversation_id', 'title', 'topic', 'created_at']].to_excel(
                    writer, sheet_name='Conversations', index=False
                )
                
                # Messages sheet
                df[['conversation_id', 'role', 'content', 'timestamp']].to_excel(
                    writer, sheet_name='Messages', index=False
                )
                
                # Statistics sheet
                stats = self.get_conversation_stats()
                stats.to_excel(writer, sheet_name='Statistics', index=False)
            
            return output.getvalue()
        elif format == "csv":
            output = io.StringIO()
            df.to_csv(output, index=False)
            return output.getvalue()
        elif format == "json":
            conversations = []
            for _, row in df.iterrows():
                conv = {
                    'id': row['conversation_id'],
                    'title': row['title'],
                    'topic': row['topic'],
                    'created_at': row['created_at'],
                    'messages': []
                }
                
                messages = self.get_messages(row['conversation_id'])
                for msg in messages:
                    conv['messages'].append({
                        'role': msg['role'],
                        'content': msg['content'],
                        'timestamp': msg['timestamp']
                    })
                    
                conversations.append(conv)
                
            return json.dumps({'conversations': conversations}, indent=2)
        else:
            raise ValueError(f"Unsupported format for bulk export: {format}")

    def export_all(self, output_path: str, format: str = "excel") -> None:
        """Export all conversations to a file."""
        df = self.to_dataframe()
        
        if df.empty:
            return
            
        if format.lower() == "excel":
            df.to_excel(output_path, index=False, engine='openpyxl')
        elif format.lower() == "json":
            # Get all conversations with their messages
            conversations = []
            for _, row in df.iterrows():
                conv = {
                    'id': row['conversation_id'],
                    'title': row['title'],
                    'topic': row['topic'],
                    'created_at': row['created_at'],
                    'messages': []
                }
                
                messages = self.get_messages(row['conversation_id'])
                for msg in messages:
                    conv['messages'].append({
                        'role': msg['role'],
                        'content': msg['content'],
                        'timestamp': msg['timestamp']
                    })
                    
                conversations.append(conv)
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({'conversations': conversations}, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format for bulk export: {format}")
