import typer
from typing import Optional
import getpass
import sys
import tty
import termios
import json
from .api import query_perplexity
from .config import PerplexityModel, Config, save_api_key, load_api_key
from pathlib import Path
from typing import Optional, List
from .notes import NotesDB
from .chat_history import ChatHistoryDB

app = typer.Typer()

def get_model_from_name(name: str) -> Optional[PerplexityModel]:
    model_mapping = {
        "small": PerplexityModel.SONAR,
        "large": PerplexityModel.SONAR_REASONING, 
        "huge": PerplexityModel.SONAR_DEEP_RESEARCH
    }
    
    model = model_mapping.get(name.lower())
    if model is None:
        raise typer.BadParameter(f"Model must be one of: small, large, huge")
    return model

def get_masked_input(prompt: str = "Enter password: ") -> str:
    """Get password input with asterisk masking."""
    password = []
    sys.stdout.write(prompt)
    sys.stdout.flush()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        
        while True:
            char = sys.stdin.read(1)
            
            if char == '\r' or char == '\n':
                sys.stdout.write('\n')
                break
            
            if char == '\x03':  # Ctrl+C
                raise KeyboardInterrupt
            
            if char == '\x7f':  # Backspace
                if password:
                    password.pop()
                    sys.stdout.write('\b \b')  # Erase character
                    sys.stdout.flush()
            else:
                password.append(char)
                sys.stdout.write('*')
                sys.stdout.flush()
                
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
    return ''.join(password)

@app.command()
def setup():
    """Configure your Perplexity API key."""
    try:
        api_key = get_masked_input("Please enter your Perplexity API key: ")
        
        if not api_key:
            typer.echo("API key cannot be empty", err=True)
            raise typer.Exit(code=1)
        
        save_api_key(api_key)
        typer.echo("\n✨ API key saved successfully! ✨", color=typer.colors.GREEN)
        typer.echo("You can now use the Perplexity CLI to ask questions.\n")
    except KeyboardInterrupt:
        typer.echo("\nSetup cancelled.", err=True)
        raise typer.Exit(code=1)

def ensure_api_key():
    """Check if API key is configured and prompt for it if not."""
    config = Config.get_instance()
    if not config.api_key:
        typer.echo("No API key found. Please set up your API key first.")
        setup()
        # Reload the configuration after setup
        config.api_key = load_api_key()
        if not config.api_key:
            typer.echo("Failed to save API key", err=True)
            raise typer.Exit(code=1)

@app.command()
def ask(
    query: str = typer.Argument(..., help="The question to ask Perplexity AI"),
    topic: str = typer.Option(None, "--topic", "-t", help="Topic for the conversation"),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use for the query (small, large, huge)",
        case_sensitive=False
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    save_history: bool = typer.Option(True, "--save-history/--no-save-history", help="Save the conversation to history")
):
    """Ask a question to Perplexity AI and get a response."""
    try:
        ensure_api_key()
        
        selected_model = get_model_from_name(model) if model else None
        
        if verbose:
            typer.echo(f"Using model: {selected_model.value if selected_model else Config.get_instance().model.value}")
        
        typer.echo("Querying Perplexity AI...")
        response = query_perplexity(query, selected_model)
        
        # Save to chat history if enabled
        if save_history:
            db = ChatHistoryDB()
            conversation_id = db.create_conversation(title=query[:50] + "..." if len(query) > 50 else query, topic=topic)
            db.add_message(conversation_id, "user", query)
            db.add_message(conversation_id, "assistant", response)
        
        typer.echo(f"Answer: {response}")
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(code=1)

@app.command(name="list-models")
def list_models():
    """List all available Perplexity AI models."""
    for model in PerplexityModel:
        typer.echo(f"{model.name.lower()}: {model.value}")

@app.command()
def note(
    title: str = typer.Option(..., "--title", "-t", help="Title of the note"),
    content: str = typer.Option(..., "--content", "-c", help="Content of the note"),
    tags: Optional[List[str]] = typer.Option(None, "--tag", help="Tags for the note"),
    directory: Optional[Path] = typer.Option(
        None, "--dir", "-d", 
        help="Directory to store notes (default: ~/.local/share/perplexity/notes)"
    )
):
    """Add a new note."""
    config = Config.get_instance()
    notes_dir = directory or config.notes_dir
    db = NotesDB(notes_dir)
    
    note_id = db.add_note(title, content, tags)
    typer.echo(f"✨ Note saved successfully with ID: {note_id} ✨")

@app.command()
def list_notes(
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter notes by tag"),
    directory: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Directory to read notes from (default: ~/.local/share/perplexity/notes)"
    )
):
    """List all notes."""
    config = Config.get_instance()
    notes_dir = directory or config.notes_dir
    db = NotesDB(notes_dir)
    
    notes = db.list_notes(tag)
    if not notes:
        typer.echo("No notes found.")
        return
    
    for note in notes:
        tags = json.loads(note['tags'])
        tag_str = f" [Tags: {', '.join(tags)}]" if tags else ""
        typer.echo(f"\nID: {note['id']}{tag_str}")
        typer.echo(f"Title: {note['title']}")
        typer.echo(f"Created: {note['created_at']}")
        typer.echo("-" * 40)

@app.command()
def view_note(
    note_id: int = typer.Argument(..., help="ID of the note to view"),
    directory: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Directory to read notes from (default: ~/.local/share/perplexity/notes)"
    )
):
    """View a specific note by its ID."""
    config = Config.get_instance()
    notes_dir = directory or config.notes_dir
    db = NotesDB(notes_dir)
    
    note = db.get_note(note_id)
    if not note:
        typer.echo(f"Note with ID {note_id} not found.")
        raise typer.Exit(code=1)
    
    tags = json.loads(note['tags'])
    tag_str = f" [Tags: {', '.join(tags)}]" if tags else ""
    
    typer.echo(f"\n📝 Note {note_id}{tag_str}")
    typer.echo("=" * 40)
    typer.echo(f"Title: {note['title']}")
    typer.echo(f"Created: {note['created_at']}")
    typer.echo(f"Updated: {note['updated_at']}")
    typer.echo("-" * 40)
    typer.echo(f"Content:\n{note['content']}")
    typer.echo("=" * 40)

@app.command()
def ask_notes(
    query: str = typer.Argument(..., help="The question to ask about your notes"),
    top_k: int = typer.Option(3, "--top", "-k", help="Number of most relevant notes to consider"),
    directory: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Directory to read notes from (default: ~/.local/share/perplexity/notes)"
    )
):
    """Ask questions about your notes using RAG."""
    config = Config.get_instance()
    notes_dir = directory or config.notes_dir
    db = NotesDB(notes_dir)
    
    # Find relevant notes
    similar_notes = db.search_similar_notes(query, top_k=top_k)
    
    if not similar_notes:
        typer.echo("No relevant notes found.")
        return
    
    # Prepare context from similar notes
    context = "\n\n".join([
        f"Note {note['id']} - {note['title']}:\n{note['content']}"
        for note, similarity in similar_notes
    ])
    
    # Prepare the prompt with context
    prompt = f"""Based on the following notes, please answer this question: {query}

Relevant Notes:
{context}

Please provide a comprehensive answer based solely on the information in these notes."""
    
    # Get response from Perplexity AI
    try:
        response = query_perplexity(prompt)
        
        # Print the response with relevant note references
        typer.echo("\n🤖 Answer based on your notes:")
        typer.echo("=" * 40)
        typer.echo(response)
        typer.echo("\n📚 Based on these notes:")
        for note, similarity in similar_notes:
            typer.echo(f"- Note {note['id']}: {note['title']} (relevance: {similarity:.2f})")
    except Exception as e:
        typer.echo(f"Error getting response: {str(e)}")
        raise typer.Exit(code=1)

@app.command(name="history")
def list_history(
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Filter conversations by topic"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search conversations by content")
):
    """List chat history."""
    db = ChatHistoryDB()
    
    if search:
        conversations = db.search_conversations(search)
    else:
        conversations = db.list_conversations(topic)
    
    if not conversations:
        typer.echo("No conversations found.")
        return
    
    for conv in conversations:
        typer.echo(f"\nID: {conv['id']}")
        typer.echo(f"Title: {conv['title']}")
        if conv['topic']:
            typer.echo(f"Topic: {conv['topic']}")
        typer.echo(f"Created: {conv['created_at']}")
        typer.echo("-" * 40)

@app.command(name="show-chat")
def show_conversation(
    conversation_id: int = typer.Argument(..., help="ID of the conversation to view")
):
    """View a specific conversation."""
    db = ChatHistoryDB()
    conversation = db.get_conversation(conversation_id)
    
    if not conversation:
        typer.echo(f"Conversation {conversation_id} not found.")
        return
    
    typer.echo(f"\n📝 Conversation {conversation_id}")
    typer.echo("=" * 40)
    typer.echo(f"Title: {conversation['title']}")
    if conversation['topic']:
        typer.echo(f"Topic: {conversation['topic']}")
    typer.echo(f"Created: {conversation['created_at']}")
    typer.echo("-" * 40)
    
    for msg in conversation['messages']:
        typer.echo(f"\n[{msg['role'].upper()}] ({msg['timestamp']})")
        typer.echo(msg['content'])
    typer.echo("=" * 40)

@app.command(name="export-chat")
def export_conversation(
    conversation_id: int = typer.Argument(..., help="ID of the conversation to export"),
    format: str = typer.Option("markdown", "--format", "-f", help="Export format (markdown, json, txt, csv, excel)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path")
):
    """Export a conversation to a file."""
    db = ChatHistoryDB()
    
    try:
        content = db.export_conversation(conversation_id, format)
        
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            if format == "excel":
                with open(output, "wb") as f:
                    f.write(content)
            else:
                output.write_text(content)
            typer.echo(f"Conversation exported to {output}")
        else:
            if format == "excel":
                typer.echo("Excel format requires an output file path")
                raise typer.Exit(code=1)
            typer.echo(content)
            
    except Exception as e:
        typer.echo(f"Error exporting conversation: {str(e)}", err=True)
        raise typer.Exit(code=1)

@app.command(name="export-all")
def export_all_conversations(
    format: str = typer.Option("excel", "--format", "-f", help="Export format (excel, csv)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file path")
):
    """Export all conversations to a file."""
    db = ChatHistoryDB()
    
    try:
        content = db.export_all_conversations(format)
        
        output.parent.mkdir(parents=True, exist_ok=True)
        if format == "excel":
            with open(output, "wb") as f:
                f.write(content)
        else:
            output.write_text(content)
        
        typer.echo(f"All conversations exported to {output}")
    except Exception as e:
        typer.echo(f"Error exporting conversations: {str(e)}", err=True)
        raise typer.Exit(code=1)

@app.command(name="chat-stats")
def show_chat_stats():
    """Show statistics about your chat history."""
    db = ChatHistoryDB()
    
    try:
        stats = db.get_conversation_stats()
        
        typer.echo("\n📊 Chat History Statistics")
        typer.echo("=" * 40)
        
        total_conversations = len(stats)
        
        if total_conversations == 0:
            typer.echo("No conversations found in history.")
            return
            
        total_messages = int(stats['message_count'].sum())
        avg_messages_per_conv = float(stats['message_count'].mean())
        avg_message_length = float(stats['avg_message_length'].mean())
        
        typer.echo(f"Total Conversations: {total_conversations}")
        typer.echo(f"Total Messages: {total_messages}")
        typer.echo(f"Average Messages per Conversation: {avg_messages_per_conv:.1f}")
        typer.echo(f"Average Message Length: {int(avg_message_length)} characters")
        
        if total_conversations > 0:
            typer.echo("\nTop 5 Most Active Conversations:")
            typer.echo("-" * 40)
            
            top_conversations = stats.nlargest(5, 'message_count')
            for _, row in top_conversations.iterrows():
                typer.echo(
                    f"ID: {row['conversation_id']} | "
                    f"Title: {row['title']} | "
                    f"Messages: {int(row['message_count'])} | "
                    f"Created: {row['created_at']}"
                )
        
    except Exception as e:
        typer.echo(f"Error getting statistics: {str(e)}", err=True)
        raise typer.Exit(code=1)