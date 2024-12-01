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

app = typer.Typer()

def get_model_from_name(name: str) -> Optional[PerplexityModel]:
    try:
        return PerplexityModel[name.upper()]
    except KeyError:
        raise typer.BadParameter(f"Model must be one of: small, large, huge")

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
            raise typer.Exit(1)
        
        save_api_key(api_key)
        typer.echo("\n✨ API key saved successfully! ✨", color=typer.colors.GREEN)
        typer.echo("You can now use the Perplexity CLI to ask questions.\n")
    except KeyboardInterrupt:
        typer.echo("\nSetup cancelled.", err=True)
        raise typer.Exit(1)

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
            raise typer.Exit(1)

@app.command()
def ask(
    query: str = typer.Argument(..., help="The question to ask Perplexity AI"),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use for the query (small, large, huge)",
        case_sensitive=False
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Ask a question to Perplexity AI and get a response."""
    try:
        ensure_api_key()
        
        selected_model = get_model_from_name(model) if model else None
        
        if verbose:
            typer.echo(f"Using model: {selected_model.value if selected_model else Config.get_instance().model.value}")
        
        typer.echo("Querying Perplexity AI...")
        response = query_perplexity(query, selected_model)
        
        typer.echo(f"Answer: {response}")
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)

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
        raise typer.Exit(1)
    
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