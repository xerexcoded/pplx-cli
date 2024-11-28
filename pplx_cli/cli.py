import typer
from typing import Optional
import getpass
import sys
import tty
import termios
from .api import query_perplexity
from .config import PerplexityModel, Config, save_api_key, load_api_key

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