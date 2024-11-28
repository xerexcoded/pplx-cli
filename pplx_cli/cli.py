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

    # Save the terminal settings
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        # Set the terminal to raw mode
        tty.setraw(sys.stdin.fileno())
        
        while True:
            char = sys.stdin.read(1)
            
            # Handle backspace
            if char in ('\x7f', '\x08'):  # backspace in unix/windows
                if password:
                    password.pop()
                    # Erase the last asterisk
                    sys.stdout.write('\b \b')
                    sys.stdout.flush()
                continue
                
            # Handle enter/return
            if char in ('\r', '\n'):
                sys.stdout.write('\n')
                sys.stdout.flush()
                break
                
            # Handle ctrl+c
            if char == '\x03':
                raise KeyboardInterrupt
                
            # Handle regular characters
            if char.isprintable():
                password.append(char)
                sys.stdout.write('*')
                sys.stdout.flush()
                
    finally:
        # Restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
    return ''.join(password)

@app.command()
def setup():
    """
    Configure your Perplexity API key.
    """
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
    if not Config.API_KEY:
        typer.echo("No API key found. Please set up your API key first.")
        setup()
        # Reload the configuration after setup
        Config.API_KEY = load_api_key()
        if not Config.API_KEY:
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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    max_retries: int = typer.Option(3, "--max-retries", "-r", help="Maximum number of retries on failure"),
):
    """
    Ask a question to Perplexity AI and get a response.
    """
    ensure_api_key()
    
    if verbose:
        typer.echo(f"Using model: {model if model else 'default'}")
        typer.echo(f"Sending query to Perplexity AI: {query}")

    selected_model = get_model_from_name(model) if model else None
    
    for attempt in range(max_retries):
        try:
            typer.echo("Querying Perplexity AI...")
            answer = query_perplexity(query, selected_model)
            typer.echo(f"Answer: {answer}")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                typer.echo(f"An error occurred (attempt {attempt + 1}/{max_retries}): {str(e)}", err=True)
                typer.echo("Retrying...")
            else:
                typer.echo(f"Failed after {max_retries} attempts. Last error: {str(e)}", err=True)
                raise typer.Exit(code=1)

@app.command()
def list_models():
    """
    List all available Perplexity AI models and their specifications.
    """
    typer.echo("Available Models:")
    for model in PerplexityModel:
        info = Config.get_model_info(model)
        typer.echo(f"\n{model.value}:")
        typer.echo(f"  Parameters: {info['parameters']}")
        typer.echo(f"  Context Length: {info['context_length']}")