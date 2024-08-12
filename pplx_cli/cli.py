import typer
from .api import query_perplexity

app = typer.Typer()

@app.command()
def ask(
    query: str = typer.Argument(..., help="The question to ask Perplexity AI"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """
    Ask a question to Perplexity AI and get a response.
    """
    if verbose:
        typer.echo(f"Sending query to Perplexity AI: {query}")

    try:
        typer.echo("Querying Perplexity AI...")
        answer = query_perplexity(query)
        typer.echo(f"Answer: {answer}")
    except Exception as e:
        typer.echo(f"An error occurred: {str(e)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
