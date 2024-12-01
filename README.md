# Perplexity CLI

A command-line interface for interacting with the Perplexity AI API. Ask questions and get answers directly from your terminal!

## Features

- 🚀 Easy to use CLI interface
- 🔑 Secure API key management
- 🔄 Multiple model support
- ⚡ Fast and efficient responses
- 🛠️ Configurable settings

## Installation

### Using pip (Recommended)

```bash
pip install pplx-cli
```

### From Source

1. Clone the repository:
```bash
git clone https://github.com/xerexcoded/pplx-cli.git
cd pplx-cli
```

2. Install dependencies:
```bash
poetry install
```

3. Build the package:
```bash
poetry build
```

4. Install the package:
```bash
pip install dist/*.whl
```

## Setup

Before using the CLI, you need to configure your Perplexity API key:

1. Get your API key from [Perplexity AI](https://docs.perplexity.ai)
2. Run the setup command:
```bash
perplexity setup
```

## Usage

### Ask a Question

```bash
perplexity ask "What is quantum computing?"
```

### Select a Specific Model

```bash
perplexity ask "Explain neural networks" --model large
```

Available models:
- `small` (8B parameters)
- `large` (70B parameters)
- `huge` (405B parameters)

### Additional Options

- Use verbose output:
```bash
perplexity ask "Your question" -v
```

- Set custom retry attempts:
```bash
perplexity ask "Your question" -r 5
```

### List Available Models

```bash
perplexity list-models
```

## Development

1. Clone the repository
2. Install dependencies:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

4. Make your changes and test:
```bash
poetry run perplexity ask "Test question"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Typer](https://typer.tiangolo.com/)
- Powered by [Perplexity AI](https://www.perplexity.ai/)
