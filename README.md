# Perplexity CLI

A command-line interface for interacting with Perplexity AI's API, featuring chat history management, note-taking, and AI search capabilities.

## Features

- 🤖 Direct interaction with Perplexity AI models
- 📝 Local note-taking with AI-powered search
- 💬 Comprehensive chat history management
- 📊 Conversation analytics and statistics
- 📤 Multiple export formats (Markdown, JSON, Excel)

## Installation

### From PyPI
```bash
pip install pplx-cli
```

### From Source (Development)
```bash
git clone https://github.com/xerexcoded/pplx-cli.git
cd pplx-cli

# Using Poetry (recommended)
poetry install
poetry run perplexity --help

# Using pip
pip install -e .
```

## Configuration

Set up your Perplexity API key:

```bash
perplexity setup
```

Or set the environment variable:

```bash
export PERPLEXITY_API_KEY='your-api-key'
```

## Usage

### Basic Commands

Ask a question:
```bash
perplexity ask "What is the capital of France?"

# With a specific model
perplexity ask "Complex reasoning question" --model sonar-reasoning

# With a topic
perplexity ask "What are the main differences between Python lists and tuples?" --topic programming
```

List available models:
```bash
perplexity list-models
```

**Available Models:**
- `sonar` - Lightweight, cost-effective search model with grounding (default)
- `sonar-reasoning` - Fast, real-time reasoning model for quick problem-solving with search
- `sonar-deep-research` - Expert-level research model conducting exhaustive searches and comprehensive reports

### Chat History Management

View chat history:
```bash
perplexity history
```

Show detailed chat statistics:
```bash
perplexity chat-stats
```

View a specific conversation:
```bash
perplexity show-chat <conversation-id>
```

Export a conversation:
```bash
# Export to markdown
perplexity export-chat <conversation-id> --format markdown -o conversation.md

# Export to JSON
perplexity export-chat <conversation-id> --format json -o conversation.json
```

Export all conversations:
```bash
# Export to Excel
perplexity export-all --format excel -o chat_history.xlsx

# Export to JSON
perplexity export-all --format json -o chat_history.json
```

### Note Management

Create a note:
```bash
perplexity note "My note content" --title "My Note" --tags "tag1,tag2"
```

List notes:
```bash
perplexity list-notes
```

View a note:
```bash
perplexity view-note <note-id>
```

Ask questions about your notes using AI:
```bash
perplexity ask-notes "What did I write about machine learning?"
```

## Features in Detail

### Chat History Features

- **Conversation Tracking**: Automatically saves all conversations with timestamps
- **Topic Organization**: Add topics to conversations for better organization
- **Rich Statistics**: View detailed statistics about your chat history
- **Flexible Export Options**: Export conversations in multiple formats
  - Markdown: Great for documentation
  - JSON: Perfect for data analysis
  - Excel: Ideal for spreadsheet analysis
  - CSV: Simple tabular format

### Note-Taking Features

- **Local Storage**: All notes are stored locally
- **Tag Support**: Organize notes with tags
- **AI-Powered Search**: Find notes using natural language queries
- **Markdown Support**: Write notes in markdown format

## Development

Clone the repository:
```bash
git clone https://github.com/xerexcoded/pplx-cli.git
cd pplx-cli
```

Install dependencies:
```bash
# Using Poetry (recommended)
poetry install

# Run tests
poetry run pytest

# Run CLI in development
poetry run perplexity --help

# Or using pip
pip install -e .
```

### Running with Poetry

Poetry is the recommended way to manage dependencies and run the CLI:

```bash
# Install dependencies
poetry install

# Run commands through Poetry
poetry run perplexity ask "test question"
poetry run perplexity list-models

# Or activate Poetry shell
poetry shell
perplexity --help  # Now you can run directly
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
