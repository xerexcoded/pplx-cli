# Perplexity CLI

A command-line interface for interacting with Perplexity AI's API, featuring chat history management, note-taking, and AI search capabilities.

## Features

- 🤖 Direct interaction with Perplexity AI models
- 📝 Local note-taking with AI-powered search
- 💬 Comprehensive chat history management
- 📊 Conversation analytics and statistics
- 📤 Multiple export formats (Markdown, JSON, Excel)

## Installation

```bash
pip install pplx-cli
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

# With a topic
perplexity ask "What are the main differences between Python lists and tuples?" --topic programming
```

List available models:
```bash
perplexity list-models
```

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
perplexity notes
```

View a note:
```bash
perplexity show-note <note-id>
```

Search notes:
```bash
perplexity search-notes "search query"
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
git clone https://github.com/yourusername/pplx-cli.git
cd pplx-cli
```

Install dependencies:
```bash
pip install -e .
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
