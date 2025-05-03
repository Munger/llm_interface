# LLM Interface

A flexible Python interface for locally hosted LLMs, focusing on Ollama integration with RAG capabilities.

## Features

- Connect to locally hosted LLMs (Ollama)
- Support both single-shot queries and persistent chat sessions
- Built-in RAG (Retrieval Augmented Generation) for web research
- Lightweight with minimal dependencies
- Cross-platform compatibility
- Extensible architecture for future capabilities (SSH control, etc.)
- Tool-based research system with ReAct pattern

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/Munger/llm_interface.git

# Or install in development mode
git clone https://github.com/Munger/llm_interface.git
cd llm_interface
pip install -e .
```

## Quick Start

### Python API

```python
from llm_interface import LLMClient

# Create a client with default Ollama model
client = LLMClient()

# Simple query
response = client.query("Explain quantum computing in simple terms")
print(response)

# Start a chat session
session = client.create_session()
response = session.chat("Tell me about Python's asyncio library")
print(response)

# Follow-up in the same session
response = session.chat("How does it compare to threading?")
print(response)

# Use research capabilities
response = session.research("What are the latest developments in fusion energy?")
print(response)
```

### CLI Usage

```bash
# Single query
llm-cli ask "What is the capital of France?"

# Start interactive chat session
llm-cli chat

# Research with web search
llm-cli research "Recent advances in quantum computing"
```

## API Key Management

LLM Interface uses API keys for various services when performing research. Keys are stored securely in `~/.llm_interface/api_keys.json`.

### Available Services

The following API keys are supported:
- youtube
- google
- google_custom_search
- vimeo
- dailymotion
- github
- twitter
- bing

### Managing Keys

Keys can be set programmatically:

```python
from llm_interface.config.api_keys import api_key_manager

# Set a key
api_key_manager.set_key("youtube", "your-api-key-here")

# Check if a key exists
if api_key_manager.has_key("youtube"):
    # Use the key
    youtube_key = api_key_manager.get_key("youtube")
```

## Architecture

LLM Interface is built with a modular architecture:

- **Core Client**: Interfaces with local LLMs (currently focused on Ollama)
- **Session Management**: Handles persistent chat sessions with history
- **Research Capabilities**: Implements RAG for enhanced responses
- **Tools**: Pluggable research tools for web search, video content, and more
- **Configuration**: Flexible configuration with sensible defaults

## Configuration

Default configuration is stored in `~/.llm_interface/config.json`. You can override settings when initializing:

```python
from llm_interface import LLMClient
from llm_interface.config import Config

# Create custom config
config_override = {
    "ollama_host": "localhost",
    "ollama_port": 11434,
    "default_model": "llama2:13b"
}

# Initialize with custom config
client = LLMClient(config_override=config_override)
```

## Tools

LLM Interface includes various tools for research:

- **Web Tools**: Search the web, fetch webpage content
- **Video Tools**: Search for videos, extract metadata
- **List Tools**: Process and aggregate lists of information

The ReAct (Reasoning + Acting) system combines these tools with LLM reasoning for powerful research capabilities.

## Command-Line Interface

```bash
# Show help
llm-cli --help

# List available sessions
llm-cli list-sessions

# Delete a session
llm-cli delete-session SESSION_ID

# Show current configuration
llm-cli show-config
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
