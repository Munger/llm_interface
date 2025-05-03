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

# Install from GitHub
pip install git+https://github.com/Munger/llm_interface.git

# Or install in development mode
git clone https://github.com/Munger/llm_interface.git
cd llm_interface
pip install -e .

## Quick Start

### Python API

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

### CLI Usage

# Single query
llm-cli ask "What is the capital of France?"

# Start interactive chat session
llm-cli chat

# Research with web search
llm-cli research "Recent advances in quantum computing"

## Chat Research Commands

During a chat session, you can initiate research using the `/research` command:

### Python API

from llm_interface import LLMClient

# Create a client
client = LLMClient()

# Create a session
session = client.create_session()

# Start a normal chat
response = session.chat("Tell me about quantum computing")
print(response)

# Use the research command in chat
response = session.chat("/research latest developments in quantum computing")
print(response)

# Continue the conversation
response = session.chat("How does that compare to previous approaches?")
print(response)

### CLI Usage

In the interactive chat session, you can use the `/research` command:

$ llm-cli chat
Created new session 123e4567-e89b-12d3-a456-426614174000
Enter your messages (Ctrl+D or type 'exit' to quit):
Use /research <query> to perform research on a topic

You> Tell me about quantum computing
LLM> Quantum computing is a type of computing that uses quantum-mechanical phenomena...

You> /research latest breakthroughs in quantum computing
LLM> Researching: latest breakthroughs in quantum computing...
[Research results will appear here]

You> That's fascinating! How will this impact cryptography?
LLM> The developments in quantum computing will have significant implications for cryptography...

The `/research` command triggers the ReAct pattern which performs comprehensive research by:
1. Breaking down the research query
2. Using appropriate tools to gather information
3. Synthesising the findings into a coherent response

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

from llm_interface.config.api_keys import api_key_manager

# Set a key
api_key_manager.set_key("youtube", "your-api-key-here")

# Check if a key exists
if api_key_manager.has_key("youtube"):
    # Use the key
    youtube_key = api_key_manager.get_key("youtube")

## Architecture

LLM Interface is built with a modular architecture:

- **Core Client**: Interfaces with local LLMs (currently focused on Ollama)
- **Session Management**: Handles persistent chat sessions with history
- **Research Capabilities**: Implements RAG for enhanced responses
- **Tools**: Pluggable research tools for web search, video content, and more
- **Configuration**: Flexible configuration with sensible defaults

## Configuration

Default configuration is stored in `~/.llm_interface/config.json`. You can override settings when initialising:

from llm_interface import LLMClient
from llm_interface.config import Config

# Create custom config
config_override = {
    "ollama_host": "localhost",
    "ollama_port": 11434,
    "default_model": "llama2:13b"
}

# Initialise with custom config
client = LLMClient(config_override=config_override)

## Tools

LLM Interface includes various tools for research:

- **Web Tools**: Search the web, fetch webpage content
- **Video Tools**: Search for videos, extract metadata
- **List Tools**: Process and aggregate lists of information

The ReAct (Reasoning + Acting) system combines these tools with LLM reasoning for powerful research capabilities.

## Command-Line Interface

# Show help
llm-cli --help

# List available sessions
llm-cli list-sessions

# Delete a session
llm-cli delete-session SESSION_ID

# Show current configuration
llm-cli show-config

## Licence

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.