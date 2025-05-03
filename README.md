# LLM Interface

A flexible Python interface for locally hosted LLMs, focusing on Ollama integration with RAG capabilities.

- Connect to locally hosted LLMs (Ollama)
- Support for both single-shot queries and persistent chat sessions
- Built-in RAG (Retrieval Augmented Generation) for web research
- Tool-based research system with ReAct pattern
- Lightweight with minimal dependencies and cross-platform compatibility
- Extensible architecture for future capabilities

## Getting Started

### Installation

# Install from GitHub
pip install git+https://github.com/Munger/llm_interface.git

# Or install in development mode
git clone https://github.com/Munger/llm_interface.git
cd llm_interface
pip install -e .

### Basic Usage (Python API)

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

### Command-Line Usage

Single query:

llm-cli ask "What is the capital of France?"

Start interactive chat session:

llm-cli chat

Research with web search:

llm-cli research "Recent advances in quantum computing"

## CLI Reference

### Global Options

-m, --model MODEL     Specify Ollama model to use (default: from config)
-h, --host HOST       Specify Ollama host (default: localhost)
-p, --port PORT       Specify Ollama port (default: 11434)
-c, --config PATH     Path to custom config file
-d, --debug           Enable debug mode for verbose output

### Commands

ask PROMPT            Send a single query to the LLM

chat                  Start an interactive chat session
  -s, --session ID    Session ID (creates new if not provided)

research QUERY        Perform intelligent research with ReAct
  -s, --session ID    Session ID (creates temporary if not provided)

list-sessions         List all available sessions

delete-session ID     Delete a chat session
  -f, --force         Force deletion without confirmation

show-config           Show current configuration
  -s, --save          Save config to user config file

list-tools            List available research tools

## Research Capabilities

You can initiate research during a chat session using the `/research` command:

$ llm-cli chat
Created new session 123e4567-e89b-12d3-a456-426614174000
Enter your messages (Ctrl+D or type 'exit' to quit):
Use /research <query> to perform research on a topic

You> Tell me about quantum computing
LLM> Quantum computing is a type of computing that uses quantum-mechanical phenomena...

You> /research latest breakthroughs in quantum computing
LLM> Researching: latest breakthroughs in quantum computing...
[Research results will appear here]

The `/research` command triggers the ReAct pattern which:
- Breaks down the research query
- Uses appropriate tools to gather information
- Synthesises the findings into a coherent response

You can also use the research command programmatically:

# Use the research command in chat
response = session.chat("/research latest developments in quantum computing")

## Configuration and APIs

### API Key Management

LLM Interface uses API keys for various services when performing research. Keys are stored securely in `~/.llm_interface/api_keys.json`.

Supported APIs:
- youtube, google, google_custom_search
- vimeo, dailymotion
- github, twitter, bing

Keys can be set programmatically:

from llm_interface.config.api_keys import api_key_manager

# Set a key
api_key_manager.set_key("youtube", "your-api-key-here")

# Check if a key exists
if api_key_manager.has_key("youtube"):
    # Use the key
    youtube_key = api_key_manager.get_key("youtube")

### Configuration Options

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

## Architecture

LLM Interface is built with a modular architecture:

- **Core Client**: Interfaces with local LLMs (currently focused on Ollama)
- **Session Management**: Handles persistent chat sessions with history
- **Research Capabilities**: Implements RAG for enhanced responses
- **Tools**: Pluggable research tools for web search, video content, and list processing
- **Configuration**: Flexible configuration with sensible defaults

The ReAct (Reasoning + Acting) system combines these tools with LLM reasoning for powerful research capabilities.

---

**Licence**: MIT

**Contributing**: Contributions are welcome! Please feel free to submit a Pull Request.