# LLM Interface

A flexible Python interface for locally hosted LLMs, focusing on Ollama integration with RAG capabilities.

## Features

- Connect to locally hosted LLMs (Ollama)
- Support both single-shot queries and persistent chat sessions
- Built-in RAG (Retrieval Augmented Generation) for web research
- Lightweight with minimal dependencies
- Cross-platform compatibility
- Extensible architecture for future capabilities (SSH control, etc.)

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/Munger/llm_interface.git

# Or install in development mode
git clone https://github.com/Munger/llm_interface.git
cd llm_interface
pip install -e .

Quick Start

python

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

CLI Usage

bash

# Single query
llm-cli ask "What is the capital of France?"

# Start interactive chat session
llm-cli chat

# Research with web search
llm-cli research "Recent advances in quantum computing"

License
MIT

