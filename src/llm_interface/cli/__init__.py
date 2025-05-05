"""
Command-line interface for LLM Interface.

This module provides a CLI for interacting with locally hosted LLMs,
allowing for queries, chat sessions, and research.

Author: Tim Hosking (https://github.com/Munger)
"""

from llm_interface.cli.main import cli, main

__all__ = [
    'cli',
    'main'
]