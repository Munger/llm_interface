"""
LLM Interface

A flexible Python interface for locally hosted LLMs, focusing on Ollama integration with RAG capabilities.
"""

from llm_interface.llm.ollama import OllamaClient

# Main client class alias for easy imports
LLMClient = OllamaClient

__version__ = "0.1.0"
__all__ = ["LLMClient"]
