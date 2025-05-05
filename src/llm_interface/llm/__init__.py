"""
LLM client implementations.

This module provides implementations for interacting with different LLMs,
with a focus on Ollama.

Author: Tim Hosking (https://github.com/Munger)
"""

from llm_interface.llm.base import BaseLLMClient, LLMSession
from llm_interface.llm.ollama import OllamaClient
from llm_interface.llm.ollama_session import OllamaSession
from llm_interface.llm.research_capabilities import OllamaResearch

__all__ = [
    'BaseLLMClient',
    'LLMSession',
    'OllamaClient',
    'OllamaSession',
    'OllamaResearch'
]