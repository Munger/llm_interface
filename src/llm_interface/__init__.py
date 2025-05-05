"""
LLM Interface

A flexible Python interface for locally hosted LLMs, focusing on Ollama integration 
with RAG capabilities and ReAct pattern for intelligent research.

Author: Tim Hosking (https://github.com/Munger)
"""

# Import necessary modules to make them available at package level
from llm_interface.llm.ollama import OllamaClient
from llm_interface.config.config import Config

# Make sure tool modules are imported so tools get registered
try:
    import llm_interface.tools.web_tools
except ImportError:
    pass  # Web tools are optional

# Main client class alias for easy imports
LLMClient = OllamaClient

__version__ = "0.1.0"
__all__ = ["LLMClient", "Config"]