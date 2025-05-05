"""
Research module for LLM Interface.

This module provides functionality for research, retrieval,
and enhancing LLM responses with external information.

Author: Tim Hosking (https://github.com/Munger)
"""

from llm_interface.research.document import Document, DocumentProcessor
from llm_interface.research.retrieval import (
    SimpleVectorStore, Embedder, RetrieverRag
)
from llm_interface.research.web import WebSearch, WebResearcher
from llm_interface.research.react import ReActResearcher

__all__ = [
    'Document',
    'DocumentProcessor',
    'SimpleVectorStore',
    'Embedder',
    'RetrieverRag',
    'WebSearch',
    'WebResearcher',
    'ReActResearcher'
]