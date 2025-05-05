"""
Tools module for LLM Interface.

This module provides tools for research and information gathering
that can be used by the LLM in a ReAct pattern.

Author: Tim Hosking (https://github.com/Munger)
"""

from llm_interface.tools.base import Tool, ToolRegistry, register_tool, registry

# Import tool modules to ensure they get registered
import llm_interface.tools.web_tools
import llm_interface.tools.video_tools
import llm_interface.tools.list_tools

__all__ = [
    'Tool',
    'ToolRegistry',
    'register_tool',
    'registry'
]