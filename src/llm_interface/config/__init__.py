"""
Configuration module for LLM Interface.

This module provides configuration management and prompt templates.

Author: Tim Hosking (https://github.com/Munger)
"""

from llm_interface.config.config import Config
from llm_interface.config.prompt_manager import (
    PromptManager, get_prompt_manager, get_prompt, get_prompt_value, format_prompt
)
from llm_interface.config.api_keys import api_key_manager

__all__ = [
    'Config',
    'PromptManager',
    'get_prompt_manager',
    'get_prompt',
    'get_prompt_value',
    'format_prompt',
    'api_key_manager'
]
