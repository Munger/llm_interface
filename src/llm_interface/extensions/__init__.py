"""
Extensions for LLM Interface.

This module provides optional extensions and additional capabilities
that can be integrated with the core functionality.

Author: Tim Hosking (https://github.com/Munger)
"""

from llm_interface.extensions.ssh import SSHController

__all__ = [
    'SSHController'
]