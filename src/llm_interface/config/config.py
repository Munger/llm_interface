"""
Configuration settings for the LLM Interface.

This module contains default settings and configuration options that can be
overridden by the user.
"""

import os
from typing import Dict, Any, Optional, Union


# Default configuration
DEFAULT_CONFIG = {
    # Ollama settings
    "ollama_host": "localhost",
    "ollama_port": 11434,
    "default_model": "deepseek-coder:33b",  # Default model to use
    "timeout": 60,  # Request timeout in seconds
    
    # Session settings
    "session_dir": os.path.expanduser("~/.llm_interface/sessions"),
    "max_history": 20,  # Maximum number of messages to keep in history
    
    # Research settings
    "embeddings_model": "all-MiniLM-L6-v2",  # Sentence transformer model
    "embeddings_dir": os.path.expanduser("~/.llm_interface/embeddings"),
    "chunk_size": 1000,  # Size of text chunks for embeddings
    "chunk_overlap": 200,  # Overlap between chunks
    "max_search_results": 5,  # Maximum number of search results to return
}

# User config will be loaded from ~/.llm_interface/config.json if it exists
USER_CONFIG_PATH = os.path.expanduser("~/.llm_interface/config.json")

class Config:
    """Configuration manager for the LLM Interface."""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with default values and optional overrides.
        
        Args:
            config_override: Optional dictionary with configuration overrides
        """
        self._config = DEFAULT_CONFIG.copy()
        
        # Load user config if it exists
        if os.path.exists(USER_CONFIG_PATH):
            import json
            try:
                with open(USER_CONFIG_PATH, 'r') as f:
                    user_config = json.load(f)
                self._config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load user config: {e}")
        
        # Apply runtime overrides
        if config_override and isinstance(config_override, dict):
            self._config.update(config_override)
        
        # Create necessary directories
        os.makedirs(self._config["session_dir"], exist_ok=True)
        os.makedirs(self._config["embeddings_dir"], exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
    
    def update(self, config: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            config: Dictionary with configuration values
        """
        if isinstance(config, dict):
            self._config.update(config)
    
    def save(self) -> None:
        """Save current configuration to user config file."""
        import json
        os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
        with open(USER_CONFIG_PATH, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def __getitem__(self, key: str) -> Any:
        """Allow access to configuration using dictionary syntax."""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow setting configuration using dictionary syntax."""
        self._config[key] = value
