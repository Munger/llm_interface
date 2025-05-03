"""
API key management for external services.

This module provides secure handling of API keys for various services
while ensuring they are not committed to version control.
"""

import os
import json
from typing import Dict, Any, Optional

# Default location for API keys file (in user's home directory)
DEFAULT_API_KEYS_PATH = os.path.expanduser("~/.llm_interface/api_keys.json")


class ApiKeyManager:
    """
    Manager for API keys used by the LLM Interface.
    
    This class handles loading and accessing API keys for
    various services while keeping them secure.
    """
    
    def __init__(self, api_keys_path: Optional[str] = None):
        """
        Initialize the API key manager.
        
        Args:
            api_keys_path: Path to the API keys file (optional)
        """
        self.api_keys_path = api_keys_path or DEFAULT_API_KEYS_PATH
        self.api_keys = {}
        self._load_api_keys()
    
    def _load_api_keys(self) -> None:
        """Load API keys from the configuration file."""
        if os.path.exists(self.api_keys_path):
            try:
                with open(self.api_keys_path, 'r') as f:
                    self.api_keys = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load API keys from {self.api_keys_path}: {e}")
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.api_keys_path), exist_ok=True)
    
    def get_key(self, service: str) -> Optional[str]:
        """
        Get an API key for a specific service.
        
        Args:
            service: The service name (e.g., 'youtube', 'google')
            
        Returns:
            The API key if available, None otherwise
        """
        return self.api_keys.get(service)
    
    def has_key(self, service: str) -> bool:
        """
        Check if an API key is available for a specific service.
        
        Args:
            service: The service name
            
        Returns:
            True if the API key is available, False otherwise
        """
        return service in self.api_keys and self.api_keys[service]
    
    def set_key(self, service: str, key: str) -> None:
        """
        Set an API key for a specific service.
        
        Args:
            service: The service name
            key: The API key
        """
        self.api_keys[service] = key
        self._save_api_keys()
    
    def _save_api_keys(self) -> None:
        """Save API keys to the configuration file."""
        try:
            with open(self.api_keys_path, 'w') as f:
                json.dump(self.api_keys, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save API keys to {self.api_keys_path}: {e}")
    
    def get_available_services(self) -> Dict[str, bool]:
        """
        Get a dictionary of available services and their status.
        
        Returns:
            Dictionary mapping service names to availability status
        """
        # List of known services
        known_services = [
            "youtube",
            "google",
            "google_custom_search",
            "vimeo",
            "dailymotion",
            "github",
            "twitter",
            "bing"
        ]
        
        # Check availability for each service
        availability = {}
        for service in known_services:
            availability[service] = self.has_key(service)
        
        # Add any additional keys not in the known list
        for service in self.api_keys:
            if service not in availability:
                availability[service] = True
        
        return availability


# Global API key manager
api_key_manager = ApiKeyManager()