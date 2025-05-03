"""
API key management for external services.

This module provides secure handling of API keys for various services
while ensuring they are not committed to version control.
"""

import os
import json
from typing import Dict, Any, Optional, Union

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
                    loaded_keys = json.load(f)
                    # Convert from old format if needed
                    self.api_keys = self._convert_to_new_format(loaded_keys)
            except Exception as e:
                print(f"Warning: Failed to load API keys from {self.api_keys_path}: {e}")
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.api_keys_path), exist_ok=True)
            
            # Initialize with known keys (disabled by default)
            self._initialize_default_keys()
            
            # Save the initial keys
            self._save_api_keys()
    
    def _convert_to_new_format(self, loaded_keys: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Convert keys from old format to new format with enabled flag.
        
        Args:
            loaded_keys: The loaded keys data
            
        Returns:
            Converted keys dictionary
        """
        converted_keys = {}
        
        for service, value in loaded_keys.items():
            if isinstance(value, dict) and "key" in value:
                # Already in new format
                converted_keys[service] = value
            else:
                # Convert to new format
                converted_keys[service] = {
                    "key": value,
                    "enabled": bool(value and value != f"your-{service}-api-key")
                }
                
        return converted_keys
    
    def _initialize_default_keys(self) -> None:
        """Initialize default keys (all disabled by default)."""
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
        
        for service in known_services:
            self.api_keys[service] = {
                "key": f"your-{service}-api-key",
                "enabled": False
            }
    
    def get_key(self, service: str) -> Optional[str]:
        """
        Get an API key for a specific service.
        
        Args:
            service: The service name (e.g., 'youtube', 'google')
            
        Returns:
            The API key if available and enabled, None otherwise
        """
        if service in self.api_keys:
            key_data = self.api_keys[service]
            if isinstance(key_data, dict) and key_data.get("enabled", False):
                return key_data.get("key")
        
        return None
    
    def has_key(self, service: str) -> bool:
        """
        Check if an API key is available and enabled for a specific service.
        
        Args:
            service: The service name
            
        Returns:
            True if the API key is available and enabled, False otherwise
        """
        if service in self.api_keys:
            key_data = self.api_keys[service]
            if isinstance(key_data, dict):
                return key_data.get("enabled", False) and bool(key_data.get("key"))
        
        return False
    
    def set_key(self, service: str, key: str, enabled: bool = True) -> None:
        """
        Set an API key for a specific service.
        
        Args:
            service: The service name
            key: The API key
            enabled: Whether the key should be enabled
        """
        self.api_keys[service] = {
            "key": key,
            "enabled": enabled
        }
        self._save_api_keys()
    
    def enable_key(self, service: str, enabled: bool = True) -> bool:
        """
        Enable or disable an API key.
        
        Args:
            service: The service name
            enabled: Whether to enable or disable the key
            
        Returns:
            True if the operation was successful, False otherwise
        """
        if service in self.api_keys:
            key_data = self.api_keys[service]
            if isinstance(key_data, dict):
                key_data["enabled"] = enabled
                self._save_api_keys()
                return True
        
        return False
    
    def is_default_key(self, service: str) -> bool:
        """
        Check if a key has the default placeholder value.
        
        Args:
            service: The service name
            
        Returns:
            True if the key has the default value, False otherwise
        """
        if service in self.api_keys:
            key_data = self.api_keys[service]
            if isinstance(key_data, dict):
                return key_data.get("key") == f"your-{service}-api-key"
        
        return False
    
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
                availability[service] = self.has_key(service)
        
        return availability


# Global API key manager
api_key_manager = ApiKeyManager()