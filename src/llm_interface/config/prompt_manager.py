"""
Prompt management for the LLM Interface.

This module provides tools for managing and formatting prompts used across the system.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
import string


class PromptManager:
    """
    Manager for prompt templates used throughout the LLM interface.
    
    Loads prompt templates from JSON files and provides access to them
    with variable interpolation support.
    """
    
    def __init__(self, prompt_file: Optional[str] = None):
        """
        Initialize the prompt manager.
        
        Args:
            prompt_file: Path to the JSON file containing prompt templates
                         (default: ~/.llm_interface/prompts.json)
        """
        if prompt_file is None:
            prompt_file = os.path.expanduser("~/.llm_interface/prompts.json")
        
        self.prompt_file = prompt_file
        self.prompts = {}
        self.load_prompts()
    
    def load_prompts(self) -> None:
        """Load prompts from the JSON file."""
        # First, load the default prompts if they exist
        default_prompts_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", 
            "prompts.json"
        )
        
        if os.path.exists(default_prompts_file):
            try:
                with open(default_prompts_file, 'r') as f:
                    self.prompts = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load default prompts: {e}")
        
        # Then load user prompts, overriding defaults if they exist
        if os.path.exists(self.prompt_file):
            try:
                with open(self.prompt_file, 'r') as f:
                    user_prompts = json.load(f)
                
                # Merge user prompts with default prompts (user prompts take precedence)
                self._merge_prompts(user_prompts)
            except Exception as e:
                print(f"Warning: Failed to load user prompts: {e}")
    
    def _merge_prompts(self, new_prompts: Dict[str, Any]) -> None:
        """
        Merge new prompts into existing prompts.
        
        Args:
            new_prompts: Dictionary of new prompts to merge
        """
        for category, items in new_prompts.items():
            if category not in self.prompts:
                self.prompts[category] = {}
            
            for key, value in items.items():
                self.prompts[category][key] = value
    
    def get_prompt(self, category: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a prompt by category and name.
        
        Args:
            category: The category of the prompt (e.g., "research")
            name: The name of the prompt (e.g., "system_message")
            
        Returns:
            The prompt dictionary, or None if not found
        """
        if category in self.prompts and name in self.prompts[category]:
            return self.prompts[category][name]
        return None
    
    def get_prompt_value(self, category: str, name: str) -> Optional[Any]:
        """
        Get the value of a prompt by category and name.
        
        Args:
            category: The category of the prompt (e.g., "research")
            name: The name of the prompt (e.g., "system_message")
            
        Returns:
            The prompt value, or None if not found
        """
        prompt = self.get_prompt(category, name)
        if prompt:
            return prompt.get("value")
        return None
    
    def format_prompt(self, category: str, name: str, **kwargs) -> Optional[str]:
        """
        Format a prompt with the given variables.
        
        Args:
            category: The category of the prompt (e.g., "research")
            name: The name of the prompt (e.g., "system_message")
            **kwargs: Variables to substitute in the prompt
            
        Returns:
            The formatted prompt, or None if the prompt was not found
        """
        prompt_value = self.get_prompt_value(category, name)
        
        if prompt_value is None:
            return None
            
        # Handle both string templates and list values
        if isinstance(prompt_value, str):
            # Use string.format for formatting
            try:
                return prompt_value.format(**kwargs)
            except KeyError as e:
                print(f"Warning: Missing variable in prompt '{category}.{name}': {e}")
                # Return the original prompt as a fallback
                return prompt_value
        elif isinstance(prompt_value, list):
            # Return the list directly
            return prompt_value
        else:
            # For any other type, just return as is
            return prompt_value
    
    def save_prompts(self, save_path: Optional[str] = None) -> None:
        """
        Save the current prompts to a JSON file.
        
        Args:
            save_path: Path to save the prompts to (default: self.prompt_file)
        """
        save_path = save_path or self.prompt_file
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(self.prompts, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save prompts: {e}")


# Global prompt manager instance
_prompt_manager = None

def get_prompt_manager(prompt_file: Optional[str] = None) -> PromptManager:
    """
    Get the global prompt manager instance.
    
    Args:
        prompt_file: Path to the JSON file containing prompt templates
                     (only used if the manager hasn't been initialized yet)
                     
    Returns:
        The global prompt manager instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager(prompt_file)
    return _prompt_manager


# Function aliases for convenience
def get_prompt(category: str, name: str) -> Optional[Dict[str, Any]]:
    """Get a prompt by category and name."""
    return get_prompt_manager().get_prompt(category, name)

def get_prompt_value(category: str, name: str) -> Optional[Any]:
    """Get the value of a prompt by category and name."""
    return get_prompt_manager().get_prompt_value(category, name)

def format_prompt(category: str, name: str, **kwargs) -> Optional[Any]:
    """Format a prompt with the given variables."""
    return get_prompt_manager().format_prompt(category, name, **kwargs)