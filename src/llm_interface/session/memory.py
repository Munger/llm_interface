"""
Conversation history and context management.

This module provides tools for managing conversation history and context
within LLM sessions.

Author: Tim Hosking (https://github.com/Munger)
"""

from typing import Dict, List, Any, Optional


class ConversationMemory:
    """
    Manages conversation history and context.
    
    This class provides methods for manipulating conversation history
    and extracting relevant context.
    """
    
    def __init__(self, max_history: Optional[int] = None):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of messages to keep in history
                         (None means no limit)
        """
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender (user, assistant, system)
            content: The message content
        """
        self.history.append({"role": role, "content": content})
        self._trim_history()
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return self.history.copy()
    
    def clear_history(self, keep_system: bool = True) -> None:
        """
        Clear the conversation history.
        
        Args:
            keep_system: Whether to keep system messages
        """
        if keep_system:
            self.history = [msg for msg in self.history if msg["role"] == "system"]
        else:
            self.history = []
    
    def _trim_history(self) -> None:
        """
        Trim history to stay within the maximum history limit.
        
        Keeps system messages and the most recent messages up to max_history.
        """
        if self.max_history is None or self.max_history <= 0:
            return
        
        if len(self.history) <= self.max_history * 2:
            return
        
        # Keep system messages and the most recent messages
        system_messages = [msg for msg in self.history if msg["role"] == "system"]
        recent_messages = self.history[-self.max_history * 2:]
        
        # Ensure we're not duplicating system messages
        recent_non_system = [msg for msg in recent_messages if msg["role"] != "system"]
        
        self.history = system_messages + recent_non_system
    
    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the conversation history.
        
        Args:
            content: The message content
        """
        self.add_message("system", content)
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            content: The message content
        """
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation history.
        
        Args:
            content: The message content
        """
        self.add_message("assistant", content)
    
    def get_context_window(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get the most recent messages as context window.
        
        Args:
            max_messages: Maximum number of messages to include
                         (None means all messages)
        
        Returns:
            List of recent messages
        """
        if max_messages is None:
            return self.get_history()
        
        # Always include system messages
        system_messages = [msg for msg in self.history if msg["role"] == "system"]
        
        # Get most recent non-system messages
        non_system = [msg for msg in self.history if msg["role"] != "system"]
        recent_non_system = non_system[-max_messages:] if non_system else []
        
        return system_messages + recent_non_system