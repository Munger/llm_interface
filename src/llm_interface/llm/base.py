"""
Base LLM client interface.

This module defines the abstract base class for all LLM clients.
"""

import abc
from typing import Dict, List, Optional, Any, Union


class BaseLLMClient(abc.ABC):
    """
    Abstract base class for LLM clients.
    
    All LLM clients should implement this interface.
    """
    
    @abc.abstractmethod
    def query(self, prompt: str, **kwargs) -> str:
        """
        Send a single query to the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional keyword arguments for the LLM
            
        Returns:
            The LLM's response as a string
        """
        pass
    
    @abc.abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Send a chat history to the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional keyword arguments for the LLM
            
        Returns:
            The LLM's response as a string
        """
        pass
    
    @abc.abstractmethod
    def create_session(self, session_id: Optional[str] = None) -> 'LLMSession':
        """
        Create a new chat session.
        
        Args:
            session_id: Optional session identifier. If not provided, a random one will be generated.
            
        Returns:
            A new LLMSession object
        """
        pass
    
    @abc.abstractmethod
    def get_session(self, session_id: str) -> 'LLMSession':
        """
        Retrieve an existing chat session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            The requested LLMSession object
            
        Raises:
            ValueError: If session does not exist
        """
        pass
    
    @abc.abstractmethod
    def list_sessions(self) -> List[str]:
        """
        List all available session IDs.
        
        Returns:
            List of session identifiers
        """
        pass
    
    @abc.abstractmethod
    def delete_session(self, session_id: str) -> None:
        """
        Delete a chat session.
        
        Args:
            session_id: The session identifier
            
        Raises:
            ValueError: If session does not exist
        """
        pass


class LLMSession(abc.ABC):
    """
    Abstract base class for LLM sessions.
    
    An LLM session maintains conversation history and context.
    """
    
    @abc.abstractmethod
    def chat(self, message: str, **kwargs) -> str:
        """
        Send a message to the LLM within this session.
        
        Args:
            message: The message to send
            **kwargs: Additional keyword arguments for the LLM
            
        Returns:
            The LLM's response as a string
        """
        pass
    
    @abc.abstractmethod
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        pass
    
    @abc.abstractmethod
    def clear_history(self) -> None:
        """Clear the conversation history."""
        pass
    
    @abc.abstractmethod
    def save(self) -> None:
        """Save the session state."""
        pass
    
    @abc.abstractmethod
    def research(self, query: str, **kwargs) -> str:
        """
        Perform web research and respond with RAG-enhanced knowledge.
        
        Args:
            query: The research question
            **kwargs: Additional keyword arguments for research
            
        Returns:
            The LLM's response enhanced with web research
        """
        pass
