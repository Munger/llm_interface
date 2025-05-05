"""
Ollama session management.

This module provides the OllamaSession class for maintaining conversation
history and context with Ollama models.

Author: Tim Hosking (https://github.com/Munger)
"""

import re
import time
from typing import Dict, List, Any, Optional, Union

from llm_interface.config import Config
from llm_interface.llm.base import LLMSession
from llm_interface.session.manager import FileSessionManager
# Direct import of the prompt_manager module
import llm_interface.config.prompt_manager as prompt_manager
from llm_interface.llm.research_capabilities import OllamaResearch


class OllamaSession(LLMSession):
    """
    Ollama chat session.
    
    Maintains conversation history and provides methods for interacting
    with the LLM within a session context.
    """
    
    def __init__(self, 
                 client, 
                 session_id: str, 
                 config: Config,
                 history: Optional[List[Dict[str, str]]] = None):
        """
        Initialize a chat session.
        
        Args:
            client: The OllamaClient instance
            session_id: The session identifier
            config: The configuration instance
            history: Optional chat history
        """
        self.client = client
        self.session_id = session_id
        self.config = config
        self.history = history or []
        self.session_manager = FileSessionManager(config)
        
        # Initialize research attributes
        self.research_history = []
        self.last_research_time = None
        self._last_research_query = None
        self._research_urls = []
        
        # Create the research capabilities
        self.research_capabilities = OllamaResearch(client, config)
        
        # Load from saved session data if available
        if session_id and self.session_manager.exists(session_id):
            try:
                session_data = self.session_manager.load(session_id)
                if "research_history" in session_data:
                    self.research_history = session_data["research_history"]
                if "last_research_time" in session_data:
                    self.last_research_time = session_data["last_research_time"]
                if "last_research_query" in session_data:
                    self._last_research_query = session_data["last_research_query"]
                if "research_urls" in session_data:
                    self._research_urls = session_data["research_urls"]
            except Exception as e:
                print(f"WARNING - Error loading research history: {e}")
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            content: The message content
        """
        self.history.append({"role": "user", "content": content})
    
    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the conversation history.
        
        Args:
            content: The message content
        """
        self.history.append({"role": "system", "content": content})
    
    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation history.
        
        Args:
            content: The message content
        """
        self.history.append({"role": "assistant", "content": content})
    
    def chat(self, message: str, model: Optional[str] = None, debug: bool = False, **kwargs) -> str:
        """
        Send a message to the LLM within this session.
        
        Args:
            message: The message to send
            model: The model to use (defaults to default_model in config)
            debug: Whether to print debug information
            **kwargs: Additional parameters to pass to the Ollama API
            
        Returns:
            The LLM's response as a string
        """
        # Check if the message is asking about research
        research_detection_keywords = prompt_manager.get_prompt_value("research", "research_detection_keywords")
        if self.research_history and research_detection_keywords and any(q in message.lower() for q in research_detection_keywords):
            # Add a research context reminder 
            self._add_research_context_reminder()
        
        # Check if the message is asking about sources or research
        source_detection_keywords = prompt_manager.get_prompt_value("research", "source_detection_keywords")
        if hasattr(self, '_research_urls') and self._research_urls and source_detection_keywords and any(
            keyword in message.lower() for keyword in source_detection_keywords
        ):
            # Add a reminder about the research conducted
            if hasattr(self, '_last_research_query'):
                self.add_research_reminder(self._last_research_query)
            else:
                self.add_research_reminder("previous topic")
                
        # Check if this is a research request
        if message.strip().lower().startswith("/research "):
            # Extract the research query
            research_query = message[10:].strip()
            if debug:
                print(f"DEBUG - Detected research command. Query: {research_query}")
            
            # Add user message to history
            self.history.append({"role": "user", "content": message})
            
            # Use the research_with_react method to get research results
            try:
                research_response = self.research_with_react(research_query, debug=debug)
                return research_response
                
            except ImportError as e:
                if debug:
                    print(f"DEBUG - ReAct module not available: {e}")
                
                # Fallback to regular research
                research_response = self.research(research_query, debug=debug, **kwargs)
                
                # Add assistant response to history
                self.history.append({"role": "assistant", "content": research_response})
                
                # Save session state
                self.save()
                
                return research_response
        
        # Regular chat processing continues here...
        # Add user message to history
        self.history.append({"role": "user", "content": message})
        
        # Get response from Ollama - use explicit model if provided and pass debug flag
        response = self.client.chat(self.history, model=model, debug=debug, **kwargs)
        
        # Add assistant response to history
        self.history.append({"role": "assistant", "content": response})
        
        # Limit history size
        max_history = self.config["max_history"]
        if max_history > 0 and len(self.history) > max_history * 2:
            # Keep system messages and the most recent messages
            system_messages = [msg for msg in self.history if msg["role"] == "system"]
            recent_messages = self.history[-max_history * 2:]
            self.history = system_messages + recent_messages
        
        # Save session state
        self.save()
        
        return response
    
    def _add_research_context_reminder(self):
        """Add a reminder about the research context to the conversation."""
        if not self.research_history:
            return
        
        # Get the most recent research
        last_research = self.research_history[-1]
        query = last_research.get("query", "")
        timestamp = last_research.get("timestamp", 0)
        sources = last_research.get("sources", [])
        
        time_ago = self._format_time_elapsed(time.time() - timestamp)
        
        # Use the prompt manager to get the reminder template
        reminder = prompt_manager.format_prompt("research", "research_reminder", 
                                time_ago=time_ago,
                                query=query,
                                num_sources=len(sources))
        
        self.history.append({"role": "system", "content": reminder})
    
    def _format_time_elapsed(self, seconds):
        """Format elapsed time in a human-readable way."""
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes > 1 else ''}"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours > 1 else ''}"
        else:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days > 1 else ''}"
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return self.history.copy()
    
    def clear_history(self) -> None:
        """Clear the conversation history, keeping only system messages."""
        self.history = [msg for msg in self.history if msg["role"] == "system"]
        self.save()
        
    def get_research_urls(self) -> List[Dict[str, str]]:
        """
        Get the URLs found during research.
        
        Returns:
            List of URL dictionaries with 'index', 'title', and 'url' keys
        """
        if hasattr(self, '_research_urls'):
            return self._research_urls
        return []
        
    def add_research_reminder(self, query: str) -> None:
        """
        Add a reminder about research that was conducted.
        This helps prevent the LLM from forgetting it did research.
        
        Args:
            query: The research query
        """
        # Check if we have research URLs
        if not hasattr(self, '_research_urls') or not self._research_urls:
            return
            
        # Create a reminder about the research
        url_list = "\n".join([f"[Source {url['index']}] {url['title']}: {url['url']}" 
                             for url in self._research_urls])
        
        # Use the prompt manager to get the reminder template
        reminder_msg = prompt_manager.format_prompt("research", "research_sources_reminder", 
                                    query=query,
                                    url_list=url_list)
        
        # Add to history as a system message
        self.history.append({"role": "system", "content": reminder_msg})
    
    def save(self) -> None:
        """Save the session state."""
        session_data = {
            "history": self.history,
            "research_history": self.research_history,
            "last_research_time": self.last_research_time,
            "last_research_query": getattr(self, '_last_research_query', None),
            "research_urls": getattr(self, '_research_urls', [])
        }
        self.session_manager.save(self.session_id, session_data)
    
    def research(self, query: str, debug: bool = False, **kwargs) -> str:
        """
        Perform web research and respond with RAG-enhanced knowledge.
        
        Args:
            query: The research question
            debug: Whether to print debug information
            **kwargs: Additional keyword arguments for research
            
        Returns:
            The LLM's response enhanced with web research
        """
        return self.research_capabilities.perform_research(
            query, self, debug=debug, **kwargs
        )
    
    def research_with_react(self, query: str, debug: bool = False, **kwargs) -> str:
        """
        Perform in-depth research using the ReAct pattern.
        
        This method uses Reasoning + Acting to conduct comprehensive
        research on the query topic, using tools as needed.
        
        Args:
            query: The research question
            debug: Whether to print debug information
            **kwargs: Additional keyword arguments
            
        Returns:
            The LLM's response enhanced with ReAct research
        """
        return self.research_capabilities.perform_react_research(
            query, self, debug=debug, **kwargs
        )