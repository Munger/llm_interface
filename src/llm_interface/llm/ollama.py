"""
Ollama LLM client implementation.

This module provides a client for interacting with the Ollama API.

Author: Tim Hosking (https://github.com/Munger)
"""

import json
import os
import requests
import uuid
from typing import Dict, List, Any, Optional, Union

from llm_interface.config import Config
from llm_interface.llm.base import BaseLLMClient
from llm_interface.session.manager import FileSessionManager
from llm_interface.llm.ollama_session import OllamaSession

class OllamaClient(BaseLLMClient):
    """
    Client for interacting with Ollama LLMs.
    """
    
    def __init__(self, 
                 model: Optional[str] = None, 
                 host: Optional[str] = None, 
                 port: Optional[int] = None,
                 config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the Ollama client.
        
        Args:
            model: The Ollama model to use (overrides config)
            host: Ollama host (overrides config)
            port: Ollama port (overrides config)
            config_override: Optional configuration overrides
        """
        self.config = Config(config_override)
        
        # Override config with constructor parameters - ensure they take priority
        if model:
            self.config["default_model"] = model
        if host:
            self.config["ollama_host"] = host
        if port:
            self.config["ollama_port"] = port
            
        self.base_url = f"http://{self.config['ollama_host']}:{self.config['ollama_port']}"
        self.session_manager = FileSessionManager(self.config)
    
    def query(self, prompt: str, model: Optional[str] = None, debug: bool = False, **kwargs) -> str:
        """
        Send a single query to the Ollama API.
        
        Args:
            prompt: The prompt to send to the LLM
            model: The model to use (defaults to default_model in config)
            debug: Whether to print debug information
            **kwargs: Additional parameters to pass to the Ollama API
            
        Returns:
            The LLM's response as a string
        """
        # Make sure model parameter takes priority over config
        if model is None:
            model = self.config["default_model"]
        
        url = f"{self.base_url}/api/generate"
        
        # Prepare request payload with an empty system message to avoid restrictions
        payload = {
            "model": model,
            "prompt": prompt,
            "system": "",  # Use empty system message to allow general questions
            **kwargs
        }
        
        if debug:
            print(f"DEBUG - Sending payload to Ollama: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(
                url, 
                json=payload,
                timeout=self.config["timeout"]
            )
            response.raise_for_status()
            
            # Ollama streaming response - collect all chunks
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    full_response += json_response.get("response", "")
            
            return full_response.strip()
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error querying Ollama: {e}")
    
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, debug: bool = False, **kwargs) -> str:
        """
        Send a chat history to the Ollama API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: The model to use (defaults to default_model in config)
            debug: Whether to print debug information
            **kwargs: Additional parameters to pass to the Ollama API
            
        Returns:
            The LLM's response as a string
        """
        # Make sure model parameter takes priority over config
        if model is None:
            model = self.config["default_model"]
        
        url = f"{self.base_url}/api/chat"
        
        # Replace any system message with an empty one to avoid restrictions
        new_messages = []
        has_system = False
        
        for msg in messages:
            if msg.get("role") == "system":
                has_system = True
                new_messages.append({"role": "system", "content": ""})
            else:
                new_messages.append(msg)
        
        # Add empty system message if none exists
        if not has_system:
            new_messages = [{"role": "system", "content": ""}] + new_messages
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": new_messages,
            "stream": False,  # Request non-streaming response
            **kwargs
        }
        
        if debug:
            print(f"DEBUG - Sending payload to Ollama chat: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(
                url, 
                json=payload,
                timeout=self.config["timeout"]
            )
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            if debug:
                print(f"DEBUG - Response status: {response.status_code}")
                print(f"DEBUG - Response content type: {content_type}")
            
            # Handle different response formats based on content type
            if 'application/json' in content_type:
                # Standard JSON response
                response_json = response.json()
                return response_json.get("message", {}).get("content", "")
                
            elif 'application/x-ndjson' in content_type:
                # Streaming NDJSON response
                if debug:
                    print("DEBUG - Handling streaming NDJSON response")
                full_content = ""
                
                # Process each line as a separate JSON object
                lines = response.text.strip().split('\n')
                for line in lines:
                    if not line.strip():
                        continue
                        
                    try:
                        json_obj = json.loads(line.strip())
                        if "message" in json_obj and "content" in json_obj["message"]:
                            content = json_obj["message"]["content"]
                            full_content += content
                    except json.JSONDecodeError:
                        if debug:
                            print(f"DEBUG - Failed to parse line: {line[:50]}...")
                
                return full_content
                
            else:
                # Unknown format, return raw text
                if debug:
                    print(f"DEBUG - Unknown content type: {content_type}")
                return f"Unknown response format. Raw: {response.text[:200]}"
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error chatting with Ollama: {e}")
    
    def create_session(self, session_id: Optional[str] = None) -> 'OllamaSession':
        """
        Create a new chat session.
        
        Args:
            session_id: Optional session identifier. If not provided, a random one will be generated.
            
        Returns:
            A new OllamaSession object
        """
        session_id = session_id or str(uuid.uuid4())
        session = OllamaSession(
            client=self,
            session_id=session_id,
            config=self.config
        )
        session.save()
        return session
    
    def get_session(self, session_id: str) -> 'OllamaSession':
        """
        Retrieve an existing chat session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            The requested OllamaSession object
            
        Raises:
            ValueError: If session does not exist
        """
        if not self.session_manager.exists(session_id):
            raise ValueError(f"Session {session_id} does not exist")
        
        session_data = self.session_manager.load(session_id)
        return OllamaSession(
            client=self,
            session_id=session_id,
            config=self.config,
            history=session_data.get("history", [])
        )
    
    def list_sessions(self) -> List[str]:
        """
        List all available session IDs.
        
        Returns:
            List of session identifiers
        """
        return self.session_manager.list_sessions()
    
    def delete_session(self, session_id: str) -> None:
        """
        Delete a chat session.
        
        Args:
            session_id: The session identifier
            
        Raises:
            ValueError: If session does not exist
        """
        if not self.session_manager.exists(session_id):
            raise ValueError(f"Session {session_id} does not exist")
        
        self.session_manager.delete(session_id)