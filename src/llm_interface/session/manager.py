"""
Session management functionality.

This module provides tools for managing LLM chat sessions.

Author: Tim Hosking (https://github.com/Munger)
"""

import json
import os
import glob
from typing import Dict, List, Any, Optional


class FileSessionManager:
    """
    Manages LLM sessions using the filesystem.
    
    Sessions are stored as JSON files in the configured session directory.
    """
    
    def __init__(self, config):
        """
        Initialize the session manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.session_dir = config["session_dir"]
        os.makedirs(self.session_dir, exist_ok=True)
    
    def _get_session_path(self, session_id: str) -> str:
        """
        Get the file path for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            The file path
        """
        # Sanitize session_id to prevent path traversal
        session_id = session_id.replace('/', '_').replace('\\', '_')
        return os.path.join(self.session_dir, f"{session_id}.json")
    
    def exists(self, session_id: str) -> bool:
        """
        Check if a session exists.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if the session exists, False otherwise
        """
        return os.path.exists(self._get_session_path(session_id))
    
    def save(self, session_id: str, data: Dict[str, Any]) -> None:
        """
        Save session data.
        
        Args:
            session_id: The session identifier
            data: The session data to save
        """
        path = self._get_session_path(session_id)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, session_id: str) -> Dict[str, Any]:
        """
        Load session data.
        
        Args:
            session_id: The session identifier
            
        Returns:
            The session data
            
        Raises:
            ValueError: If the session does not exist
        """
        path = self._get_session_path(session_id)
        if not os.path.exists(path):
            raise ValueError(f"Session {session_id} does not exist")
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def delete(self, session_id: str) -> None:
        """
        Delete a session.
        
        Args:
            session_id: The session identifier
            
        Raises:
            ValueError: If the session does not exist
        """
        path = self._get_session_path(session_id)
        if not os.path.exists(path):
            raise ValueError(f"Session {session_id} does not exist")
        
        os.remove(path)
    
    def list_sessions(self) -> List[str]:
        """
        List all available session IDs.
        
        Returns:
            List of session identifiers
        """
        pattern = os.path.join(self.session_dir, "*.json")
        session_files = glob.glob(pattern)
        return [os.path.splitext(os.path.basename(f))[0] for f in session_files]