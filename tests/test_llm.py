"""
Tests for the LLM client.

This module contains unit tests for the LLM client functionality.
"""

import unittest
import os
import json
from unittest.mock import patch, MagicMock

from llm_interface.llm.ollama import OllamaClient
from llm_interface.config import Config


class TestOllamaClient(unittest.TestCase):
    """Tests for the OllamaClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test config
        self.config = Config({
            "ollama_host": "localhost",
            "ollama_port": 11434,
            "default_model": "test-model",
            "session_dir": "/tmp/llm_interface_test_sessions",
            "timeout": 10
        })
        
        # Ensure test directory exists
        os.makedirs(self.config["session_dir"], exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up test files
        if os.path.exists(self.config["session_dir"]):
            import shutil
            shutil.rmtree(self.config["session_dir"])
    
    @patch('requests.post')
    def test_query(self, mock_post):
        """Test the query method."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'{"response": "Hello", "done": false}',
            b'{"response": " world", "done": false}',
            b'{"response": "!", "done": true}'
        ]
        mock_post.return_value = mock_response
        
        # Create client
        client = OllamaClient(config_override=self.config)
        
        # Call method
        response = client.query("Test prompt")
        
        # Check response
        self.assertEqual(response, "Hello world!")
        
        # Verify request
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["model"], "test-model")
        self.assertEqual(kwargs["json"]["prompt"], "Test prompt")
    
    @patch('requests.post')
    def test_chat(self, mock_post):
        """Test the chat method."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "Hello, how can I help you?"
            }
        }
        mock_post.return_value = mock_response
        
        # Create client
        client = OllamaClient(config_override=self.config)
        
        # Create messages
        messages = [
            {"role": "user", "content": "Hi"}
        ]
        
        # Call method
        response = client.chat(messages)
        
        # Check response
        self.assertEqual(response, "Hello, how can I help you?")
        
        # Verify request
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["model"], "test-model")
        self.assertEqual(kwargs["json"]["messages"], messages)
    
    def test_create_session(self):
        """Test the create_session method."""
        # Create client
        client = OllamaClient(config_override=self.config)
        
        # Create session
        session = client.create_session()
        
        # Check session
        self.assertIsNotNone(session)
        self.assertIsNotNone(session.session_id)
        self.assertEqual(session.client, client)
        
        # Check session file
        session_path = os.path.join(
            self.config["session_dir"], 
            f"{session.session_id}.json"
        )
        self.assertTrue(os.path.exists(session_path))
        
        # Check session data
        with open(session_path, 'r') as f:
            data = json.load(f)
            self.assertIn("history", data)
            self.assertEqual(data["history"], [])
    
    def test_get_session(self):
        """Test the get_session method."""
        # Create client
        client = OllamaClient(config_override=self.config)
        
        # Create session
        session1 = client.create_session()
        session_id = session1.session_id
        
        # Get session
        session2 = client.get_session(session_id)
        
        # Check session
        self.assertEqual(session2.session_id, session_id)
        
        # Check invalid session
        with self.assertRaises(ValueError):
            client.get_session("invalid_session_id")
    
    def test_list_sessions(self):
        """Test the list_sessions method."""
        # Create client
        client = OllamaClient(config_override=self.config)
        
        # Create sessions
        session1 = client.create_session()
        session2 = client.create_session()
        
        # List sessions
        sessions = client.list_sessions()
        
        # Check sessions
        self.assertIn(session1.session_id, sessions)
        self.assertIn(session2.session_id, sessions)
    
    def test_delete_session(self):
        """Test the delete_session method."""
        # Create client
        client = OllamaClient(config_override=self.config)
        
        # Create session
        session = client.create_session()
        session_id = session.session_id
        
        # Delete session
        client.delete_session(session_id)
        
        # Check session file
        session_path = os.path.join(
            self.config["session_dir"], 
            f"{session_id}.json"
        )
        self.assertFalse(os.path.exists(session_path))
        
        # Check sessions list
        sessions = client.list_sessions()
        self.assertNotIn(session_id, sessions)
        
        # Check invalid session
        with self.assertRaises(ValueError):
            client.delete_session("invalid_session_id")


if __name__ == '__main__':
    unittest.main()
