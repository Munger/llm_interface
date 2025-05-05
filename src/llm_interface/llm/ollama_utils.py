"""
Utility functions for Ollama integration.

This module provides helper functions used by the Ollama client and session.

Author: Tim Hosking (https://github.com/Munger)
"""

import re
from typing import Dict, List, Any, Optional


def extract_content_blocks(text: str, block_type: str) -> List[str]:
    """
    Extract content blocks of a specific type from text.
    
    Args:
        text: The text to search
        block_type: The type of block to extract (e.g., 'code', 'json')
        
    Returns:
        List of extracted block contents
    """
    pattern = rf"```{block_type}(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    return [block.strip() for block in blocks]


def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """
    Extract JSON objects from text.
    
    Args:
        text: The text to search
        
    Returns:
        List of extracted JSON objects
    """
    import json
    
    # Try to extract code blocks first
    json_blocks = extract_content_blocks(text, "json")
    
    # If no code blocks, try to find JSON-like patterns
    if not json_blocks:
        # Look for patterns like { ... }
        json_pattern = r"\{(?:[^{}]|(?R))*\}"
        matches = re.findall(json_pattern, text)
        json_blocks = [match.strip() for match in matches]
    
    # Parse each block
    json_objects = []
    for block in json_blocks:
        try:
            json_obj = json.loads(block)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            # Skip invalid JSON
            continue
    
    return json_objects


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: The text to search
        
    Returns:
        List of extracted URLs
    """
    # Simple URL pattern
    url_pattern = r"https?://[^\s<>\"'()]+"
    return re.findall(url_pattern, text)


def format_conversation_for_context(history: List[Dict[str, str]], max_msgs: int = 10) -> str:
    """
    Format conversation history for inclusion in prompts.
    
    Args:
        history: List of message dictionaries
        max_msgs: Maximum number of messages to include
        
    Returns:
        Formatted conversation string
    """
    # Take the most recent messages
    recent_msgs = history[-max_msgs:] if len(history) > max_msgs else history
    
    # Format messages
    formatted = []
    for msg in recent_msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            formatted.append(f"Human: {content}")
        elif role == "assistant":
            formatted.append(f"Assistant: {content}")
        elif role == "system":
            # Skip system messages in this format
            continue
    
    return "\n\n".join(formatted)


def truncate_chat_history(messages: List[Dict[str, str]], 
                         max_tokens: int = 8000, 
                         token_estimator=None) -> List[Dict[str, str]]:
    """
    Truncate chat history to fit within token limit.
    
    Args:
        messages: List of message dictionaries
        max_tokens: Maximum tokens allowed
        token_estimator: Function to estimate tokens (optional)
        
    Returns:
        Truncated message list
    """
    if not messages:
        return []
    
    # Set default token estimator if none provided
    if token_estimator is None:
        def token_estimator(text):
            # Simple approximation: 1 token ~ 4 characters
            return len(text) // 4
    
    # Get system messages (these stay)
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    
    # Get non-system messages
    chat_messages = [msg for msg in messages if msg.get("role") != "system"]
    
    # Estimate tokens in system messages
    system_tokens = sum(token_estimator(msg.get("content", "")) for msg in system_messages)
    
    # Maximum tokens available for chat messages
    available_tokens = max_tokens - system_tokens
    
    # Keep as many recent chat messages as possible
    kept_messages = []
    token_count = 0
    
    for msg in reversed(chat_messages):
        msg_tokens = token_estimator(msg.get("content", ""))
        
        if token_count + msg_tokens <= available_tokens:
            kept_messages.insert(0, msg)
            token_count += msg_tokens
        else:
            # Can't add more messages without exceeding the limit
            break
    
    # Combine system messages and kept chat messages
    return system_messages + kept_messages
