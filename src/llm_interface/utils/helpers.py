"""
Utility functions for the LLM Interface.

This module provides common utility functions used across the package.

Author: Tim Hosking (https://github.com/Munger)
"""

import os
import json
import uuid
import time
from typing import Dict, List, Any, Optional, Union


def generate_id() -> str:
    """
    Generate a unique identifier.
    
    Returns:
        A unique string identifier
    """
    return str(uuid.uuid4())


def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def save_json(data: Any, path: str) -> None:
    """
    Save data as JSON.
    
    Args:
        data: Data to save
        path: File path
    """
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    """
    Load data from JSON.
    
    Args:
        path: File path
        
    Returns:
        Loaded data
        
    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(path, 'r') as f:
        return json.load(f)


def truncate_text(text: str, max_length: int = 100, add_ellipsis: bool = True) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        add_ellipsis: Whether to add ellipsis if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    if add_ellipsis:
        truncated += "..."
        
    return truncated


def format_time(timestamp: Optional[float] = None) -> str:
    """
    Format a timestamp as a human-readable string.
    
    Args:
        timestamp: Unix timestamp (default: current time)
        
    Returns:
        Formatted time string
    """
    if timestamp is None:
        timestamp = time.time()
        
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def parse_bool(value: Union[str, bool]) -> bool:
    """
    Parse a string as a boolean.
    
    Args:
        value: String or boolean value
        
    Returns:
        Boolean value
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        return value.lower() in ('true', 'yes', 'y', '1', 'on')
    
    return bool(value)


def format_exception(e: Exception) -> str:
    """
    Format an exception as a string.
    
    Args:
        e: Exception object
        
    Returns:
        Formatted exception string
    """
    return f"{type(e).__name__}: {str(e)}"


def is_valid_url(url: str) -> bool:
    """
    Check if a string is a valid URL.
    
    Args:
        url: URL string
        
    Returns:
        True if valid URL, False otherwise
    """
    import re
    
    # Simple URL validation regex
    pattern = re.compile(
        r'^(?:http|https)://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return bool(re.match(pattern, url))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to be safe for filesystem use.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Ensure filename is not too long
    max_length = 255
    if len(filename) > max_length:
        base, ext = os.path.splitext(filename)
        filename = base[:max_length - len(ext)] + ext
    
    return filename
