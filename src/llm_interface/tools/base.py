"""
Base tool registry system for LLM Interface.

This module provides the foundation for implementing tools that can be
used by the LLM to gather information during research.

Author: Tim Hosking (https://github.com/Munger)
"""

from typing import Any, Callable, Dict, List, Optional


class Tool:
    """
    A tool that can be used by the LLM for research or information gathering.
    
    Tools provide specific functionality like web searching, code analysis,
    or document parsing that can be used during research.
    """
    
    def __init__(self, name: str, description: str, function: Callable):
        """
        Initialize a tool.
        
        Args:
            name: The tool name
            description: Description of what the tool does
            function: The function that implements the tool
        """
        self.name = name
        self.description = description
        self.function = function
    
    def execute(self, **params) -> Dict[str, Any]:
        """
        Execute the tool with the given parameters.
        
        Args:
            **params: Parameters for the tool
            
        Returns:
            Result of the tool execution
        """
        return self.function(**params)


class ToolRegistry:
    """
    Registry for all available tools.
    
    The tool registry manages the available tools and provides
    access to them for the ReAct system.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools = {}
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: The tool to register
        """
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            name: The tool name
            
        Returns:
            The tool if found, None otherwise
        """
        return self.tools.get(name)
    
    def execute_tool(self, name: str, **params) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            name: The tool name
            **params: Parameters for the tool
            
        Returns:
            Result of the tool execution
            
        Raises:
            ValueError: If the tool doesn't exist
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        
        return tool.execute(**params)
    
    def list_tools(self) -> List[Dict[str, str]]:
        """
        List all available tools.
        
        Returns:
            List of tool information dictionaries
        """
        return [{"name": tool.name, "description": tool.description} 
                for tool in self.tools.values()]


# Global tool registry
registry = ToolRegistry()


def register_tool(name: str, description: str):
    """
    Decorator for registering a function as a tool.
    
    Args:
        name: The tool name
        description: Description of what the tool does
    """
    def decorator(func):
        tool = Tool(name=name, description=description, function=func)
        registry.register_tool(tool)
        return func
    
    return decorator