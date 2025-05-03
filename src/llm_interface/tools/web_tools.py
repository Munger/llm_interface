"""
Web-based tools for LLM research.

This module provides tools for web searching, content extraction,
and other web-based information gathering.
"""

import time
from typing import Dict, List, Any, Optional

from llm_interface.tools.base import register_tool
from llm_interface.config import Config


@register_tool(
    name="web_search",
    description="Search the web for information on a specific topic"
)
def web_search(query: str, max_results: int = 5):
    """
    Search the web for information.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with search results
    """
    from llm_interface.research.web import WebSearch
    
    search = WebSearch()
    results = search.search(query, max_results=max_results)
    
    return {
        "query": query,
        "results": results,
        "timestamp": time.time()
    }


@register_tool(
    name="fetch_webpage",
    description="Fetch and extract content from a webpage"
)
def fetch_webpage(url: str):
    """
    Fetch and extract content from a webpage.
    
    Args:
        url: The URL to fetch
        
    Returns:
        Dictionary with extracted content
    """
    from llm_interface.research.web import WebSearch
    
    search = WebSearch()
    content = search.fetch_content(url)
    
    return {
        "url": url,
        "content": content,
        "timestamp": time.time()
    }


@register_tool(
    name="search_and_read",
    description="Search the web and read the most relevant page"
)
def search_and_read(query: str, max_results: int = 3):
    """
    Search the web and extract content from the most relevant result.
    
    Args:
        query: The search query
        max_results: Maximum number of results to consider
        
    Returns:
        Dictionary with search results and page content
    """
    from llm_interface.research.web import WebSearch
    
    search = WebSearch()
    results = search.search(query, max_results=max_results)
    
    if not results:
        return {
            "query": query,
            "error": "No search results found",
            "timestamp": time.time()
        }
    
    # Get the most relevant result
    top_result = results[0]
    url = top_result.get("url", "")
    
    if not url:
        return {
            "query": query,
            "results": results,
            "error": "No URL in top result",
            "timestamp": time.time()
        }
    
    # Fetch the content
    content = search.fetch_content(url)
    
    return {
        "query": query,
        "url": url,
        "title": top_result.get("title", ""),
        "content": content,
        "timestamp": time.time()
    }


@register_tool(
    name="find_list",
    description="Find a list of items on a specific topic"
)
def find_list(topic: str, item_type: str = "examples"):
    """
    Search for and extract lists related to a specific topic.
    
    Args:
        topic: The topic to find a list about
        item_type: The type of items in the list (examples, products, etc.)
        
    Returns:
        Dictionary with list information
    """
    from llm_interface.research.web import WebSearch
    
    # Form a search query designed to find lists
    queries = [
        f"list of {item_type} {topic}",
        f"top {item_type} {topic}",
        f"{topic} {item_type} list"
    ]
    
    search = WebSearch()
    all_results = []
    
    # Try different queries
    for query in queries:
        results = search.search(query, max_results=3)
        
        if results:
            all_results.extend(results)
            
            # Try to fetch content from the top result
            top_url = results[0].get("url", "")
            if top_url:
                content = search.fetch_content(top_url)
                
                # Check if content contains list indicators
                list_indicators = ["1.", "2.", "•", "-", "Top", "Best"]
                if any(indicator in content for indicator in list_indicators):
                    return {
                        "topic": topic,
                        "url": top_url,
                        "title": results[0].get("title", ""),
                        "content": content,
                        "timestamp": time.time()
                    }
    
    # If no good list found, return the best results
    if all_results:
        return {
            "topic": topic,
            "results": all_results[:5],
            "message": "No specific list found, but here are relevant results",
            "timestamp": time.time()
        }
    
    return {
        "topic": topic,
        "error": "No relevant list information found",
        "timestamp": time.time()
    }