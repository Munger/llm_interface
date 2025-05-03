"""
List processing tools for LLM research.

This module provides tools for generating, processing, and aggregating
lists of information for more comprehensive research results.
"""

import time
from typing import Dict, List, Any, Optional

from llm_interface.tools.base import register_tool


@register_tool(
    name="aggregate_list",
    description="Aggregate and deduplicate list items from multiple sources"
)
def aggregate_list(items: List[Dict[str, Any]], target_count: int = 100):
    """
    Aggregate and deduplicate list items.
    
    Args:
        items: List of items to process, each with at least a "url" or "id" field
        target_count: Target number of items to aim for
        
    Returns:
        Dictionary with aggregated list
    """
    # Track unique items by URL or other unique identifier
    unique_items = {}
    
    for item in items:
        # Determine unique key
        key = item.get("url", None)
        if key is None:
            key = item.get("id", None)
        if key is None:
            # If no unique key, use the entire item as the key
            key = str(item)
        
        # Add to unique items, overwriting if item already exists
        # This ensures we keep only the most complete version of each item
        if key not in unique_items or _is_better_item(item, unique_items[key]):
            unique_items[key] = item
    
    # Sort items by relevance or completeness
    sorted_items = sorted(
        unique_items.values(),
        key=lambda x: _calculate_item_score(x),
        reverse=True
    )
    
    # Calculate statistics
    stats = {
        "total_input_items": len(items),
        "unique_items": len(unique_items),
        "returned_items": min(len(sorted_items), target_count),
        "completeness": len(sorted_items) / target_count if target_count > 0 else 1.0
    }
    
    return {
        "items": sorted_items[:target_count],
        "stats": stats,
        "timestamp": time.time()
    }


@register_tool(
    name="enhance_list_items",
    description="Enhance list items with additional information"
)
def enhance_list_items(items: List[Dict[str, Any]], fields_to_enhance: List[str]):
    """
    Enhance list items with additional information.
    
    Args:
        items: List of items to enhance
        fields_to_enhance: Fields to enhance (title, description, etc.)
        
    Returns:
        Dictionary with enhanced items
    """
    from llm_interface.research.web import WebSearch
    
    enhanced_items = []
    search = WebSearch()
    
    for item in items:
        # Skip if already has all fields
        if all(field in item and item[field] for field in fields_to_enhance):
            enhanced_items.append(item)
            continue
        
        # Try to enhance by fetching the URL content if available
        url = item.get("url")
        if url:
            content = search.fetch_content(url)
            
            # Extract title if needed
            if "title" in fields_to_enhance and not item.get("title"):
                from llm_interface.tools.video_tools import _extract_title
                item["title"] = _extract_title(content, url)
            
            # Extract description if needed
            if "description" in fields_to_enhance and not item.get("description"):
                from llm_interface.tools.video_tools import _extract_description
                item["description"] = _extract_description(content, url)
            
            # Add other fields as needed
            # ...
        
        enhanced_items.append(item)
    
    return {
        "items": enhanced_items,
        "enhanced_fields": fields_to_enhance,
        "timestamp": time.time()
    }


def _is_better_item(new_item: Dict[str, Any], existing_item: Dict[str, Any]) -> bool:
    """Determine if the new item is more complete than the existing one."""
    # More fields is better
    if len(new_item) > len(existing_item):
        return True
    
    # More complete critical fields is better
    critical_fields = ["title", "description", "url"]
    new_critical_count = sum(1 for field in critical_fields if field in new_item and new_item[field])
    existing_critical_count = sum(1 for field in critical_fields if field in existing_item and existing_item[field])
    
    if new_critical_count > existing_critical_count:
        return True
    
    # Longer description is usually better
    if "description" in new_item and "description" in existing_item:
        if len(new_item["description"]) > len(existing_item["description"]):
            return True
    
    return False


def _calculate_item_score(item: Dict[str, Any]) -> float:
    """Calculate a relevance/completeness score for sorting."""
    score = 0.0
    
    # Score for having critical fields
    critical_fields = ["title", "description", "url"]
    for field in critical_fields:
        if field in item and item[field]:
            score += 1.0
    
    # Bonus for longer description (up to a point)
    if "description" in item and item["description"]:
        desc_len = len(item["description"])
        score += min(desc_len / 200, 1.0)  # Cap at 1.0 for descriptions of 200+ chars
    
    # Bonus for having additional metadata
    additional_fields = ["channel", "duration", "platform"]
    for field in additional_fields:
        if field in item and item[field]:
            score += 0.2
    
    return score
