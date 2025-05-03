"""
Video-specific tools for LLM research.

This module provides tools for searching videos and extracting
video metadata from various platforms.
"""

import re
import time
import urllib.parse
from typing import Dict, List, Any, Optional

from llm_interface.tools.base import register_tool
from llm_interface.config import Config
from llm_interface.config.api_keys import api_key_manager


@register_tool(
    name="search_videos",
    description="Search for videos on a specific topic across video platforms"
)
def search_videos(query: str, platform: str = "youtube", max_results: int = 10):
    """
    Search for videos on platforms like YouTube.
    
    Args:
        query: The search query
        platform: Video platform to search (youtube, vimeo, etc.)
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with video search results
    """
    from llm_interface.research.web import WebSearch
    
    # Modify query to specifically target videos on the specified platform
    platform_query = f"{query} {platform} video"
    
    search = WebSearch()
    results = search.search(platform_query, max_results=max_results * 2)  # Request more to filter down
    
    videos = []
    for result in results:
        url = result.get("url", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        
        # Check if this is likely a video result
        if _is_video_url(url, platform):
            videos.append({
                "title": title,
                "description": snippet,
                "url": url,
                "platform": platform
            })
            
            if len(videos) >= max_results:
                break
    
    # If we don't have enough results, try fetching top result pages to extract more video links
    if len(videos) < max_results and results:
        # Try to extract video links from top search results
        for result in results[:3]:  # Only check top 3 results to avoid too many requests
            url = result.get("url", "")
            if not url or any(v["url"] == url for v in videos):
                continue
                
            # Fetch the page content
            content = search.fetch_content(url)
            
            # Extract video links from the content
            extracted_videos = _extract_video_links(content, platform, query)
            
            # Add new videos to the list
            for video in extracted_videos:
                if len(videos) >= max_results:
                    break
                    
                if not any(v["url"] == video["url"] for v in videos):
                    videos.append(video)
    
    return {
        "query": query,
        "platform": platform,
        "videos": videos,
        "timestamp": time.time()
    }


@register_tool(
    name="search_youtube",
    description="Search for videos on YouTube using API when available"
)
def search_youtube(query: str, max_results: int = 10):
    """
    Search for videos on YouTube, using the API if available.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with search results
    """
    # Try using the YouTube API if available
    if api_key_manager.has_key("youtube"):
        try:
            return _search_youtube_api(query, max_results, api_key_manager.get_key("youtube"))
        except Exception as e:
            print(f"YouTube API search failed: {e}")
            # Fall back to web search
    
    # Fallback to web search
    return _search_youtube_web(query, max_results)


@register_tool(
    name="extract_video_metadata",
    description="Extract detailed metadata from a video URL"
)
def extract_video_metadata(url: str):
    """
    Extract detailed metadata from a video URL.
    
    Args:
        url: The video URL
        
    Returns:
        Dictionary with video metadata
    """
    from llm_interface.research.web import WebSearch
    
    if not _is_video_url(url):
        return {
            "url": url,
            "error": "Not a recognized video URL",
            "timestamp": time.time()
        }
    
    search = WebSearch()
    content = search.fetch_content(url)
    
    # Extract metadata
    title = _extract_title(content, url)
    description = _extract_description(content, url)
    
    # Extract additional metadata if available
    channel = _extract_channel(content, url)
    duration = _extract_duration(content, url)
    
    return {
        "url": url,
        "title": title,
        "description": description,
        "channel": channel,
        "duration": duration,
        "timestamp": time.time()
    }


@register_tool(
    name="find_video_playlists",
    description="Find playlists of videos related to a specific topic"
)
def find_video_playlists(topic: str, platform: str = "youtube", max_results: int = 5):
    """
    Find playlists of videos related to a specific topic.
    
    Args:
        topic: The topic to find video playlists about
        platform: Video platform to search (youtube, vimeo, etc.)
        max_results: Maximum number of playlists to return
        
    Returns:
        Dictionary with playlist information
    """
    from llm_interface.research.web import WebSearch
    
    # Modify query to specifically target playlists
    playlist_query = f"{topic} {platform} playlist"
    
    search = WebSearch()
    results = search.search(playlist_query, max_results=max_results * 2)
    
    playlists = []
    for result in results:
        url = result.get("url", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        
        # Check if this is likely a playlist result
        if _is_playlist_url(url, platform):
            playlists.append({
                "title": title,
                "description": snippet,
                "url": url,
                "platform": platform
            })
            
            if len(playlists) >= max_results:
                break
    
    return {
        "topic": topic,
        "platform": platform,
        "playlists": playlists,
        "timestamp": time.time()
    }


def _search_youtube_api(query: str, max_results: int, api_key: str) -> Dict[str, Any]:
    """Search YouTube using the API."""
    import requests
    
    api_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(max_results, 50),  # API limit is 50
        "key": api_key
    }
    
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    data = response.json()
    
    videos = []
    for item in data.get("items", []):
        video_id = item.get("id", {}).get("videoId")
        if not video_id:
            continue
            
        snippet = item.get("snippet", {})
        title = snippet.get("title", "")
        description = snippet.get("description", "")
        channel = snippet.get("channelTitle", "")
        
        videos.append({
            "title": title,
            "description": description,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "channel": channel,
            "platform": "youtube"
        })
    
    return {
        "query": query,
        "platform": "youtube",
        "videos": videos,
        "timestamp": time.time()
    }


def _search_youtube_web(query: str, max_results: int) -> Dict[str, Any]:
    """Search YouTube using web search."""
    from llm_interface.research.web import WebSearch
    
    # Modify query to specifically target YouTube videos
    youtube_query = f"{query} site:youtube.com"
    
    search = WebSearch()
    results = search.search(youtube_query, max_results=max_results * 2)
    
    videos = []
    for result in results:
        url = result.get("url", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        
        # Check if this is a YouTube video URL
        if "youtube.com/watch" in url or "youtu.be/" in url:
            videos.append({
                "title": title,
                "description": snippet,
                "url": url,
                "platform": "youtube"
            })
            
            if len(videos) >= max_results:
                break
    
    # If we don't have enough results, try fetching content to extract more videos
    if len(videos) < max_results and results:
        for result in results[:3]:
            url = result.get("url", "")
            if not url or not ("youtube.com" in url and not "/watch" in url):
                continue
                
            # Fetch the page content
            content = search.fetch_content(url)
            
            # Extract video links
            extracted_videos = _extract_youtube_links(content, query)
            
            # Add new videos
            for video in extracted_videos:
                if len(videos) >= max_results:
                    break
                    
                if not any(v["url"] == video["url"] for v in videos):
                    videos.append(video)
    
    return {
        "query": query,
        "platform": "youtube",
        "videos": videos,
        "timestamp": time.time()
    }


def _is_video_url(url: str, platform: str = None) -> bool:
    """Check if a URL is likely a video URL."""
    if not url:
        return False
        
    if platform == "youtube" or platform is None:
        if "youtube.com/watch" in url or "youtu.be/" in url:
            return True
            
    if platform == "vimeo" or platform is None:
        if "vimeo.com/" in url and not "/channels/" in url:
            return True
            
    if platform == "dailymotion" or platform is None:
        if "dailymotion.com/video/" in url:
            return True
    
    # Generic check for video-like URLs
    video_patterns = [
        r'/watch\?',
        r'/video/',
        r'/embed/',
        r'\.mp4',
        r'\.webm'
    ]
    
    return any(re.search(pattern, url) for pattern in video_patterns)


def _is_playlist_url(url: str, platform: str = None) -> bool:
    """Check if a URL is likely a playlist URL."""
    if not url:
        return False
        
    if platform == "youtube" or platform is None:
        if "youtube.com/playlist" in url or "list=" in url:
            return True
            
    if platform == "vimeo" or platform is None:
        if "vimeo.com/channels/" in url or "vimeo.com/album/" in url:
            return True
    
    # Generic check for playlist-like URLs
    playlist_patterns = [
        r'/playlist',
        r'/channel/',
        r'/collection/',
        r'/album/'
    ]
    
    return any(re.search(pattern, url) for pattern in playlist_patterns)


def _extract_video_links(content: str, platform: str, query: str) -> List[Dict[str, str]]:
    """Extract video links from HTML content."""
    videos = []
    
    # Extract YouTube video links
    if platform == "youtube" or platform is None:
        youtube_patterns = [
            r'href="(https://www\.youtube\.com/watch\?v=[^"&]+)"',
            r'href="(https://youtu\.be/[^"]+)"'
        ]
        
        for pattern in youtube_patterns:
            matches = re.findall(pattern, content)
            for url in matches:
                # Clean up URL
                url = url.split("&")[0]  # Remove additional parameters
                
                # Try to extract title
                title_pattern = r'title="([^"]+)"[^>]*href="[^"]*' + re.escape(url.split("/")[-1])
                title_match = re.search(title_pattern, content)
                
                title = title_match.group(1) if title_match else f"Video about {query}"
                
                videos.append({
                    "title": title,
                    "description": f"Found video about {query}",
                    "url": url,
                    "platform": "youtube"
                })
    
    # Extract Vimeo video links
    if platform == "vimeo" or platform is None:
        vimeo_pattern = r'href="(https://vimeo\.com/\d+[^"/]*)"'
        matches = re.findall(vimeo_pattern, content)
        
        for url in matches:
            # Try to extract title
            title_pattern = r'title="([^"]+)"[^>]*href="[^"]*' + re.escape(url.split("/")[-1])
            title_match = re.search(title_pattern, content)
            
            title = title_match.group(1) if title_match else f"Vimeo video about {query}"
            
            videos.append({
                "title": title,
                "description": f"Found video about {query}",
                "url": url,
                "platform": "vimeo"
            })
    
    return videos


def _extract_youtube_links(content: str, query: str) -> List[Dict[str, str]]:
    """Extract YouTube video links from HTML content."""
    videos = []
    
    # Extract YouTube video links
    youtube_patterns = [
        r'href="(https://www\.youtube\.com/watch\?v=[^"&]+)"',
        r'href="(https://youtu\.be/[^"]+)"'
    ]
    
    for pattern in youtube_patterns:
        matches = re.findall(pattern, content)
        for url in matches:
            # Clean up URL
            url = url.split("&")[0]  # Remove additional parameters
            
            # Try to extract title
            title_pattern = r'title="([^"]+)"[^>]*href="[^"]*' + re.escape(url.split("/")[-1].split("?")[0])
            title_match = re.search(title_pattern, content)
            
            title = title_match.group(1) if title_match else f"Video about {query}"
            
            videos.append({
                "title": title,
                "description": f"Found video about {query}",
                "url": url,
                "platform": "youtube"
            })
    
    return videos


def _extract_title(content: str, url: str) -> str:
    """Extract title from page content or URL."""
    # Try to extract title from HTML
    title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()
        # Clean up title
        title = re.sub(r'\s+', ' ', title)
        title = re.sub(r'\s*[|:-]\s*YouTube$', '', title)
        title = re.sub(r'\s*[|:-]\s*Vimeo$', '', title)
        return title
    
    # Extract from URL if title not found
    if "youtube.com" in url or "youtu.be" in url:
        # Try to extract video ID
        video_id = None
        if "youtube.com/watch" in url:
            match = re.search(r'v=([^&]+)', url)
            if match:
                video_id = match.group(1)
        elif "youtu.be/" in url:
            video_id = url.split("/")[-1].split("?")[0]
        
        if video_id:
            return f"YouTube Video (ID: {video_id})"
    
    elif "vimeo.com" in url:
        video_id = url.split("/")[-1].split("?")[0]
        if video_id.isdigit():
            return f"Vimeo Video (ID: {video_id})"
    
    # Generic fallback
    return "Video (URL: " + url.split("/")[-1] + ")"


def _extract_description(content: str, url: str) -> str:
    """Extract description from page content."""
    # Try to extract meta description
    meta_desc_match = re.search(r'<meta\s+name="description"\s+content="([^"]+)"', content, re.IGNORECASE)
    if meta_desc_match:
        return meta_desc_match.group(1).strip()
    
    # Try to extract Open Graph description
    og_desc_match = re.search(r'<meta\s+property="og:description"\s+content="([^"]+)"', content, re.IGNORECASE)
    if og_desc_match:
        return og_desc_match.group(1).strip()
    
    # Look for a description in the content
    if "youtube.com" in url:
        desc_match = re.search(r'<div id="description[^"]*"[^>]*>(.*?)</div>', content, re.DOTALL)
        if desc_match:
            desc = desc_match.group(1)
            # Clean up HTML
            desc = re.sub(r'<[^>]+>', ' ', desc)
            desc = re.sub(r'\s+', ' ', desc)
            if len(desc) > 200:
                desc = desc[:197] + "..."
            return desc.strip()
    
    # Generic fallback
    return "No description available"


def _extract_channel(content: str, url: str) -> str:
    """Extract channel information from page content."""
    if "youtube.com" in url or "youtu.be" in url:
        channel_match = re.search(r'<link\s+rel="canonical"\s+href="https://www\.youtube\.com/channel/([^"]+)"', content)
        if channel_match:
            return f"YouTube Channel: {channel_match.group(1)}"
        
        # Alternative pattern
        alt_channel_match = re.search(r'<span[^>]*itemprop="author"[^>]*>.*?<link[^>]*href="([^"]+)"', content, re.DOTALL)
        if alt_channel_match:
            return f"Channel: {alt_channel_match.group(1).split('/')[-1]}"
    
    elif "vimeo.com" in url:
        channel_match = re.search(r'<a[^>]*href="[^"]*/user/([^"]+)"', content)
        if channel_match:
            return f"Vimeo User: {channel_match.group(1)}"
    
    return "Unknown channel"


def _extract_duration(content: str, url: str) -> str:
    """Extract video duration from page content."""
    # Look for duration metadata
    duration_match = re.search(r'<meta\s+itemprop="duration"\s+content="([^"]+)"', content, re.IGNORECASE)
    if duration_match:
        duration = duration_match.group(1)
        # Convert ISO 8601 duration if needed (PT1H30M15S)
        if duration.startswith("PT"):
            duration = duration[2:]
            hours = re.search(r'(\d+)H', duration)
            minutes = re.search(r'(\d+)M', duration)
            seconds = re.search(r'(\d+)S', duration)
            
            duration_parts = []
            if hours:
                duration_parts.append(f"{hours.group(1)} hour(s)")
            if minutes:
                duration_parts.append(f"{minutes.group(1)} minute(s)")
            if seconds and not (hours or minutes):
                duration_parts.append(f"{seconds.group(1)} second(s)")
                
            if duration_parts:
                return "Duration: " + ", ".join(duration_parts)
    
    # Look for length in text
    length_match = re.search(r'>([\d:]+)</span>\s*<span[^>]*>Length', content)
    if length_match:
        return f"Duration: {length_match.group(1)}"
    
    return "Unknown duration"