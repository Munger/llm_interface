"""
Web search functionality for research.

This module provides tools for searching the web and retrieving relevant information.
"""

import json
import requests
from typing import List, Dict, Any, Optional, Set
import urllib.parse
import time
import re

from llm_interface.config import Config


class WebSearch:
    """
    Web search client for retrieving information from the internet.
    
    Uses the DuckDuckGo API for searches without requiring API keys.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the web search client.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or Config()
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    
    def search(self, query: str, max_results: Optional[int] = None, debug: bool = False) -> List[Dict[str, str]]:
        """
        Search the web for information.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
                         (None means use config default)
            debug: Whether to print debug information
                         
        Returns:
            List of search result dictionaries with title, snippet, and url keys
        """
        max_results = max_results or self.config["max_search_results"]
        
        # Try to import duckduckgo_search if available
        try:
            from duckduckgo_search import DDGS
            
            if debug:
                print(f"DEBUG - Using DuckDuckGo Search library for query: {query}")
            
            ddgs = DDGS()
            results = []
            
            # Use the DuckDuckGo search library
            try:
                # Request more results than needed to ensure we get enough after filtering
                ddg_results = list(ddgs.text(query, max_results=max_results*3))
                
                for result in ddg_results:
                    if len(results) >= max_results:
                        break
                    
                    results.append({
                        "title": result.get("title", "No title"),
                        "snippet": result.get("body", "No snippet available"),
                        "url": result.get("href", "")
                    })
                
                if debug:
                    print(f"DEBUG - Found {len(results)} results using DuckDuckGo Search")
                
                return results[:max_results]
                
            except Exception as e:
                if debug:
                    print(f"DEBUG - Error with DuckDuckGo Search library: {e}")
                    print("DEBUG - Falling back to alternative search method")
                
                # If DDGS fails, fall back to the alternative method
                pass
        
        except ImportError:
            if debug:
                print("DEBUG - DuckDuckGo Search library not available, using alternative method")
        
        # Fallback method: Direct DuckDuckGo API-like interface
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&pretty=0"
        
        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, headers=headers, timeout=self.config["timeout"])
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Process abstract text if available
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", "DuckDuckGo Abstract"),
                    "snippet": data.get("Abstract"),
                    "url": data.get("AbstractURL")
                })
            
            # Process related topics
            for topic in data.get("RelatedTopics", []):
                if "Topics" in topic:
                    # This is a category, process its subtopics
                    for subtopic in topic.get("Topics", []):
                        if len(results) >= max_results:
                            break
                        
                        results.append({
                            "title": subtopic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                            "snippet": subtopic.get("Text", ""),
                            "url": subtopic.get("FirstURL")
                        })
                else:
                    if len(results) >= max_results:
                        break
                    
                    results.append({
                        "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL")
                    })
            
            if debug:
                print(f"DEBUG - Found {len(results)} results using DuckDuckGo API")
            
            return results[:max_results]
            
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            if debug:
                print(f"DEBUG - Error during web search: {e}")
            return []
    
    def fetch_content(self, url: str, debug: bool = False) -> str:
        """
        Fetch the content of a webpage.
        
        Args:
            url: The URL to fetch
            debug: Whether to print debug information
            
        Returns:
            The plain text content of the webpage
        """
        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, headers=headers, timeout=self.config["timeout"])
            response.raise_for_status()
            
            # Import BeautifulSoup for HTML parsing
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
                element.extract()
            
            # Extract links since they might be important
            links = []
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if href.startswith("http") and len(href) > 10:
                    link_text = a.get_text().strip()
                    if link_text:
                        links.append(f"Link: {link_text} - {href}")
            
            # Get text
            text = soup.get_text()
            
            # Process text to make it more readable
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Add important links to the beginning of the text if there aren't too many
            if links and len(links) < 20:
                text = "IMPORTANT LINKS FOUND:\n" + "\n".join(links[:10]) + "\n\n" + text
            
            # Limit content length to avoid excessive text
            max_content_length = 12000
            if len(text) > max_content_length:
                if debug:
                    print(f"DEBUG - Truncating content from {len(text)} to {max_content_length} chars")
                text = text[:max_content_length] + "..."
            
            return text
            
        except Exception as e:
            if debug:
                print(f"DEBUG - Error fetching content from {url}: {e}")
            return ""


class WebResearcher:
    """
    Web researcher that combines search and knowledge retrieval.
    
    This class provides methods for researching topics on the web
    and formatting the results for use with an LLM.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the web researcher.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or Config()
        self.web_search = WebSearch(self.config)
    
    def research(self, query: str, debug: bool = False) -> Dict[str, Any]:
        """
        Research a topic on the web with enhanced depth.
        
        Args:
            query: The research query
            debug: Whether to print debug information
            
        Returns:
            Dictionary with search results and extracted content
        """
        if debug:
            print(f"DEBUG - Starting deep web research for query: {query}")
        
        # First round of search with original query
        primary_results = self.web_search.search(query, max_results=15, debug=debug)
        
        if debug:
            print(f"DEBUG - Found {len(primary_results)} primary search results")
        
        # Track all URLs we've seen to avoid duplicates
        all_urls = set(r.get("url", "") for r in primary_results if r.get("url"))
        all_results = primary_results.copy()
        
        # Extract key terms from initial results
        key_terms = self._extract_key_terms(primary_results, query, debug=debug)
        
        # Generate a variety of follow-up queries based on the query type
        follow_up_queries = self._generate_follow_up_queries(query, key_terms, primary_results)
        
        if debug:
            print(f"DEBUG - Generated {len(follow_up_queries)} follow-up queries: {follow_up_queries}")
        
        # Try different query transformations
        for follow_up_query in follow_up_queries[:7]:  # Limit to top 7 follow-up queries
            if debug:
                print(f"DEBUG - Performing follow-up search: {follow_up_query}")
                
            secondary_results = self.web_search.search(follow_up_query, max_results=7, debug=debug)
            
            # Add new results that aren't duplicates
            new_results = []
            for result in secondary_results:
                url = result.get("url", "")
                if url and url not in all_urls:
                    all_urls.add(url)
                    new_results.append(result)
            
            all_results.extend(new_results)
            
            if debug:
                print(f"DEBUG - Found {len(new_results)} new results from follow-up query")
        
        # Sort results by relevance (we'll use a simple heuristic)
        sorted_results = self._sort_results_by_relevance(all_results, query, key_terms)
        
        # Extract content from top results, prioritizing diverse sources
        content = []
        processed_domains = set()
        
        # First pass: Try to get content from the most relevant results
        for result in sorted_results[:20]:  # Examine top 20 results
            url = result.get("url", "")
            if not url:
                continue
                
            # Extract domain to ensure diversity
            domain = self._extract_domain(url)
            
            # Skip if we already have content from this domain
            if domain in processed_domains and len(processed_domains) >= 7:
                continue
                
            if debug:
                print(f"DEBUG - Fetching content from {url}")
            
            # Add a small delay to avoid overwhelming the server
            time.sleep(0.5)
            
            text = self.web_search.fetch_content(url, debug=debug)
            if text:
                processed_domains.add(domain)
                content.append({
                    "title": result.get("title", ""),
                    "url": url,
                    "content": text[:7500]  # Allow for more content per source
                })
                
                if debug:
                    print(f"DEBUG - Added content from {url} ({len(text)} chars)")
                
                # Once we have 12 diverse sources, stop processing
                if len(content) >= 15:
                    break
        
        # Second pass: If query suggests needing many items, look for list-type content
        if self._needs_list_content(query) and len(content) < 8:
            list_query = f"{query} list comprehensive"
            if debug:
                print(f"DEBUG - Searching specifically for list content: {list_query}")
                
            list_results = self.web_search.search(list_query, max_results=8, debug=debug)
            
            for result in list_results:
                url = result.get("url", "")
                if not url or url in all_urls:
                    continue
                    
                domain = self._extract_domain(url)
                if domain in processed_domains and len(processed_domains) >= 10:
                    continue
                    
                time.sleep(0.5)
                text = self.web_search.fetch_content(url, debug=debug)
                
                if text and self._contains_list(text):
                    processed_domains.add(domain)
                    content.append({
                        "title": result.get("title", ""),
                        "url": url,
                        "content": text[:10000]  # Allow longer content for lists
                    })
                    
                    if debug:
                        print(f"DEBUG - Added list content from {url}")
                    
                    if len(content) >= 20:
                        break
        
        # If we still don't have enough content, try more general sources
        if len(content) < 3:
            if debug:
                print(f"DEBUG - Insufficient content found. Searching for more general sources")
                
            general_query = self._generalize_query(query)
            general_results = self.web_search.search(general_query, max_results=5, debug=debug)
            
            for result in general_results:
                url = result.get("url", "")
                if not url or url in all_urls:
                    continue
                    
                time.sleep(0.5)
                text = self.web_search.fetch_content(url, debug=debug)
                
                if text:
                    content.append({
                        "title": result.get("title", ""),
                        "url": url,
                        "content": text[:7500]
                    })
                    
                    if debug:
                        print(f"DEBUG - Added general content from {url}")
                    
                    if len(content) >= 20:
                        break
        
        if debug:
            print(f"DEBUG - Research complete. Found {len(all_results)} total results and {len(content)} content sources")
        
        return {
            "query": query,
            "search_results": sorted_results,
            "content": content,
            "timestamp": time.time()
        }
    
    def _extract_key_terms(self, 
                         results: List[Dict[str, str]], 
                         original_query: str,
                         debug: bool = False) -> Set[str]:
        """
        Extract key terms from search results.
        
        Args:
            results: Search results to analyze
            original_query: The original search query
            debug: Whether to print debug information
            
        Returns:
            Set of key terms
        """
        terms = set()
        term_counts = {}
        
        # Add terms from the original query
        query_terms = set(re.findall(r'\b\w{4,}\b', original_query.lower()))
        terms.update(query_terms)
        
        # Extract terms from titles and snippets
        for result in results:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            
            # Extract potential key terms
            text = f"{title} {snippet}"
            # Remove punctuation and split into words
            words = re.findall(r'\b\w+\b', text.lower())
            
            for word in words:
                # Skip very short words and common stop words
                if len(word) <= 3 or word in [
                    "the", "and", "for", "with", "that", "this", "what", "where", 
                    "when", "how", "who", "why", "which", "about", "from"
                ]:
                    continue
                    
                term_counts[word] = term_counts.get(word, 0) + 1
        
        # Get terms that appear multiple times
        for term, count in term_counts.items():
            if count >= 2:
                terms.add(term)
        
        # Add capitalized terms (potential proper nouns)
        for result in results:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            
            text = f"{title} {snippet}"
            capitalized_terms = re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', text)
            for term in capitalized_terms:
                terms.add(term.lower())
        
        if debug:
            print(f"DEBUG - Extracted {len(terms)} key terms: {', '.join(list(terms)[:10])}...")
            
        return terms
    
    def _generate_follow_up_queries(self, 
                                   original_query: str, 
                                   key_terms: Set[str],
                                   results: List[Dict[str, str]]) -> List[str]:
        """
        Generate follow-up queries based on the original query and key terms.
        
        Args:
            original_query: The original search query
            key_terms: Set of extracted key terms
            results: Initial search results
            
        Returns:
            List of follow-up queries
        """
        follow_up_queries = []
        
        # Determine what type of query this is
        is_list_query = self._needs_list_content(original_query)
        is_how_to_query = "how" in original_query.lower() and "to" in original_query.lower()
        is_comparison_query = any(term in original_query.lower() for term in ["versus", " vs ", "compare", "difference"])
        is_definition_query = any(term in original_query.lower() for term in ["what is", "definition", "meaning", "explain"])
        
        # Use key terms to create more targeted queries
        if key_terms:
            # Group the most relevant terms
            top_terms = list(key_terms)[:7]
            
            # Create combinations of the original query with key terms
            if len(top_terms) >= 3:
                follow_up_queries.append(f"{original_query} {top_terms[0]} {top_terms[1]} {top_terms[2]}")
            
            if len(top_terms) >= 2:
                follow_up_queries.append(f"{original_query} {top_terms[0]} {top_terms[1]}")
            
            if len(top_terms) >= 1:
                follow_up_queries.append(f"{original_query} {top_terms[0]}")
        
        # Handle different query types appropriately
        if is_list_query:
            follow_up_queries.append(f"{original_query} list")
            follow_up_queries.append(f"{original_query} comprehensive")
            follow_up_queries.append(f"{original_query} examples")
            follow_up_queries.append(f"{original_query} directory")
            
            # Check if the query has a number in it
            number_match = re.search(r'\b(\d+)\b', original_query)
            if number_match:
                number = int(number_match.group(1))
                # Also search for smaller numbers that might yield more results
                if number > 50:
                    follow_up_queries.append(original_query.replace(str(number), "50"))
                if number > 20:
                    follow_up_queries.append(original_query.replace(str(number), "20"))
                if number > 10:
                    follow_up_queries.append(original_query.replace(str(number), "10"))
        
        if is_how_to_query:
            follow_up_queries.append(f"{original_query} tutorial")
            follow_up_queries.append(f"{original_query} guide")
            follow_up_queries.append(f"{original_query} steps")
        
        if is_comparison_query:
            follow_up_queries.append(f"{original_query} comparison")
            follow_up_queries.append(f"{original_query} differences")
            follow_up_queries.append(f"{original_query} similarities")
        
        if is_definition_query:
            follow_up_queries.append(f"{original_query} definition")
            follow_up_queries.append(f"{original_query} meaning")
            follow_up_queries.append(f"{original_query} examples")
        
        # Add general enhancement queries based on result analysis
        entity_types = self._analyze_entity_types(results)
        for entity_type in entity_types:
            follow_up_queries.append(f"{original_query} {entity_type}")
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for query in follow_up_queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)
        
        return unique_queries
    
    def _analyze_entity_types(self, results: List[Dict[str, str]]) -> List[str]:
        """
        Analyze search results to determine likely entity types.
        
        Args:
            results: Search results to analyze
            
        Returns:
            List of likely entity types
        """
        # Count occurrences of certain words that suggest entity types
        entity_words = {
            "people": 0,
            "places": 0,
            "products": 0,
            "companies": 0,
            "examples": 0,
            "guides": 0,
            "articles": 0,
            "resources": 0,
            "reviews": 0,
        }
        
        for result in results:
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            text = f"{title} {snippet}"
            
            for entity_type in entity_words:
                # Check singular and plural forms
                singular = entity_type[:-1] if entity_type.endswith("s") else entity_type
                if singular in text or entity_type in text:
                    entity_words[entity_type] += 1
        
        # Return the top entity types
        sorted_entities = sorted(entity_words.items(), key=lambda x: x[1], reverse=True)
        return [entity for entity, count in sorted_entities if count > 0][:3]
    
    def _sort_results_by_relevance(self, 
                                  results: List[Dict[str, str]], 
                                  query: str,
                                  key_terms: Set[str]) -> List[Dict[str, str]]:
        """
        Sort search results by estimated relevance.
        
        Args:
            results: Search results to sort
            query: The original search query
            key_terms: Key terms extracted from initial results
            
        Returns:
            Sorted list of search results
        """
        # Calculate relevance scores
        scored_results = []
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        for result in results:
            score = 0
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            url = result.get("url", "").lower()
            
            # Score based on query terms in title
            for term in query_terms:
                if term in title:
                    score += 5
                if term in snippet:
                    score += 2
                if term in url:
                    score += 1
            
            # Score based on key terms
            for term in key_terms:
                if term in title:
                    score += 3
                if term in snippet:
                    score += 1
            
            # Bonus for list content if query needs a list
            if self._needs_list_content(query):
                if any(term in title.lower() for term in ["list", "top", "best", "comprehensive"]):
                    score += 8
                if re.search(r'\b\d+\s+', title):  # Title contains a number followed by space
                    score += 5
            
            scored_results.append((score, result))
        
        # Sort by score (descending)
        scored_results.sort(reverse=True, key=lambda x: x[0])
        
        # Return just the results
        return [result for score, result in scored_results]
    
    def _extract_domain(self, url: str) -> str:
        """
        Extract the domain from a URL.
        
        Args:
            url: The URL
            
        Returns:
            The domain
        """
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            # Simple fallback
            parts = url.split("/")
            if len(parts) > 2:
                return parts[2]
            return url
    
    def _needs_list_content(self, query: str) -> bool:
        """
        Determine if the query is asking for a list of items.
        
        Args:
            query: The search query
            
        Returns:
            True if the query needs list content, False otherwise
        """
        list_indicators = [
            "list", "top", "best", "examples", "ways to", 
            "things", "items", "reasons", "methods", "techniques",
            "tips", "ideas", "options", "alternatives", "types"
        ]
        
        # Check for numbers which often indicate list requests
        has_number = bool(re.search(r'\b\d+\b', query))
        
        for indicator in list_indicators:
            if indicator in query.lower():
                return True
        
        return has_number
    
    def _contains_list(self, text: str) -> bool:
        """
        Check if the text contains a list.
        
        Args:
            text: The text to check
            
        Returns:
            True if the text contains a list, False otherwise
        """
        # Check for numbered lists (1., 2., etc.)
        has_numbered_list = bool(re.search(r'\b\d+\.\s+\w+', text))
        
        # Check for bullet lists
        has_bullet_list = bool(re.search(r'[-•*]\s+\w+', text))
        
        # Check for list keywords
        list_indicators = ["list", "top", "following", "items", "examples"]
        has_list_keywords = any(indicator in text.lower() for indicator in list_indicators)
        
        return has_numbered_list or has_bullet_list or has_list_keywords
    
    def _generalize_query(self, query: str) -> str:
        """
        Create a more general version of the query.
        
        Args:
            query: The specific query
            
        Returns:
            A more general version of the query
        """
        # Remove specific numbers
        query = re.sub(r'\b\d+\b', '', query)
        
        # Remove list-specific terms
        for term in ["list of", "top", "best", "most popular"]:
            query = query.replace(term, "")
        
        # Add "about" to make it more general
        query = f"about {query.strip()}"
        
        return query
    
    def format_research_for_prompt(self, research_data: Dict[str, Any]) -> str:
        """
        Format research data for inclusion in an LLM prompt.
        
        Args:
            research_data: Research data from the research method
            
        Returns:
            Formatted research text for inclusion in an LLM prompt
        """
        prompt_parts = [
            f"WEB RESEARCH RESULTS",
            f"Query: '{research_data['query']}'",
            "\n===== SEARCH RESULTS =====\n"
        ]
        
        # Include the most relevant search results
        for i, result in enumerate(research_data.get("search_results", [])[:20], 1):
            prompt_parts.append(
                f"[{i}] {result.get('title', 'No title')}\n"
                f"URL: {result.get('url', 'No URL')}\n"
                f"Snippet: {result.get('snippet', 'No snippet')}\n"
            )
        
        prompt_parts.append("\n===== WEBPAGE CONTENT =====\n")
        
        # Format content extracts
        for i, item in enumerate(research_data.get("content", []), 1):
            # Truncate content to a reasonable size but larger for list content
            content = item.get("content", "")
            is_list_content = self._contains_list(content)
            
            # Allow more content for lists
            max_length = 3000 if is_list_content else 1500
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            prompt_parts.append(
                f"[SOURCE {i}] {item.get('title', 'No title')}\n"
                f"URL: {item.get('url', 'No URL')}\n\n"
                f"{content}\n"
            )
        
        # Add a reminder about citing sources
        prompt_parts.append(
            "\n===== INSTRUCTIONS =====\n"
            "When using the information above, please cite the sources by referencing the [SOURCE X] numbers."
        )
        
        return "\n".join(prompt_parts)