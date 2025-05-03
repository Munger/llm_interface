"""
Ollama LLM client implementation.

This module provides a client for interacting with the Ollama API.
"""

import json
import os
import re
import requests
import uuid
from typing import Dict, List, Optional, Any, Union

from llm_interface.config import Config
from llm_interface.llm.base import BaseLLMClient, LLMSession
from llm_interface.session.manager import FileSessionManager


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


class OllamaSession(LLMSession):
    """
    Ollama chat session.
    
    Maintains conversation history and provides methods for interacting
    with the LLM within a session context.
    """
    
    def __init__(self, 
                 client: OllamaClient, 
                 session_id: str, 
                 config: Config,
                 history: Optional[List[Dict[str, str]]] = None):
        """
        Initialize a chat session.
        
        Args:
            client: The OllamaClient instance
            session_id: The session identifier
            config: The configuration instance
            history: Optional chat history
        """
        self.client = client
        self.session_id = session_id
        self.config = config
        self.history = history or []
        self.session_manager = FileSessionManager(config)
    
    def chat(self, message: str, model: Optional[str] = None, debug: bool = False, **kwargs) -> str:
        """
        Send a message to the LLM within this session.
        
        Args:
            message: The message to send
            model: The model to use (defaults to default_model in config)
            debug: Whether to print debug information
            **kwargs: Additional parameters to pass to the Ollama API
            
        Returns:
            The LLM's response as a string
        """
        # Add user message to history
        self.history.append({"role": "user", "content": message})
        
        # Get response from Ollama - use explicit model if provided and pass debug flag
        response = self.client.chat(self.history, model=model, debug=debug, **kwargs)
        
        # Add assistant response to history
        self.history.append({"role": "assistant", "content": response})
        
        # Limit history size
        max_history = self.config["max_history"]
        if max_history > 0 and len(self.history) > max_history * 2:
            # Keep system messages and the most recent messages
            system_messages = [msg for msg in self.history if msg["role"] == "system"]
            recent_messages = self.history[-max_history * 2:]
            self.history = system_messages + recent_messages
        
        # Save session state
        self.save()
        
        return response
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return self.history.copy()
    
    def clear_history(self) -> None:
        """Clear the conversation history, keeping only system messages."""
        self.history = [msg for msg in self.history if msg["role"] == "system"]
        self.save()
    
    def save(self) -> None:
        """Save the session state."""
        session_data = {
            "history": self.history
        }
        self.session_manager.save(self.session_id, session_data)
    
    def research(self, query: str, debug: bool = False, **kwargs) -> str:
        """
        Perform web research and respond with RAG-enhanced knowledge.
        
        Args:
            query: The research question
            debug: Whether to print debug information
            **kwargs: Additional keyword arguments for research
            
        Returns:
            The LLM's response enhanced with web research
        """
        try:
            from llm_interface.research.web import WebResearcher
            
            if debug:
                print(f"DEBUG - Performing web research for query: {query}")
            
            # Step 1: Ask the LLM to generate search strategies focused on finding specific items
            search_strategy_prompt = (
                f"I need to research to create: \"{query}\"\n\n"
                f"This is a request that may require finding specific items or examples. Please suggest 5-7 specific search queries "
                f"that would help find the exact items requested rather than just general information about the topic. "
                f"Include different phrasings and specific platforms or sources where these items might be found. "
                f"Just list the search queries directly, one per line, starting with a hyphen."
            )
            
            if debug:
                print(f"DEBUG - Asking LLM for search strategies")
            
            strategy_response = self.client.query(search_strategy_prompt, debug=debug)
            suggested_queries = self._extract_search_terms(strategy_response, query)
            
            if debug and suggested_queries:
                print(f"DEBUG - LLM suggested {len(suggested_queries)} search queries: {suggested_queries}")
            
            # Initialize web researcher
            researcher = WebResearcher(self.config)
            
            # Step 2: Perform the primary research with original query
            primary_results = researcher.research(query, debug=debug)
            
            # Step 3: Perform additional research with LLM-suggested queries if available
            all_results = {
                "query": query,
                "search_results": primary_results.get("search_results", []).copy(),
                "content": primary_results.get("content", []).copy(),
                "timestamp": primary_results.get("timestamp", 0)
            }
            
            # Track URLs we've already seen
            seen_urls = {item.get("url", "") for item in all_results["content"]}
            
            # Use LLM-suggested queries for additional research if we need more content
            if suggested_queries and len(all_results["content"]) < 10:
                # Limit to top 3 suggested queries to keep latency reasonable
                for suggested_query in suggested_queries[:3]:
                    if debug:
                        print(f"DEBUG - Researching with LLM-suggested query: {suggested_query}")
                    
                    additional_results = researcher.research(suggested_query, debug=debug)
                    
                    # Add new content that we haven't seen before
                    for item in additional_results.get("content", []):
                        url = item.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_results["content"].append(item)
                    
                    # Add new search results that we haven't seen before
                    for result in additional_results.get("search_results", []):
                        url = result.get("url", "")
                        if url and not any(r.get("url", "") == url for r in all_results["search_results"]):
                            all_results["search_results"].append(result)
                    
                    # If we've found enough content, stop researching
                    if len(all_results["content"]) >= 15:
                        break
            
            # Format research for prompt
            research_context = researcher.format_research_for_prompt(all_results)
            
            if debug:
                print(f"DEBUG - Research context generated ({len(research_context)} chars)")
            
            # General purpose research prompt that informs the LLM about its RAG capabilities
            # and encourages it to focus on fulfilling specific requests
            enhanced_query = (
                f"I need you to create exactly what was requested: {query}\n\n"
                f"I've provided you with web research data specifically to help with this task:\n\n{research_context}\n\n"
                f"IMPORTANT INSTRUCTIONS FOR YOU TO FOLLOW:\n"
                f"1. You have access to web research capabilities that have been used to gather the information above.\n"
                f"2. When asked for a list of specific items (like videos, books, people, etc.), provide the actual list "
                f"of items found in the research, not just general information or guidance on where to find them.\n"
                f"3. Focus on extracting and compiling the specific items requested from the research data.\n"
                f"4. If the research doesn't provide enough items to fulfill the exact request (e.g., 100 items), "
                f"provide as many as you can find in the data, and be clear about how many you found.\n"
                f"5. Cite your sources using the [SOURCE X] references for each item when possible.\n\n"
                f"Based on the provided research data, please fulfill the request: {query}\n"
                f"Be direct and deliver exactly what was asked for, focusing on the specific items requested."
            )
            
            # Use a direct query instead of chat-based approach
            response = self.client.query(enhanced_query, debug=debug, **kwargs)
            
            return response
        
        except ImportError as e:
            if debug:
                print(f"DEBUG - Research module not available: {e}")
            
            # Fallback if research module isn't available
            return self.chat(f"Please answer this question with your knowledge: {query}", debug=debug, **kwargs)
    
    def research_with_react(self, query: str, debug: bool = False, **kwargs) -> str:
        """
        Perform in-depth research using the ReAct pattern.
        
        This method uses Reasoning + Acting to conduct comprehensive
        research on the query topic, using tools as needed.
        
        Args:
            query: The research question
            debug: Whether to print debug information
            **kwargs: Additional keyword arguments
            
        Returns:
            The LLM's response enhanced with ReAct research
        """
        try:
            from llm_interface.research.react import ReActResearcher
            
            if debug:
                print(f"DEBUG - Starting ReAct research for query: {query}")
            
            # Initialize ReAct researcher
            researcher = ReActResearcher(self.client, self.config)
            
            # Conduct research
            research_context = researcher.research(query, debug=debug)
            
            if debug:
                print(f"DEBUG - Research complete, synthesizing results")
            
            # Synthesize findings
            response = researcher.synthesize(research_context, debug=debug)
            
            return response
            
        except ImportError as e:
            if debug:
                print(f"DEBUG - ReAct module not available: {e}")
            
            # Fallback to regular research
            return self.research(query, debug=debug, **kwargs)
    
    def _extract_search_terms(self, llm_response: str, original_query: str) -> List[str]:
        """
        Extract potential search terms from LLM's verbose response.
        
        Args:
            llm_response: The LLM's response text
            original_query: The original research query
            
        Returns:
            List of extracted search terms
        """
        suggested_terms = []
        
        # Look for list items with hyphens, bullets, or numbers
        lines = llm_response.split('\n')
        for line in lines:
            line = line.strip()
            # Match lines starting with hyphens, bullets, numbers, etc.
            if re.match(r'^[-*•]|\d+\.|\d+\)', line):
                # Extract the actual term (removing the prefix)
                term = re.sub(r'^[-*•]|\d+\.|\d+\)\s*', '', line).strip()
                # Skip very short terms, quotes, and duplicates of original query
                if len(term) > 5 and term.lower() != original_query.lower():
                    suggested_terms.append(term)
        
        # If no list items found, try to extract phrases using more generic patterns
        if not suggested_terms:
            # Look for quoted phrases
            quoted_phrases = re.findall(r'["\']([^"\']+)["\']', llm_response)
            for phrase in quoted_phrases:
                if len(phrase) > 5 and phrase.lower() != original_query.lower():
                    suggested_terms.append(phrase)
            
            # Look for phrases after certain keywords
            keyword_phrases = re.findall(r'(?:try|query|search|research|use|topic|explore|investigate)\s+["\'"]?([^.,;:"\'\n]{5,})["\'"]?', llm_response, re.IGNORECASE)
            for phrase in keyword_phrases:
                phrase = phrase.strip()
                if len(phrase) > 5 and phrase.lower() != original_query.lower():
                    suggested_terms.append(phrase)
        
        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in suggested_terms:
            normalized_term = term.lower()
            if normalized_term not in seen:
                seen.add(normalized_term)
                unique_terms.append(term)
        
        return unique_terms