"""
Research capabilities for Ollama sessions.

This module provides research functionality for enhancing LLM responses
with web search and retrieval-augmented generation.

Author: Tim Hosking (https://github.com/Munger)
"""

import re
import time
from typing import Dict, List, Any, Optional, Union

from llm_interface.config import Config
# Direct import of the prompt_manager module
import llm_interface.config.prompt_manager as prompt_manager


class OllamaResearch:
    """
    Research capabilities for Ollama sessions.
    
    This class provides methods for conducting web research and
    integrating the results with LLM responses.
    """
    
    def __init__(self, client, config: Optional[Config] = None):
        """
        Initialize the research capabilities.
        
        Args:
            client: The OllamaClient instance
            config: Optional configuration object
        """
        self.client = client
        self.config = config or Config()
    
    def perform_research(self, query: str, session, debug: bool = False, **kwargs) -> str:
        """
        Perform web research and respond with RAG-enhanced knowledge.
        
        Args:
            query: The research question
            session: The OllamaSession instance
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
            # Use the prompt manager to get the search strategy prompt
            search_strategy_prompt = prompt_manager.format_prompt("research", "search_strategy_prompt", query=query)
            
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
            
            # Check if we have any useful findings
            valid_findings = len(all_results["content"]) > 0 and \
                             any(len(item.get("content", "")) > 0 for item in all_results["content"])
            
            # Store URLs for future reference
            all_urls = []
            for item in all_results["content"]:
                url = item.get("url", "")
                title = item.get("title", "")
                if url and not any(source.get("url") == url for source in all_urls):
                    all_urls.append({"url": url, "title": title or url})
            
            # Store in the _research_urls attribute
            if all_urls:
                if not hasattr(session, '_research_urls'):
                    session._research_urls = []
                
                for i, source in enumerate(all_urls, 1):
                    session._research_urls.append({
                        "index": i, 
                        "title": source.get("title", "Unknown"), 
                        "url": source.get("url", "")
                    })
            
            # Record research request in research history
            research_entry = {
                "query": query,
                "timestamp": time.time(),
                "sources": all_urls,
                "findings": research_context
            }
            
            session.last_research_time = time.time()
            session.research_history.append(research_entry)
            
            # Store the research query
            session._last_research_query = query
            
            # Create sources section for the system message
            sources_list = ""
            for i, source in enumerate(all_urls, 1):
                source_title = source.get('title', 'Unknown title')
                source_url = source.get('url', '')
                sources_list += prompt_manager.format_prompt("web_research", "format_source_entry",
                                            index=i, title=source_title, url=source_url)
            
            sources_section = prompt_manager.format_prompt("web_research", "format_sources_section",
                                          sources_list=sources_list)
            
            # Add system message based on whether valid findings were found
            if valid_findings:
                # Use the prompt manager to get the system message template
                research_system_msg = prompt_manager.format_prompt("research", "system_message", 
                                                   query=query,
                                                   findings_text=research_context,
                                                   sources_section=sources_section)
            else:
                # Use the prompt manager to get the system message template for no results
                research_system_msg = prompt_manager.format_prompt("research", "system_message_no_results", query=query)
            
            # Add the research system message to history
            session.history.append({"role": "system", "content": research_system_msg})
            
            # Generate the prompt for the LLM
            if valid_findings:
                # Use the prompt manager to get the enhanced query template
                enhanced_query = prompt_manager.format_prompt("research", "enhanced_query_template", query=query)
            else:
                # Use the prompt manager to get the no results query template
                enhanced_query = prompt_manager.format_prompt("research", "no_results_query_template", query=query)
            
            # Use a direct query instead of chat-based approach
            response = self.client.query(enhanced_query, debug=debug, **kwargs)
            
            # Format the final response
            if valid_findings:
                # Create sources list for the response
                sources_list = ""
                for i, source in enumerate(all_urls, 1):
                    source_title = source.get('title', 'Unknown title')
                    source_url = source.get('url', '')
                    sources_list += f"{i}. {source_title}: {source_url}\n"
                
                # Use the prompt manager to format the response
                immediate_response = prompt_manager.format_prompt("response_formats", "research_response",
                                                 query=query,
                                                 response_content=response,
                                                 num_sources=len(all_urls),
                                                 sources_list=sources_list)
            else:
                # Use the prompt manager to format the response for no results
                immediate_response = prompt_manager.format_prompt("response_formats", "no_results_response",
                                                 query=query,
                                                 response_content=response)
            
            # Save the assistant's response to history
            session.history.append({"role": "assistant", "content": immediate_response})
            
            # Save session state
            session.save()
            
            return immediate_response
            
        except ImportError as e:
            if debug:
                print(f"DEBUG - Research module not available: {e}")
            
            # Fallback if research module isn't available
            return session.chat(f"Please answer this question with your knowledge: {query}", debug=debug, **kwargs)
    
    def perform_react_research(self, query: str, session, debug: bool = False, **kwargs) -> str:
        """
        Perform in-depth research using the ReAct pattern.
        
        This method uses Reasoning + Acting to conduct comprehensive
        research on the query topic, using tools as needed.
        
        Args:
            query: The research question
            session: The OllamaSession instance
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
                print(f"DEBUG - Research complete, synthesising results")
            
            # Extract all URLs from the research
            all_urls = []
            valid_findings = False
            
            # Format findings for the LLM's reference
            findings_text = ""
            for i, finding in enumerate(research_context.get("findings", []), 1):
                need = finding.get("need", "")
                tool = finding.get("tool", "")
                result = finding.get("result", {})
                
                # Check if we have any valid search results
                if (tool == "web_search" and result.get("results") and len(result.get("results", [])) > 0) or \
                   (tool == "search_and_read" and result.get("content")) or \
                   (tool == "fetch_webpage" and result.get("content")):
                    valid_findings = True
                
                # Format finding based on tool type
                tool_specific_content = ""
                if tool == "web_search":
                    results = result.get("results", [])
                    search_results_content = ""
                    for j, res in enumerate(results[:5], 1):
                        url = res.get("url", "")
                        title = res.get("title", "")
                        snippet = res.get("snippet", "")
                        
                        # Format the search result using the template
                        result_format = prompt_manager.format_prompt("web_research", "format_search_result", 
                                                     i=i, j=j, title=title, snippet=snippet, url=url)
                        search_results_content += result_format
                        
                        # Add URL to list
                        if url and not any(source.get("url") == url for source in all_urls):
                            all_urls.append({"url": url, "title": title})
                    
                    # Format the web search results
                    tool_specific_content = prompt_manager.format_prompt("web_research", "format_web_search_results",
                                                        count=len(results),
                                                        results_content=search_results_content)
                
                elif tool == "fetch_webpage" or tool == "search_and_read":
                    url = result.get("url", "")
                    title = result.get("title", "")
                    content = result.get("content", "")
                    
                    # Add URL to list
                    if url and not any(source.get("url") == url for source in all_urls):
                        all_urls.append({"url": url, "title": title or url})
                    
                    if content:
                        # Include a brief excerpt of the content
                        content_summary = content[:500] + "..." if len(content) > 500 else content
                        tool_specific_content = prompt_manager.format_prompt("web_research", "format_webpage_content",
                                                           i=i, title=title, url=url, content_summary=content_summary)
                
                # Format the finding using the template
                finding_format = prompt_manager.format_prompt("web_research", "format_findings",
                                             index=i, need=need, tool=tool, 
                                             tool_specific_content=tool_specific_content)
                findings_text += finding_format
            
            # Record research request in research history
            research_entry = {
                "query": query,
                "timestamp": time.time(),
                "sources": all_urls,
                "findings": findings_text
            }
            
            session.last_research_time = time.time()
            session.research_history.append(research_entry)
            
            # Store the last research query
            session._last_research_query = query
            
            # Create dedicated section for source reference
            sources_list = ""
            for i, source in enumerate(all_urls, 1):
                source_title = source.get('title', 'Unknown title')
                source_url = source.get('url', '')
                
                # Format source entry using the template
                source_entry = prompt_manager.format_prompt("web_research", "format_source_entry",
                                          index=i, title=source_title, url=source_url)
                sources_list += source_entry
                
                # Also save URL separately to make sure we can reference it later
                if not hasattr(session, '_research_urls'):
                    session._research_urls = []
                
                session._research_urls.append({"index": i, "title": source_title, "url": source_url})
            
            # Format the sources section using the template
            sources_section = prompt_manager.format_prompt("web_research", "format_sources_section",
                                          sources_list=sources_list)
            
            # Create message based on whether valid findings were found
            if valid_findings:
                # Use the prompt manager to get the system message template
                system_msg = prompt_manager.format_prompt("research", "system_message", 
                                           query=query, 
                                           findings_text=findings_text,
                                           sources_section=sources_section)
            else:
                # Use the prompt manager to get the system message template for no results
                system_msg = prompt_manager.format_prompt("research", "system_message_no_results", 
                                           query=query)
            
            # Add to history as a system message
            session.history.append({"role": "system", "content": system_msg})
            
            # Synthesise findings
            synthesised_response = researcher.synthesize(research_context, debug=debug)
            
            # Create sources list for the response
            sources_list = ""
            for i, source in enumerate(all_urls, 1):
                source_title = source.get('title', 'Unknown title')
                source_url = source.get('url', '')
                sources_list += f"{i}. {source_title}: {source_url}\n"
            
            # Create an immediate response based on whether we had valid findings
            if valid_findings:
                # Use the prompt manager to format the response
                immediate_response = prompt_manager.format_prompt("response_formats", "research_response",
                                                 query=query,
                                                 response_content=synthesised_response,
                                                 num_sources=len(all_urls),
                                                 sources_list=sources_list)
            else:
                # Use the prompt manager to format the response for no results
                immediate_response = prompt_manager.format_prompt("response_formats", "no_results_response",
                                                 query=query,
                                                 response_content=synthesised_response)
            
            # Save the assistant's response to history
            session.history.append({"role": "assistant", "content": immediate_response})
            
            # Save session state
            session.save()
            
            return immediate_response
            
        except ImportError as e:
            if debug:
                print(f"DEBUG - ReAct module not available: {e}")
            
            # Fallback to regular research
            return self.perform_research(query, session, debug=debug, **kwargs)
    
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