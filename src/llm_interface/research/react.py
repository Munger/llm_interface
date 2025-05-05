"""
ReAct (Reasoning + Acting) implementation for LLM research.

This module provides a research system that enables LLMs to
think about what they know, what they need to find out, and
take actions to gather that information.
"""

import re
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from llm_interface.config import Config
from llm_interface.tools.base import registry as tool_registry


class ReActResearcher:
    """
    ReAct-based research system.
    
    This class implements the Reasoning + Acting pattern for
    conducting in-depth research using LLMs and tools.
    """
    
    def __init__(self, llm_client, config: Optional[Config] = None):
        """
        Initialize the ReAct researcher.
        
        Args:
            llm_client: LLM client for queries
            config: Optional configuration object
        """
        self.llm_client = llm_client
        self.config = config or Config()
        self.max_iterations = self.config.get("react_max_iterations", 5)
    
    def research(self, query: str, debug: bool = False) -> Dict[str, Any]:
        """
        Conduct research using the ReAct pattern.
        
        Args:
            query: The research query
            debug: Whether to print debug information
            
        Returns:
            Research results
        """
        if debug:
            print(f"DEBUG - Starting ReAct research for: {query}")
        
        # Initial thinking step
        thinking_prompt = self._create_thinking_prompt(query)
        thinking_result = self.llm_client.query(thinking_prompt, debug=debug)
        research_needs = self._extract_research_needs(thinking_result)
        
        if debug:
            print(f"DEBUG - Initial research needs: {research_needs}")
        
        # Research context to accumulate findings
        context = {
            "query": query,
            "iterations": [],
            "tools_used": [],
            "findings": []
        }
        
        # ReAct loop
        for iteration in range(self.max_iterations):
            if debug:
                print(f"DEBUG - Starting iteration {iteration + 1}/{self.max_iterations}")
            
            # For each research need, determine and use appropriate tool
            iteration_context = {"needs": [], "actions": [], "observations": []}
            
            for need in research_needs:
                # Skip if we've already researched this need
                if any(finding["need"] == need for finding in context["findings"]):
                    if debug:
                        print(f"DEBUG - Skipping already researched need: {need}")
                    continue
                
                # Determine which tool to use
                tool_selection_prompt = self._create_tool_selection_prompt(need, tool_registry.list_tools())
                tool_selection_result = self.llm_client.query(tool_selection_prompt, debug=debug)
                tool_name, params = self._extract_tool_selection(tool_selection_result)
                
                # Fix for empty query parameters - use the research need as the query if empty
                if tool_name in ['web_search', 'search_and_read', 'find_list'] and (
                    'query' not in params or not params['query'].strip()
                ):
                    # Create a search query from the research need
                    search_query = self._create_search_query_from_need(need)
                    params['query'] = search_query
                    if debug:
                        print(f"DEBUG - Empty query detected. Using research need as query: {search_query}")
                
                # Fix parameter names for search_and_read
                if tool_name == 'search_and_read':
                    # Convert num_results to max_results
                    if 'num_results' in params:
                        params['max_results'] = params.pop('num_results')
                    
                    # Remove unsupported parameters
                    for param in list(params.keys()):
                        if param not in ['query', 'max_results']:
                            params.pop(param, None)
                
                if debug:
                    print(f"DEBUG - Selected tool: {tool_name}, params: {params}")
                
                # Execute the tool
                try:
                    result = tool_registry.execute_tool(tool_name, **params)
                    context["tools_used"].append(tool_name)
                    
                    # Add to iteration context
                    iteration_context["needs"].append(need)
                    iteration_context["actions"].append({"tool": tool_name, "params": params})
                    iteration_context["observations"].append(result)
                    
                    # Add finding
                    context["findings"].append({
                        "need": need,
                        "tool": tool_name,
                        "result": result
                    })
                    
                    if debug:
                        print(f"DEBUG - Tool execution successful")
                
                except Exception as e:
                    if debug:
                        print(f"DEBUG - Tool execution failed: {e}")
                    
                    # Record the failure but continue with research
                    iteration_context["observations"].append(
                        {"error": f"Tool execution failed: {str(e)}"}
                    )
                    
                    # Try to fallback to web_search if another tool failed
                    if tool_name != 'web_search':
                        try:
                            if debug:
                                print(f"DEBUG - Attempting fallback to web_search")
                            search_query = self._create_search_query_from_need(need)
                            result = tool_registry.execute_tool('web_search', query=search_query)
                            
                            # Add to context
                            context["tools_used"].append('web_search')
                            iteration_context["actions"].append({"tool": 'web_search', "params": {"query": search_query}})
                            iteration_context["observations"].append(result)
                            
                            # Add finding
                            context["findings"].append({
                                "need": need,
                                "tool": 'web_search',
                                "result": result
                            })
                            
                            if debug:
                                print(f"DEBUG - Fallback to web_search successful")
                                
                        except Exception as fallback_err:
                            if debug:
                                print(f"DEBUG - Fallback to web_search failed: {fallback_err}")
            
            # Add iteration to context
            context["iterations"].append(iteration_context)
            
            # Check if we have enough information
            evaluation_prompt = self._create_evaluation_prompt(query, context)
            evaluation_result = self.llm_client.query(evaluation_prompt, debug=debug)
            is_complete, missing_info = self._extract_completion_status(evaluation_result)
            
            if debug:
                if is_complete:
                    print(f"DEBUG - Research complete")
                else:
                    print(f"DEBUG - Research incomplete. Missing: {missing_info}")
            
            if is_complete:
                if debug:
                    print(f"DEBUG - Research complete after {iteration + 1} iterations")
                break
            
            # Generate new research needs based on missing information
            research_needs = missing_info
            
            if not research_needs:
                # If no specific missing info was identified but research is not complete,
                # generate new research needs based on what we've learned so far
                thinking_prompt = self._create_iteration_thinking_prompt(query, context)
                thinking_result = self.llm_client.query(thinking_prompt, debug=debug)
                research_needs = self._extract_research_needs(thinking_result)
            
            if debug:
                print(f"DEBUG - New research needs: {research_needs}")
        
        # Add timestamp
        context["timestamp"] = time.time()
        
        return context
    
    def synthesize(self, research_context: Dict[str, Any], debug: bool = False) -> str:
        """
        Synthesize research findings into a coherent response.
        
        Args:
            research_context: The research context from research method
            debug: Whether to print debug information
            
        Returns:
            Synthesized research response
        """
        synthesis_prompt = self._create_synthesis_prompt(research_context)
        return self.llm_client.query(synthesis_prompt, debug=debug)
    
    def _create_thinking_prompt(self, query: str) -> str:
        """Create prompt for initial thinking."""
        available_tools = tool_registry.list_tools()
        tool_descriptions = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in available_tools
        ])
        
        return (
            f"I need to research: {query}\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            f"Based on this query, what specific information do I need to find out? "
            f"List 3-5 specific research needs or questions that will help answer this query comprehensively.\n\n"
            f"Format your response as a numbered list (1., 2., etc.) of specific questions or information needs."
        )
    
    def _extract_research_needs(self, thinking_result: str) -> List[str]:
        """Extract research needs from thinking result."""
        # Look for numbered or bulleted items
        needs = []
        
        # Match numbered items (1. Item)
        numbered_pattern = r'\d+\.\s+(.*?)(?=(?:\d+\.)|$)'
        numbered_matches = re.findall(numbered_pattern, thinking_result, re.DOTALL)
        needs.extend([match.strip() for match in numbered_matches if match.strip()])
        
        # Match bulleted items (- Item or * Item)
        if not needs:
            bulleted_pattern = r'[-*]\s+(.*?)(?=(?:[-*])|$)'
            bulleted_matches = re.findall(bulleted_pattern, thinking_result, re.DOTALL)
            needs.extend([match.strip() for match in bulleted_matches if match.strip()])
        
        # If no structured items found, try to split by sentences or newlines
        if not needs:
            # Try splitting by double newlines first
            lines = thinking_result.split('\n\n')
            if len(lines) > 1:
                needs = [line.strip() for line in lines if line.strip()]
            else:
                # Try splitting by sentences
                sentences = thinking_result.split('.')
                needs = [s.strip() + '.' for s in sentences if len(s.strip()) > 10]
        
        return needs
    
    def _create_tool_selection_prompt(self, need: str, available_tools: List[Dict[str, str]]) -> str:
        """Create prompt for tool selection."""
        tool_descriptions = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in available_tools
        ])
        
        return (
            f"I need to find information about: {need}\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            f"Which tool should I use for this specific information need, and with what parameters?\n\n"
            f"Select exactly one tool and specify the parameters to use.\n"
            f"Format your response as follows:\n"
            f"Tool: [tool_name]\n"
            f"Parameters: {{\n"
            f"  \"param1\": \"value1\",\n"
            f"  \"param2\": \"value2\"\n"
            f"}}\n\n"
            f"IMPORTANT: For search tools like web_search or search_and_read, you MUST provide a specific, "
            f"non-empty search query that includes relevant keywords from the research need."
        )
    
    def _extract_tool_selection(self, tool_selection_result: str) -> Tuple[str, Dict[str, Any]]:
        """Extract tool selection from LLM response."""
        # Extract tool name
        tool_pattern = r'Tool:\s*(\w+)'
        tool_match = re.search(tool_pattern, tool_selection_result)
        
        if not tool_match:
            # Default to web search if no tool specified
            return "web_search", {"query": ""}
        
        tool_name = tool_match.group(1)
        
        # Extract parameters
        params = {}
        params_pattern = r'Parameters:\s*{(.*?)}'
        params_match = re.search(params_pattern, tool_selection_result, re.DOTALL)
        
        if params_match:
            # Try to parse JSON
            try:
                params_str = "{" + params_match.group(1) + "}"
                params = json.loads(params_str)
            except:
                # If JSON parsing fails, try to extract key-value pairs
                param_pairs = re.findall(r'"(\w+)":\s*"([^"]*)"', params_match.group(1))
                for key, value in param_pairs:
                    params[key] = value
        
        # Handle cases where parameters are listed in a different format
        if not params:
            # Look for key-value pairs in the format "param: value"
            param_pairs = re.findall(r'(\w+):\s*"?([^",\n]+)"?', tool_selection_result)
            for key, value in param_pairs:
                if key.lower() != "tool" and key.lower() != "parameters":
                    params[key] = value
        
        # Ensure required parameters for the selected tool
        if tool_name == "web_search" and "query" not in params:
            # Extract potential query from the context
            query_pattern = r'(?:query|search for|research):\s*"?([^"]+)"?'
            query_match = re.search(query_pattern, tool_selection_result, re.IGNORECASE)
            
            if query_match:
                params["query"] = query_match.group(1)
            else:
                # If no query found, use an empty string
                params["query"] = ""
        
        elif tool_name == "fetch_webpage" and "url" not in params:
            # Extract URL from the context
            url_pattern = r'(?:url|webpage|site|link):\s*"?(https?://[^\s"]+)"?'
            url_match = re.search(url_pattern, tool_selection_result, re.IGNORECASE)
            
            if url_match:
                params["url"] = url_match.group(1)
            else:
                # Default to empty URL
                params["url"] = ""
                
        elif tool_name == "search_and_read" and "query" not in params:
            # Extract potential query from the context
            query_pattern = r'(?:query|search for|research):\s*"?([^"]+)"?'
            query_match = re.search(query_pattern, tool_selection_result, re.IGNORECASE)
            
            if query_match:
                params["query"] = query_match.group(1)
            else:
                # If no query found, use an empty string
                params["query"] = ""
        
        return tool_name, params
    
    def _create_search_query_from_need(self, need: str) -> str:
        """
        Create a search query from a research need statement.
        This is used as a fallback when no query is provided.
        
        Args:
            need: Research need statement
            
        Returns:
            Search query string
        """
        # Clean up the need to create a search query
        query = need
        
        # Remove prefixes like "What is" or "How to"
        query = re.sub(r'^(?:what|how|why|when|where|who|which)\s+(?:is|are|does|do|can|should|would|will|has|have)\s+', '', query, flags=re.IGNORECASE)
        
        # Remove question marks
        query = query.replace('?', '')
        
        # Extract key phrases using regex
        key_phrase_pattern = r'"([^"]+)"'
        key_phrases = re.findall(key_phrase_pattern, query)
        
        if key_phrases:
            # Use quoted phrases in the query if found
            return ' '.join(key_phrases)
        
        # Extract key terms (longer words more likely to be important)
        words = re.findall(r'\b\w{4,}\b', query)
        
        # Remove common stop words
        stop_words = {'this', 'that', 'these', 'those', 'what', 'which', 'when', 'where', 'who', 'whose', 'whom', 'how', 'why'}
        keywords = [word for word in words if word.lower() not in stop_words]
        
        # If we don't have enough keywords, use the original query
        if len(keywords) < 3:
            return query
        
        # Otherwise join keywords
        return ' '.join(keywords)
    
    def _create_evaluation_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Create prompt for evaluating research progress."""
        # Format findings in a readable way
        findings_text = ""
        for i, finding in enumerate(context.get("findings", []), 1):
            need = finding.get("need", "")
            tool = finding.get("tool", "")
            result = finding.get("result", {})
            
            # Format result based on tool type
            result_text = ""
            if tool == "web_search":
                results = result.get("results", [])
                result_text = f"Found {len(results)} search results"
                if results:
                    result_text += ":\n"
                    for j, res in enumerate(results[:3], 1):
                        result_text += f"  {j}. {res.get('title', '')}\n"
                    if len(results) > 3:
                        result_text += f"  ...and {len(results) - 3} more results\n"
            
            elif tool == "fetch_webpage":
                content = result.get("content", "")
                result_text = f"Fetched webpage content ({len(content)} chars)"
            
            elif tool == "search_and_read":
                content = result.get("content", "")
                result_text = f"Searched and read content ({len(content)} chars)"
            
            else:
                # Generic result formatting
                result_text = str(result)
                if len(result_text) > 100:
                    result_text = result_text[:100] + "..."
            
            findings_text += f"Finding {i}:\n"
            findings_text += f"- Need: {need}\n"
            findings_text += f"- Tool: {tool}\n"
            findings_text += f"- Result: {result_text}\n\n"
        
        return (
            f"I'm researching: {query}\n\n"
            f"So far, I've gathered the following information:\n\n{findings_text}\n"
            f"Based on this information, do I have enough to provide a comprehensive answer to the original query?\n\n"
            f"If yes, explain why the research is sufficient.\n"
            f"If no, list the specific information that is still missing and needs to be researched.\n\n"
            f"Format your response as follows:\n"
            f"Research complete: [Yes/No]\n"
            f"Reasoning: [Your explanation]\n"
            f"Missing information: [List of missing information needs, if any]"
        )
    
    def _extract_completion_status(self, evaluation_result: str) -> Tuple[bool, List[str]]:
        """Extract research completion status and missing information."""
        # Check if research is complete
        complete_pattern = r'Research complete:\s*(Yes|No)'
        complete_match = re.search(complete_pattern, evaluation_result, re.IGNORECASE)
        
        is_complete = False
        if complete_match:
            is_complete = complete_match.group(1).lower() == "yes"
        else:
            # If no explicit completion status, check for keywords
            is_complete = "research is complete" in evaluation_result.lower() or "sufficient information" in evaluation_result.lower()
        
        # Extract missing information
        missing_info = []
        
        if not is_complete:
            # Look for a "Missing information" section
            missing_section_pattern = r'Missing information:(.*?)(?:$|(?:\n\n))'
            missing_section_match = re.search(missing_section_pattern, evaluation_result, re.DOTALL)
            
            if missing_section_match:
                missing_section = missing_section_match.group(1).strip()
                
                # Look for numbered or bulleted items
                item_pattern = r'(?:^|\n)\s*(?:\d+\.|-|\*)\s+(.*?)(?=(?:\n\s*(?:\d+\.|-|\*))|$)'
                items = re.findall(item_pattern, missing_section, re.DOTALL)
                
                if items:
                    missing_info = [item.strip() for item in items if item.strip()]
                else:
                    # If no structured items, split by newlines
                    lines = missing_section.split('\n')
                    missing_info = [line.strip() for line in lines if line.strip()]
        
        return is_complete, missing_info
    
    def _create_iteration_thinking_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Create prompt for thinking about next research steps."""
        # Summarize findings so far
        findings_summary = ""
        for i, finding in enumerate(context.get("findings", []), 1):
            need = finding.get("need", "")
            findings_summary += f"{i}. {need}\n"
        
        return (
            f"I'm researching: {query}\n\n"
            f"So far, I've investigated these questions:\n{findings_summary}\n"
            f"Based on what I've learned, what additional information should I look for next to complete my research?\n\n"
            f"List 2-3 specific new research questions that would help fill in the gaps in my current knowledge.\n\n"
            f"Format your response as a numbered list of specific questions or information needs."
        )
    
    def _create_synthesis_prompt(self, research_context: Dict[str, Any]) -> str:
        """Create prompt for synthesizing research findings."""
        query = research_context.get("query", "")
        
        # Format findings in a readable way
        findings_text = ""
        
        # Check if we actually found any useful information
        useful_findings = [f for f in research_context.get("findings", []) 
                           if f.get("tool") in ["web_search", "fetch_webpage", "search_and_read", "find_list"] and 
                              isinstance(f.get("result", {}), dict)]
                              
        has_useful_info = len(useful_findings) > 0
        
        if has_useful_info:
            for i, finding in enumerate(research_context.get("findings", []), 1):
                need = finding.get("need", "")
                tool = finding.get("tool", "")
                result = finding.get("result", {})
                
                findings_text += f"Source {i}: {need}\n"
                
                # Format result based on tool type
                if tool == "web_search":
                    results = result.get("results", [])
                    for j, res in enumerate(results[:5], 1):
                        findings_text += f"- {res.get('title', '')}: {res.get('snippet', '')}\n"
                        findings_text += f"  URL: {res.get('url', '')}\n"
                
                elif tool == "fetch_webpage":
                    url = result.get("url", "")
                    content = result.get("content", "")
                    findings_text += f"- Webpage: {url}\n"
                    
                    # Add a brief excerpt of the content
                    if content:
                        content_excerpt = content[:500] + "..." if len(content) > 500 else content
                        findings_text += f"- Content excerpt: {content_excerpt}\n"
                
                elif tool == "search_and_read" or tool == "find_list":
                    url = result.get("url", "")
                    title = result.get("title", "")
                    content = result.get("content", "")
                    
                    findings_text += f"- Source: {title}\n"
                    findings_text += f"- URL: {url}\n"
                    
                    # Add a brief excerpt of the content
                    if content:
                        content_excerpt = content[:500] + "..." if len(content) > 500 else content
                        findings_text += f"- Content excerpt: {content_excerpt}\n"
                
                findings_text += "\n"
        else:
            findings_text = "No relevant information was found during the research process. Please provide a response based on your general knowledge while acknowledging the limitations of the research."
        
        if has_useful_info:
            return (
                f"I've researched the question: {query}\n\n"
                f"Based on my research, I have gathered the following information:\n\n{findings_text}\n"
                f"Using this information, provide a comprehensive, well-organized answer to the original question.\n\n"
                f"Important guidelines:\n"
                f"1. When referencing information from a source, include the source URL directly in your text.\n"
                f"2. If there are conflicting pieces of information, acknowledge them and explain which seems most reliable.\n"
                f"3. If some aspects of the question couldn't be answered by the research, acknowledge those limitations.\n"
                f"4. Start your response with: 'Based on my web research about \"{query}\", here's what I found:'\n\n"
                f"Format your response in a clear, structured way that directly answers the original question."
            )
        else:
            return (
                f"I've researched the question: {query}\n\n"
                f"Unfortunately, my research didn't yield specific information about this topic. "
                f"Based on this outcome, please provide a response that:\n\n"
                f"1. Acknowledges the limitations of the research\n"
                f"2. Explains that specific information about communicating with an HP power supply using an ESP32 wasn't found\n"
                f"3. Offers general information about ESP32 communication capabilities and what might be needed\n"
                f"4. Suggests next steps the user could take to find more specific information\n\n"
                f"Start your response with: 'Based on my web research about \"{query}\", I wasn't able to find specific information.'\n\n"
                f"Format your response in a clear, structured way that acknowledges the limitations while still being helpful."
            )