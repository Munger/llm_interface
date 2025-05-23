"""
Command-line interface for the LLM Interface.

This module provides a CLI for interacting with the LLM Interface.
"""

import os
import sys
import click
import json
import traceback
from typing import Optional

from llm_interface import LLMClient
from llm_interface.config import Config


@click.group()
@click.option(
    "--model", "-m",
    help="Ollama model to use",
)
@click.option(
    "--host", "-h",
    help="Ollama host",
)
@click.option(
    "--port", "-p",
    type=int,
    help="Ollama port",
)
@click.option(
    "--config", "-c",
    help="Path to config file",
)
@click.option(
    "--debug", "-d",
    is_flag=True,
    help="Enable debug mode",
)
@click.pass_context
def cli(ctx, model, host, port, config, debug):
    """LLM Interface CLI - Interact with locally hosted LLMs."""
    # Initialize config
    config_obj = Config()
    
    # Set debug flag
    ctx.obj = {
        "debug": debug
    }
    
    try:
        # Override with config file if provided
        if config and os.path.exists(config):
            with open(config, 'r') as f:
                config_override = json.load(f)
                config_obj.update(config_override)
        
        # Initialize client - pass model, host, port directly to ensure they take priority
        client = LLMClient(model=model, host=host, port=port, config_override=config_obj)
        
        # Store in context
        ctx.obj["client"] = client
        ctx.obj["config"] = config_obj
        
        if debug:
            click.echo("Debug mode enabled")
            click.echo(f"Using model: {client.config['default_model']}")
            click.echo(f"Ollama host: {client.config['ollama_host']}:{client.config['ollama_port']}")
    
    except Exception as e:
        if debug:
            click.echo(f"Error during initialisation: {e}", err=True)
            click.echo(traceback.format_exc(), err=True)
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("prompt")
@click.pass_context
def ask(ctx, prompt):
    """Send a single query to the LLM."""
    client = ctx.obj.get("client")
    debug = ctx.obj.get("debug", False)
    
    if not client:
        click.echo("Error: Client not initialised", err=True)
        sys.exit(1)
    
    try:
        if debug:
            click.echo(f"Sending query: {prompt}")
        
        response = client.query(prompt, debug=debug)
        click.echo(response)
        
    except Exception as e:
        if debug:
            click.echo(f"Error during query: {e}", err=True)
            click.echo(traceback.format_exc(), err=True)
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--session", "-s",
    help="Session ID (creates a new session if not provided)",
)
@click.pass_context
def chat(ctx, session):
    """Start an interactive chat session."""
    client = ctx.obj["client"]
    debug = ctx.obj.get("debug", False)
    
    try:
        # Get or create session
        if session and client.session_manager.exists(session):
            session_obj = client.get_session(session)
            click.echo(f"Resuming session {session}")
        else:
            session_obj = client.create_session(session)
            if session:
                click.echo(f"Created new session {session}")
            else:
                click.echo(f"Created new session {session_obj.session_id}")
        
        # Print instructions
        click.echo("Enter your messages (Ctrl+D or type 'exit' to quit):")
        click.echo("Use /research <query> to perform research on a topic")
        
        # Import prompt_toolkit here to avoid dependency if not using chat
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import InMemoryHistory
            
            # Create prompt session with history
            prompt_history = InMemoryHistory()
            prompt_session = PromptSession(history=prompt_history)
            
            # Interactive loop with prompt_toolkit
            while True:
                try:
                    # Get user input with prompt_toolkit
                    prompt = prompt_session.prompt("You> ")
                    
                    # Check for exit command
                    if prompt.lower() in ("exit", "quit", "q"):
                        break
                    
                    # Check for research command
                    if prompt.strip().lower().startswith("/research "):
                        query = prompt[10:].strip()
                        if query:
                            click.echo(f"\nResearching: {query}...")
                            
                            # Key change: Add the research command to conversation history
                            # This makes the LLM aware that research was requested
                            research_request_msg = f"I want to research: {query}"
                            session_obj.add_user_message(research_request_msg)
                            
                            # Show spinner if not in debug mode
                            if not debug:
                                with click.progressbar(length=100, label='Researching') as bar:
                                    # Update progress bar as a simple spinner
                                    for i in range(10):
                                        bar.update(10)
                                        # Perform the actual research when we're at 50%
                                        if i == 5:
                                            research_result = session_obj.research_with_react(query, debug=debug)
                            else:
                                research_result = session_obj.research_with_react(query, debug=debug)
                            
                            # Print the actual research response
                            click.echo(f"\nLLM> {research_result}\n")
                        else:
                            click.echo("Please provide a research query, e.g. /research quantum computing")
                        continue
                    
                    # Send to LLM
                    response = session_obj.chat(prompt, debug=debug)
                    
                    # Print response
                    click.echo("\nLLM> " + response + "\n")
                    
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
                
        except ImportError:
            # Fallback to click.prompt if prompt_toolkit is not available
            if debug:
                click.echo("DEBUG - prompt_toolkit not available, using click.prompt")
                
            # Interactive loop with click.prompt
            while True:
                try:
                    # Get user input
                    prompt = click.prompt("You", prompt_suffix="> ")
                    
                    # Check for exit command
                    if prompt.lower() in ("exit", "quit", "q"):
                        break
                    
                    # Check for research command
                    if prompt.strip().lower().startswith("/research "):
                        query = prompt[10:].strip()
                        if query:
                            click.echo(f"\nResearching: {query}...")
                            
                            # Key change: Add the research command to conversation history
                            # This makes the LLM aware that research was requested
                            research_request_msg = f"I want to research: {query}"
                            session_obj.add_user_message(research_request_msg)
                            
                            # Show spinner if not in debug mode
                            if not debug:
                                with click.progressbar(length=100, label='Researching') as bar:
                                    # Update progress bar as a simple spinner
                                    for i in range(10):
                                        bar.update(10)
                                        # Perform the actual research when we're at 50%
                                        if i == 5:
                                            research_result = session_obj.research_with_react(query, debug=debug)
                            else:
                                research_result = session_obj.research_with_react(query, debug=debug)
                            
                            # Print the actual research response
                            click.echo(f"\nLLM> {research_result}\n")
                        else:
                            click.echo("Please provide a research query, e.g. /research quantum computing")
                        continue
                    
                    # Send to LLM
                    response = session_obj.chat(prompt, debug=debug)
                    
                    # Print response
                    click.echo("\nLLM> " + response + "\n")
                    
                except click.exceptions.Abort:
                    break
        
        click.echo(f"Session {session_obj.session_id} saved.")
        
    except Exception as e:
        if debug:
            click.echo(f"Error: {e}", err=True)
            click.echo(traceback.format_exc(), err=True)
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option(
    "--session", "-s",
    help="Session ID (uses a temporary session if not provided)",
)
@click.pass_context
def research(ctx, query, session):
    """Perform intelligent research with ReAct pattern."""
    client = ctx.obj["client"]
    debug = ctx.obj.get("debug", False)
    
    try:
        # Get or create session
        if session and client.session_manager.exists(session):
            session_obj = client.get_session(session)
            click.echo(f"Using session {session} for research")
        else:
            session_obj = client.create_session(session)
            if session:
                click.echo(f"Created new session {session} for research")
            else:
                click.echo(f"Created temporary session for research")
        
        # Add the research command to conversation history
        research_request_msg = f"I want to research: {query}"
        session_obj.add_user_message(research_request_msg)
                
        # Perform research using ReAct
        click.echo(f"Researching: {query}\n")
        
        # Show spinner if not in debug mode
        if not debug:
            with click.progressbar(length=100, label='Researching') as bar:
                # Update progress bar as a simple spinner
                for i in range(10):
                    bar.update(10)
                    # Perform the actual research when we're at 50%
                    if i == 5:
                        response = session_obj.research_with_react(query, debug=debug)
        else:
            response = session_obj.research_with_react(query, debug=debug)
        
        # Print response
        click.echo("\nResearch Results:\n")
        click.echo(response)
        
    except Exception as e:
        if debug:
            click.echo(f"Error: {e}", err=True)
            click.echo(traceback.format_exc(), err=True)
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def list_sessions(ctx):
    """List all available sessions."""
    client = ctx.obj["client"]
    debug = ctx.obj.get("debug", False)
    
    try:
        sessions = client.list_sessions()
        
        if sessions:
            click.echo("Available sessions:")
            for session_id in sessions:
                click.echo(f"  {session_id}")
        else:
            click.echo("No sessions found.")
        
    except Exception as e:
        if debug:
            click.echo(f"Error: {e}", err=True)
            click.echo(traceback.format_exc(), err=True)
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("session_id")
@click.option(
    "--force", "-f", 
    is_flag=True,
    help="Force deletion without confirmation",
)
@click.pass_context
def delete_session(ctx, session_id, force):
    """Delete a session."""
    client = ctx.obj["client"]
    debug = ctx.obj.get("debug", False)
    
    try:
        if not client.session_manager.exists(session_id):
            click.echo(f"Session {session_id} does not exist.", err=True)
            sys.exit(1)
        
        if not force and not click.confirm(f"Delete session {session_id}?"):
            click.echo("Aborted.")
            return
        
        client.delete_session(session_id)
        click.echo(f"Session {session_id} deleted.")
        
    except Exception as e:
        if debug:
            click.echo(f"Error: {e}", err=True)
            click.echo(traceback.format_exc(), err=True)
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--save", "-s", 
    is_flag=True,
    help="Save config to user config file",
)
@click.pass_context
def show_config(ctx, save):
    """Show current configuration."""
    config = ctx.obj["config"]
    debug = ctx.obj.get("debug", False)
    
    # Show config
    click.echo(json.dumps(config._config, indent=2))
    
    # Save if requested
    if save:
        config.save()
        click.echo(f"Configuration saved to {config.USER_CONFIG_PATH}")


@cli.command()
@click.pass_context
def list_tools(ctx):
    """List available research tools."""
    debug = ctx.obj.get("debug", False)
    
    try:
        # Import tool registry
        from llm_interface.tools.base import registry
        
        tools = registry.list_tools()
        
        if tools:
            click.echo("Available research tools:")
            for tool in tools:
                click.echo(f"  {tool['name']}: {tool['description']}")
        else:
            click.echo("No research tools registered.")
        
    except ImportError as e:
        if debug:
            click.echo(f"Error loading tools: {e}", err=True)
            click.echo(traceback.format_exc(), err=True)
        else:
            click.echo("Error: Tools module not available", err=True)
        sys.exit(1)


def main():
    """CLI entry point."""
    try:
        cli(obj={})
    except Exception as e:
        click.echo(f"Error in main function: {e}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()