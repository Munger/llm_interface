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
            click.echo(f"Error during initialization: {e}", err=True)
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
        click.echo("Error: Client not initialized", err=True)
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
    """Perform research with web search and RAG."""
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
        
        # Perform research
        click.echo(f"Researching: {query}\n")
        
        # Show spinner if not in debug mode
        if not debug:
            with click.progressbar(length=100, label='Researching') as bar:
                # Update progress bar as a simple spinner
                for i in range(10):
                    bar.update(10)
                    # Perform the actual research when we're at 50%
                    if i == 5:
                        response = session_obj.research(query, debug=debug)
        else:
            response = session_obj.research(query, debug=debug)
        
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
def list_models(ctx):
    """List available models from Ollama."""
    client = ctx.obj["client"]
    debug = ctx.obj.get("debug", False)
    
    try:
        # Make a request to Ollama's API to list models
        import requests
        
        base_url = f"http://{client.config['ollama_host']}:{client.config['ollama_port']}"
        url = f"{base_url}/api/tags"
        
        if debug:
            click.echo(f"DEBUG - Requesting models from: {url}")
        
        response = requests.get(url, timeout=client.config["timeout"])
        response.raise_for_status()
        
        models_data = response.json()
        models = models_data.get("models", [])
        
        if models:
            click.echo("Available models:")
            # Calculate column widths for nice formatting
            name_width = max(len(model.get("name", "")) for model in models) + 2
            
            # Print header
            click.echo(f"{'NAME':{name_width}} {'SIZE':<10} {'MODIFIED':<15}")
            
            # Print each model
            for model in models:
                name = model.get("name", "")
                size = model.get("size", 0)
                modified = model.get("modified_at", "")
                
                # Format size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f} KB"
                elif size < 1024 * 1024 * 1024:
                    size_str = f"{size/(1024*1024):.1f} MB"
                else:
                    size_str = f"{size/(1024*1024*1024):.1f} GB"
                
                # Format modified date
                import datetime
                try:
                    dt = datetime.datetime.fromisoformat(modified.replace("Z", "+00:00"))
                    now = datetime.datetime.now(datetime.timezone.utc)
                    diff = now - dt
                    
                    if diff.days < 1:
                        modified_str = "Today"
                    elif diff.days < 2:
                        modified_str = "Yesterday"
                    elif diff.days < 7:
                        modified_str = f"{diff.days} days ago"
                    elif diff.days < 30:
                        modified_str = f"{diff.days // 7} weeks ago"
                    elif diff.days < 365:
                        modified_str = f"{diff.days // 30} months ago"
                    else:
                        modified_str = f"{diff.days // 365} years ago"
                except:
                    modified_str = modified
                
                click.echo(f"{name:{name_width}} {size_str:<10} {modified_str:<15}")
        else:
            click.echo("No models found.")
        
    except Exception as e:
        if debug:
            click.echo(f"Error listing models: {e}", err=True)
            click.echo(traceback.format_exc(), err=True)
        else:
            click.echo(f"Error listing models: {e}", err=True)
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