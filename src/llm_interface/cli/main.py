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