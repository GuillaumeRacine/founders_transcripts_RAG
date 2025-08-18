"""
Command Line Interface for the RAG Knowledge Base
Provides interactive terminal commands for querying and managing the knowledge base
"""

import typer
from typing import Optional, List
from pathlib import Path
import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

from config_manager import ConfigManager
from document_processor import DocumentProcessor
from rag_engine import RAGEngine

app = typer.Typer(name="founder-rag", help="Terminal-based RAG Knowledge Base for Founder Psychology Analysis")
console = Console()

# Global state
config_manager = None
rag_engine = None

def initialize_system():
    """Initialize the RAG system with configuration"""
    global config_manager, rag_engine
    
    if not config_manager:
        config_manager = ConfigManager()
        rag_engine = RAGEngine(config_manager)
    
    return config_manager, rag_engine

@app.command()
def setup():
    """Interactive setup wizard for first-time configuration"""
    console.print("[bold blue]🧠 Founder Psychology RAG Setup Wizard[/bold blue]")
    console.print()
    
    # Check for API keys
    openai_key = Prompt.ask("Enter your OpenAI API key (or press Enter to skip)", password=True, default="")
    anthropic_key = Prompt.ask("Enter your Anthropic API key (or press Enter to skip)", password=True, default="")
    
    # Set environment variables
    if openai_key:
        import os
        os.environ["OPENAI_API_KEY"] = openai_key
    if anthropic_key:
        import os
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    
    # Initialize system
    config_mgr, _ = initialize_system()
    
    # Check for Ollama
    has_ollama = Confirm.ask("Do you have Ollama installed locally?", default=False)
    
    console.print("\n[green]✅ Setup complete![/green]")
    console.print("You can now:")
    console.print("• Add documents with: [bold]python main.py ingest /path/to/documents[/bold]")
    console.print("• Query the knowledge base with: [bold]python main.py query 'your question'[/bold]")
    console.print("• Start interactive mode with: [bold]python main.py chat[/bold]")

@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to documents directory"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
    recursive: bool = typer.Option(True, "--recursive", "-r", help="Process subdirectories"),
    force: bool = typer.Option(False, "--force", "-f", help="Reprocess existing documents")
):
    """Ingest documents into the knowledge base"""
    config_mgr, rag_eng = initialize_system()
    
    if model:
        config_mgr.set_active_model(model)
    
    doc_path = Path(path)
    if not doc_path.exists():
        console.print(f"[red]❌ Path does not exist: {path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]📚 Ingesting documents from: {path}[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing documents...", total=None)
        
        try:
            result = asyncio.run(rag_eng.ingest_documents(doc_path, recursive=recursive, force=force))
            
            console.print(f"\n[green]✅ Successfully processed {result['processed']} documents[/green]")
            if result['skipped'] > 0:
                console.print(f"[yellow]⚠️  Skipped {result['skipped']} documents (already processed)[/yellow]")
            if result['errors'] > 0:
                console.print(f"[red]❌ Failed to process {result['errors']} documents[/red]")
                
        except Exception as e:
            console.print(f"[red]❌ Error during ingestion: {str(e)}[/red]")
            raise typer.Exit(1)

@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask the knowledge base"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
    template: Optional[str] = typer.Option(None, "--template", "-t", help="Prompt template to use"),
    max_results: int = typer.Option(5, "--max-results", "-n", help="Maximum number of source documents"),
    show_sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source documents")
):
    """Query the knowledge base with a specific question"""
    config_mgr, rag_eng = initialize_system()
    
    if model:
        config_mgr.set_active_model(model)
    
    console.print(f"[blue]🤔 Querying: {question}[/blue]")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Searching knowledge base...", total=None)
        
        try:
            result = asyncio.run(rag_eng.query(
                question, 
                template=template, 
                max_results=max_results
            ))
            
            # Display response
            response_panel = Panel(
                Markdown(result['response']),
                title="🧠 AI Response",
                border_style="blue"
            )
            console.print(response_panel)
            
            # Display sources if requested
            if show_sources and result['sources']:
                console.print("\n[bold]📚 Sources:[/bold]")
                for i, source in enumerate(result['sources'], 1):
                    source_info = f"**{source['title']}** (Score: {source['score']:.3f})"
                    if source.get('page'):
                        source_info += f" - Page {source['page']}"
                    console.print(f"{i}. {source_info}")
                    console.print(f"   {source['content'][:200]}...")
                    console.print()
                    
        except Exception as e:
            console.print(f"[red]❌ Error during query: {str(e)}[/red]")
            raise typer.Exit(1)

@app.command()
def chat(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
    template: Optional[str] = typer.Option(None, "--template", "-t", help="Prompt template to use")
):
    """Start interactive chat mode with the knowledge base"""
    config_mgr, rag_eng = initialize_system()
    
    if model:
        config_mgr.set_active_model(model)
    
    current_model = config_mgr.get_active_model()
    console.print(f"[green]🚀 Starting chat mode with {current_model}[/green]")
    console.print("[dim]Type 'quit' or 'exit' to leave chat mode[/dim]")
    console.print("[dim]Type '/help' for available commands[/dim]")
    console.print()
    
    while True:
        try:
            question = Prompt.ask("[bold blue]You[/bold blue]")
            
            if question.lower() in ['quit', 'exit']:
                console.print("[yellow]👋 Goodbye![/yellow]")
                break
            
            if question.startswith('/'):
                handle_chat_command(question, config_mgr, rag_eng)
                continue
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Thinking...", total=None)
                
                result = asyncio.run(rag_eng.query(question, template=template))
                
                console.print(f"[bold green]AI[/bold green]: {result['response']}")
                console.print()
                
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")
            console.print()

def handle_chat_command(command: str, config_mgr: ConfigManager, rag_eng: RAGEngine):
    """Handle special chat commands"""
    if command == '/help':
        console.print("[bold]Available commands:[/bold]")
        console.print("• /models - List available models")
        console.print("• /switch <model> - Switch to a different model")
        console.print("• /templates - List available prompt templates")
        console.print("• /status - Show current configuration")
        console.print("• /help - Show this help message")
    
    elif command == '/models':
        models = config_mgr.list_available_models()
        table = Table(title="Available Models")
        table.add_column("Provider", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Status", style="yellow")
        
        for provider, model_info in models.items():
            status = "✅ Ready" if model_info.get('available', False) else "❌ Not configured"
            table.add_row(provider, model_info['model'], status)
        
        console.print(table)
    
    elif command.startswith('/switch '):
        model_name = command.split(' ', 1)[1]
        try:
            config_mgr.set_active_model(model_name)
            console.print(f"[green]✅ Switched to {model_name}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to switch model: {str(e)}[/red]")
    
    elif command == '/templates':
        templates = config_mgr.list_prompt_templates()
        table = Table(title="Available Prompt Templates")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        
        for template in templates:
            table.add_row(template['name'], template['description'])
        
        console.print(table)
    
    elif command == '/status':
        status = config_mgr.get_system_status()
        console.print(f"[bold]Current Model:[/bold] {status['active_model']}")
        console.print(f"[bold]Documents:[/bold] {status['document_count']}")
        console.print(f"[bold]Vector DB:[/bold] {status['vector_db_status']}")

@app.command()
def models():
    """List and manage available LLM models"""
    config_mgr, _ = initialize_system()
    
    models = config_mgr.list_available_models()
    current_model = config_mgr.get_active_model()
    
    table = Table(title="Available LLM Models")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Active", style="magenta")
    
    for provider, model_info in models.items():
        status = "✅ Ready" if model_info.get('available', False) else "❌ Not configured"
        active = "🔥" if provider == current_model else ""
        table.add_row(provider, model_info['model'], status, active)
    
    console.print(table)

@app.command()
def status():
    """Show system status and configuration"""
    config_mgr, rag_eng = initialize_system()
    
    status = config_mgr.get_system_status()
    
    console.print("[bold blue]📊 System Status[/bold blue]")
    console.print()
    
    # Configuration panel
    config_info = f"""
**Active Model:** {status['active_model']}
**Document Count:** {status['document_count']}
**Vector Database:** {status['vector_db_status']}
**Data Directory:** {config_mgr.config['app']['data_dir']}
**Vector DB Directory:** {config_mgr.config['app']['vector_db_dir']}
    """
    
    config_panel = Panel(
        Markdown(config_info),
        title="Configuration",
        border_style="green"
    )
    console.print(config_panel)

if __name__ == "__main__":
    app()
