# Founder Psychology RAG Knowledge Base

A terminal-based RAG (Retrieval-Augmented Generation) system specifically designed for analyzing founder psychology documents. Perfect for researchers, content creators, and anyone studying entrepreneurial psychology patterns.

## Features

🧠 **Specialized for Psychology Analysis** - Built-in prompts for psychological pattern recognition, founder behavior analysis, and content creation

🔄 **Multi-LLM Support** - Switch between OpenAI (GPT-4), Anthropic (Claude), and local Ollama models on-the-fly

📚 **Smart Document Processing** - Handles PDFs, DOCX, TXT, and MD files with intelligent chunking and metadata preservation

🎯 **Content Creation Tools** - Generate story ideas and research angles specifically for newsletters and content creation

⚡ **Production-Ready** - Designed to handle 1000+ documents efficiently with ChromaDB vector storage

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/GuillaumeRacine/founders_transcripts_RAG.git
cd founders_transcripts_RAG

# Install dependencies
pip install typer rich PyPDF2 pymupdf python-docx chromadb openai anthropic pyyaml requests
```

### 2. Setup

```bash
# Interactive setup - adds your API keys
python main.py setup
```

### 3. Add Documents

Place your documents (PDFs, books, transcripts, research papers) in the `data/` folder, then:

```bash
# Ingest all documents
python main.py ingest data/ --recursive
```

### 4. Start Researching

```bash
# Interactive chat mode
python main.py chat

# Single queries with templates
python main.py query "What psychological patterns make founders successful?" --template psychology_analysis
```

## Available Research Templates

- **`psychology_analysis`** - Deep psychological pattern analysis
- **`biography_insights`** - Extract insights from founder biographies
- **`story_research`** - Generate content ideas for newsletters/blogs
- **`trend_identification`** - Identify patterns across multiple sources
- **`comparison_analysis`** - Compare different founders or approaches

## Command Reference

### Document Management
```bash
# Ingest documents
python main.py ingest /path/to/docs --recursive --force

# Check system status
python main.py status
```

### Querying
```bash
# Basic query
python main.py query "your question here"

# With specific template and model
python main.py query "your question" --template psychology_analysis --model anthropic

# Interactive chat mode
python main.py chat --model openai
```

### Model Management
```bash
# List available models
python main.py models

# Chat mode commands:
# /models - List available models
# /switch claude - Switch to Claude
# /switch openai - Switch to GPT-4
# /templates - List research templates
# /status - System status
```

## Configuration

The system is configured via `config.yaml`. Key settings:

- **Chunk size**: 400 tokens (optimal for context preservation)
- **Chunk overlap**: 50 tokens (maintains continuity)
- **Vector database**: ChromaDB (persistent local storage)
- **Embedding model**: OpenAI text-embedding-3-large

## Supported File Formats

- **PDF** - Using PyMuPDF and PyPDF2 for robust extraction
- **DOCX** - Microsoft Word documents
- **TXT** - Plain text files
- **MD** - Markdown files

## API Keys Required

- **OpenAI**: For GPT models and embeddings (required)
- **Anthropic**: For Claude models (optional)
- **Ollama**: Local models (optional, no API key needed)

Set these as environment variables or add them during setup:
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

## Architecture

```
├── src/
│   ├── cli.py              # Command-line interface
│   ├── config_manager.py   # Configuration and model management
│   ├── document_processor.py # PDF/DOCX processing and chunking
│   ├── llm_manager.py      # Multi-provider LLM handling
│   ├── rag_engine.py       # Main RAG orchestration
│   └── vector_store.py     # ChromaDB operations
├── templates/
│   ├── research_prompts.yaml # Specialized research prompts
│   └── model_configs.yaml    # Model-specific optimizations
├── data/                   # Your documents go here
├── vector_db/             # ChromaDB storage
└── config.yaml           # Main configuration
```

## Example Use Cases

### Research Story Ideas
```bash
python main.py query "Generate story ideas about founder resilience during economic downturns" --template story_research
```

### Psychological Analysis
```bash
python main.py query "What are the key psychological differences between successful and failed founders?" --template psychology_analysis
```

### Extract Biography Insights
```bash
python main.py query "What can we learn from Elon Musk's early psychological development?" --template biography_insights
```

## Advanced Features

- **Multi-query research** - Analyze topics from multiple angles
- **Source citation** - All responses include document sources and relevance scores
- **Deduplication** - Prevents reprocessing the same documents
- **Batch processing** - Efficiently handles large document collections
- **Hot-swappable models** - Change AI models mid-conversation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions or issues, please open a GitHub issue or contact the maintainers.