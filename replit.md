# Overview

This is a Terminal-based RAG (Retrieval-Augmented Generation) Knowledge Base system specifically designed for Founder Psychology Analysis. The system processes various document formats (PDF, DOCX, etc.), creates semantic embeddings, and provides intelligent querying capabilities through a command-line interface. It's built to analyze psychological patterns, behavioral traits, and cognitive frameworks of entrepreneurs and business founders.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Core Architecture Pattern
The system follows a modular, component-based architecture with clear separation of concerns:

**Problem Addressed**: Need for a specialized knowledge base system that can ingest, process, and analyze documents related to founder psychology while providing intelligent querying capabilities.

**Chosen Solution**: Multi-layered RAG architecture with dedicated modules for document processing, vector storage, LLM management, and CLI interaction.

**Key Benefits**: 
- Modular design allows for easy component replacement and testing
- Specialized for psychological analysis with custom prompt templates
- Terminal-based interface for developer-friendly interaction

## Frontend Architecture
- **CLI Interface**: Built with Typer for command-line operations
- **Rich Console**: Uses Rich library for enhanced terminal UI with tables, panels, and progress indicators
- **Interactive Setup**: Wizard-based configuration for first-time users

## Backend Architecture

### Document Processing Layer
- **Multi-format Support**: Handles PDF (PyPDF2, PyMuPDF), DOCX, and text files
- **Smart Chunking**: Configurable chunk sizes with overlap for context preservation
- **Deduplication**: Hash-based tracking to avoid reprocessing documents
- **Metadata Preservation**: Maintains source file, page numbers, and processing timestamps

### Vector Database Layer
- **ChromaDB Integration**: Persistent vector storage with embedding management
- **Embedding Strategy**: Configurable embedding models with hot-swappable providers
- **Collection Management**: Organized storage with metadata filtering capabilities

### LLM Management Layer
- **Multi-Provider Support**: Abstract provider pattern supporting OpenAI, Anthropic, and extensible to other providers
- **Hot-Swappable Models**: Runtime model switching without system restart
- **Configuration Management**: YAML-based configuration with environment variable support

### RAG Orchestration
- **Async Processing**: Asynchronous document ingestion and processing
- **Query Pipeline**: Combines vector search with LLM generation
- **Prompt Specialization**: Custom templates for founder psychology analysis

## Configuration Management
- **YAML Configuration**: Centralized configuration with environment variable integration
- **Model Availability Checking**: Automatic detection of available LLM providers
- **Logging Integration**: Structured logging across all components

# External Dependencies

## LLM Providers
- **OpenAI**: GPT models for text generation and embeddings
- **Anthropic**: Claude models as alternative LLM provider
- **Extensible Provider System**: Architecture supports adding additional LLM providers

## Vector Database
- **ChromaDB**: Primary vector database for embedding storage and similarity search
- **Persistent Storage**: Local file-based persistence for embeddings and metadata

## Document Processing
- **PyPDF2**: PDF text extraction and processing
- **PyMuPDF (fitz)**: Enhanced PDF processing with layout preservation
- **python-docx**: Microsoft Word document processing

## CLI and UI Libraries
- **Typer**: Modern CLI framework for command-line interface
- **Rich**: Enhanced terminal UI with formatting, tables, and progress indicators

## Configuration and Data Management
- **PyYAML**: YAML configuration file parsing
- **Python Standard Library**: Pathlib, logging, hashlib for core functionality

## Development and Deployment
- **Environment Variables**: API key management and configuration
- **Logging**: Standard Python logging with configurable levels
- **Async Support**: Asyncio for concurrent document processing

The system is designed to be self-contained with minimal external service dependencies, primarily requiring only API access to LLM providers for intelligent query responses.