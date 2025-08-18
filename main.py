#!/usr/bin/env python3
"""
Terminal-based RAG Knowledge Base for Founder Psychology Analysis
Main entry point for the CLI application
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from cli import app

if __name__ == "__main__":
    app()
