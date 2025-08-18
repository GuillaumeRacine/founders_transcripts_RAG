"""
Document Processing Module for the RAG Knowledge Base
Handles PDF extraction, text chunking, and metadata preservation
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
import logging
import mimetypes
from datetime import datetime

# Document processing libraries
import PyPDF2
import pymupdf  # fitz
from docx import Document as DocxDocument

@dataclass
class DocumentChunk:
    """Represents a chunk of processed document content"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_file: str
    page_number: Optional[int] = None
    chunk_index: int = 0

class DocumentProcessor:
    """Processes various document formats and creates semantic chunks"""
    
    def __init__(self, config_manager):
        self.config = config_manager.get_document_config()
        self.chunk_size = self.config['chunk_size']
        self.chunk_overlap = self.config['chunk_overlap']
        self.separators = self.config['separators']
        self.supported_formats = set(self.config['supported_formats'])
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize processed documents tracking
        self.processed_docs_file = Path("processed_documents.txt")
        self.processed_hashes = self._load_processed_hashes()
    
    def _load_processed_hashes(self) -> set:
        """Load hashes of previously processed documents"""
        if self.processed_docs_file.exists():
            with open(self.processed_docs_file, 'r') as f:
                return set(line.strip() for line in f)
        return set()
    
    def _save_processed_hash(self, file_hash: str):
        """Save hash of processed document"""
        self.processed_hashes.add(file_hash)
        with open(self.processed_docs_file, 'a') as f:
            f.write(f"{file_hash}\n")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file to track processing"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def should_process_file(self, file_path: Path, force: bool = False) -> bool:
        """Check if file should be processed"""
        if force:
            return True
        
        file_hash = self._get_file_hash(file_path)
        return file_hash not in self.processed_hashes
    
    def extract_text_from_pdf(self, file_path: Path) -> tuple[List[Dict], Dict[str, Any]]:
        """Extract text from PDF using PyMuPDF for better quality"""
        try:
            doc = pymupdf.open(file_path)
            text_content = []
            metadata = {
                'page_count': len(doc),
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', '')
            }
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    text_content.append({
                        'page': page_num + 1,
                        'content': text
                    })
            
            doc.close()
            return text_content, metadata
            
        except Exception as e:
            self.logger.warning(f"PyMuPDF failed for {file_path}, trying PyPDF2: {e}")
            # Fallback to PyPDF2
            return self._extract_pdf_pypdf2(file_path)
    
    def _extract_pdf_pypdf2(self, file_path: Path) -> tuple[List[Dict], Dict[str, Any]]:
        """Fallback PDF extraction using PyPDF2"""
        text_content = []
        metadata = {}
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract metadata
            if pdf_reader.metadata:
                metadata = {
                    'title': pdf_reader.metadata.get('/Title', ''),
                    'author': pdf_reader.metadata.get('/Author', ''),
                    'subject': pdf_reader.metadata.get('/Subject', ''),
                    'creator': pdf_reader.metadata.get('/Creator', ''),
                    'producer': pdf_reader.metadata.get('/Producer', ''),
                    'creation_date': str(pdf_reader.metadata.get('/CreationDate', '')),
                    'modification_date': str(pdf_reader.metadata.get('/ModDate', ''))
                }
            
            metadata['page_count'] = len(pdf_reader.pages)
            
            # Extract text
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_content.append({
                        'page': page_num + 1,
                        'content': text
                    })
        
        return text_content, metadata
    
    def extract_text_from_docx(self, file_path: Path) -> tuple[List[Dict], Dict[str, Any]]:
        """Extract text from DOCX files"""
        doc = DocxDocument(str(file_path))
        
        # Extract text
        text_content = []
        full_text = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text)
        
        text_content.append({
            'page': 1,
            'content': '\n'.join(full_text)
        })
        
        # Extract metadata
        core_props = doc.core_properties
        metadata = {
            'title': core_props.title or '',
            'author': core_props.author or '',
            'subject': core_props.subject or '',
            'creator': core_props.author or '',
            'creation_date': str(core_props.created) if core_props.created else '',
            'modification_date': str(core_props.modified) if core_props.modified else '',
            'page_count': 1
        }
        
        return text_content, metadata
    
    def extract_text_from_txt(self, file_path: Path) -> tuple[List[Dict], Dict[str, Any]]:
        """Extract text from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError("Could not decode file with any common encoding")
        
        text_content = [{'page': 1, 'content': content}]
        
        # Get file stats for metadata
        stat = file_path.stat()
        metadata = {
            'title': file_path.stem,
            'author': '',
            'subject': '',
            'creator': '',
            'creation_date': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modification_date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'page_count': 1
        }
        
        return text_content, metadata
    
    def extract_text_from_file(self, file_path: Path) -> tuple[List[Dict], Dict[str, Any]]:
        """Extract text from supported file formats"""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext in ['.txt', '.md']:
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def recursive_text_splitter(self, text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        """Recursively split text using different separators"""
        if not separators:
            # Base case: split by character if no separators left
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunks.append(text[i:i + chunk_size])
            return chunks
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        parts = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            # If adding this part would exceed chunk size, process current chunk
            if len(current_chunk) + len(part) + len(separator) > chunk_size and current_chunk:
                if len(current_chunk) > chunk_size:
                    # Current chunk is too big, split it recursively
                    sub_chunks = self.recursive_text_splitter(
                        current_chunk, remaining_separators, chunk_size, chunk_overlap
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if chunks and chunk_overlap > 0:
                    overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                    current_chunk = overlap_text + separator + part
                else:
                    current_chunk = part
            else:
                # Add part to current chunk
                if current_chunk:
                    current_chunk += separator + part
                else:
                    current_chunk = part
        
        # Add final chunk
        if current_chunk.strip():
            if len(current_chunk) > chunk_size:
                sub_chunks = self.recursive_text_splitter(
                    current_chunk, remaining_separators, chunk_size, chunk_overlap
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_chunks(self, text_pages: List[Dict], file_metadata: Dict[str, Any], file_path: Path) -> List[DocumentChunk]:
        """Create semantic chunks from extracted text"""
        chunks = []
        
        for page_data in text_pages:
            page_content = page_data['content']
            page_number = page_data.get('page', 1)
            
            if not page_content.strip():
                continue
            
            # Split page content into chunks
            text_chunks = self.recursive_text_splitter(
                page_content, 
                self.separators, 
                self.chunk_size, 
                self.chunk_overlap
            )
            
            for chunk_idx, chunk_text in enumerate(text_chunks):
                if not chunk_text.strip():
                    continue
                
                # Create unique chunk ID
                chunk_id = hashlib.md5(
                    f"{file_path.name}_{page_number}_{chunk_idx}_{chunk_text[:50]}".encode()
                ).hexdigest()
                
                # Prepare chunk metadata
                chunk_metadata = {
                    'source_file': str(file_path),
                    'file_name': file_path.name,
                    'file_type': file_path.suffix.lower(),
                    'page_number': page_number,
                    'chunk_index': chunk_idx,
                    'chunk_id': chunk_id,
                    'processing_date': datetime.now().isoformat(),
                    **file_metadata  # Include document metadata
                }
                
                # Add contextual header for better retrieval
                contextual_content = f"Document: {file_path.name}"
                if file_metadata.get('title'):
                    contextual_content += f" | Title: {file_metadata['title']}"
                if file_metadata.get('author'):
                    contextual_content += f" | Author: {file_metadata['author']}"
                if page_number > 1:
                    contextual_content += f" | Page: {page_number}"
                
                contextual_content += f"\n\n{chunk_text}"
                
                chunk = DocumentChunk(
                    content=contextual_content,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                    source_file=str(file_path),
                    page_number=page_number,
                    chunk_index=chunk_idx
                )
                
                chunks.append(chunk)
        
        return chunks
    
    def process_file(self, file_path: Path, force: bool = False) -> List[DocumentChunk]:
        """Process a single file and return chunks"""
        if not self.should_process_file(file_path, force):
            self.logger.info(f"Skipping already processed file: {file_path.name}")
            return []
        
        if file_path.suffix.lower() not in self.supported_formats:
            self.logger.warning(f"Unsupported file format: {file_path}")
            return []
        
        try:
            self.logger.info(f"Processing file: {file_path.name}")
            
            # Extract text and metadata
            text_pages, file_metadata = self.extract_text_from_file(file_path)
            
            if not text_pages:
                self.logger.warning(f"No text extracted from: {file_path}")
                return []
            
            # Create chunks
            chunks = self.create_chunks(text_pages, file_metadata, file_path)
            
            # Mark file as processed
            file_hash = self._get_file_hash(file_path)
            self._save_processed_hash(file_hash)
            
            self.logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return []
    
    def process_directory(self, directory_path: Path, recursive: bool = True, force: bool = False) -> Generator[DocumentChunk, None, None]:
        """Process all supported files in a directory"""
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                chunks = self.process_file(file_path, force=force)
                for chunk in chunks:
                    yield chunk
    
    def get_processing_stats(self) -> Dict[str, int]:
        """Get statistics about processed documents"""
        return {
            'processed_documents': len(self.processed_hashes),
            'supported_formats': len(self.supported_formats)
        }
