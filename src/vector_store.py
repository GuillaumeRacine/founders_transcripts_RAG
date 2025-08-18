"""
Vector Store Module for the RAG Knowledge Base
Manages embeddings and vector database operations using ChromaDB
"""

import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import hashlib
import json

from document_processor import DocumentChunk

class VectorStore:
    """Manages vector database operations with ChromaDB"""
    
    def __init__(self, config_manager):
        self.config = config_manager.get_vector_db_config()
        self.embedding_config = config_manager.get_embedding_config()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB
        self._initialize_chroma()
        
        # Initialize embedding function
        self._initialize_embeddings()
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection"""
        persist_dir = Path(self.config['persist_directory'])
        persist_dir.mkdir(exist_ok=True)
        
        # Create ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        collection_name = self.config['collection_name']
        try:
            self.collection = self.client.get_collection(collection_name)
            self.logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Founder psychology knowledge base"}
            )
            self.logger.info(f"Created new collection: {collection_name}")
    
    def _initialize_embeddings(self):
        """Initialize embedding function"""
        embedding_model = self.embedding_config['model']
        api_key = os.getenv(self.embedding_config['api_key_env'])
        
        if not api_key:
            raise ValueError(f"API key not found for embeddings: {self.embedding_config['api_key_env']}")
        
        if embedding_model.startswith('text-embedding'):
            # OpenAI embeddings
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=api_key)
            self.embedding_model = embedding_model
            self.embed_function = self._embed_openai
        else:
            raise ValueError(f"Unsupported embedding model: {embedding_model}")
        
        self.logger.info(f"Initialized embeddings with model: {embedding_model}")
    
    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        response = self.openai_client.embeddings.create(
            input=texts,
            model=self.embedding_model
        )
        
        return [embedding.embedding for embedding in response.data]
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Add document chunks to the vector store"""
        if not chunks:
            return {"added": 0, "skipped": 0, "errors": 0}
        
        added = 0
        skipped = 0
        errors = 0
        
        # Check which chunks already exist
        existing_ids = set()
        try:
            existing_data = self.collection.get(include=[])
            existing_ids = set(existing_data['ids'])
        except:
            pass  # Collection might be empty
        
        # Prepare data for batch insertion
        batch_size = 100  # Process in batches to manage memory
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Filter out existing chunks
            new_chunks = [chunk for chunk in batch_chunks if chunk.chunk_id not in existing_ids]
            
            if not new_chunks:
                skipped += len(batch_chunks)
                continue
            
            try:
                # Prepare data for ChromaDB
                documents = [chunk.content for chunk in new_chunks]
                metadatas = [chunk.metadata for chunk in new_chunks]
                ids = [chunk.chunk_id for chunk in new_chunks]
                
                # Generate embeddings
                embeddings = self.embed_function(documents)
                
                # Add to collection
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids
                )
                
                added += len(new_chunks)
                skipped += len(batch_chunks) - len(new_chunks)
                
                self.logger.info(f"Added batch of {len(new_chunks)} chunks to vector store")
                
            except Exception as e:
                self.logger.error(f"Error adding batch to vector store: {str(e)}")
                errors += len(batch_chunks)
        
        self.logger.info(f"Vector store update complete: {added} added, {skipped} skipped, {errors} errors")
        
        return {
            "added": added,
            "skipped": skipped,
            "errors": errors
        }
    
    def similarity_search(self, query: str, k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform similarity search in the vector store"""
        try:
            # Generate query embedding
            query_embedding = self.embed_function([query])[0]
            
            # Prepare where clause for filtering
            where_clause = None
            if filter_metadata:
                where_clause = filter_metadata
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            search_results = []
            
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'source': results['metadatas'][0][i].get('source_file', ''),
                        'title': results['metadatas'][0][i].get('file_name', ''),
                        'page': results['metadatas'][0][i].get('page_number', 1)
                    }
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and keyword search"""
        # For now, implement as pure vector search
        # In future versions, could combine with keyword search using ChromaDB's metadata filtering
        return self.similarity_search(query, k)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_size = min(10, count)
            if sample_size > 0:
                sample = self.collection.peek(limit=sample_size)
                
                # Analyze document types
                doc_types = {}
                authors = set()
                
                for metadata in sample['metadatas']:
                    file_type = metadata.get('file_type', 'unknown')
                    doc_types[file_type] = doc_types.get(file_type, 0) + 1
                    
                    author = metadata.get('author', '')
                    if author:
                        authors.add(author)
                
                return {
                    'total_chunks': count,
                    'document_types': doc_types,
                    'unique_authors': len(authors),
                    'collection_name': self.config['collection_name']
                }
            else:
                return {
                    'total_chunks': 0,
                    'document_types': {},
                    'unique_authors': 0,
                    'collection_name': self.config['collection_name']
                }
                
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            return {
                'total_chunks': 0,
                'document_types': {},
                'unique_authors': 0,
                'collection_name': self.config['collection_name'],
                'error': str(e)
            }
    
    def delete_by_source(self, source_file: str) -> int:
        """Delete all chunks from a specific source file"""
        try:
            # Find all chunks from this source
            results = self.collection.get(
                where={"source_file": source_file},
                include=["documents"]
            )
            
            if results['ids']:
                # Delete the chunks
                self.collection.delete(ids=results['ids'])
                deleted_count = len(results['ids'])
                self.logger.info(f"Deleted {deleted_count} chunks from {source_file}")
                return deleted_count
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"Error deleting chunks from {source_file}: {str(e)}")
            return 0
    
    def reset_collection(self) -> bool:
        """Reset the entire collection (delete all data)"""
        try:
            collection_name = self.config['collection_name']
            self.client.delete_collection(collection_name)
            
            # Recreate the collection
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Founder psychology knowledge base"}
            )
            
            self.logger.info("Collection reset successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting collection: {str(e)}")
            return False
    
    def export_data(self, output_path: Path) -> bool:
        """Export collection data to JSON file"""
        try:
            # Get all data from collection
            results = self.collection.get(include=['documents', 'metadatas'])
            
            export_data = {
                'collection_name': self.config['collection_name'],
                'total_chunks': len(results['ids']),
                'export_timestamp': str(pd.Timestamp.now()),
                'data': []
            }
            
            for i in range(len(results['ids'])):
                export_data['data'].append({
                    'id': results['ids'][i],
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported {len(results['ids'])} chunks to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            return False
