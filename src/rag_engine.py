"""
RAG Engine for the Knowledge Base
Orchestrates document processing, vector search, and LLM generation
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from config_manager import ConfigManager
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_manager import LLMManager

class RAGEngine:
    """Main orchestrator for the RAG knowledge base system"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.document_processor = DocumentProcessor(config_manager)
        self.vector_store = VectorStore(config_manager)
        self.llm_manager = LLMManager(config_manager)
        
        self.logger.info("RAG Engine initialized successfully")
    
    async def ingest_documents(self, path: Path, recursive: bool = True, force: bool = False) -> Dict[str, int]:
        """Ingest documents from a path into the knowledge base"""
        self.logger.info(f"Starting document ingestion from: {path}")
        
        stats = {
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'chunks_added': 0
        }
        
        try:
            if path.is_file():
                # Process single file
                chunks = self.document_processor.process_file(path, force=force)
                if chunks:
                    vector_stats = self.vector_store.add_chunks(chunks)
                    stats['processed'] = 1
                    stats['chunks_added'] = vector_stats['added']
                else:
                    stats['skipped'] = 1
            else:
                # Process directory
                all_chunks = []
                
                for chunk in self.document_processor.process_directory(path, recursive=recursive, force=force):
                    all_chunks.append(chunk)
                    
                    # Process in batches to manage memory
                    if len(all_chunks) >= 100:
                        vector_stats = self.vector_store.add_chunks(all_chunks)
                        stats['chunks_added'] += vector_stats['added']
                        stats['processed'] += len(all_chunks)
                        all_chunks = []
                
                # Process remaining chunks
                if all_chunks:
                    vector_stats = self.vector_store.add_chunks(all_chunks)
                    stats['chunks_added'] += vector_stats['added']
                    stats['processed'] += len(all_chunks)
            
            self.logger.info(f"Ingestion complete: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error during document ingestion: {str(e)}")
            stats['errors'] = 1
            return stats
    
    async def query(self, 
                   question: str, 
                   template: Optional[str] = None,
                   max_results: int = 5,
                   filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the knowledge base and generate response"""
        
        self.logger.info(f"Processing query: {question[:100]}...")
        
        try:
            # Step 1: Retrieve relevant documents
            search_results = self.vector_store.similarity_search(
                query=question,
                k=max_results,
                filter_metadata=filter_metadata
            )
            
            if not search_results:
                return {
                    'response': "I couldn't find any relevant information in the knowledge base to answer your question. Please ensure documents have been ingested or try rephrasing your question.",
                    'sources': [],
                    'query': question
                }
            
            # Step 2: Prepare context from search results
            context_parts = []
            sources = []
            
            for i, result in enumerate(search_results):
                # Add source information
                source_info = f"[Source {i+1}: {result['title']}"
                if result.get('page'):
                    source_info += f", Page {result['page']}"
                source_info += f" (Relevance: {result['score']:.3f})]"
                
                context_parts.append(f"{source_info}\n{result['content']}")
                
                sources.append({
                    'title': result['title'],
                    'content': result['content'][:300] + "..." if len(result['content']) > 300 else result['content'],
                    'score': result['score'],
                    'page': result.get('page'),
                    'source_file': result.get('source', '')
                })
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Step 3: Get research instructions
            instructions = self.config_manager.get_research_instructions()
            
            # Step 4: Create messages for LLM
            messages = self.llm_manager.create_research_messages(
                query=question,
                context=context,
                instructions=instructions,
                template=template
            )
            
            # Step 5: Generate response
            response = self.llm_manager.generate_response(messages)
            
            return {
                'response': response,
                'sources': sources,
                'query': question,
                'context_used': len(context_parts),
                'model_used': self.llm_manager.get_active_provider()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                'response': f"An error occurred while processing your query: {str(e)}",
                'sources': [],
                'query': question,
                'error': str(e)
            }
    
    async def multi_query_research(self, queries: List[str], template: Optional[str] = None) -> Dict[str, Any]:
        """Process multiple related queries for comprehensive research"""
        
        results = {}
        all_sources = set()
        
        for query in queries:
            result = await self.query(query, template=template)
            results[query] = result
            
            # Collect unique sources
            for source in result.get('sources', []):
                all_sources.add(source['source_file'])
        
        # Generate summary
        summary_query = f"Based on the research queries: {'; '.join(queries)}, provide a comprehensive summary of the key insights and patterns."
        summary_result = await self.query(summary_query, template=template)
        
        return {
            'individual_results': results,
            'summary': summary_result,
            'unique_sources': len(all_sources),
            'total_queries': len(queries)
        }
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge base"""
        
        # Vector store stats
        vector_stats = self.vector_store.get_collection_stats()
        
        # Document processor stats
        doc_stats = self.document_processor.get_processing_stats()
        
        # LLM provider stats
        llm_stats = self.llm_manager.get_provider_status()
        
        # Configuration info
        config = self.config_manager.config
        
        return {
            'vector_database': vector_stats,
            'document_processing': doc_stats,
            'llm_providers': llm_stats,
            'configuration': {
                'data_directory': config['app']['data_dir'],
                'chunk_size': config['document_processing']['chunk_size'],
                'chunk_overlap': config['document_processing']['chunk_overlap'],
                'embedding_model': config['embeddings']['model']
            }
        }
    
    async def research_story_ideas(self, topic: str, num_angles: int = 5) -> Dict[str, Any]:
        """Generate story ideas for content creation based on knowledge base"""
        
        # Generate different research angles
        research_queries = [
            f"What are the key psychological patterns in {topic}?",
            f"What success factors are most important for {topic}?",
            f"What common mistakes or failures occur with {topic}?",
            f"What unique insights or contrarian views exist about {topic}?",
            f"What trends or changes are happening in {topic}?"
        ]
        
        research_results = await self.multi_query_research(research_queries[:num_angles])
        
        # Generate story angle suggestions
        story_prompt = f"""Based on the research about {topic}, suggest specific story ideas for a Substack newsletter focused on founder psychology. 

Each story idea should:
1. Have a compelling angle or hook
2. Be based on concrete examples from the research
3. Provide actionable insights for founders
4. Be suitable for a 1000-1500 word article

Format as a numbered list with brief descriptions."""
        
        story_ideas = await self.query(story_prompt, template="story_research")
        
        return {
            'topic': topic,
            'research_foundation': research_results,
            'story_ideas': story_ideas,
            'recommended_sources': list(research_results.get('unique_sources', []))
        }
    
    async def analyze_founder_patterns(self, founder_name: str) -> Dict[str, Any]:
        """Analyze psychological patterns for a specific founder"""
        
        analysis_queries = [
            f"What are the key psychological traits of {founder_name}?",
            f"What mental frameworks or decision-making patterns does {founder_name} use?",
            f"What challenges or obstacles has {founder_name} overcome?",
            f"What leadership style or management approach does {founder_name} demonstrate?",
            f"What lessons can other founders learn from {founder_name}?"
        ]
        
        return await self.multi_query_research(analysis_queries, template="psychology_analysis")
    
    async def trend_analysis(self, timeframe: str = "recent") -> Dict[str, Any]:
        """Analyze trends in founder psychology and entrepreneurship"""
        
        trend_queries = [
            f"What are the emerging trends in founder psychology and entrepreneurship?",
            f"How are founder mental health and wellbeing approaches changing?",
            f"What new success patterns are emerging among modern founders?",
            f"What shifts in investor expectations or market conditions affect founder psychology?",
            f"What technological or social changes are impacting how founders think and operate?"
        ]
        
        return await self.multi_query_research(trend_queries, template="trend_identification")
    
    def reset_knowledge_base(self) -> bool:
        """Reset the entire knowledge base (use with caution)"""
        self.logger.warning("Resetting knowledge base - all data will be lost")
        
        success = self.vector_store.reset_collection()
        
        if success:
            # Also reset document processing tracking
            if self.document_processor.processed_docs_file.exists():
                self.document_processor.processed_docs_file.unlink()
            self.document_processor.processed_hashes = set()
        
        return success
