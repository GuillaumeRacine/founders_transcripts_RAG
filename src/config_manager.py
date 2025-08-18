"""
Configuration Manager for the RAG Knowledge Base
Handles model configurations, API keys, and system settings
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class ModelConfig:
    """Configuration for a specific LLM model"""
    provider: str
    model: str
    temperature: float
    max_tokens: int
    api_key_env: str
    base_url: Optional[str] = None
    available: bool = False

class ConfigManager:
    """Manages configuration for the RAG system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.active_model = self.config['models']['default']
        self.model_configs = self._load_model_configs()
        self._check_model_availability()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model_configs(self) -> Dict[str, ModelConfig]:
        """Load model configurations"""
        configs = {}
        
        for provider, config in self.config['models'].items():
            if provider == 'default':
                continue
                
            configs[provider] = ModelConfig(
                provider=provider,
                model=config['model'],
                temperature=config['temperature'],
                max_tokens=config['max_tokens'],
                api_key_env=config['api_key_env'],
                base_url=config.get('base_url')
            )
        
        return configs
    
    def _check_model_availability(self):
        """Check which models are available based on API keys and services"""
        for provider, model_config in self.model_configs.items():
            if provider == 'ollama':
                # Check if Ollama is running
                try:
                    import requests
                    response = requests.get(f"{model_config.base_url}/api/tags", timeout=5)
                    model_config.available = response.status_code == 200
                except:
                    model_config.available = False
            else:
                # Check for API key
                api_key = os.getenv(model_config.api_key_env)
                model_config.available = bool(api_key)
    
    def get_model_config(self, provider: str) -> ModelConfig:
        """Get configuration for a specific model provider"""
        if provider not in self.model_configs:
            raise ValueError(f"Unknown model provider: {provider}")
        
        config = self.model_configs[provider]
        if not config.available:
            if provider == 'ollama':
                raise RuntimeError(f"Ollama is not running. Start it with: ollama serve")
            else:
                raise RuntimeError(f"API key not found for {provider}. Set {config.api_key_env} environment variable.")
        
        return config
    
    def set_active_model(self, provider: str):
        """Set the active model provider"""
        if provider not in self.model_configs:
            raise ValueError(f"Unknown model provider: {provider}")
        
        # Validate model is available
        self.get_model_config(provider)  # This will raise if not available
        
        self.active_model = provider
        self.logger.info(f"Switched to model: {provider}")
    
    def get_active_model(self) -> str:
        """Get the currently active model provider"""
        return self.active_model
    
    def get_active_model_config(self) -> ModelConfig:
        """Get configuration for the currently active model"""
        return self.get_model_config(self.active_model)
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their status"""
        models = {}
        
        for provider, config in self.model_configs.items():
            models[provider] = {
                'model': config.model,
                'available': config.available,
                'temperature': config.temperature,
                'max_tokens': config.max_tokens
            }
        
        return models
    
    def list_prompt_templates(self) -> List[Dict[str, str]]:
        """List available prompt templates"""
        return self.config['research']['prompt_templates']
    
    def get_prompt_template(self, template_name: str) -> Optional[str]:
        """Get a specific prompt template"""
        templates_path = Path("templates/research_prompts.yaml")
        
        if not templates_path.exists():
            return None
        
        with open(templates_path, 'r') as f:
            templates = yaml.safe_load(f)
        
        return templates.get(template_name)
    
    def get_research_instructions(self) -> str:
        """Get default research instructions"""
        return self.config['research']['default_instructions']
    
    def get_document_config(self) -> Dict[str, Any]:
        """Get document processing configuration"""
        return self.config['document_processing']
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration"""
        return self.config['vector_db']
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration"""
        return self.config['embeddings']
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        from vector_store import VectorStore
        
        # Count documents in data directory
        data_dir = Path(self.config['app']['data_dir'])
        doc_count = 0
        if data_dir.exists():
            supported_formats = self.config['document_processing']['supported_formats']
            for fmt in supported_formats:
                doc_count += len(list(data_dir.glob(f"**/*{fmt}")))
        
        # Check vector database status
        try:
            vector_store = VectorStore(self)
            vector_db_status = "Connected"
        except:
            vector_db_status = "Not initialized"
        
        return {
            'active_model': self.active_model,
            'document_count': doc_count,
            'vector_db_status': vector_db_status,
            'available_models': len([m for m in self.model_configs.values() if m.available])
        }
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration and save to file"""
        def deep_update(base: dict, updates: dict):
            for key, value in updates.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
        
        deep_update(self.config, updates)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        self.logger.info("Configuration updated and saved")
