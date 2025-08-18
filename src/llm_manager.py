"""
LLM Manager for the RAG Knowledge Base
Handles multiple LLM providers with hot-swappable configurations
"""

import os
from typing import Dict, Any, Optional, List
import logging
from abc import ABC, abstractmethod

# LLM provider imports
from openai import OpenAI
import anthropic
import requests

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from the LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, model_config):
        self.config = model_config
        api_key = os.getenv(model_config.api_key_env)
        
        if not api_key:
            raise ValueError(f"API key not found: {model_config.api_key_env}")
        
        self.client = OpenAI(api_key=api_key)
        self.logger = logging.getLogger(__name__)
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                stream=False
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        try:
            # Make a minimal test request
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except:
            return False

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, model_config):
        self.config = model_config
        api_key = os.getenv(model_config.api_key_env)
        
        if not api_key:
            raise ValueError(f"API key not found: {model_config.api_key_env}")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.logger = logging.getLogger(__name__)
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Anthropic API"""
        try:
            # Convert messages format for Anthropic
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)
            
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                system=system_message,
                messages=user_messages
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if Anthropic is available"""
        try:
            # Make a minimal test request
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except:
            return False

class OllamaProvider(LLMProvider):
    """Local Ollama provider"""
    
    def __init__(self, model_config):
        self.config = model_config
        self.base_url = model_config.base_url or "http://localhost:11434"
        self.logger = logging.getLogger(__name__)
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Ollama API"""
        try:
            # Convert messages to prompt format for Ollama
            prompt = self._convert_messages_to_prompt(messages)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get('temperature', self.config.temperature),
                        "num_predict": kwargs.get('max_tokens', self.config.max_tokens)
                    }
                },
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result.get('response', '').strip()
            
        except Exception as e:
            self.logger.error(f"Ollama API error: {str(e)}")
            raise
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt for Ollama"""
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

class LLMManager:
    """Manages multiple LLM providers with hot-swapping capabilities"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.providers = {}
        self.active_provider = None
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured LLM providers"""
        model_configs = self.config_manager.model_configs
        
        for provider_name, model_config in model_configs.items():
            if not model_config.available:
                continue
            
            try:
                if provider_name == 'openai':
                    self.providers[provider_name] = OpenAIProvider(model_config)
                elif provider_name == 'anthropic':
                    self.providers[provider_name] = AnthropicProvider(model_config)
                elif provider_name == 'ollama':
                    self.providers[provider_name] = OllamaProvider(model_config)
                else:
                    self.logger.warning(f"Unknown provider: {provider_name}")
                    continue
                
                self.logger.info(f"Initialized provider: {provider_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize provider {provider_name}: {str(e)}")
        
        # Set active provider
        active_model = self.config_manager.get_active_model()
        if active_model in self.providers:
            self.active_provider = active_model
        elif self.providers:
            # Fallback to first available provider
            self.active_provider = list(self.providers.keys())[0]
            self.logger.info(f"Using fallback provider: {self.active_provider}")
        else:
            raise RuntimeError("No LLM providers available")
    
    def switch_provider(self, provider_name: str):
        """Switch to a different LLM provider"""
        if provider_name not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(f"Provider {provider_name} not available. Available: {available}")
        
        self.active_provider = provider_name
        self.config_manager.set_active_model(provider_name)
        self.logger.info(f"Switched to provider: {provider_name}")
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using the active provider"""
        if not self.active_provider or self.active_provider not in self.providers:
            raise RuntimeError("No active LLM provider")
        
        provider = self.providers[self.active_provider]
        return provider.generate_response(messages, **kwargs)
    
    def get_active_provider(self) -> str:
        """Get the name of the active provider"""
        return self.active_provider
    
    def get_provider_config(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a provider"""
        name = provider_name or self.active_provider
        
        if name not in self.providers:
            raise ValueError(f"Provider {name} not available")
        
        model_config = self.config_manager.get_model_config(name)
        
        return {
            'provider': name,
            'model': model_config.model,
            'temperature': model_config.temperature,
            'max_tokens': model_config.max_tokens,
            'available': model_config.available
        }
    
    def list_available_providers(self) -> List[str]:
        """List all available providers"""
        return list(self.providers.keys())
    
    def test_provider(self, provider_name: str) -> bool:
        """Test if a provider is working"""
        if provider_name not in self.providers:
            return False
        
        try:
            provider = self.providers[provider_name]
            return provider.is_available()
        except:
            return False
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        status = {}
        
        for name, provider in self.providers.items():
            config = self.get_provider_config(name)
            status[name] = {
                'available': True,
                'working': self.test_provider(name),
                'model': config['model'],
                'active': name == self.active_provider
            }
        
        return status
    
    def create_research_messages(self, query: str, context: str, instructions: str, template: Optional[str] = None) -> List[Dict[str, str]]:
        """Create properly formatted messages for research queries"""
        
        # Base system message for founder psychology research
        system_content = instructions
        
        # Add template-specific instructions if provided
        if template:
            template_prompt = self.config_manager.get_prompt_template(template)
            if template_prompt:
                system_content += f"\n\n{template_prompt}"
        
        # Create messages
        messages = [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": f"""Based on the following context from the knowledge base, please answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. Cites specific sources and examples from the context
3. Highlights key psychological insights or patterns
4. Provides actionable insights for understanding founder psychology

Answer:"""
            }
        ]
        
        return messages
