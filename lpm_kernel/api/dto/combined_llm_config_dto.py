from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from lpm_kernel.api.dto.user_llm_config_dto import UserLLMConfigDTO
from lpm_kernel.api.dto.thinking_model_dto import ThinkingModelDTO


class CombinedLLMConfigDTO(BaseModel):
    """Combined DTO containing both UserLLMConfig and ThinkingModel fields"""
    # UserLLMConfig fields
    id: Optional[int] = None
    provider_type: str = 'openai'
    key: Optional[str] = None
    
    # Chat configuration
    chat_endpoint: Optional[str] = None
    chat_api_key: Optional[str] = None
    chat_model_name: Optional[str] = None
    
    # Embedding configuration
    embedding_endpoint: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_model_name: Optional[str] = None
    
    # Thinking configuration from UserLLMConfig
    thinking_endpoint: Optional[str] = None
    thinking_api_key: Optional[str] = None
    thinking_model_name: Optional[str] = None
    
    # ThinkingModel specific fields (if different from UserLLMConfig)
    thinking_model_id: Optional[int] = None
    thinking_model_endpoint: Optional[str] = None
    thinking_model_api_key: Optional[str] = None
    thinking_model_name: Optional[str] = None
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_dtos(cls, llm_config: Optional[UserLLMConfigDTO], thinking_model: Optional[ThinkingModelDTO]):
        """Create combined DTO from UserLLMConfigDTO and ThinkingModelDTO"""
        if not llm_config:
            return None
            
        return cls(
            # UserLLMConfig fields
            id=llm_config.id,
            provider_type=llm_config.provider_type,
            key=llm_config.key,
            chat_endpoint=llm_config.chat_endpoint,
            chat_api_key=llm_config.chat_api_key,
            chat_model_name=llm_config.chat_model_name,
            embedding_endpoint=llm_config.embedding_endpoint,
            embedding_api_key=llm_config.embedding_api_key,
            embedding_model_name=llm_config.embedding_model_name,
            thinking_endpoint=llm_config.thinking_endpoint,
            thinking_api_key=llm_config.thinking_api_key,
            thinking_model_name=llm_config.thinking_model_name,
            created_at=llm_config.created_at,
            updated_at=llm_config.updated_at,
            
            # ThinkingModel specific fields
            thinking_model_id=thinking_model.id if thinking_model else None,
            thinking_model_endpoint=thinking_model.thinking_endpoint if thinking_model else None,
            thinking_model_api_key=thinking_model.thinking_api_key if thinking_model else None,
            thinking_model_name=thinking_model.thinking_model_name if thinking_model else None
        )
