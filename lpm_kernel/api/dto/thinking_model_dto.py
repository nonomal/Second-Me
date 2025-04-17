from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class BaseThinkingModelDTO(BaseModel):
    """Base Thinking Model DTO"""
    thinking_model_name: str
    thinking_endpoint: str
    thinking_api_key: Optional[str] = None
    
    def dict(self, *args, **kwargs):
        result = super().dict(*args, **kwargs)
        return result


class CreateThinkingModelDTO(BaseThinkingModelDTO):
    """Create Thinking Model DTO"""
    pass


class UpdateThinkingModelDTO(BaseModel):
    """Update Thinking Model DTO"""
    thinking_model_name: Optional[str] = None
    thinking_endpoint: Optional[str] = None
    thinking_api_key: Optional[str] = None
    
    def dict(self, *args, **kwargs):
        result = super().dict(*args, **kwargs)
        return result


class ThinkingModelDTO(BaseThinkingModelDTO):
    """Thinking Model DTO"""
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_model(cls, model):
        """Create DTO from model"""
        if not model:
            return None
        return cls(
            id=model.id,
            thinking_model_name=model.thinking_model_name,
            thinking_endpoint=model.thinking_endpoint,
            thinking_api_key=model.thinking_api_key,
            created_at=model.created_at,
            updated_at=model.updated_at
        )


class ThinkingModelListDTO(BaseModel):
    """Thinking Model List DTO"""
    items: List[ThinkingModelDTO]
