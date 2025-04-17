from typing import List, Optional
from lpm_kernel.api.repositories.thinking_model_repository import ThinkingModelRepository
from lpm_kernel.api.dto.thinking_model_dto import (
    ThinkingModelDTO,
    UpdateThinkingModelDTO
)
from datetime import datetime
from lpm_kernel.common.logging import logger


class ThinkingModelService:
    """Thinking Model Service"""

    def __init__(self):
        self.repository = ThinkingModelRepository()

    def get_thinking_model(self, model_id: int) -> Optional[ThinkingModelDTO]:
        """Get thinking model by ID
        
        Args:
            model_id: Thinking model ID
            
        Returns:
            Thinking model or None if not found
        """
        return self.repository._get_by_id(model_id)
    
    def get_all_thinking_models(self) -> List[ThinkingModelDTO]:
        """Get all thinking models
        
        Returns:
            List of all thinking models
        """
        return self.repository.get_all()

    def update_thinking_model(
        self, 
        model_id: int, 
        dto: UpdateThinkingModelDTO
    ) -> ThinkingModelDTO:
        """Update thinking model or create if not exists
        
        Args:
            model_id: Thinking model ID
            dto: UpdateThinkingModelDTO object
            
        Returns:
            Updated or created thinking model
        """
        return self.repository.update(model_id, dto)
    
    def create_thinking_model(
        self,
        dto: ThinkingModelDTO
    ) -> ThinkingModelDTO:
        """Create a new thinking model
        
        Args:
            dto: ThinkingModelDTO object
            
        Returns:
            Created thinking model
        """
        return self.repository.create(dto)
    
    def delete_thinking_model(self, model_id: int) -> Optional[ThinkingModelDTO]:
        """Delete thinking model
        
        Args:
            model_id: Thinking model ID
            
        Returns:
            Deleted thinking model or None if not found
        """
        return self.repository.delete(model_id)
