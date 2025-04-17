from typing import Optional, List
from lpm_kernel.api.repositories.user_llm_config_repository import UserLLMConfigRepository
from lpm_kernel.api.repositories.thinking_model_repository import ThinkingModelRepository
from lpm_kernel.api.dto.user_llm_config_dto import (
    UserLLMConfigDTO,
    UpdateUserLLMConfigDTO
)
from lpm_kernel.api.dto.thinking_model_dto import (
    ThinkingModelDTO,
    UpdateThinkingModelDTO
)
from lpm_kernel.api.dto.combined_llm_config_dto import CombinedLLMConfigDTO
from datetime import datetime
from lpm_kernel.common.logging import logger


class UserLLMConfigService:
    """User LLM Configuration Service"""

    def __init__(self):
        self.repository = UserLLMConfigRepository()
        self.thinking_repository = ThinkingModelRepository()

    def get_available_llm(self) -> Optional[CombinedLLMConfigDTO]:
        """Get available LLM configuration with thinking model
        Returns a combined DTO with fields from both UserLLMConfig and ThinkingModel
        """
        llm_config = self.repository.get_default_config()
        thinking_model = self.thinking_repository.get_default_model()
        
        return CombinedLLMConfigDTO.from_dtos(llm_config, thinking_model)
    

    def update_config(
        self, 
        config_id: int, 
        dto: UpdateUserLLMConfigDTO
    ) -> UserLLMConfigDTO:
        """Update configuration or create if not exists
        
        This method ensures that only one configuration record exists in the database.
        If the configuration with the given ID doesn't exist, it will be created.
        
        Args:
            config_id: Configuration ID (should be 1)
            dto: UpdateUserLLMConfigDTO object
            
        Returns:
            Updated or created configuration
        """
        # Check if we need to clean up extra records
        self._ensure_single_record()
        
        # Update or create the configuration
        return self.repository.update(config_id, dto)
    
    def delete_key(self, config_id: int = 1) -> Optional[UserLLMConfigDTO]:
        """Delete API key from the configuration
        
        This method removes the API key and related fields from the configuration.
        
        Args:
            config_id: Configuration ID (default is 1)
            
        Returns:
            Updated configuration with key removed
        """
        # Check if we need to clean up extra records
        self._ensure_single_record()
        
        # Get the current configuration
        config = self.repository.get_default_config()
        if not config:
            # If no configuration exists, return None
            return None
        
        # delete 
        return self.repository.delete(config_id)
        
    def _ensure_single_record(self):
        """Ensure that only one configuration record exists in the database"""
        # This is a safety measure to ensure we only have one record
        # In normal operation, this should never be needed
        count = self.repository.count()
        if count != 1:
            # If we have more than one record, we need to clean up
            # This is a rare case that should not happen in normal operation
            # Implementation would depend on how we want to handle this case
            # For now, we'll just log a warning
            logger.warning(f"Found {count} LLM configurations in the database. Only one should exist.")
            # Future implementation could delete extra records
            
    # Thinking Model methods
    def get_thinking_model(self, model_id: int) -> Optional[ThinkingModelDTO]:
        """Get thinking model by ID
        
        Args:
            model_id: Thinking model ID
            
        Returns:
            Thinking model or None if not found
        """
        return self.thinking_repository._get_by_id(model_id)
    
    def get_all_thinking_models(self) -> List[ThinkingModelDTO]:
        """Get all thinking models
        
        Returns:
            List of all thinking models
        """
        return self.thinking_repository.get_all()

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
        return self.thinking_repository.update(model_id, dto)
    
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
        return self.thinking_repository.create(dto)
    
    def delete_thinking_model(self, model_id: int) -> Optional[ThinkingModelDTO]:
        """Delete thinking model
        
        Args:
            model_id: Thinking model ID
            
        Returns:
            Deleted thinking model or None if not found
        """
        return self.thinking_repository.delete(model_id)
