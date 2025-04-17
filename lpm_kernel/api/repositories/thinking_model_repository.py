from typing import List, Optional, Union
from datetime import datetime
from sqlalchemy import select, and_
from lpm_kernel.common.repository.base_repository import BaseRepository
from lpm_kernel.api.models.thinking_model import ThinkingModel
from lpm_kernel.api.dto.thinking_model_dto import ThinkingModelDTO, UpdateThinkingModelDTO


class ThinkingModelRepository(BaseRepository[ThinkingModel]):
    def __init__(self):
        super().__init__(ThinkingModel)

    def get_default_model(self) -> Optional[ThinkingModelDTO]:
        """Get default thinking model (ID=1)"""
        return self._get_by_id(1)

    def _get_by_id(self, id: int) -> Optional[ThinkingModelDTO]:
        """Get thinking model by ID"""
        with self._db.session() as session:
            result = session.get(ThinkingModel, id)
            return ThinkingModelDTO.from_model(result) if result else None
            
    def count(self) -> int:
        """Count total number of thinking models"""
        with self._db.session() as session:
            return session.query(ThinkingModel).count()

    def create(self, dto: ThinkingModelDTO) -> ThinkingModelDTO:
        """Create a new thinking model
        
        Args:
            dto: ThinkingModelDTO object
            
        Returns:
            Created thinking model
        """
        with self._db.session() as session:
            # Convert DTO to dictionary, filtering out None values
            create_dict = {k: v for k, v in dto.dict().items() if v is not None and k != 'id'}
            
            # Set timestamps
            now = datetime.now()
            create_dict['created_at'] = now
            create_dict['updated_at'] = now
            
            # Create entity
            entity = ThinkingModel(**create_dict)
            
            session.add(entity)
            session.commit()
            return ThinkingModelDTO.from_model(entity)
    
    def update(self, id: int, dto: Union[ThinkingModelDTO, UpdateThinkingModelDTO]) -> ThinkingModelDTO:
        """Update thinking model or create if not exists
        
        Args:
            id: Thinking model ID
            dto: ThinkingModelDTO or UpdateThinkingModelDTO object
            
        Returns:
            Updated or created thinking model
        """
        with self._db.session() as session:
            entity = session.get(ThinkingModel, id)
            
            if not entity:
                # If entity doesn't exist, create a new one
                session.commit()  # Close current transaction
                return self.create(ThinkingModelDTO(**dto.dict()))
            
            # Convert DTO to dictionary, filtering out None values
            update_dict = {k: v for k, v in dto.dict().items() if v is not None}
            
            # Update entity attributes
            for key, value in update_dict.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            
            # Update timestamp
            entity.updated_at = datetime.now()
            
            session.commit()
            return ThinkingModelDTO.from_model(entity)

    def delete(self, id: int) -> Optional[ThinkingModelDTO]:
        """Delete specified thinking model
        
        Args:
            id: ID of thinking model to delete
        
        Returns:
            Deleted thinking model or None if not found
        """
        with self._db.session() as session:
            entity = session.get(ThinkingModel, id)
            if not entity:
                return None
                
            model_dto = ThinkingModelDTO.from_model(entity)
            session.delete(entity)
            session.commit()
            return model_dto
            
    def get_all(self) -> List[ThinkingModelDTO]:
        """Get all thinking models
        
        Returns:
            List of all thinking models
        """
        with self._db.session() as session:
            results = session.query(ThinkingModel).all()
            return [ThinkingModelDTO.from_model(result) for result in results]
