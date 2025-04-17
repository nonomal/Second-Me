from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime
from lpm_kernel.common.repository.database_session import Base


class ThinkingModel(Base):
    """Thinking Model configuration model"""
    __tablename__ = 'thinking_models'

    id = Column(Integer, primary_key=True)
    thinking_model_name = Column(String(200), nullable=False, comment='Thinking model name')
    thinking_endpoint = Column(String(200), nullable=False, comment='Thinking API endpoint')
    thinking_api_key = Column(String(200), nullable=True, comment='Thinking API key')
    
    created_at = Column(DateTime, default=datetime.utcnow, comment='Creation time')
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment='Update time')

    def __repr__(self):
        return f'<ThinkingModel {self.id}>'

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'thinking_model_name': self.thinking_model_name,
            'thinking_endpoint': self.thinking_endpoint,
            'thinking_api_key': self.thinking_api_key,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    @classmethod
    def from_dict(cls, data):
        """Create instance from dictionary"""
        return cls(
            id=data.get('id'),
            thinking_model_name=data.get('thinking_model_name'),
            thinking_endpoint=data.get('thinking_endpoint'),
            thinking_api_key=data.get('thinking_api_key'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )
