"""
Cloud Inference DTO objects
"""
from typing import Dict, List, Optional

from pydantic import BaseModel

class CloudInferenceRequest(BaseModel):
    """Cloud inference request in OpenAI-compatible format"""
    # Core OpenAI API fields
    messages: List[Dict[str, str]]  # OpenAI compatible messages array
    model_id: str  # Model identifier (deployment ID)
    temperature: float = 0.1  # Temperature parameter for controlling randomness
    max_tokens: int = 2000  # Maximum tokens to generate
    stream: bool = False  # Whether to stream response
    
    # Knowledge retrieval parameters for hybrid architecture
    enable_l0_retrieval: bool = False  # Enable L0 knowledge retrieval
    enable_l1_retrieval: bool = False  # Enable L1 knowledge retrieval
    role_id: Optional[str] = None  # Role ID for role-based retrieval
