from enum import Enum
from typing import List


class CloudProcessStep(Enum):
    """Cloud training process steps"""

    # 数据处理步骤
    LIST_DOCUMENTS = "list_documents"
    GENERATE_DOCUMENT_EMBEDDINGS = "generate_document_embeddings"
    CHUNK_DOCUMENT = "process_chunks"
    CHUNK_EMBEDDING = "chunk_embedding"
    EXTRACT_DIMENSIONAL_TOPICS = "extract_dimensional_topics"
    GENERATE_BIOGRAPHY = "generate_biography"
    MAP_YOUR_ENTITY_NETWORK = "map_your_entity_network"
    DECODE_PREFERENCE_PATTERNS = "decode_preference_patterns"
    REINFORCE_IDENTITY = "reinforce_identity"
    AUGMENT_CONTENT_RETENTION = "augment_content_retention"
    
    # 云端训练步骤
    PREPARE_TRAINING_DATA = "prepare_training_data"
    UPLOAD_TRAINING_DATA = "upload_training_data"
    CREATE_FINE_TUNE_JOB = "create_fine_tune_job"
    WAIT_FOR_FINE_TUNE_COMPLETION = "wait_for_fine_tune_completion"

    @classmethod
    def get_ordered_steps(cls) -> List["CloudProcessStep"]:
        """Get ordered steps"""
        return [
            cls.LIST_DOCUMENTS,
            cls.GENERATE_DOCUMENT_EMBEDDINGS,
            cls.CHUNK_DOCUMENT,
            cls.CHUNK_EMBEDDING,
            cls.EXTRACT_DIMENSIONAL_TOPICS,
            cls.GENERATE_BIOGRAPHY,
            cls.MAP_YOUR_ENTITY_NETWORK,
            cls.DECODE_PREFERENCE_PATTERNS,
            cls.REINFORCE_IDENTITY,
            cls.AUGMENT_CONTENT_RETENTION,
            # 云端训练步骤
            cls.PREPARE_TRAINING_DATA,
            cls.UPLOAD_TRAINING_DATA,
            cls.CREATE_FINE_TUNE_JOB,
            cls.WAIT_FOR_FINE_TUNE_COMPLETION,
        ]
        
    def get_method_name(self) -> str:
        """Get the corresponding method name for this step"""
        return self.value
