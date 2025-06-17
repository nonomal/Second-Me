from enum import Enum
from typing import List


class CloudProcessStep(Enum):
    """Cloud training process steps"""
    
    # Cloud training steps

    UPLOAD_TRAINING_DATA = "upload_training_data"
    CREATE_FINE_TUNE_JOB = "create_fine_tune_job"
    WAIT_FOR_FINE_TUNE_COMPLETION = "wait_for_fine_tune_completion"

    @classmethod
    def get_ordered_steps(cls) -> List["CloudProcessStep"]:
        """Get ordered steps"""
        return [
            # Cloud training steps

            cls.UPLOAD_TRAINING_DATA,
            cls.CREATE_FINE_TUNE_JOB,
            cls.WAIT_FOR_FINE_TUNE_COMPLETION,
        ]
        
    def get_method_name(self) -> str:
        """Get the corresponding method name for this step"""
        return self.value
