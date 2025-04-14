from typing import Dict, Optional
import json
from dataclasses import dataclass, asdict
from enum import Enum


class Status(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"


@dataclass
class Step:
    name: str = ""
    completed: bool = False
    status: Status = Status.PENDING


@dataclass
class Stage:
    name: str
    progress: float = 0
    status: Status = Status.PENDING
    steps: Dict[str, Step] = None
    current_step: Optional[str] = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = {}


class TrainProgress:
    def __init__(self):
        self.stages = {
            "downloading_the_base_model": Stage(
                name="Downloading the Base Model",
                steps={
                    "model_download": Step(name="Model Download")
                }
            ),
            "activating_the_memory_matrix": Stage(
                name="Activating the Memory Matrix",
                steps={
                    "list_documents": Step(name="List Documents"),
                    "generate_document_embeddings": Step(name="Generate Document Embeddings"),
                    "process_chunks": Step(name="Process Chunks"),
                    "chunk_embedding": Step(name="Chunk Embedding"),
                }
            ),
            "synthesize_your_life_narrative": Stage(
                name="Synthesize Your Life Narrative",
                steps={
                    "extract_dimensional_topics": Step(name="Extract Dimensional Topics"),
                    "map_your_entity_network": Step(name="Map Your Entity Network"),
                }
            ),
            "prepare_training_data_for_deep_comprehension": Stage(
                name="Prepare Training Data for Deep Comprehension",
                steps={
                    "decode_preference_patterns": Step(name="Decode Preference Patterns"),
                    "reinforce_identity": Step(name="Reinforce Identity"),
                    "augment_content_retention": Step(name="Augment Content Retention"),
                }
            ),
            "training_to_create_second_me": Stage(
                name="Training to create Second Me",
                steps={
                    "train": Step(name="Train"),
                    "merge_weights": Step(name="Merge Weights"),
                    "convert_model": Step(name="Convert Model"),                    
                }
            )
        }
        self.overall_progress: float = 0
        self.current_stage: Optional[str] = None
        self.status: Status = Status.PENDING

    def update_progress(self, stage: str, step: str, status: Status, progress: Optional[float] = None):
        """Update progress status
        Args:
            stage: Stage key (snake_case format)
            step: Step key (snake_case format)
            status: Status
            progress: Optional progress value (0-100)
        """
        if stage not in self.stages:
            raise ValueError(f"Invalid stage: {stage}")
        
        stage_obj = self.stages[stage]
        if step not in stage_obj.steps:
            raise ValueError(f"Invalid step {step} for stage {stage}")
        
        # Update step status
        step_obj = stage_obj.steps[step]
        step_obj.status = status
        step_obj.completed = status == Status.COMPLETED
        
        # Update stage progress
        if progress is not None:
            # If progress value is provided, use it directly
            stage_obj.progress = progress
        else:
            # Otherwise calculate progress based on the proportion of completed steps
            completed_steps = sum(1 for s in stage_obj.steps.values() if s.completed)
            total_steps = len(stage_obj.steps)
            stage_obj.progress = (completed_steps / total_steps) * 100
        
        # Update stage status
        if all(s.completed for s in stage_obj.steps.values()):
            stage_obj.status = Status.COMPLETED
            stage_obj.current_step = None
            
            # If current stage is completed, find the next uncompleted stage
            next_stage = None
            for stage_name, stage_data in self.stages.items():
                if stage_data.status != Status.COMPLETED:
                    next_stage = stage_name
                    break
            self.current_stage = next_stage
        elif any(s.status == Status.FAILED for s in stage_obj.steps.values()):
            stage_obj.status = Status.FAILED
        elif any(s.status == Status.SUSPENDED for s in stage_obj.steps.values()):
            stage_obj.status = Status.SUSPENDED
        else:
            stage_obj.status = Status.IN_PROGRESS
            stage_obj.current_step = step
            self.current_stage = stage
        
        # Update overall progress
        completed_progress = sum(s.progress for s in self.stages.values())
        self.overall_progress = completed_progress / len(self.stages)
        
        # Update overall status
        if all(s.status == Status.COMPLETED for s in self.stages.values()):
            self.status = Status.COMPLETED
        elif any(s.status == Status.FAILED for s in self.stages.values()):
            self.status = Status.FAILED
        elif any(s.status == Status.SUSPENDED for s in self.stages.values()):
            self.status = Status.SUSPENDED
        elif any(s.status == Status.IN_PROGRESS for s in self.stages.values()):
            self.status = Status.IN_PROGRESS
        else:
            self.status = Status.PENDING

    def to_dict(self) -> dict:
        """Convert progress status to dictionary format"""
        result = {
            "stages": {},
            "overall_progress": self.overall_progress,
            "current_stage": self.current_stage,
            "status": self.status.value
        }
        
        for stage_name, stage in self.stages.items():
            stage_dict = asdict(stage)
            # Convert enum values to strings
            stage_dict["status"] = stage.status.value
            stage_dict["steps"] = {
                step_key: {
                    "name": step.name,
                    "completed": step.completed,
                    "status": step.status.value
                }
                for step_key, step in stage.steps.items()
            }
            result["stages"][stage_name] = stage_dict
        
        return result
    
    def reset(self):
        """Reset all progress statuses"""
        self.__init__()
