from typing import Dict, Optional
import json
from dataclasses import dataclass, asdict
from enum import Enum


class Status(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Step:
    name: str = ""
    step_status: Status = Status.PENDING


@dataclass
class Stage:
    name: str
    progress: float = 0
    stage_status: Status = Status.PENDING
    steps: Dict[str, Step] = None
    current_step: Optional[str] = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = {}


class TrainProgress:
    def __init__(self):
        # 不仅仅是一个预存，还会根据状态变化而变化，是带了状态的
        # 预设一个template起始状态
        # copyFrom（template）
        #
        self.stages = {
            "??": Stage(
                name="Downloading the Base Model",
                steps={
                    "model_download": Step(name="Model Download")
                }
            ),
            "xx": Stage(
                name="Activating the Memory Matrix",
                steps={
                    "list_documents": Step(name="List Documents"),
                    "generate_document_embeddings": Step(name="Generate Document Embeddings"),
                    "process_chunks": Step(name="Process Chunks"),
                    "chunk_embedding": Step(name="Chunk Embedding"),
                }
            ),
            "aa": Stage(
                name="Synthesize Your Life Narrative",
                steps={
                    "extract_dimensional_topics": Step(name="Extract Dimensional Topics"),
                    "map_your_entity_network": Step(name="Map Your Entity Network"),
                }
            ),
            "bb": Stage(
                name="Prepare Training Data for Deep Comprehension",
                steps={
                    "decode_preference_patterns": Step(name="Decode Preference Patterns"),
                    "reinforce_identity": Step(name="Reinforce Identity"),
                    "augment_content_retention": Step(name="Augment Content Retention"),
                }
            ),
            "dd": Stage(
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
        self.overall_status: Status = Status.PENDING

    def update_progress(self, stage: str, step: str, current_step_status: Status, stageProgress: Optional[float] = None):

        current_stage_obj = self.stages[stage]

        current_step_obj = current_stage_obj.steps[step]
        current_step_obj.step_status = current_step_status

        self._update_stage_progress(current_stage_obj, stageProgress)
        self._update_stage_status(current_stage_obj, stage, step)

        # Update overall progress
        completed_progress = sum(s.progress for s in self.stages.values())
        self.overall_progress = completed_progress / len(self.stages)

        self._update_overall_status()

    def _update_overall_status(self):
        # Update overall status
        if all(stage.stage_status == Status.COMPLETED for stage in self.stages.values()):
            self.overall_status = Status.COMPLETED
        elif any(stage.stage_status == Status.FAILED for stage in self.stages.values()):
            self.overall_status = Status.FAILED
        elif any(stage.stage_status == Status.IN_PROGRESS for stage in self.stages.values()):
            self.overall_status = Status.IN_PROGRESS
        else:
            self.overall_status = Status.PENDING

    def _update_stage_status(self, current_stage_obj, stage, step):
        # Update stage status
        if all(step.overall_status == Status.COMPLETED for step in current_stage_obj.steps.values()):
            current_stage_obj.overall_status = Status.COMPLETED
            current_stage_obj.current_step = None

            # If current stage is completed, find the next uncompleted stage
            next_stage = None
            # wrong fixme right order.
            for stage_name, stage_data in self.stages.items():
                if stage_data.stage_status != Status.COMPLETED:
                    next_stage = stage_name
                    break
            self.current_stage = next_stage
        elif any(s.overall_status == Status.FAILED for s in current_stage_obj.steps.values()):
            current_stage_obj.overall_status = Status.FAILED
        else:
            current_stage_obj.overall_status = Status.IN_PROGRESS
            current_stage_obj.current_step = step
            self.current_stage = stage

    def _update_stage_progress(self, current_stage_obj, stageProgress):
        # Update stage progress
        if stageProgress is not None:
            # If progress value is provided, use it directly
            current_stage_obj.progress = stageProgress
        else:
            # Otherwise calculate progress based on the proportion of completed steps
            completed_steps = sum(1 for s in current_stage_obj.steps.values() if s.completed)
            total_steps = len(current_stage_obj.steps)
            current_stage_obj.progress = (completed_steps / total_steps) * 100

    def to_dict(self) -> dict:
        """Convert progress status to dictionary format"""
        # Create result dictionary with basic properties
        result = {
            "stages": [],
            "overall_progress": self.overall_progress,
            "current_stage": self.current_stage,
            "status": self.overall_status.value
        }
        
        # Define the order of stages
        stage_order = [
            "downloading_the_base_model",
            "activating_the_memory_matrix",
            "synthesize_your_life_narrative",
            "prepare_training_data_for_deep_comprehension",
            "training_to_create_second_me"
        ]
        
        # Process stages in the defined order
        for stage_name in stage_order:
            if stage_name in self.stages:
                stage = self.stages[stage_name]
                stage_dict = asdict(stage)
                
                # Convert enum values to strings
                stage_dict["status"] = stage.overall_status.value
                
                # Create steps as a list of dictionaries
                steps_list = []
                for step_key, step in stage.steps.items():
                    steps_list.append({
                        "name": step.name,
                        "completed": step.completed,
                        "status": step.overall_status.value
                    })
                
                # Replace steps dict with steps list
                stage_dict["steps"] = steps_list
                
                # Add stage to stages list with its name
                result["stages"].append({
                    "name": stage.name,
                    "progress": stage.progress,
                    "status": stage.overall_status.value,
                    "current_step": stage.current_step,
                    "steps": steps_list
                })
        
        return result
    
    def reset(self):
        """Reset all progress statuses"""
        self.__init__()
