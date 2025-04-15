from typing import List, Dict, Optional, Union, Any
import json
from dataclasses import dataclass, field
from enum import Enum


class Status(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"


class TrainProgress:
    def __init__(self):
        # Define the complete data structure directly in the format matching the desired JSON output
        self.data = {
            "stages": [
                {
                    "name": "Downloading the Base Model",
                    "progress": 0.0,
                    "status": "pending",
                    "current_step": None,
                    "steps": [
                        {
                            "name": "Model Download",
                            "completed": False,
                            "status": "pending"
                        }
                    ]
                },
                {
                    "name": "Activating the Memory Matrix",
                    "progress": 0.0,
                    "status": "pending",
                    "current_step": None,
                    "steps": [
                        {
                            "name": "List Documents",
                            "completed": False,
                            "status": "pending"
                        },
                        {
                            "name": "Generate Document Embeddings",
                            "completed": False,
                            "status": "pending"
                        },
                        {
                            "name": "Process Chunks",
                            "completed": False,
                            "status": "pending"
                        },
                        {
                            "name": "Chunk Embedding",
                            "completed": False,
                            "status": "pending"
                        }
                    ]
                },
                {
                    "name": "Synthesize Your Life Narrative",
                    "progress": 0.0,
                    "status": "pending",
                    "current_step": None,
                    "steps": [
                        {
                            "name": "Extract Dimensional Topics",
                            "completed": False,
                            "status": "pending"
                        },
                        {
                            "name": "Map Your Entity Network",
                            "completed": False,
                            "status": "pending"
                        }
                    ]
                },
                {
                    "name": "Prepare Training Data for Deep Comprehension",
                    "progress": 0.0,
                    "status": "pending",
                    "current_step": None,
                    "steps": [
                        {
                            "name": "Decode Preference Patterns",
                            "completed": False,
                            "status": "pending"
                        },
                        {
                            "name": "Reinforce Identity",
                            "completed": False,
                            "status": "pending"
                        },
                        {
                            "name": "Augment Content Retention",
                            "completed": False,
                            "status": "pending"
                        }
                    ]
                },
                {
                    "name": "Training to create Second Me",
                    "progress": 0.0,
                    "status": "pending",
                    "current_step": None,
                    "steps": [
                        {
                            "name": "Train",
                            "completed": False,
                            "status": "pending"
                        },
                        {
                            "name": "Merge Weights",
                            "completed": False,
                            "status": "pending"
                        },
                        {
                            "name": "Convert Model",
                            "completed": False,
                            "status": "pending"
                        }
                    ]
                }
            ],
            "overall_progress": 0.0,
            "current_stage": None,
            "status": "pending"
        }
        
        # Create stage name to stage data mapping
        self.stage_map = {}
        for stage in self.data["stages"]:
            stage_name = stage["name"].lower().replace(" ", "_")
            self.stage_map[stage_name] = stage
            
        # Create step name to step data mapping for each stage
        self.steps_map = {}
        for stage_name, stage in self.stage_map.items():
            self.steps_map[stage_name] = {}
            for step in stage["steps"]:
                step_name = step["name"].lower().replace(" ", "_")
                self.steps_map[stage_name][step_name] = step

    def update_progress(self, stage: str, step: str, status: Union[Status, str], progress: Optional[float] = None):
        """Update progress status
        Args:
            stage: Stage key (snake_case format)
            step: Step key (snake_case format)
            status: Status (enum or string)
            progress: Optional progress value (0-100)
        """
        if stage not in self.stage_map:
            raise ValueError(f"Invalid stage: {stage}")
            
        stage_data = self.stage_map[stage]
        
        # Convert status to string if it's an enum
        status_value = status.value if isinstance(status, Status) else status
        
        # Find step in the stage
        if stage not in self.steps_map or step not in self.steps_map[stage]:
            raise ValueError(f"Invalid step {step} for stage {stage}")
            
        step_data = self.steps_map[stage][step]
        
        # Update step status
        step_data["status"] = status_value
        step_data["completed"] = status_value == "completed"
        
        # Update stage progress
        if progress is not None:
            # If progress value is provided, use it directly
            stage_data["progress"] = progress
        else:
            # Otherwise calculate progress based on the proportion of completed steps
            completed_steps = sum(1 for s in stage_data["steps"] if s["completed"])
            total_steps = len(stage_data["steps"])
            stage_data["progress"] = (completed_steps / total_steps) * 100.0
        
        # Update stage status and current step
        if all(s["completed"] for s in stage_data["steps"]):
            stage_data["status"] = "completed"
            stage_data["current_step"] = None
            
            # If current stage is completed, find the next uncompleted stage
            next_stage = None
            for stage_name, stage_info in self.stage_map.items():
                if stage_info["status"] != "completed":
                    next_stage = stage_name
                    break
            self.data["current_stage"] = next_stage
        elif any(s["status"] == "failed" for s in stage_data["steps"]):
            stage_data["status"] = "failed"
            stage_data["current_step"] = step_data["name"]
            self.data["current_stage"] = stage_data["name"]
        elif any(s["status"] == "suspended" for s in stage_data["steps"]):
            stage_data["status"] = "suspended"
            stage_data["current_step"] = step_data["name"]
            self.data["current_stage"] = stage_data["name"]
        else:
            stage_data["status"] = "in_progress"
            stage_data["current_step"] = step_data["name"]
            self.data["current_stage"] = stage_data["name"]
        
        # Update overall progress
        completed_progress = sum(s["progress"] for s in self.data["stages"])
        self.data["overall_progress"] = completed_progress / len(self.data["stages"])
        
        # Update overall status
        if all(s["status"] == "completed" for s in self.data["stages"]):
            self.data["status"] = "completed"
        elif any(s["status"] == "failed" for s in self.data["stages"]):
            self.data["status"] = "failed"
        elif any(s["status"] == "suspended" for s in self.data["stages"]):
            self.data["status"] = "suspended"
        elif any(s["status"] == "in_progress" for s in self.data["stages"]):
            self.data["status"] = "in_progress"
        else:
            self.data["status"] = "pending"

    def to_dict(self) -> dict:
        """Convert progress status to dictionary format"""
        return self.data
    
    def reset(self):
        """Reset all progress statuses"""
        self.__init__()