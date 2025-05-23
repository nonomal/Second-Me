import os
import json
from typing import Optional, Dict, Any
from pathlib import Path

from lpm_kernel.api.domains.cloud_service.cloud_process_step import CloudProcessStep
from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()

class CloudStatus:
    """Cloud training status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"

class CloudProgressHolder:
    """Cloud training progress holder"""
    
    def __init__(self, model_name: str):
        """
        Initialize progress holder
        
        Args:
            model_name: Model name
        """
        self.model_name = model_name
        self.progress_dir = Path("data/cloud_progress")
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.progress_dir / f"{model_name}.json"
        
        # Initialize progress data
        self.progress = self._load_progress()
        
    def _load_progress(self) -> Dict[str, Any]:
        """
        Load progress from file
        
        Returns:
            Dict: Progress data
        """
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading progress: {str(e)}")
        
        # Initialize new progress data
        # 使用CloudStatus的字符串值而不是类属性
        pending_status = CloudStatus.PENDING  # 这是字符串 "pending"
        return {
            "model_name": self.model_name,
            "status": pending_status,
            "steps": {step.value: {"status": pending_status, "progress": 0} for step in CloudProcessStep},
            "current_step": None,
            "message": "Initialized",
            "timestamp": None
        }
    
    def save_progress(self):
        """Save progress to file"""
        try:
            # 确保所有值都是可序列化的
            # 如果有CloudStatus类的属性，将其转换为字符串
            progress_copy = self._ensure_serializable(self.progress)
            
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_copy, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress: {str(e)}")
            
    def _ensure_serializable(self, obj):
        """Ensure all values in the object are JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._ensure_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):  # 如果是自定义对象，转换为字符串
            return str(obj)
        else:
            return obj
    
    def mark_step_status(self, step: CloudProcessStep, status: str, message: str = None):
        """
        Mark step status
        
        Args:
            step: Process step
            status: Status (string value)
            message: Optional message
        """
        if step.value not in self.progress["steps"]:
            self.progress["steps"][step.value] = {"status": CloudStatus.PENDING, "progress": 0}
        
        # 确保使用字符串值
        self.progress["steps"][step.value]["status"] = status
        
        # 使用字符串比较和赋值
        if status == CloudStatus.IN_PROGRESS:
            self.progress["current_step"] = step.value
            self.progress["status"] = status  # 使用传入的status参数
        elif status == CloudStatus.COMPLETED:
            self.progress["steps"][step.value]["progress"] = 100
        elif status == CloudStatus.FAILED:
            self.progress["status"] = status  # 使用传入的status参数
        elif status == CloudStatus.SUSPENDED:
            self.progress["status"] = status  # 使用传入的status参数
        
        if message:
            self.progress["steps"][step.value]["message"] = message
            self.progress["message"] = message
        
        import time
        self.progress["timestamp"] = time.time()
        
        self.save_progress()
    
    def update_step_progress(self, step: CloudProcessStep, progress: float, message: str = None):
        """
        Update step progress
        
        Args:
            step: Process step
            progress: Progress percentage (0-100)
            message: Optional message
        """
        if step.value not in self.progress["steps"]:
            self.progress["steps"][step.value] = {"status": CloudStatus.PENDING, "progress": 0}
        
        self.progress["steps"][step.value]["progress"] = progress
        
        if progress < 100:
            self.progress["steps"][step.value]["status"] = CloudStatus.IN_PROGRESS
            self.progress["current_step"] = step.value
            self.progress["status"] = CloudStatus.IN_PROGRESS
        else:
            self.progress["steps"][step.value]["status"] = CloudStatus.COMPLETED
        
        if message:
            self.progress["steps"][step.value]["message"] = message
            self.progress["message"] = message
        
        import time
        self.progress["timestamp"] = time.time()
        
        self.save_progress()
    
    def get_last_successful_step(self) -> Optional[CloudProcessStep]:
        """
        Get the last successfully completed step
        
        Returns:
            Optional[CloudProcessStep]: Last successful step or None
        """
        ordered_steps = CloudProcessStep.get_ordered_steps()
        last_successful = None
        
        for step in ordered_steps:
            if (step.value in self.progress["steps"] and 
                self.progress["steps"][step.value]["status"] == CloudStatus.COMPLETED):
                last_successful = step
            else:
                break
        
        return last_successful
    
    def reset_progress(self):
        """Reset progress"""
        self.progress = {
            "model_name": self.model_name,
            "status": CloudStatus.PENDING,
            "steps": {step.value: {"status": CloudStatus.PENDING, "progress": 0} for step in CloudProcessStep},
            "current_step": None,
            "message": "Reset",
            "timestamp": None
        }
        self.save_progress()
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get progress data
        
        Returns:
            Dict: Progress data
        """
        return self.progress
