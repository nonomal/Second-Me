import os
import json
from typing import Optional, Dict, Any, Union
from pathlib import Path
import time

from lpm_kernel.api.domains.cloud_service.cloud_process_step import CloudProcessStep
from lpm_kernel.configs.logging import get_train_process_logger
from lpm_kernel.api.domains.trainprocess.process_step import ProcessStep

logger = get_train_process_logger()

class CloudStatus:
    """Cloud training status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"

class CloudProgress:
    """Cloud training progress data structure"""
    
    def __init__(self):
        # Define the complete data structure for cloud training progress
        self.data = {
            "stages": [
                {
                    "name": "Activating the Memory Matrix",
                    "progress": 0.0,
                    "status": CloudStatus.PENDING,
                    "current_step": None,
                    "steps": [
                        {
                            "name": "list_documents",
                            "completed": False,
                            "status": CloudStatus.PENDING,
                            "have_output": False,
                            "path": None
                        },
                        {
                            "name": "generate_document_embeddings",
                            "completed": False,
                            "status": CloudStatus.PENDING,
                            "have_output": False,
                            "path": None
                        },
                        {
                            "name": "process_chunks",
                            "completed": False,
                            "status": CloudStatus.PENDING,
                            "have_output": False,
                            "path": None
                        },
                        {
                            "name": "chunk_embedding",
                            "completed": False,
                            "status": CloudStatus.PENDING,
                            "have_output": False,
                            "path": None
                        }
                    ]
                },
                {
                    "name": "Synthesize Your Life Narrative",
                    "progress": 0.0,
                    "status": CloudStatus.PENDING,
                    "current_step": None,
                    "steps": [
                        {
                            "name": "extract_dimensional_topics",
                            "completed": False,
                            "status": CloudStatus.PENDING,
                            "have_output": True,
                            "path": "resources/L2/data_pipeline/raw_data/topics.json"
                        },
                        {
                            "name": "generate_biography",
                            "completed": False,
                            "status": CloudStatus.PENDING,
                            "have_output": True,
                            "path": "From database"
                        },
                        {
                            "name": "map_your_entity_network",
                            "completed": False,
                            "status": CloudStatus.PENDING,
                            "have_output": True,
                            "path": "resources/L1/graphrag_indexing_output/subjective/entities.parquet"
                        }
                    ]
                },
                {
                    "name": "Prepare Training Data for Deep Comprehension",
                    "progress": 0.0,
                    "status": CloudStatus.PENDING,
                    "current_step": None,
                    "steps": [
                        {
                            "name": "decode_preference_patterns",
                            "completed": False,
                            "status": CloudStatus.PENDING,
                            "have_output": True,
                            "path": "resources/L2/data/preference.json"
                        },
                        {
                            "name": "reinforce_identity",
                            "completed": False,
                            "status": CloudStatus.PENDING,
                            "have_output": True,
                            "path": "resources/L2/data/selfqa.json"
                        },
                        {
                            "name": "augment_content_retention",
                            "completed": False,
                            "status": CloudStatus.PENDING,
                            "have_output": True,
                            "path": "resources/L2/data/diversity.json"
                        }
                    ]
                },
                {
                    "name": "Training to create Second Me",
                    "progress": 0.0,
                    "status": CloudStatus.PENDING,
                    "current_step": None,
                    "steps": [
                        {
                            "name": "upload_training_data",
                            "completed": False,
                            "status": CloudStatus.PENDING,
                            "have_output": False,
                            "path": None
                        },
                        {
                            "name": "create_fine_tune_job",
                            "completed": False,
                            "status": CloudStatus.PENDING,
                            "have_output": True,
                            "path": "job_id.json"
                        },
                        {
                            "name": "wait_for_fine_tune_completion",
                            "completed": False,
                            "status": CloudStatus.PENDING,
                            "have_output": True,
                            "path": None
                        }
                    ]
                }
            ],
            "overall_progress": 0.0,
            "current_stage": None,
            "status": CloudStatus.PENDING,
            "message": "Initialized",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": None,
            "job_id": None
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
                
        
        # Stage mapping for process steps
        self._stage_mapping = {
            # Data processing step mapping (using ProcessStep to maintain consistency with local)
            ProcessStep.LIST_DOCUMENTS: "activating_the_memory_matrix",
            ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS: "activating_the_memory_matrix",
            ProcessStep.CHUNK_DOCUMENT: "activating_the_memory_matrix",
            ProcessStep.CHUNK_EMBEDDING: "activating_the_memory_matrix",

            ProcessStep.EXTRACT_DIMENSIONAL_TOPICS: "synthesize_your_life_narrative",
            ProcessStep.GENERATE_BIOGRAPHY: "synthesize_your_life_narrative",
            ProcessStep.MAP_ENTITY_NETWORK: "synthesize_your_life_narrative",

            ProcessStep.DECODE_PREFERENCE_PATTERNS: "prepare_training_data_for_deep_comprehension",
            ProcessStep.REINFORCE_IDENTITY: "prepare_training_data_for_deep_comprehension",
            ProcessStep.AUGMENT_CONTENT_RETENTION: "prepare_training_data_for_deep_comprehension",
            
            # Cloud training step mapping
            CloudProcessStep.UPLOAD_TRAINING_DATA: "training_to_create_second_me",
            CloudProcessStep.CREATE_FINE_TUNE_JOB: "training_to_create_second_me",
            CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION: "training_to_create_second_me",
        }
    
    def update_progress(self, stage: str, step: str, current_step_status: Union[str, CloudStatus], 
                       stage_progress: Optional[float] = None, extra_info: Optional[Dict[str, Any]] = None):
        """Update progress status
        
        Args:
            stage: Stage key (snake_case format)
            step: Step key (snake_case format)
            current_step_status: Status (string or CloudStatus)
            stage_progress: Optional progress value (0-100)
            extra_info: Optional extra information to update (job_id, model_id, etc.)
        """
        stage_data = self.stage_map[stage]
        status_value = current_step_status if isinstance(current_step_status, str) else current_step_status
        step_data = self.steps_map[stage][step]
        
        # Update step status
        step_data["status"] = status_value
        step_data["completed"] = status_value == CloudStatus.COMPLETED
        
        # Update extra info if provided
        if extra_info:
            for key, value in extra_info.items():
                if key in self.data:
                    self.data[key] = value
        
        # Update stage progress
        self._update_stage_progress(stage_data, stage_progress)
        
        # Update stage status and current step
        self._update_stage_status(stage_data, step_data)
        
        # Update overall progress
        self._update_overall_progress()
        
        # Update overall status
        self._update_overall_status()
        
        # Update timestamp
        self.data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    def _update_stage_progress(self, stage_data: Dict, stage_progress: Optional[float] = None):
        """Update the progress of a stage
        
        Args:
            stage_data: Stage data dictionary
            stage_progress: Optional progress value (0-100)
        """
        if stage_progress is not None:
            stage_data["progress"] = stage_progress
        else:
            completed_steps = sum(1 for s in stage_data["steps"] if s["completed"])
            total_steps = len(stage_data["steps"])
            stage_data["progress"] = (completed_steps / total_steps) * 100.0
    
    def _update_stage_status(self, stage_data: Dict, step_data: Dict):
        """Update the status and current step of a stage
        
        Args:
            stage_data: Stage data dictionary
            step_data: Step data dictionary
        """
        if all(step["completed"] for step in stage_data["steps"]):
            stage_data["status"] = CloudStatus.COMPLETED
            stage_data["current_step"] = None
            next_stage = None
            for stage_name, stage_info in self.stage_map.items():
                if stage_info["status"] != CloudStatus.COMPLETED:
                    next_stage = stage_name
                    break
            self.data["current_stage"] = next_stage
        elif any(step["status"] == CloudStatus.FAILED for step in stage_data["steps"]):
            stage_data["status"] = CloudStatus.FAILED
            stage_data["current_step"] = step_data["name"]
            self.data["current_stage"] = stage_data["name"]
        elif any(step["status"] == CloudStatus.CANCELLED for step in stage_data["steps"]):
            stage_data["status"] = CloudStatus.CANCELLED
            stage_data["current_step"] = step_data["name"]
            self.data["current_stage"] = stage_data["name"]
        elif any(step["status"] == CloudStatus.SUSPENDED for step in stage_data["steps"]):
            stage_data["status"] = CloudStatus.SUSPENDED
            stage_data["current_step"] = step_data["name"]
            self.data["current_stage"] = stage_data["name"]
        elif step_data["status"] == CloudStatus.IN_PROGRESS:
            # Only set stage to IN_PROGRESS when current step is in IN_PROGRESS state
            stage_data["status"] = CloudStatus.IN_PROGRESS
            stage_data["current_step"] = step_data["name"]
            self.data["current_stage"] = stage_data["name"]
        else:
            # Current step is not in IN_PROGRESS state, keep stage status unchanged
            # But still update the current step
            stage_data["current_step"] = step_data["name"]
    
    def _update_overall_progress(self):
        """Update the overall progress based on all stages"""
        completed_progress = sum(s["progress"] for s in self.data["stages"])
        self.data["overall_progress"] = completed_progress / len(self.data["stages"])
    
    def _update_overall_status(self):
        """Update the overall status based on all stages"""
        if all(s["status"] == CloudStatus.COMPLETED for s in self.data["stages"]):
            self.data["status"] = CloudStatus.COMPLETED
        elif any(s["status"] == CloudStatus.FAILED for s in self.data["stages"]):
            self.data["status"] = CloudStatus.FAILED
        elif any(s["status"] == CloudStatus.CANCELLED for s in self.data["stages"]):
            self.data["status"] = CloudStatus.CANCELLED
        elif any(s["status"] == CloudStatus.SUSPENDED for s in self.data["stages"]):
            self.data["status"] = CloudStatus.SUSPENDED
        elif any(s["status"] == CloudStatus.IN_PROGRESS for s in self.data["stages"]):
            self.data["status"] = CloudStatus.IN_PROGRESS
        else:
            self.data["status"] = CloudStatus.PENDING
    
    def to_dict(self) -> dict:
        """Convert progress status to dictionary format"""
        return self.data
    
    def reset(self):
        """Reset all progress statuses"""
        self.__init__()

class CloudProgressHolder:
    """Cloud training progress holder"""
    
    def __init__(self, model_name=None, job_id=None):
        self.model_name = model_name
        self.job_id = job_id
        self.progress = CloudProgress()
        
        # Set identifiers in progress data
        if model_name:
            self.progress.data["model_name"] = model_name
        if job_id:
            self.progress.data["job_id"] = job_id
            
        # Set progress file path - always use the same file
        progress_dir = Path("data/cloud_progress")
        progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Use fixed filename
        self.progress_file = progress_dir / "cloud_progress.json"
        self._load_progress()
    
    @staticmethod
    def get_latest_progress():
        """
        Get the latest progress data
        
        Returns:
            tuple: (CloudProgressHolder, job_id) Returns (CloudProgressHolder instance, job_id) if progress data is found; otherwise returns (None, None)
        """
        progress_file = Path("data/cloud_progress/cloud_progress.json")
        if not progress_file.exists():
            return None, None
            
        try:
            # Read file content
            with open(progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Get model name and job_id
            model_name = data.get("model_name")
            job_id = data.get("job_id")
            
            # Create progress holder instance
            holder = CloudProgressHolder(model_name=model_name, job_id=job_id)
            holder.progress.data = data
            holder._rebuild_mappings()
            holder._reset_in_progress_status()
            
            return holder, job_id
        except Exception as e:
            logger.error(f"Error reading progress file {progress_file}: {str(e)}")
            return None, None
    
    def _rebuild_mappings(self):
        """Rebuild progress data mappings"""
        # 重新创建映射
        self.progress.stage_map = {}
        for stage in self.progress.data["stages"]:
            stage_name = stage["name"].lower().replace(" ", "_")
            self.progress.stage_map[stage_name] = stage
            
        self.progress.steps_map = {}
        for stage_name, stage in self.progress.stage_map.items():
            self.progress.steps_map[stage_name] = {}
            for step in stage["steps"]:
                step_name = step["name"].lower().replace(" ", "_")
                self.progress.steps_map[stage_name][step_name] = step
        
    def _load_progress(self):
        """
        Load progress data from file
        """
        if not self.progress_file or not os.path.exists(self.progress_file):
            return
            
        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                
                # Load progress data
                self.progress.data = saved_data
                
                # If model name or job_id not specified, load from file
                if not self.model_name:
                    self.model_name = saved_data.get("model_name")
                if not self.job_id:
                    self.job_id = saved_data.get("job_id")
                
                # Rebuild mappings
                self._rebuild_mappings()
                
                # Reset in-progress status
                self._reset_in_progress_status()
                
                logger.debug(f"Loaded progress data from {self.progress_file}")
        except Exception as e:
            logger.error(f"Error loading progress: {str(e)}")
    
    def _reset_in_progress_status(self):
        """
        Reset any in_progress status to failed after loading from file
        """
        need_save = False
        
        # Check overall status
        if self.progress.data["status"] == CloudStatus.IN_PROGRESS:
            self.progress.data["status"] = CloudStatus.FAILED
            need_save = True
            logger.info("Reset overall in_progress status to failed")
        
        # Check each stage
        for stage in self.progress.data["stages"]:
            if stage["status"] == CloudStatus.IN_PROGRESS:
                stage["status"] = CloudStatus.FAILED
                need_save = True
                logger.info(f"Reset stage '{stage['name']}' in_progress status to failed")
            
            # Check each step in the stage
            for step in stage["steps"]:
                if step["status"] == CloudStatus.IN_PROGRESS:
                    step["status"] = CloudStatus.FAILED
                    step["completed"] = False
                    need_save = True
                    logger.info(f"Reset step '{step['name']}' in_progress status to failed")
        
        # If there are any changes, save progress
        if need_save:
            self.save_progress()
            logger.info("Saved progress after resetting in_progress statuses")
    
    def save_progress(self):
        """
        Update progress in memory (no file saving)
        """
        try:
            # Update timestamp
            self.progress.data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Ensure progress file path exists
            progress_dir = Path("data/cloud_progress")
            progress_dir.mkdir(parents=True, exist_ok=True)
            self.progress_file = progress_dir / "cloud_progress.json"
            
            # Save progress data
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(self.progress.data, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"Progress saved to {self.progress_file}")
        except Exception as e:
            logger.error(f"Error saving progress: {str(e)}")
    
    def mark_step_status(self, step, status: str, step_name: str = None):
        """
        Mark step status
        
        Args:
            step: Process step (CloudProcessStep/ProcessStep enum or stage name string)
            status: Status (string value)
            step_name: Optional step name (only used when step is a string)
        """
        try:
            # 处理不同类型的step参数
            if hasattr(step, 'value') and isinstance(step.value, str):
                # If it's an enum type (object with value attribute)
                stage_name = self.progress._stage_mapping[step]
                step_name = step.value
            else:
                # 如果是字符串，直接使用
                stage_name = step
                # 如果没有提供step_name，使用stage_name
                if step_name is None:
                    step_name = stage_name
            
            # Ensure status is string type
            if hasattr(status, 'value') and isinstance(status.value, str):
                status_str = status.value
            else:
                status_str = str(status)
                
            # Update step status
            self.progress.update_progress(stage_name, step_name, status_str)
            
            # 保存进度
            self.save_progress()
        except Exception as e:
            logger.error(f"Error marking step status: {str(e)}")
            logger.debug(f"Error details: step={step}, stage_name={stage_name if 'stage_name' in locals() else 'unknown'}, step_name={step_name if 'step_name' in locals() else 'unknown'}")


    def update_step_progress(self, step, progress: float, message: str = None, step_name: str = None):
        """
        Update step progress
        
        Args:
            step: Process step (CloudProcessStep/ProcessStep enum or stage name string)
            progress: Progress value (0-100)
            message: Optional message
            step_name: Optional step name (only used when step is a string)
        """
        try:
            # 准备额外信息
            extra_info = {}
            if message:
                extra_info["message"] = message
            
            # 确定状态
            status = None
            if progress >= 100:
                status = CloudStatus.COMPLETED
            elif progress > 0:
                status = CloudStatus.IN_PROGRESS
            else:
                status = CloudStatus.PENDING
            
            # 处理不同类型的step参数
            if hasattr(step, 'value') and isinstance(step.value, str):
                # 如果是枚举类型（具有value属性的对象）
                if hasattr(self.progress, '_stage_mapping') and step in self.progress._stage_mapping:
                    # 如果是CloudProcessStep枚举且在映射中
                    stage_name = self.progress._stage_mapping.get(step)
                    if not stage_name:
                        logger.error(f"No stage mapping found for step: {step}")
                        return
                else:
                    # 如果是CloudProcessStep枚举但不在映射中，使用默认阶段名称
                    if isinstance(step, CloudProcessStep):
                        # 对于MAP_YOUR_ENTITY_NETWORK步骤，使用synthesize_your_life_narrative阶段
                        if step == CloudProcessStep.MAP_YOUR_ENTITY_NETWORK:
                            stage_name = "synthesize_your_life_narrative"
                        # 对于DECODE_PREFERENCE_PATTERNS步骤，使用prepare_training_data_for_deep_comprehension阶段
                        elif step == CloudProcessStep.DECODE_PREFERENCE_PATTERNS:
                            stage_name = "prepare_training_data_for_deep_comprehension"
                        else:
                            # 其他CloudProcessStep枚举使用默认阶段
                            stage_name = "prepare_training_data_for_deep_comprehension"
                            logger.warning(f"CloudProcessStep {step} not found in _stage_mapping, using default stage")
                    else:
                        # 如果是ProcessStep枚举或其他枚举，直接使用其值
                        stage_name = step.value.lower().replace(" ", "_")
                
                # 获取步骤名称
                step_name = step.value.lower().replace(" ", "_")
            else:
                # 如果是字符串，直接使用
                stage_name = step
                # 如果没有提供step_name，使用stage_name
                if step_name is None:
                    step_name = stage_name
            
            # 更新进度
            self.progress.update_progress(stage_name, step_name, status, progress, extra_info)
            
            # 保存进度
            self.save_progress()
        except Exception as e:
            logger.error(f"Error updating step progress: {str(e)}")
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get progress data
        
        Returns:
            Dict: Progress data
        """
        return self.progress.data
        
    def is_stage_completed(self, stage_name):
        """
        检查指定阶段是否已完成
        
        Args:
            stage_name: 阶段名称（格式化后的，如"activating_the_memory_matrix"）
            
        Returns:
            bool: 如果阶段已完成返回True，否则返回False
        """
        try:
            # 加载最新的进度数据
            self._load_progress()
            
            # 获取阶段对象
            stage = self.progress.stage_map.get(stage_name)
            if not stage:
                logger.warning(f"Stage {stage_name} not found in progress data")
                return False
                
            # 检查阶段状态
            return stage.get("status") == CloudStatus.COMPLETED and stage.get("progress") == 100.0
        except Exception as e:
            logger.error(f"Error checking stage completion status: {str(e)}")
            return False
            
    def is_step_completed(self, stage_name, step_name):
        """
        检查指定阶段中的特定步骤是否已完成
        
        Args:
            stage_name: 阶段名称（格式化后的，如"activating_the_memory_matrix"）
            step_name: 步骤名称（如"list_documents"）
            
        Returns:
            bool: 如果步骤已完成返回True，否则返回False
        """
        try:
            # 加载最新的进度数据
            self._load_progress()
            
            # 获取阶段对象
            stage = self.progress.stage_map.get(stage_name)
            if not stage:
                logger.warning(f"Stage {stage_name} not found in progress data")
                return False
                
            # 查找步骤
            for step in stage.get("steps", []):
                if step.get("name") == step_name:
                    return step.get("completed", False) and step.get("status") == CloudStatus.COMPLETED
                    
            logger.warning(f"Step {step_name} not found in stage {stage_name}")
            return False
        except Exception as e:
            logger.error(f"Error checking step completion status: {str(e)}")
            return False
            
    def update_message(self, message: str):
        """
        更新进度消息
        
        Args:
            message: 新的消息内容
        """
        try:
            self.progress.data["message"] = message
            self.save_progress()
            logger.info(f"Updated progress message: {message}")
        except Exception as e:
            logger.error(f"Error updating progress message: {str(e)}")
