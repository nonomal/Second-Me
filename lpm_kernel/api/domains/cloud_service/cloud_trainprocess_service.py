import os
import re
import time
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
import threading
import os

from lpm_kernel.api.domains.cloud_service.service import CloudService
from lpm_kernel.api.domains.cloud_service.cloud_process_step import CloudProcessStep
from lpm_kernel.api.domains.cloud_service.cloud_progress_holder import CloudProgressHolder, CloudStatus
from lpm_kernel.api.domains.trainprocess.training_params_manager import TrainingParamsManager
from lpm_kernel.api.domains.trainprocess.trainprocess_service import TrainProcessService
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.config import Config

from lpm_kernel.kernel.l1.l1_manager import (
    extract_notes_from_documents,
    document_service,
    get_latest_status_bio,
    get_latest_global_bio,
    generate_l1_from_l0
)
from lpm_kernel.kernel.chunk_service import ChunkService
from lpm_kernel.file_data.chunker import DocumentChunker
from lpm_kernel.configs.logging import get_train_process_logger
from lpm_kernel.kernel.note_service import NoteService

logger = get_train_process_logger()

class CloudTrainProcessService(TrainProcessService):
    """Cloud training process service (singleton pattern)
    
    This class extends the TrainProcessService to add cloud-specific functionality.
    It reuses the data processing methods from TrainProcessService but adds cloud training methods.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, current_model_name: str, base_model, training_type, hyper_parameters):
        """
        Initialize cloud training process service
        
        Args:
            current_model_name: Model name
            base_model: Base model for cloud training
            training_type: Type of training (e.g., efficient_sft)
            hyper_parameters: Training hyperparameters
        """
        # Initialize parent class
        super().__init__(current_model_name)
        
        # Override progress holder
        self.progress = CloudProgressHolder(current_model_name)
        
        # Initialize cloud-specific attributes
        self.training_data_path = None
        self.base_model = base_model
        self.training_type = training_type
        self.hyper_parameters = hyper_parameters
        self.job_id = None
        
        # Initialize cloud service
        self.cloud_service = CloudService()

    @classmethod
    def get_instance(cls, current_model_name: str = None, base_model=None, training_type=None, hyper_parameters=None):
        """
        Get the current instance of CloudTrainProcessService
        
        Args:
            current_model_name: Optional model name to update the instance with
            base_model: Base model for cloud training (only used when creating a new instance)
            training_type: Type of training (only used when creating a new instance)
            hyper_parameters: Training hyperparameters (only used when creating a new instance)
            
        Returns:
            CloudTrainProcessService: The singleton instance
        """
        if cls._instance is None:
            if current_model_name is None:
                logger.warning("current_model_name must be provided when creating a new instance")
                return None
            if base_model is None or training_type is None or hyper_parameters is None:
                logger.warning("base_model, training_type, and hyper_parameters must be provided when creating a new instance")
                return None
            return cls(current_model_name=current_model_name, base_model=base_model, 
                      training_type=training_type, hyper_parameters=hyper_parameters)
        
        if current_model_name is not None:
            # Update the existing instance with new model name
            cls._instance.model_name = current_model_name
            cls._instance.progress = CloudProgressHolder(current_model_name)
            
        return cls._instance


            
    def prepare_training_data(self) -> bool:
        """Prepare training data for cloud training"""
        try:
            # 执行第一阶段的步骤（Activating the Memory Matrix）
            logger.info("Executing memory matrix activation steps...")
            stage_name = "activating_the_memory_matrix"
            stage = self.progress.progress.stage_map.get(stage_name)
            total_steps = len(stage["steps"]) if stage else 4
            
            # 1. 列出文档
            logger.info("Step 1.1: Listing documents...")
            if not super().list_documents():
                logger.error("Failed to list documents")
                return False
            
            # 更新第一步完成后的进度
            if stage:
                stage["progress"] = 25.0  # 第一步完成，进度25%
                stage["status"] = CloudStatus.IN_PROGRESS
                # 更新步骤状态
                if len(stage["steps"]) > 0:
                    stage["steps"][0]["completed"] = True
                    stage["steps"][0]["status"] = CloudStatus.COMPLETED
                logger.info(f"Updated {stage_name} progress to 25% after completing list_documents")
                self._update_overall_progress()
            
            # 2. 生成文档嵌入
            logger.info("Step 1.2: Generating document embeddings...")
            if not super().generate_document_embeddings():
                logger.error("Failed to generate document embeddings")
                return False
            
            # 更新第二步完成后的进度
            if stage:
                stage["progress"] = 50.0  # 第二步完成，进度50%
                # 更新步骤状态
                if len(stage["steps"]) > 1:
                    stage["steps"][1]["completed"] = True
                    stage["steps"][1]["status"] = CloudStatus.COMPLETED
                logger.info(f"Updated {stage_name} progress to 50% after completing generate_document_embeddings")
                self._update_overall_progress()
            
            # 3. 处理文档分块
            logger.info("Step 1.3: Processing document chunks...")
            if not super().process_chunks():
                logger.error("Failed to process document chunks")
                return False
            
            # 更新第三步完成后的进度
            if stage:
                stage["progress"] = 75.0  # 第三步完成，进度75%
                # 更新步骤状态
                if len(stage["steps"]) > 2:
                    stage["steps"][2]["completed"] = True
                    stage["steps"][2]["status"] = CloudStatus.COMPLETED
                logger.info(f"Updated {stage_name} progress to 75% after completing process_chunks")
                self._update_overall_progress()
            
            # 4. 生成分块嵌入
            logger.info("Step 1.4: Generating chunk embeddings...")
            if not super().chunk_embedding():
                logger.error("Failed to generate chunk embeddings")
                return False
            
            # 更新第一阶段完成后的进度为100%并标记为已完成
            if stage:
                stage["progress"] = 100.0  # 全部完成，进度100%
                stage["status"] = CloudStatus.COMPLETED
                # 更新最后一个步骤状态
                if len(stage["steps"]) > 3:
                    stage["steps"][3]["completed"] = True
                    stage["steps"][3]["status"] = CloudStatus.COMPLETED
                logger.info(f"Updated {stage_name} progress to 100% and status to COMPLETED")
                self._update_overall_progress()
            
            # 执行第二阶段的步骤（Synthesize Your Life Narrative）
            logger.info("Executing life narrative synthesis steps...")
            stage_name = "synthesize_your_life_narrative"
            stage = self.progress.progress.stage_map.get(stage_name)
            total_steps = len(stage["steps"]) if stage else 3
            
            # 1. 提取维度主题
            logger.info("Step 2.1: Extracting dimensional topics...")
            if not super().extract_dimensional_topics():
                logger.error("Failed to extract dimensional topics")
                return False
            
            # 更新第一步完成后的进度
            if stage:
                stage["progress"] = 33.0  # 第一步完成，进度33%
                stage["status"] = CloudStatus.IN_PROGRESS
                # 更新步骤状态
                if len(stage["steps"]) > 0:
                    stage["steps"][0]["completed"] = True
                    stage["steps"][0]["status"] = CloudStatus.COMPLETED
                logger.info(f"Updated {stage_name} progress to 33% after completing extract_dimensional_topics")
                self._update_overall_progress()
            
            # 2. 生成传记
            logger.info("Step 2.2: Generating biography...")
            if not super().generate_biography():
                logger.error("Failed to generate biography")
                return False
            
            # 更新第二步完成后的进度
            if stage:
                stage["progress"] = 66.0  # 第二步完成，进度66%
                # 更新步骤状态
                if len(stage["steps"]) > 1:
                    stage["steps"][1]["completed"] = True
                    stage["steps"][1]["status"] = CloudStatus.COMPLETED
                logger.info(f"Updated {stage_name} progress to 66% after completing generate_biography")
                self._update_overall_progress()
            
            # 3. 映射实体网络
            logger.info("Step 2.3: Mapping entity network...")
            if not super().map_your_entity_network():
                logger.error("Failed to map entity network")
                return False
            
            # 更新第二阶段完成后的进度为100%并标记为已完成
            if stage:
                stage["progress"] = 100.0  # 全部完成，进度100%
                stage["status"] = CloudStatus.COMPLETED
                # 更新最后一个步骤状态
                if len(stage["steps"]) > 2:
                    stage["steps"][2]["completed"] = True
                    stage["steps"][2]["status"] = CloudStatus.COMPLETED
                logger.info(f"Updated {stage_name} progress to 100% and status to COMPLETED")
                self._update_overall_progress()
            
            # 执行第三阶段的步骤（Prepare Training Data for Deep Comprehension）
            logger.info("Executing training data preparation steps...")
            stage_name = "prepare_training_data_for_deep_comprehension"
            stage = self.progress.progress.stage_map.get(stage_name)
            total_steps = len(stage["steps"]) if stage else 3
            
            # 1. 解码偏好模式
            logger.info("Step 3.1: Decoding preference patterns...")
            if not super().decode_preference_patterns():
                logger.error("Failed to decode preference patterns")
                return False
            
            # 更新第一步完成后的进度
            if stage:
                stage["progress"] = 33.0  # 第一步完成，进度33%
                stage["status"] = CloudStatus.IN_PROGRESS
                # 更新步骤状态
                if len(stage["steps"]) > 0:
                    stage["steps"][0]["completed"] = True
                    stage["steps"][0]["status"] = CloudStatus.COMPLETED
                logger.info(f"Updated {stage_name} progress to 33% after completing decode_preference_patterns")
                self._update_overall_progress()
            
            # 2. 强化身份
            logger.info("Step 3.2: Reinforcing identity...")
            if not super().reinforce_identity():
                logger.error("Failed to reinforce identity")
                return False
            
            # 更新第二步完成后的进度
            if stage:
                stage["progress"] = 66.0  # 第二步完成，进度66%
                # 更新步骤状态
                if len(stage["steps"]) > 1:
                    stage["steps"][1]["completed"] = True
                    stage["steps"][1]["status"] = CloudStatus.COMPLETED
                logger.info(f"Updated {stage_name} progress to 66% after completing reinforce_identity")
                self._update_overall_progress()
            
            # 3. 增强内容保留
            logger.info("Step 3.3: Augmenting content retention...")
            if not super().augment_content_retention():
                logger.error("Failed to augment content retention")
                return False
            
            # 更新第三阶段完成后的进度为100%并标记为已完成
            if stage:
                stage["progress"] = 100.0  # 全部完成，进度100%
                stage["status"] = CloudStatus.COMPLETED
                # 更新最后一个步骤状态
                if len(stage["steps"]) > 2:
                    stage["steps"][2]["completed"] = True
                    stage["steps"][2]["status"] = CloudStatus.COMPLETED
                logger.info(f"Updated {stage_name} progress to 100% and status to COMPLETED")
                self._update_overall_progress()
            
            # 计算并更新整体进度
            self._update_overall_progress()
            
            logger.info("Successfully generated all necessary data using parent class methods")
            return True
        except Exception as e:
            logger.error(f"Prepare training data failed: {str(e)}")
            # 尝试标记当前正在进行的阶段为失败
            current_stage = self.progress.get_progress().get("current_stage")
            if current_stage:
                for step in self.progress.get_progress().get("stages", []):
                    if step["name"].lower().replace(" ", "_") == current_stage and step["current_step"]:
                        stage_name = current_stage
                        step_name = step["current_step"].lower().replace(" ", "_")
                        self.progress.mark_step_status(stage_name, step_name, CloudStatus.FAILED, f"Error: {str(e)}")
                        break
            return False
    

    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return self.progress.get_progress()
    
    def start_process(self) -> bool:
        """Start the cloud training process using CloudService"""
        try:
            self.is_stopped = False
            # Store the current process PID
            self.current_pid = os.getpid()
            logger.info(f"Cloud training process started with PID: {self.current_pid}")
            logger.info(f"Using base_model: {self.base_model}, training_type: {self.training_type}")
            logger.info(f"CloudService initialized with API key: {self.cloud_service.api_key is not None}")
            
            # 1. 准备训练数据（生成L2级别数据）
            logger.info("Step 1: Preparing training data...")
            from lpm_kernel.api.domains.trainprocess.process_step import ProcessStep
            self.progress.mark_step_status(ProcessStep.DECODE_PREFERENCE_PATTERNS, CloudStatus.IN_PROGRESS)
            success = self.prepare_training_data()
            logger.info(f"Training data preparation result: {success}")
            if not success:
                logger.error("Failed to prepare training data")
                self.progress.mark_step_status(ProcessStep.DECODE_PREFERENCE_PATTERNS, CloudStatus.FAILED)
                return False
            self.progress.mark_step_status(ProcessStep.DECODE_PREFERENCE_PATTERNS, CloudStatus.COMPLETED)
            
            # 7. 上传训练数据
            logger.info("Step 7: Uploading training data...")
            self.progress.mark_step_status(CloudProcessStep.UPLOAD_TRAINING_DATA, CloudStatus.IN_PROGRESS)
            try:
                # 直接使用upload_training_file的默认参数
                file_id = self.cloud_service.upload_training_file()
                logger.info(f"File upload result: file_id={file_id}")
            except Exception as e:
                logger.error(f"Exception during file upload: {str(e)}", exc_info=True)
                self.progress.mark_step_status(CloudProcessStep.UPLOAD_TRAINING_DATA, CloudStatus.FAILED)
                return False
                
            if not file_id:
                logger.error("Failed to upload training data")
                self.progress.mark_step_status(CloudProcessStep.UPLOAD_TRAINING_DATA, CloudStatus.FAILED)
                return False
            self.progress.mark_step_status(CloudProcessStep.UPLOAD_TRAINING_DATA, CloudStatus.COMPLETED)
            
            # 8. 创建微调任务
            logger.info("Step 8: Creating fine-tune job...")
            self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.IN_PROGRESS)
            
            try:
                success_id = self.cloud_service.create_fine_tune_job(
                    base_model=self.base_model,
                    training_type=self.training_type,
                    hyper_parameters=self.hyper_parameters
                )
                try:
                    current_dir = Path(__file__).parent
                    
                    job_file_path = current_dir / "job_id.json"
                    
                    job_info = {
                        "job_id": success_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "completed"
                    }
                    
                    with open(job_file_path, "w") as f:
                        json.dump(job_info, f, indent=2)
                        
                    logger.info(f"Job ID information saved to {job_file_path}")
                except Exception as e:
                    logger.error(f"Failed to write job ID to file: {str(e)}", exc_info=True)

                logger.info(f"Create fine-tune job result: {success_id}")
            except Exception as e:
                logger.error(f"Exception during fine-tune job creation: {str(e)}", exc_info=True)
                self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.FAILED)
                return False
                
            if success_id is None:
                logger.error("Failed to create fine-tune job")
                self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.FAILED)
                return False
            
            self.job_id = success_id
            logger.info(f"Job ID set: {self.job_id}")
            # 将job_id保存到进度条中
            self.progress.job_id = self.job_id
            self.progress.progress.data["job_id"] = self.job_id
            self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.COMPLETED)
            
            # 9. 等待微调完成
            logger.info("Step 9: Waiting for fine-tune job to complete...")
            self.progress.mark_step_status(CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION, CloudStatus.IN_PROGRESS)
            self._wait_for_completion_thread(self.cloud_service, self.job_id)
            
            logger.info("Cloud training process completed successfully")
            return True
        except Exception as e:
            logger.error(f"Cloud training process failed: {str(e)}", exc_info=True)
            if self.current_step:
                self.progress.mark_step_status(self.current_step, CloudStatus.FAILED, f"Error: {str(e)}")
            return False
    
    def _wait_for_completion_thread(self, cloud_service, job_id):
        try:
            logger.info(f"Async thread: waiting for job {job_id} to complete")
            
            # 定义进度回调函数
            def progress_callback(status, progress, message):
                try:
                    logger.info(f"Progress update: {status}, {progress}%, {message}")
                    
                    # 将状态映射到CloudStatus
                    status_mapping = {
                        "IN_PROGRESS": CloudStatus.IN_PROGRESS,
                        "COMPLETED": CloudStatus.COMPLETED,
                        "FAILED": CloudStatus.FAILED,
                        "CANCELLED": CloudStatus.CANCELLED
                    }
                    
                    cloud_status = status_mapping.get(status, CloudStatus.IN_PROGRESS)
                    
                    # 更新进度条
                    self.progress.update_step_progress(
                        CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION,
                        progress,
                        message
                    )
                    
                    # 如果完成或失败，更新状态
                    if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                        self.progress.mark_step_status(
                            CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION,
                            cloud_status,
                            message
                        )
                except Exception as e:
                    logger.error(f"Error in progress callback: {str(e)}", exc_info=True)
            
            # 使用回调函数调用wait_for_job_completion
            success = cloud_service.wait_for_job_completion(
                job_id=job_id,
                progress_callback=progress_callback
            )
            
            if success:
                logger.info(f"Fine-tuning job completed successfully.")
                # 更新进度消息
                self.progress.update_message("Fine-tuning job completed successfully!")
            else:
                logger.error(f"Fine-tuning job failed")
                # 进度回调函数已经处理了失败状态，这里不需要再次更新
        except Exception as e:
            logger.error(f"Error in async wait thread: {str(e)}", exc_info=True)
            self.progress.mark_step_status("cloud_training", "wait_for_fine-tune_completion", CloudStatus.FAILED, f"Error: {str(e)}")
    
    def _update_overall_progress(self):
        """Calculate and update the overall progress based on the stages' progress"""
        try:
            # 获取所有阶段
            stages = self.progress.data["stages"]
            total_stages = len(stages)
            completed_stages = 0
            total_progress = 0.0
            
            # 计算已完成的阶段数和总进度
            for stage in stages:
                total_progress += stage["progress"]
                if stage["status"] == CloudStatus.COMPLETED:
                    completed_stages += 1
            
            # 计算整体进度（所有阶段进度的平均值）
            if total_stages > 0:
                overall_progress = total_progress / total_stages
            else:
                overall_progress = 0.0
                
            # 更新整体进度
            self.progress.data["overall_progress"] = overall_progress
            logger.info(f"Updated overall progress to {overall_progress:.2f}%")
            
            # 如果所有阶段都已完成，将整体状态设置为已完成
            if completed_stages == total_stages:
                self.progress.data["status"] = CloudStatus.COMPLETED
                logger.info("All stages completed, setting overall status to COMPLETED")
            
            # 保存进度
            self.progress.save_progress()
        except Exception as e:
            logger.error(f"Error updating overall progress: {str(e)}")
    
    def stop_process(self) -> bool:
        """Stop the cloud training process
        
        This method will attempt to stop the fine-tuning job if it's in progress,
        by deleting the job, and update the progress status accordingly.
        
        Returns:
            bool: True if the process was successfully stopped, False otherwise
        """
        try:
            self.is_stopped = True
            logger.info(f"Attempting to stop cloud training process for model: {self.model_name}")
            
            any_operation_succeeded = False
            
            if not self.job_id:
                try:
                    current_dir = Path(__file__).parent
                    job_file_path = current_dir / "job_id.json"
                    
                    if job_file_path.exists():
                        with open(job_file_path, "r") as f:
                            job_info = json.load(f)
                            if "job_id" in job_info:
                                self.job_id = job_info["job_id"]
                                logger.info(f"Retrieved job_id from file: {self.job_id}")
                except Exception as e:
                    logger.error(f"Failed to read job ID from file: {str(e)}", exc_info=True)
            
            if self.job_id:
                logger.info(f"Attempting to cancel fine-tune job: {self.job_id}")
                success = self.cloud_service.cancel_fine_tune_job(self.job_id)
                
                if success:
                    logger.info(f"Successfully cancelled fine-tune job: {self.job_id}")
                    any_operation_succeeded = True
                else:
                    logger.error(f"Failed to cancel fine-tune job: {self.job_id}")
            else:
                logger.warning("No active fine-tune job found to delete")

            # 获取当前阶段和步骤
            current_stage = self.progress.get_progress().get("current_stage")
            if current_stage:
                # 找到当前阶段对应的步骤
                for stage in self.progress.get_progress().get("stages", []):
                    if stage["name"].lower().replace(" ", "_") == current_stage:
                        current_step_name = stage.get("current_step")
                        if current_step_name:
                            # 找到对应的CloudProcessStep
                            for step in CloudProcessStep:
                                if step.value == current_step_name:
                                    self.progress.mark_step_status(step, CloudStatus.CANCELLED, "Process cancelled by user")
                                    break
                        break
            
            # 设置整体消息
            self.progress.set_message("Cloud process cancelled by user")
            
            return any_operation_succeeded or not (self.job_id)
                
        except Exception as e:
            logger.error(f"Error stopping cloud process: {str(e)}", exc_info=True)
            return False
