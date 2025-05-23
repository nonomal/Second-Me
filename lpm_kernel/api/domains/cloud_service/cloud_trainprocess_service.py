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


    # 不需要重新实现map_your_entity_network, decode_preference_patterns, reinforce_identity, augment_content_retention等方法
    # 这些方法在父类TrainProcessService中已经实现，我们可以直接使用
    # 如果需要更新CloudProgressHolder，可以在start_process方法中处理

    # 不需要重新实现_prepare_l2_data方法，直接使用父类TrainProcessService的实现
            
    def prepare_training_data(self) -> bool:
        """Prepare training data for cloud training"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(CloudProcessStep.PREPARE_TRAINING_DATA, CloudStatus.IN_PROGRESS)
            
            # 首先调用父类的数据准备方法来生成基础训练数据
            logger.info("Preparing L2 data using parent class methods")
            
            # 调用父类的_prepare_l2_data方法准备基础数据
            if not super()._prepare_l2_data():
                logger.error("Failed to prepare L2 data")
                self.progress.mark_step_status(CloudProcessStep.PREPARE_TRAINING_DATA, CloudStatus.FAILED, 
                                              "Failed to prepare L2 data")
                return False
                
            # 调用父类的数据生成方法
            if not super().map_your_entity_network():
                logger.error("Failed to map entity network")
                self.progress.mark_step_status(CloudProcessStep.PREPARE_TRAINING_DATA, CloudStatus.FAILED, 
                                              "Failed to map entity network")
                return False
                
            if not super().decode_preference_patterns():
                logger.error("Failed to decode preference patterns")
                self.progress.mark_step_status(CloudProcessStep.PREPARE_TRAINING_DATA, CloudStatus.FAILED, 
                                              "Failed to decode preference patterns")
                return False
                
            if not super().reinforce_identity():
                logger.error("Failed to reinforce identity")
                self.progress.mark_step_status(CloudProcessStep.PREPARE_TRAINING_DATA, CloudStatus.FAILED, 
                                              "Failed to reinforce identity")
                return False
                
            if not super().augment_content_retention():
                logger.error("Failed to augment content retention")
                self.progress.mark_step_status(CloudProcessStep.PREPARE_TRAINING_DATA, CloudStatus.FAILED, 
                                              "Failed to augment content retention")
                return False
            
            logger.info("Successfully generated all necessary data using parent class methods")
            
            # 然后生成云端特有的训练数据格式（JSONL）
            training_data_path = self._generate_training_data()
            
            if not training_data_path or not os.path.exists(training_data_path):
                logger.error("Failed to generate training data")
                self.progress.mark_step_status(CloudProcessStep.PREPARE_TRAINING_DATA, CloudStatus.FAILED, 
                                              "Failed to generate training data")
                return False
            
            # Store the training data path for later use
            self.training_data_path = training_data_path
            
            logger.info(f"Successfully prepared training data at {training_data_path}")
            self.progress.mark_step_status(CloudProcessStep.PREPARE_TRAINING_DATA, CloudStatus.COMPLETED, 
                                          f"Training data prepared at {training_data_path}")
            return True
        except Exception as e:
            logger.error(f"Prepare training data failed: {str(e)}")
            self.progress.mark_step_status(CloudProcessStep.PREPARE_TRAINING_DATA, CloudStatus.FAILED, 
                                          f"Error: {str(e)}")
            return False
    
    def _generate_training_data(self) -> Optional[str]:
        """Generate training data for cloud training by converting merged.json to JSONL format
        
        Returns:
            Path to generated training data file in JSONL format, or None if generation failed
        """
        try:
            import os
            merged_file_path = os.path.join(os.getcwd(), "resources/L2/data/merged.json")
            if not os.path.exists(merged_file_path):
                logger.error(f"Merged data file not found at {merged_file_path}")
                return None

            # 将merged.json转换为JSONL格式用于云训练
            jsonl_path = merged_file_path
            if not jsonl_path:
                logger.error("Failed to convert merged.json to JSONL format")
                return None

            logger.info(f"Converted merged.json directly to JSONL format at {jsonl_path}")
            return jsonl_path

        except Exception as e:
            logger.error(f"Failed to generate training data: {str(e)}")
            return None
    
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
            self.progress.mark_step_status(CloudProcessStep.PREPARE_TRAINING_DATA, CloudStatus.IN_PROGRESS)
            success = self.prepare_training_data()
            logger.info(f"Training data preparation result: {success}, path: {self.training_data_path}")
            if not success or not self.training_data_path:
                logger.error("Failed to prepare training data")
                self.progress.mark_step_status(CloudProcessStep.PREPARE_TRAINING_DATA, CloudStatus.FAILED, "Failed to prepare training data")
                return False
            self.progress.mark_step_status(CloudProcessStep.PREPARE_TRAINING_DATA, CloudStatus.COMPLETED)
            
            # 2. 上传训练数据
            logger.info("Step 2: Uploading training data...")
            self.progress.mark_step_status(CloudProcessStep.UPLOAD_TRAINING_DATA, CloudStatus.IN_PROGRESS)
            try:
                file_id = self.cloud_service.upload_training_file(self.training_data_path)
                logger.info(f"File upload result: file_id={file_id}")
            except Exception as e:
                logger.error(f"Exception during file upload: {str(e)}", exc_info=True)
                self.progress.mark_step_status(CloudProcessStep.UPLOAD_TRAINING_DATA, CloudStatus.FAILED, f"Exception: {str(e)}")
                return False
                
            if not file_id:
                logger.error("Failed to upload training data")
                self.progress.mark_step_status(CloudProcessStep.UPLOAD_TRAINING_DATA, CloudStatus.FAILED, "Failed to upload training data")
                return False
            self.progress.mark_step_status(CloudProcessStep.UPLOAD_TRAINING_DATA, CloudStatus.COMPLETED)
            
            # 3. 创建微调任务
            logger.info("Step 3: Creating fine-tune job...")
            self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.IN_PROGRESS)
            
            try:
                success_id = self.cloud_service.create_fine_tune_job(
                    base_model=self.base_model,
                    training_type=self.training_type,
                    hyper_parameters=self.hyper_parameters
                )
                logger.info(f"Create fine-tune job result: {success_id}")
            except Exception as e:
                logger.error(f"Exception during fine-tune job creation: {str(e)}", exc_info=True)
                self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.FAILED, f"Exception: {str(e)}")
                return False
                
            if success_id is None:
                logger.error("Failed to create fine-tune job")
                self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.FAILED, "Failed to create fine-tune job")
                return False
            
            # 获取任务ID
            self.job_id = success_id  # 保存任务ID以便后续使用
            logger.info(f"Job ID set: {self.job_id}")
            self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.COMPLETED)
            
            # 4. 开始异步等待任务完成
            self.progress.mark_step_status(CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION, CloudStatus.IN_PROGRESS)
            # 启动异步等待任务，不阻塞主流程
            self._start_async_wait_for_completion(self.cloud_service, self.job_id)
            
            logger.info("Cloud training process completed successfully")
            return True
        except Exception as e:
            logger.error(f"Cloud training process failed: {str(e)}", exc_info=True)
            if self.current_step:
                self.progress.mark_step_status(self.current_step, CloudStatus.FAILED, f"Error: {str(e)}")
            return False
    
    def _start_async_wait_for_completion(self, cloud_service, job_id):
        """启动异步线程等待任务完成"""
        thread = threading.Thread(
            target=self._wait_for_completion_thread,
            args=(cloud_service, job_id),
            daemon=True  
        )
        thread.start()
        logger.info(f"Started async thread to monitor job {job_id}")
    
    def _wait_for_completion_thread(self, cloud_service, job_id):
        """在线程中等待任务完成"""
        try:
            logger.info(f"Async thread: waiting for job {job_id} to complete")
            success = cloud_service.wait_for_job_completion(job_id=job_id)
            
            if success:
                # 任务成功完成
                self.model_id = cloud_service.model_id  # 保存模型ID
                logger.info(f"Fine-tuning job completed successfully. Model ID: {self.model_id}")
                self.progress.mark_step_status(CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION, CloudStatus.COMPLETED)
                # 更新整体进度为完成
                self.progress.progress["status"] = CloudStatus.COMPLETED
                self.progress.progress["message"] = "Cloud training process completed successfully"
                self.progress.save_progress()
            else:
                # 任务失败
                logger.error(f"Fine-tuning job failed")
                self.progress.mark_step_status(CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION, CloudStatus.FAILED, "Fine-tuning job failed")
        except Exception as e:
            logger.error(f"Error in async wait thread: {str(e)}", exc_info=True)
            self.progress.mark_step_status(CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION, CloudStatus.FAILED, f"Error: {str(e)}")
    
    def stop_process(self) -> bool:
        """Stop the cloud training process"""
        self.is_stopped = True
        return True
