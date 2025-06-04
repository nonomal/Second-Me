import time
import json
import os
import enum
import multiprocessing
from multiprocessing import Process, Queue, Event
import signal
import psutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from lpm_kernel.api.domains.cloud_service.cloud_process_step import CloudProcessStep
from lpm_kernel.api.domains.cloud_service.cloud_progress_holder import CloudProgressHolder, CloudStatus
from lpm_kernel.api.domains.cloud_service.service import CloudService
from lpm_kernel.api.domains.trainprocess.process_step import ProcessStep
from lpm_kernel.api.domains.trainprocess.trainprocess_service import TrainProcessService
from lpm_kernel.configs.logging import get_train_process_logger
from lpm_kernel.models.memory import Memory
from lpm_kernel.common.repository.database_session import DatabaseSession

logger = get_train_process_logger()

class PrepareDataResult(enum.Enum):
    SUCCESS = "success"
    STOPPED = "stopped"
    ERROR = "error"

class CloudTrainProcessService(TrainProcessService):
    """Cloud training process service (singleton pattern)"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, current_model_name: str, base_model, training_type, hyper_parameters):
        """Initialize cloud training process service"""
        # Initialize parent class
        super().__init__(current_model_name)
        
        # Override progress holder
        self.progress = CloudProgressHolder(current_model_name)
        
        # Initialize cloud-specific attributes
        self.training_data_path = None
        self.base_model = base_model
        self.training_type = training_type
        self.hyper_parameters = hyper_parameters
        self.model_name = current_model_name
        self.job_id = None
        
        # For tracking data processing process
        self._data_processing_process = None
        self._data_processing_pid = None
        self._result_queue = None
        self._process_completed = None
        self._data_processing_result = None
        
        # For tracking task completion process
        self._wait_completion_process = None
        self._wait_completion_pid = None
        
        # Initialize cloud service
        self.cloud_service = CloudService()

    @classmethod
    def get_instance(cls):
        """Get the current instance of CloudTrainProcessService
        
        Returns:
            CloudTrainProcessService: The singleton instance
        """

        if cls._instance is not None:
            return cls._instance

        try:
            
            params_file = Path("data/cloud_progress/cloud_training_params.json")
            if params_file.exists():
                with open(params_file, "r", encoding="utf-8") as f:
                    params = json.load(f)

                model_name = params.get("model_name")
                base_model = params.get("base_model")
                training_type = params.get("training_type", "efficient_sft")
                hyper_parameters = params.get("hyper_parameters", {})
                
                if model_name and base_model:
                    logger.info(f"Loaded training parameters for model {model_name} from file")

                    cls._instance = cls(current_model_name=model_name, 
                                        base_model=base_model,
                                        training_type=training_type,
                                        hyper_parameters=hyper_parameters)
                    return cls._instance
                else:
                    logger.warning("Invalid training parameters in file: missing model_name or base_model")
        except Exception as e:
            logger.warning(f"Failed to load training parameters from file: {str(e)}")
        
        logger.warning("No valid training parameters found in file")
        return None


            
    def prepare_training_data(self) -> PrepareDataResult:
        """Prepare all necessary data for cloud training"""
        try:
            logger.info("Starting cloud training data preparation")
            
            logger.info("Executing memory matrix activation steps...")
            stage_name = "activating_the_memory_matrix"
            stage = self.progress.progress.stage_map.get(stage_name)
            
            # Check if stage is already completed
            if self.progress.is_stage_completed(stage_name):
                logger.info(f"Stage '{stage_name}' already completed, skipping...")
            else:

                logger.info("Step 1.1: Listing documents...")

                if self.progress.is_step_completed(stage_name, "list_documents"):
                    logger.info("Step 'list_documents' already completed, skipping...")
                else:
                    if not super().list_documents():
                        logger.error("Failed to list documents")
                        return PrepareDataResult.ERROR
            
                if stage:
                    stage["progress"] = 25.0  
                    stage["status"] = CloudStatus.IN_PROGRESS
                    # Update step status
                    if len(stage["steps"]) > 0:
                        stage["steps"][0]["completed"] = True
                        stage["steps"][0]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 25% after completing list_documents")
                    self._update_overall_progress()
                
                if self.is_stopped:
                    logger.info("Process has been stopped after completing list_documents, exiting.")
                    return PrepareDataResult.STOPPED
            
                # 2. Generate document embeddings
                logger.info("Step 1.2: Generating document embeddings...")
                if self.progress.is_step_completed(stage_name, "generate_document_embeddings"):
                    logger.info("Step 'generate_document_embeddings' already completed, skipping...")
                else:
                    if not super().generate_document_embeddings():
                        logger.error("Failed to generate document embeddings")
                        return PrepareDataResult.ERROR
            
                # Update progress after completing second step
                if stage:
                    stage["progress"] = 50.0  # Second step completed, progress 50%

                    if len(stage["steps"]) > 1:
                        stage["steps"][1]["completed"] = True
                        stage["steps"][1]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 50% after completing generate_document_embeddings")
                    self._update_overall_progress()
                # 步骤后检测停止信号
                if self.is_stopped:
                    logger.info("Process has been stopped after completing generate_document_embeddings, exiting.")
                    return PrepareDataResult.STOPPED
            
                # 3. Process document chunks
                logger.info("Step 1.3: Processing document chunks...")
                if not super().process_chunks():
                    logger.error("Failed to process document chunks")
                    return PrepareDataResult.ERROR
            
                # Update progress after completing third step
                if stage:
                    stage["progress"] = 75.0  # Third step completed, progress 75%
                    # Update step status
                    if len(stage["steps"]) > 2:
                        stage["steps"][2]["completed"] = True
                        stage["steps"][2]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 75% after completing process_chunks")
                    self._update_overall_progress()
            
                if self.is_stopped:
                    logger.info("Process has been stopped after completing process_chunks, exiting.")
                    return PrepareDataResult.STOPPED
            
                # 4. Generate chunk embeddings
                logger.info("Step 1.4: Generating chunk embeddings...")
                if not super().chunk_embedding():
                    logger.error("Failed to generate chunk embeddings")
                    return PrepareDataResult.ERROR
            
                # Update progress to 100% after completing first stage
                if stage:
                    stage["progress"] = 100.0  # All completed, progress 100%
                    stage["status"] = CloudStatus.COMPLETED
                    # Update last step status
                    if len(stage["steps"]) > 3:
                        stage["steps"][3]["completed"] = True
                        stage["steps"][3]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 100% and status to COMPLETED")
                    self._update_overall_progress()
                
                if self.is_stopped:
                    logger.info("Process has been stopped after completing chunk_embedding, exiting.")
                    return PrepareDataResult.STOPPED
            

            logger.info("Executing life narrative synthesis steps...")
            stage_name = "synthesize_your_life_narrative"
            stage = self.progress.progress.stage_map.get(stage_name)
            

            if self.progress.is_stage_completed(stage_name):
                logger.info(f"Stage '{stage_name}' already completed, skipping...")
            else:
 
                logger.info("Step 2.1: Extracting dimensional topics...")
 
                if self.progress.is_step_completed(stage_name, "extract_dimensional_topics"):
                    logger.info("Step 'extract_dimensional_topics' already completed, skipping...")
                else:
                    if not super().extract_dimensional_topics():
                        logger.error("Failed to extract dimensional topics")
                        return PrepareDataResult.ERROR
            
 
                if stage:
                    stage["progress"] = 33.0  
                    stage["status"] = CloudStatus.IN_PROGRESS

                    if len(stage["steps"]) > 0:
                        stage["steps"][0]["completed"] = True
                        stage["steps"][0]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 33% after completing extract_dimensional_topics")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing extract_dimensional_topics, exiting.")
                    return PrepareDataResult.STOPPED
            

                logger.info("Step 2.2: Generating biography...")
                if self.progress.is_step_completed(stage_name, "generate_biography"):
                    logger.info("Step 'generate_biography' already completed, skipping...")
                else:
                    if not super().generate_biography():
                        logger.error("Failed to generate biography")
                        return PrepareDataResult.ERROR
            
                if stage:
                    stage["progress"] = 66.0  
                    if len(stage["steps"]) > 1:
                        stage["steps"][1]["completed"] = True
                        stage["steps"][1]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 66% after completing generate_biography")
                    self._update_overall_progress()
                
                if self.is_stopped:
                    logger.info("Process has been stopped after completing generate_biography, exiting.")
                    return PrepareDataResult.STOPPED
            
                logger.info("Step 2.3: Mapping entity network...")
                if self.progress.is_step_completed(stage_name, "map_your_entity_network"):
                    logger.info("Step 'map_your_entity_network' already completed, skipping...")
                else:
                    if not super().map_your_entity_network():
                        logger.error("Failed to map entity network")
                        return PrepareDataResult.ERROR
            
                if stage:
                    stage["progress"] = 100.0  
                    stage["status"] = CloudStatus.COMPLETED
                    if len(stage["steps"]) > 2:
                        stage["steps"][2]["completed"] = True
                        stage["steps"][2]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 100% and status to COMPLETED")
                    self._update_overall_progress()
                
                if self.is_stopped:
                    logger.info("Process has been stopped after completing map_your_entity_network, exiting.")
                    return PrepareDataResult.STOPPED
            
            logger.info("Executing training data preparation steps...")
            stage_name = "prepare_training_data_for_deep_comprehension"
            stage = self.progress.progress.stage_map.get(stage_name)
            
            if self.progress.is_stage_completed(stage_name):
                logger.info(f"Stage '{stage_name}' already completed, skipping...")
            else:
                logger.info("Step 3.1: Decoding preference patterns...")
                if self.progress.is_step_completed(stage_name, "decode_preference_patterns"):
                    logger.info("Step 'decode_preference_patterns' already completed, skipping...")
                else:
                    if not super().decode_preference_patterns():
                        logger.error("Failed to decode preference patterns")
                        return PrepareDataResult.ERROR
            
                if stage:
                    stage["progress"] = 33.0  
                    stage["status"] = CloudStatus.IN_PROGRESS
                    if len(stage["steps"]) > 0:
                        stage["steps"][0]["completed"] = True
                        stage["steps"][0]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 33% after completing decode_preference_patterns")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing decode_preference_patterns, exiting.")
                    return PrepareDataResult.STOPPED
            
                logger.info("Step 3.2: Reinforcing identity...")
                
                if self.progress.is_step_completed(stage_name, "reinforce_identity"):
                    logger.info("Step 'reinforce_identity' already completed, skipping...")
                else:
                    if not super().reinforce_identity():
                        logger.error("Failed to reinforce identity")
                        return PrepareDataResult.ERROR
            
                if stage:
                    stage["progress"] = 66.0  
                    if len(stage["steps"]) > 1:
                        stage["steps"][1]["completed"] = True
                        stage["steps"][1]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 66% after completing reinforce_identity")
                    self._update_overall_progress()
                
                if self.is_stopped:
                    logger.info("Process has been stopped after completing reinforce_identity, exiting.")
                    return PrepareDataResult.STOPPED
            
                logger.info("Step 3.3: Augmenting content retention...")
                if self.progress.is_step_completed(stage_name, "augment_content_retention"):
                    logger.info("Step 'augment_content_retention' already completed, skipping...")
                else:
                    if not super().augment_content_retention():
                        logger.error("Failed to augment content retention")
                        return PrepareDataResult.ERROR
            
                if stage:
                    stage["progress"] = 100.0  
                    stage["status"] = CloudStatus.COMPLETED
                    if len(stage["steps"]) > 2:
                        stage["steps"][2]["completed"] = True
                        stage["steps"][2]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 100% and status to COMPLETED")
                    self._update_overall_progress()
                
                if self.is_stopped:
                    logger.info("Process has been stopped after completing augment_content_retention, exiting.")
                    return PrepareDataResult.STOPPED
            
            self._update_overall_progress()
            
            if self.is_stopped:
                logger.info("Data preparation completed current step, stopping as requested")
                return PrepareDataResult.STOPPED
                
            logger.info("Successfully generated all necessary data using parent class methods")
            return PrepareDataResult.SUCCESS
        except Exception as e:
            logger.error(f"Prepare training data failed: {str(e)}")
            current_stage = self.progress.get_progress().get("current_stage")
            if current_stage:
                for step in self.progress.get_progress().get("stages", []):
                    if step["name"].lower().replace(" ", "_") == current_stage and step["current_step"]:
                        stage_name = current_stage
                        step_name = step["current_step"].lower().replace(" ", "_")
                        self.progress.mark_step_status(stage_name, step_name, CloudStatus.FAILED)
                        break
            return PrepareDataResult.ERROR
    
    def start_process(self) -> bool:
        """Start the cloud training process using CloudService"""
        self.is_stopped = False
        self._data_processing_result = None

        self.current_pid = os.getpid()
        logger.info(f"Cloud training process started with PID: {self.current_pid}")
        logger.info(f"Using base_model: {self.base_model}, training_type: {self.training_type}")
        logger.info(f"CloudService initialized with API key: {self.cloud_service.api_key is not None}")

        logger.info("Step 1: Preparing training data...")
        
        try:
            if self.is_stopped:
                logger.info("Process has been stopped, will complete current stage and then stop")
                
            result = self.prepare_training_data()
            self._data_processing_result = result
            
            success = self._data_processing_result
            logger.info(f"Training data preparation result: {success}")
            
            if success == PrepareDataResult.SUCCESS:
                logger.info("Training data preparation completed successfully")
            elif success == PrepareDataResult.STOPPED:
                logger.info("Training data preparation stopped by user")
                return False
            elif success == PrepareDataResult.ERROR:
                logger.error("Failed to prepare training data")
                return False

            if self.is_stopped:
                logger.info("Process has been stopped after data preparation")
                return False
                
            deploy_success = self.cloud_deploy()
            logger.info(f"Cloud deploy result: {deploy_success}")
            if not deploy_success:
                logger.error("Failed to cloud deploy")
                return False

            return True
        except Exception as e:
            logger.error(f"Error in cloud training process: {str(e)}", exc_info=True)
            return False

    def cloud_deploy(self) -> bool:
        try:
            logger.info("Step 7: Uploading training data...")
            if self.is_stopped:
                logger.info("Process has been stopped, cancelling cloud deployment")
                return False
            
            self.progress.mark_step_status(CloudProcessStep.UPLOAD_TRAINING_DATA, CloudStatus.IN_PROGRESS)
            try:
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

            logger.info("Step 8: Creating fine-tune job...")
            self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.IN_PROGRESS)

            try:
                success_id = self.cloud_service.create_fine_tune_job(
                    base_model=self.base_model,
                    training_type=self.training_type,
                    hyper_parameters=self.hyper_parameters
                )
                try:
                    params_dir = Path("data/cloud_progress")
                    params_dir.mkdir(parents=True, exist_ok=True)
                    job_file_path = params_dir / "job_id.json"

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

            self.progress.job_id = self.job_id
            self.progress.progress.data["job_id"] = self.job_id
            self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.COMPLETED)

            logger.info("Step 9: Waiting for fine-tune job to complete...")
        
            self.progress.mark_step_status(CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION, CloudStatus.IN_PROGRESS)
        
            logger.info(f"Fine-tune job {self.job_id} has been created and is now running")
            
            # Start a separate process to monitor the job completion
            logger.info("Starting a separate process to monitor the job completion")
            self._wait_completion_process = multiprocessing.Process(
                target=self._wait_for_completion_process,
                args=(self.cloud_service, self.job_id)
            )
            self._wait_completion_process.daemon = True
            self._wait_completion_process.start()
            self._wait_completion_pid = self._wait_completion_process.pid
            logger.info(f"Job monitoring process started with PID: {self._wait_completion_pid}")
            
            self.progress.mark_step_status(CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION, CloudStatus.IN_PROGRESS, 
                                          "Fine-tune job is running in the background")
        
            logger.info("Cloud training process completed successfully")
            return True
        except Exception as e:
            logger.error(f"Cloud training process failed: {str(e)}", exc_info=True)
            if self.current_step:
                self.progress.mark_step_status(self.current_step, CloudStatus.FAILED, f"Error: {str(e)}")
            return False

    def _wait_for_completion_process(self, cloud_service, job_id):
        try:
            def handle_sigterm(signum, frame):
                logger.info(f"Wait completion process received SIGTERM signal, exiting...")
                import sys
                sys.exit(0)
                
            signal.signal(signal.SIGTERM, handle_sigterm)
            
            logger.info(f"Async process: waiting for job {job_id} to complete")
            
            def progress_callback(status, progress, message):
                try:
                    logger.info(f"Progress update: {status}, {progress}%, {message}")
                    
                    status_mapping = {
                        "IN_PROGRESS": CloudStatus.IN_PROGRESS,
                        "COMPLETED": CloudStatus.COMPLETED,
                        "FAILED": CloudStatus.FAILED,
                        "CANCELED": CloudStatus.CANCELED  
                    }
                    
                    cloud_status = status_mapping.get(status, CloudStatus.IN_PROGRESS)
                    
                    self.progress.update_step_progress(
                        CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION,
                        progress,
                        message
                    )
                    
                    if status in ["COMPLETED", "FAILED", "CANCELED"]:
                        self.progress.mark_step_status(
                            CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION,
                            cloud_status,
                            message
                        )
                except Exception as e:
                    logger.error(f"Error in progress callback: {str(e)}", exc_info=True)
            
            success = cloud_service.wait_for_job_completion(
                job_id=job_id,
                progress_callback=progress_callback
            )
            
            if success:
                self.progress.update_message("Fine-tuning job completed successfully!")
                # Update is_trained flag for memory records after successful cloud training
                self.update_memory_training_status()
            else:
                logger.error(f"Fine-tuning job failed")
        except Exception as e:
            logger.error(f"Error in async wait thread: {str(e)}", exc_info=True)
            self.progress.mark_step_status("cloud_training", "wait_for_fine-tune_completion", CloudStatus.FAILED)
    
    def update_memory_training_status(self):
        """Update is_trained flag for memory records after successful cloud training"""
        try:
            
            with DatabaseSession.session() as session:
                update_count = session.query(Memory).filter(Memory.status == "active").update(
                    {"is_trained": True},
                    synchronize_session=False  
                )
                
                session.commit()
            logger.info(f"Updated training status for {update_count} memory records after cloud training")
        except Exception as e:
            logger.error(f"Failed to update memory training status: {str(e)}", exc_info=True)

    def _update_overall_progress(self):
        """Calculate and update the overall progress based on the stages' progress"""
        try:
            stages = self.progress.progress.data["stages"]
            total_stages = len(stages)
            completed_stages = 0
            total_progress = 0.0

            for stage in stages:
                total_progress += stage["progress"]
                if stage["status"] == CloudStatus.COMPLETED:
                    completed_stages += 1

            if total_stages > 0:
                overall_progress = total_progress / total_stages
            else:
                overall_progress = 0.0

            self.progress.progress.data["overall_progress"] = overall_progress
            logger.info(f"Updated overall progress to {overall_progress:.2f}%")

            if completed_stages == total_stages:
                self.progress.progress.data["status"] = CloudStatus.COMPLETED
                logger.info("All stages completed, setting overall status to COMPLETED")
            
            self.progress.save_progress()
        except Exception as e:
            logger.error(f"Error updating overall progress: {str(e)}")
    
    def stop_process(self) -> bool:
        """Stop the cloud training process
        
        This method will attempt to stop the fine-tuning job if it's in progress,
        by deleting the job, and update the progress status accordingly.
        It will also wait for the current data processing step to complete before returning.
        
        Returns:
            bool: True if the process was successfully stopped, False otherwise
        """
        try:
            logger.info(f"Attempting to stop cloud training process for model: {self.model_name}")
            
            self.is_stopped = True
            
            max_wait_time = 300 
            wait_start = time.time()
            
            current_stage = self.progress.get_progress().get("current_stage")
            logger.info(f"Current stage when stopping: {current_stage}")
            current_step = None
            
            if current_stage:
                for stage in self.progress.get_progress().get("stages", []):
                    if stage["name"] == current_stage:
                        current_step_name = stage.get("current_step")
                        if current_step_name:
                            found = False
                            for step in CloudProcessStep:
                                if step.value == current_step_name:
                                    current_step = step
                                    found = True
                                    logger.info(f"Found step in CloudProcessStep: {current_step}")
                                    break
                            
                            if not found:
                                for step in ProcessStep:
                                    if step.value == current_step_name:
                                        current_step = step
                                        logger.info(f"Found step in ProcessStep: {current_step}")
                                        break
                        break
            
            logger.info(f"Current step when stopping: {current_step}")
            
            while time.time() - wait_start < max_wait_time:
                if current_step:
                    step_status = None
                    current_stage_data = None
                    
                    for stage in self.progress.progress.data["stages"]:
                        if stage["name"] == current_stage:
                            current_stage_data = stage
                            logger.info(f"Found current stage data: {stage['name']}")
                            break
                    
                    if current_stage_data:
                        step_name = current_step.value if hasattr(current_step, 'value') else str(current_step)
                        logger.info(f"Looking for step with name: {step_name}")
                        for step in current_stage_data["steps"]:
                            if step["name"] == step_name:
                                step_status = step["status"]
                                logger.info(f"Found step status: {step_status}")
                                break
                    
                    logger.info(f"Current step status: {step_status}")
                    if step_status in [CloudStatus.COMPLETED, CloudStatus.FAILED, CloudStatus.CANCELED]:
                        logger.info(f"Step {current_step.value} has status {step_status}, continuing with stop process")
                        break
                
                time.sleep(2)
            
            if time.time() - wait_start >= max_wait_time:
                logger.warning(f"Waited {max_wait_time} seconds for current step to complete, proceeding with stop process")
            
            if not self.job_id:
                try:
                    params_dir = Path("data/cloud_progress")
                    job_file_path = params_dir / "job_id.json"
                    
                    if job_file_path.exists():
                        with open(job_file_path, "r") as f:
                            job_info = json.load(f)
                            if "job_id" in job_info:
                                self.job_id = job_info["job_id"]
                                logger.info(f"Retrieved job_id from file: {self.job_id}")
                except Exception as e:
                    logger.error(f"Failed to read job ID from file: {str(e)}", exc_info=True)
            
            # Terminate the wait completion process if it's running
            if self._wait_completion_process and self._wait_completion_process.is_alive():
                logger.info(f"Terminating wait completion process (PID: {self._wait_completion_pid})")
                try:
                    os.kill(self._wait_completion_pid, signal.SIGTERM)
                    self._wait_completion_process.join(timeout=5)
                    if self._wait_completion_process.is_alive():
                        logger.warning(f"Wait completion process did not terminate gracefully, forcing termination")
                        self._wait_completion_process.terminate()
                    logger.info(f"Wait completion process terminated successfully")
                except Exception as e:
                    logger.error(f"Error terminating wait completion process: {str(e)}", exc_info=True)
            
            if self.job_id:
                logger.info(f"Attempting to cancel fine-tune job: {self.job_id}")
                success = self.cloud_service.cancel_fine_tune_job(self.job_id)
                
                if success:
                    logger.info(f"Successfully canceled fine-tune job: {self.job_id}")
                else:
                    logger.error(f"Failed to cancel fine-tune job: {self.job_id}")
            else:
                logger.warning("No active fine-tune job found to delete")

            if current_step:
                step_status = None
                current_stage_data = None
                
                for stage in self.progress.progress.data["stages"]:
                    if stage["name"] == current_stage:
                        current_stage_data = stage
                        break
                
                if current_stage_data:
                    step_name = current_step.value if hasattr(current_step, 'value') else str(current_step)
                    for step in current_stage_data["steps"]:
                        if step["name"] == step_name:
                            step_status = step["status"]
                            break
                
                if step_status != CloudStatus.COMPLETED:
                    logger.info(f"Marking step {current_step} as CANCELED because its status is {step_status}")
                    self.progress.mark_step_status(current_step, CloudStatus.CANCELED, "Process canceled by user")
                else:
                    logger.info(f"Step {current_step} is already COMPLETED, preserving its status")
            
            logger.info("Cloud training process has been stopped successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error stopping cloud process: {str(e)}", exc_info=True)
            return False
