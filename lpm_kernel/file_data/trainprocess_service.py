from enum import Enum
from typing import Dict, List, Optional
import json
import os
import re
import time
import psutil
from lpm_kernel.configs.config import Config
import logging
from lpm_kernel.L1.utils import save_true_topics
from lpm_kernel.L1.serializers import NotesStorage
from lpm_kernel.kernel.note_service import NoteService
from lpm_kernel.L2.l2_generator import L2Generator
from lpm_kernel.L2.utils import save_hf_model
from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.api.domains.loads.services import LoadService
from lpm_kernel.kernel.chunk_service import ChunkService
from lpm_kernel.kernel.l1.l1_manager import (
    extract_notes_from_documents,
    document_service,
    get_latest_status_bio,
    get_latest_global_bio,
)
from lpm_kernel.api.common.script_executor import ScriptExecutor
from lpm_kernel.configs.config import Config
from lpm_kernel.file_data.chunker import DocumentChunker
from lpm_kernel.kernel.l1.l1_manager import generate_l1_from_l0
import threading
from ..api.domains.trainprocess.progress import TrainProgress, Status, Step, Status
import gc

class ProcessStep(Enum):
    """Training process steps"""

    LIST_DOCUMENTS = "list_documents"
    GENERATE_DOCUMENT_EMBEDDINGS = "generate_document_embeddings"
    CHUNK_DOCUMENT = "process_chunks"
    CHUNK_EMBEDDING = "chunk_embedding"
    EXTRACT_DIMENSIONAL_TOPICS = "extract_dimensional_topics"
    MODEL_DOWNLOAD = "model_download"
    MAP_ENTITY_NETWORK = "map_your_entity_network"
    DECODE_PREFERENCE_PATTERNS = "decode_preference_patterns"
    REINFORCE_IDENTITY = "reinforce_identity"
    AUGMENT_CONTENT_RETENTION = "augment_content_retention"
    TRAIN = "train"
    MERGE_WEIGHTS = "merge_weights"
    CONVERT_MODEL = "convert_model"

    @classmethod
    def get_ordered_steps(cls) -> List["ProcessStep"]:
        """Get ordered steps"""
        return [
            cls.MODEL_DOWNLOAD,
            cls.LIST_DOCUMENTS,
            cls.GENERATE_DOCUMENT_EMBEDDINGS,
            cls.CHUNK_DOCUMENT,
            cls.CHUNK_EMBEDDING,
            cls.EXTRACT_DIMENSIONAL_TOPICS,
            cls.MAP_ENTITY_NETWORK,
            cls.DECODE_PREFERENCE_PATTERNS,
            cls.REINFORCE_IDENTITY,
            cls.AUGMENT_CONTENT_RETENTION,
            cls.TRAIN,
            cls.MERGE_WEIGHTS,
            cls.CONVERT_MODEL,
        ]
        
    def get_method_name(self) -> str:
        """Get the corresponding method name for this step"""
        # Map from step value to method name
        method_name_mapping = {
            "model_download": "model_download",
            "list_documents": "list_documents",
            "generate_document_embeddings": "generate_document_embeddings",
            "process_chunks": "process_chunks",
            "chunk_embedding": "chunk_embedding",
            "extract_dimensional_topics": "extract_dimensional_topics",
            "map_your_entity_network": "map_entity_network",
            "decode_preference_patterns": "decode_preference_patterns",
            "reinforce_identity": "reinforce_identity",
            "augment_content_retention": "augment_content_retention",
            "train": "train",
            "merge_weights": "merge_weights",
            "convert_model": "convert_model",
        }
        return method_name_mapping[self.value]


class Progress:
    """Progress management class"""

    def __init__(
        self, progress_file: str = "trainprocess_progress.json", progress_callback=None
    ):
        progress_dir = os.path.join(os.getcwd(), "data/progress")
        if not os.path.exists(progress_dir):
            os.makedirs(progress_dir)
        self.progress_file = os.path.join(progress_dir, progress_file)
        self.progress = TrainProgress()
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)
        self._load_progress()

    def _load_progress(self):
        """Load progress file"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    saved_progress = json.load(f)
                    # Restore saved progress state
                    for stage_name, stage_data in saved_progress.get("stages", {}).items():
                        if stage_name in self.progress.stages:
                            stage = self.progress.stages[stage_name]
                            # Restore stage progress
                            if "progress" in stage_data:
                                stage.progress = stage_data["progress"]
                            # Restore stage status
                            if "status" in stage_data:
                                stage.status = Status[stage_data["status"].upper()]
                            # Restore current step
                            if "current_step" in stage_data:
                                stage.current_step = stage_data["current_step"]
                            
                            # Restore step status
                            for step_name, step_data in stage_data.get("steps", {}).items():
                                if step_name in stage.steps:
                                    step = stage.steps[step_name]
                                    if "status" in step_data:
                                        status = Status[step_data["status"].upper()]
                                        self.progress.update_progress(
                                            stage_name,
                                            step_name,
                                            status,
                                            step_data.get("progress", None)
                                        )
                    
                    # Restore overall progress
                    if "overall_progress" in saved_progress:
                        self.progress.overall_progress = saved_progress["overall_progress"]
                    # Restore current stage
                    if "current_stage" in saved_progress:
                        self.progress.current_stage = saved_progress["current_stage"]
                    # Restore overall status
                    if "status" in saved_progress:
                        self.progress.status = Status[saved_progress["status"].upper()]
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to load progress file: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error loading progress: {str(e)}")

    def _save_progress(self):
        """Save progress"""
        progress_dict = self.progress.to_dict()
        with open(self.progress_file, "w") as f:
            json.dump(progress_dict, f, indent=2)
        if self.progress_callback:
            self.progress_callback(progress_dict)
        self._load_progress()

    def _get_stage_and_step(self, step: ProcessStep) -> tuple:
        """Get the stage and step name corresponding to the step"""
        step_name = step.value
        # Determine the stage based on step name
        stage_mapping = {
            ProcessStep.MODEL_DOWNLOAD: "downloading_the_base_model",
            
            ProcessStep.LIST_DOCUMENTS: "activating_the_memory_matrix",
            ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS: "activating_the_memory_matrix",
            ProcessStep.CHUNK_DOCUMENT: "activating_the_memory_matrix",
            ProcessStep.CHUNK_EMBEDDING: "activating_the_memory_matrix",
            
            ProcessStep.EXTRACT_DIMENSIONAL_TOPICS: "synthesize_your_life_narrative",
            ProcessStep.MAP_ENTITY_NETWORK: "synthesize_your_life_narrative",
            
            ProcessStep.DECODE_PREFERENCE_PATTERNS: "prepare_training_data_for_deep_comprehension",
            ProcessStep.REINFORCE_IDENTITY: "prepare_training_data_for_deep_comprehension",
            ProcessStep.AUGMENT_CONTENT_RETENTION: "prepare_training_data_for_deep_comprehension",
            
            ProcessStep.TRAIN: "training_to_create_second_me",
            ProcessStep.MERGE_WEIGHTS: "training_to_create_second_me",
            ProcessStep.CONVERT_MODEL: "training_to_create_second_me",
        }
        return stage_mapping[step], step_name

    def is_step_completed(self, step: ProcessStep) -> bool:
        """Check if a step is completed"""
        stage_name, step_name = self._get_stage_and_step(step)
        stage = self.progress.stages.get(stage_name)
        if not stage:
            return False
        step_info = stage.steps.get(step_name)
        return step_info and step_info.completed

    def mark_step_completed(self, step: ProcessStep):
        """Mark a step as completed"""
        stage_name, step_name = self._get_stage_and_step(step)
        self.progress.update_progress(stage_name, step_name, Status.COMPLETED)
        self._save_progress()
        if self.progress_callback:
            self.progress_callback({
                "stage": stage_name,
                "step": step_name,
                "status": Status.COMPLETED.value
            })

    def mark_step_failed(self, step: ProcessStep):
        """Mark a step as failed"""
        stage_name, step_name = self._get_stage_and_step(step)
        self.progress.update_progress(stage_name, step_name, Status.FAILED)
        self._save_progress()
        if self.progress_callback:
            self.progress_callback({
                "stage": stage_name,
                "step": step_name,
                "status": Status.FAILED.value
            })
            
    def mark_step_in_progress(self, step: ProcessStep):
        """Mark a step as in progress"""
        stage_name, step_name = self._get_stage_and_step(step)
        self.progress.update_progress(stage_name, step_name, Status.IN_PROGRESS)
        self._save_progress()
        if self.progress_callback:
            self.progress_callback({
                "stage": stage_name,
                "step": step_name,
                "status": Status.IN_PROGRESS.value
            })

    def reset_progress(self):
        """Reset all progress"""
        self.progress = TrainProgress()
        self._save_progress()
        if self.progress_callback:
            self.progress_callback({
                "reset": True,
                "status": Status.PENDING.value
            })

    def get_last_successful_step(self) -> Optional[ProcessStep]:
        """Get the last successfully completed step"""
        ordered_steps = ProcessStep.get_ordered_steps()
        for step in reversed(ordered_steps):
            if self.is_step_completed(step):
                return step
        return None


class TrainProcessService:
    """Training process service (singleton pattern)"""
    
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, base_url: str = None, progress_file: str = "trainprocess_progress.json", progress_callback=None, model_name: str = None):
        if not self._initialized:
            config = Config.from_env()
            self.base_url = base_url or config.KERNEL2_SERVICE_URL
            # Generate a unique progress file name based on model name
            if model_name:
                progress_file = f"trainprocess_progress_{model_name}.json"
            self.progress = Progress(progress_file, progress_callback)
            self.logger = logging.getLogger(__name__)
            self.model_name = None  # Initialize as None
            self._initialized = True
            
            # Initialize stop flag
            self.is_stopped = False
            # Initialize training process tracking
            self.training_process = None
            self.current_step = None
            
            # Initialize L2 data dictionary
            self.l2_data = {
                "notes": None,
                "basic_info": None,
                "data_output_base_dir": None,
                "topics_path": None,
                "entitys_path": None,
                "graph_path": None,
                "config_path": None
            }
            self.l2_data_prepared = False
        
        # Update callback function
        if progress_callback is not None:
            self.progress.progress_callback = progress_callback
            
        # Update model name and progress instance if model name changes
        if model_name is not None and model_name != self.model_name:
            self.model_name = model_name
            # Create new progress instance with updated progress file name
            progress_file = f"trainprocess_progress_{model_name}.json"
            self.progress = Progress(progress_file, progress_callback)

    def list_documents(self):
        """List all documents"""
        try:
            # Mark step as in progress
            self.progress.mark_step_in_progress(ProcessStep.LIST_DOCUMENTS)            
            # Directly call document service instead of API
            documents = document_service.list_documents()
            # Mark step as completed if we found documents
            self.progress.mark_step_completed(ProcessStep.LIST_DOCUMENTS)
                
            return [doc.to_dict() for doc in documents]
        except Exception as e:
            self.logger.error(f"List documents failed: {str(e)}")
            self.progress.mark_step_failed(ProcessStep.LIST_DOCUMENTS)
            return []

    def generate_document_embeddings(self) -> bool:
        """Process embeddings for all documents"""
        try:
            # Mark step as in progress
            self.progress.mark_step_in_progress(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS)
            documents = self.list_documents() 
            for doc in documents:
                doc_id = doc.get("id")

                # Directly call document service instead of API
                embedding = document_service.process_document_embedding(doc_id)
                if embedding is None:
                    self.logger.error(
                        f"Generate document embeddings failed for doc_id: {doc_id}"
                    )
                    self.progress.mark_step_failed(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS)
                    return False
                self.progress.mark_step_completed(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS)
                self.logger.info(f"Successfully generated embedding for document {doc_id}") 
            return True
        except Exception as e:
            self.logger.error(f"Generate document embeddings failed: {str(e)}")
            self.progress.mark_step_failed(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS)
            return False

    def process_chunks(self) -> bool:
        """Process document chunks"""
        try:
            # Mark step as in progress
            self.progress.mark_step_in_progress(ProcessStep.CHUNK_DOCUMENT)
            config = Config.from_env()
            chunker = DocumentChunker(
                chunk_size=int(config.get("DOCUMENT_CHUNK_SIZE")),
                overlap=int(config.get("DOCUMENT_CHUNK_OVERLAP")),
            )
            documents = document_service.list_documents()
            processed, failed = 0, 0

            chunk_service = ChunkService()
            for doc in documents:
                try:
                    if not doc.raw_content:
                        self.logger.warning(f"Document {doc.id} has no content, skipping...")
                        failed += 1
                        continue

                    # Split into chunks and save
                    chunks = chunker.split(doc.raw_content)
                    for chunk in chunks:
                        chunk.document_id = doc.id
                        chunk_service.save_chunk(chunk)

                    processed += 1
                    self.logger.info(
                        f"Document {doc.id} processed: {len(chunks)} chunks created"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to process document {doc.id}: {str(e)}")
                    failed += 1      
            self.progress.mark_step_completed(ProcessStep.CHUNK_DOCUMENT)
            return True
        except Exception as e:
            self.logger.error(f"Process chunks failed: {str(e)}")
            self.progress.mark_step_failed(ProcessStep.CHUNK_DOCUMENT)
            return False

    def chunk_embedding(self) -> bool:
        """Process embeddings for all document chunks"""
        try:
            # Mark step as in progress
            self.progress.mark_step_in_progress(ProcessStep.CHUNK_EMBEDDING)
            documents = self.list_documents()
            for doc in documents:
                doc_id = doc.get("id")
                try:
                    # Directly call document service to generate chunk embeddings
                    processed_chunks = document_service.generate_document_chunk_embeddings(doc_id)
                    if not processed_chunks:
                        self.logger.warning(f"No chunks to process for document: {doc_id}")
                        continue
                except Exception as e:
                    self.logger.error(
                        f"Generate chunk embeddings failed for doc_id: {doc_id}: {str(e)}"
                    )
                    self.progress.mark_step_failed(ProcessStep.CHUNK_EMBEDDING)
                    return False
            # All documents' chunks processed successfully
            self.progress.mark_step_completed(ProcessStep.CHUNK_EMBEDDING)
            return True
        except Exception as e:
            self.logger.error(f"Generate chunk embeddings failed: {str(e)}")
            self.progress.mark_step_failed(ProcessStep.CHUNK_EMBEDDING)
            return False

    def extract_dimensional_topics(self) -> bool:
        """Extract dimensional topics (L0 and L1)"""
        try:
            # Mark step as in progress
            self.progress.mark_step_in_progress(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS)
            self.logger.info("Starting dimensional topics extraction (L0 and L1)...")
            
            # Step 1: Generate L0 - Call document_service to analyze all documents
            self.logger.info("Generating L0 data...")
            analyzed_docs = document_service.analyze_all_documents()
            self.logger.info(f"Successfully analyzed {len(analyzed_docs)} documents for L0")
            
            # Step 2: Generate L1 - Direct call to L1 generator service
            self.logger.info("Generating L1 data...")
            generate_l1_from_l0()      
            self.logger.info("Successfully generated L1 data")
            
            # Mark step as completed
            self.progress.mark_step_completed(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS)
            self.logger.info("Dimensional topics extraction completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Extract dimensional topics failed: {str(e)}")
            self.progress.mark_step_failed(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS)
            return False

    def model_download(self) -> bool:
        """Download model"""
        try:
            # Mark step as in progress
            self.progress.mark_step_in_progress(ProcessStep.MODEL_DOWNLOAD)
            # Directly call save_hf_model function to download model
            self.logger.info(f"Starting model download: {self.model_name}")
            
            # Start monitoring the download progress in a separate thread
            monitor_thread = threading.Thread(target=self._monitor_model_download)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Start the actual download
            model_path = save_hf_model(self.model_name)
            
            if model_path and os.path.exists(model_path):
                self.logger.info(f"Model downloaded successfully to {model_path}")
                self.progress.mark_step_completed(ProcessStep.MODEL_DOWNLOAD)
                return True
            else:
                self.logger.error(f"Model path does not exist after download: {model_path}")
                self.progress.mark_step_failed(ProcessStep.MODEL_DOWNLOAD)
                return False

        except Exception as e:
            self.logger.error(f"Download model failed: {str(e)}")
            self.progress.mark_step_failed(ProcessStep.MODEL_DOWNLOAD)
            return False

    def map_entity_network(self)->bool:
        """Map entity network using notes and basic info"""
        try:
            # Mark step as in progress
            self.progress.mark_step_in_progress(ProcessStep.MAP_ENTITY_NETWORK)
            self.logger.info("Starting entity network mapping...")
        
            # Get or prepare L2 data
            self._prepare_l2_data()

            l2_generator = L2Generator(
                data_path=os.path.join(os.getcwd(), "resources")
            )
            l2_generator.data_preprocess(self.l2_data["notes"], self.l2_data["basic_info"])
            
            self.progress.mark_step_completed(ProcessStep.MAP_ENTITY_NETWORK)
            self.logger.info("Entity network mapping completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Map entity network failed: {str(e)}")
            self.progress.mark_step_failed(ProcessStep.MAP_ENTITY_NETWORK)
            self._cleanup_resources()
            return False

    def decode_preference_patterns(self)->bool:
        """Decode preference patterns using notes and related data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_in_progress(ProcessStep.DECODE_PREFERENCE_PATTERNS)
            self.logger.info("Starting preference patterns decoding...")
            # Get or prepare L2 data
            self._prepare_l2_data()

            # Use data from l2_data dictionary
            L2Generator().gen_preference_data(                
                    self.l2_data["notes"],
                    self.l2_data["basic_info"],
                    self.l2_data["data_output_base_dir"],
                    self.l2_data["topics_path"],
                    self.l2_data["entitys_path"],
                    self.l2_data["graph_path"],
                    self.l2_data["config_path"]
                    )
            
            self.progress.mark_step_completed(ProcessStep.DECODE_PREFERENCE_PATTERNS)
            self.logger.info("Preference patterns decoding completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Decode preference patterns failed: {str(e)}")
            self.progress.mark_step_failed(ProcessStep.DECODE_PREFERENCE_PATTERNS)
            return False

    def reinforce_identity(self)->bool:
        """Reinforce identity using notes and related data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_in_progress(ProcessStep.REINFORCE_IDENTITY)
            self.logger.info("Starting identity reinforcement...")
            # Get or prepare L2 data
            self._prepare_l2_data()

            # Use data from l2_data dictionary
            l2_generator = L2Generator(
                data_path=os.path.join(os.getcwd(), "resources")
                )  
            l2_generator.gen_subjective_data(                
                    self.l2_data["notes"],
                    self.l2_data["basic_info"],
                    self.l2_data["data_output_base_dir"],
                    self.l2_data["topics_path"],
                    self.l2_data["entitys_path"],
                    self.l2_data["graph_path"],
                    self.l2_data["config_path"]
                    )
            
            self.progress.mark_step_completed(ProcessStep.REINFORCE_IDENTITY)
            self.logger.info("Identity reinforcement completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Reinforce identity failed: {str(e)}")
            self.progress.mark_step_failed(ProcessStep.REINFORCE_IDENTITY)
            return False
            
    def _cleanup_resources(self):
        """Clean up resources to prevent memory leaks"""
        self.logger.info("Cleaning up resources to prevent memory leaks")
        
        # Clean up large data structures in l2_data dictionary
        for key in self.l2_data:
            self.l2_data[key] = None
        
        self.l2_data_prepared = False
        
        # Force garbage collection
        gc.collect()
        
        # Log memory usage after cleanup
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        self.logger.info(f"Memory usage after cleanup: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    def augment_content_retention(self) -> bool:
        """Augment content retention using notes, basic info and graph data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_in_progress(ProcessStep.AUGMENT_CONTENT_RETENTION)
            self.logger.info("Starting content retention augmentation...")
            # Get or prepare L2 data
            self._prepare_l2_data()

            # Use data from l2_data dictionary
            l2_generator = L2Generator(data_path=os.path.join(os.getcwd(), "resources"))
            l2_generator.gen_diversity_data(
                self.l2_data["notes"],
                self.l2_data["basic_info"],
                self.l2_data["data_output_base_dir"],
                self.l2_data["topics_path"],
                self.l2_data["entitys_path"],
                self.l2_data["graph_path"],
                self.l2_data["config_path"]
            )
            
            # Mark step as completed
            self.logger.info("Content retention augmentation completed successfully")
            self.progress.mark_step_completed(ProcessStep.AUGMENT_CONTENT_RETENTION)
            
            # Clean up resources after completion
            self._cleanup_resources()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to augment content retention: {str(e)}")
            self.progress.mark_step_failed(ProcessStep.AUGMENT_CONTENT_RETENTION)
            # Clean up resources even if there was an error
            self._cleanup_resources()
            return False

    def _prepare_l2_data(self) -> dict:
        """Prepare common data needed for L2 generation tasks using lazy loading
        
        Returns:
            Dictionary containing all L2 data:
            - notes: List of prepared notes
            - basic_info: Dict containing user information
            - data_output_base_dir: Path to output directory
            - topics_path: Path to topics data
            - entitys_path: Path to entity mapping file
            - graph_path: Path to graph data
            - config_path: Path to config file
        """
        # If data is already prepared, return cached data directly
        if self.l2_data_prepared and all(self.l2_data.values()):
            self.logger.info("Using cached L2 data")
            return self.l2_data
        
        self.logger.info("Preparing L2 data...")
        
        # Setup directories and paths
        config = Config.from_env()
        base_dir = os.path.join(
            os.getcwd(), config.get("USER_DATA_PIPELINE_DIR") + "/raw_data"
        )
        os.makedirs(base_dir, exist_ok=True)

        # get topic
        topics_path = os.path.join(base_dir, "topics.json")
        self.l2_data["topics_path"] = topics_path
        self.logger.info("Topics data not found, generating it...")
        chunk_service = ChunkService()
        topics_data = chunk_service.query_topics_data()
        save_true_topics(topics_data, topics_path)

        # Initialize storage
        storage = NotesStorage()
        self.logger.info("Notes not found, preparing them...")
        documents = document_service.list_documents_with_l0()
        self.logger.info(f"list_documents_with_l0 len: {len(documents)}")
        notes_list, _ = extract_notes_from_documents(documents)
        self.logger.info(f"extract_notes_from_documents len: {len(notes_list)}")
        note_service = NoteService()
        note_service.prepareNotes(notes_list)
        storage.save_notes(notes_list)
        self.l2_data["notes"] = storage.load_notes()

        # Get paths
        self.l2_data["config_path"] = os.path.join(
            os.getcwd(),
            "resources/L2/data_pipeline/data_prep/subjective/config/config.json",
        )
        self.l2_data["entitys_path"] = os.path.join(
            os.getcwd(),
            "resources/L2/data_pipeline/raw_data/id_entity_mapping_subjective_v2.json",
        )
        self.l2_data["graph_path"] = os.path.join(
            os.getcwd(),
            "resources/L1/graphrag_indexing_output/subjective/entities.parquet",
        )
        self.l2_data["data_output_base_dir"] = os.path.join(os.getcwd(), "resources/L2/data")

        # Lazy load user information
        self.logger.info("Loading user information...")
        status_bio = get_latest_status_bio()
        global_bio = get_latest_global_bio()
        self.l2_data["basic_info"] = {
            "username": LoadService.get_current_upload_name(),
            "aboutMe": LoadService.get_current_upload_description(),
            "statusBio": status_bio.content if status_bio else "Currently working on an AI project.",
            "globalBio": global_bio.content_third_view if global_bio 
                else "The User is a software engineer who loves programming and learning new technologies.",
            "lang": "English",
        }
        
        # Mark data as prepared
        self.l2_data_prepared = True
        
        return self.l2_data

    def train(self) -> bool:
        """Start model training"""
        try:
            # Mark step as in progress
            self.progress.mark_step_in_progress(ProcessStep.TRAIN)
            
            # Get paths for the model
            paths = self._get_model_paths(self.model_name)
            
            # Check if model exists
            if not os.path.exists(paths["base_path"]):
                self.logger.error(f"Model '{self.model_name}' does not exist, please download first")
                self.progress.mark_step_failed(ProcessStep.TRAIN)
                return False
            
            # Prepare log directory and file
            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "train.log")
            self.logger.info(f"Log file path: {log_path}")
            
            # Ensure output directory exists
            os.makedirs(paths["personal_dir"], exist_ok=True)
            
            # Set USER_NAME environment variable
            os.environ["USER_NAME"] = LoadService.get_current_upload_name()
            self.logger.info(f"USER_NAME environment variable set: {os.environ['USER_NAME']}")
            
            script_path = os.path.join(os.getcwd(), "lpm_kernel/L2/train_for_user.sh")
            
            # Start training in a separate thread
            training_thread = threading.Thread(
                target=self._start_training,
                args=(script_path, log_path),
                daemon=True
            )
            training_thread.start()
            
            self.logger.info("Training started, monitoring progress")
            # start monitoring training progress
            return self._monitor_training_progress()
            
        except Exception as e:
            self.logger.error(f"Failed to start training: {str(e)}")
            self.progress.mark_step_failed(ProcessStep.TRAIN)
            return False
            
    def _get_model_paths(self, model_name):
        """Get all relevant paths for a model and set environment variables
        
        Args:
            model_name: Model name
            
        Returns:
            Dictionary containing all related paths:
            - base_path: Base model path
            - personal_dir: Personal trained model output directory
            - merged_dir: Merged model output directory
            - gguf_dir: GGUF model output directory
        """
        base_dir = os.getcwd()
        paths = {
            "base_path": os.path.join(base_dir, "resources/L2/base_models", model_name),
            "personal_dir": os.path.join(base_dir, "resources/model/output/personal_model", model_name),
            "merged_dir": os.path.join(base_dir, "resources/model/output/merged_model", model_name),
            "gguf_dir": os.path.join(base_dir, "resources/model/output/gguf", model_name)
        }
        
        # Ensure all directories exist
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
            
        # Set environment variables
        os.environ["MODEL_BASE_PATH"] = paths["base_path"]
        os.environ["MODEL_PERSONAL_DIR"] = paths["personal_dir"]
        os.environ["MODEL_MERGED_DIR"] = paths["merged_dir"]
        os.environ["MODEL_GGUF_DIR"] = paths["gguf_dir"]
        
        # Log environment variables
        self.logger.info("Set environment variables:")
        self.logger.info(f"MODEL_BASE_PATH: {paths['base_path']}")
        self.logger.info(f"MODEL_PERSONAL_DIR: {paths['personal_dir']}")
        self.logger.info(f"MODEL_MERGED_DIR: {paths['merged_dir']}")
        self.logger.info(f"MODEL_GGUF_DIR: {paths['gguf_dir']}")
        
        return paths
        
    def _start_training(self, script_path, log_path):
        """Start training process
        
        Args:
            script_path: Path to training script
            log_path: Path to log file
            
        Returns:
            bool: True if the training process started successfully, False otherwise
        """
        try:
            # Use ScriptRunner to execute the script
            from lpm_kernel.api.common.script_runner import ScriptRunner
            runner = ScriptRunner(log_path=log_path)
            
            # Reset stop flag before starting
            self.is_stopped = False
            
            # Start the training process
            training_process = runner.execute_script(
                script_path=script_path,
                script_type="training",
                is_python=False,  # This is a bash script
            )
            
            self.logger.info(f"Training process started: {training_process}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start training process: {str(e)}")
            return False

    def _monitor_training_progress(self) -> bool:
        """Monitor training progress"""
        try:
            log_dir = os.path.join(os.getcwd(), "logs")
            log_file = os.path.join(log_dir, "train.log")
            
            # Initialize last_position to the end of file to only process new content
            try:
                with open(log_file, 'r') as f:
                    f.seek(0, 2)  # Move to the end of file
                    last_position = f.tell()
            except FileNotFoundError:
                # If file doesn't exist yet, start from beginning when it's created
                last_position = 0
            
            # variable to track training status
            total_steps = None
            current_step = 0
            last_update_time = time.time()
            training_started = False
            
            while True:
                try:
                    # read new log content
                    with open(log_file, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()
                        
                    for line in new_lines:
                        line = line.strip()
                        # Check if training has started
                        if not training_started:
                            if "***** Running training *****" in line:
                                training_started = True
                                self.logger.info("Training started")
                            continue  # Skip progress matching until training starts
                        
                        progress_match = re.search(r"(\d+)%\|[^|]+\| (\d+)/(\d+)", line)
                        if progress_match and len(progress_match.groups()) == 3:
                            percentage = int(progress_match.group(1))
                            current_step = int(progress_match.group(2))
                            total_steps = int(progress_match.group(3))
                            
                            # Update progress at most once per second
                            current_time = time.time()
                            if current_time - last_update_time >= 1.0:
                                # self.logger.info(f"Training progress: {percentage}% ({current_step}/{total_steps})")
                                if percentage == 100.0:
                                    self.progress.mark_step_completed(ProcessStep.TRAIN)
                                    return True
                                self._update_progress("training_to_create_second_me", "train", percentage, f"Current step: {current_step}/{total_steps}")
                                last_update_time = current_time
                    
                        # Check if we have exited the training record interval
                        if "=== Training Ended ===" in line:
                            # in_training_section = False  # Exit training record interval
                            self.logger.info("Exited training record interval")
                        
                    # Briefly pause to avoid excessive CPU usage
                    time.sleep(0.1)  
                    
                except IOError as e:
                    self.logger.error(f"Failed to read log file: {str(e)}")
                    time.sleep(0.1)
                    continue
                    
        except Exception as e:
            self.logger.error(f"Failed to monitor training progress: {str(e)}")
            self.progress.mark_step_failed(ProcessStep.TRAIN)
            return False

    def _update_progress(self, stage: str, step: str, percentage: float, message: str):
        """Update progress for any stage and step"""
        try:
            self.progress.progress.update_progress(
                stage,  # stage
                step,   # step
                Status.IN_PROGRESS,
                percentage
            )
            self.logger.info(f"Progress updated: {percentage}% - {message}")
        except Exception as e:
            self.logger.error(f"Failed to update progress: {str(e)}")

    def _monitor_model_download(self) -> bool:
        """Monitor model download progress"""
        try:
            log_dir = os.path.join(os.getcwd(), "logs")
            log_file = os.path.join(log_dir, "model_download.log")
            
            # Initialize last_position to the end of file to only process new content
            try:
                with open(log_file, 'r') as f:
                    f.seek(0, 2)  # Move to the end of file
                    last_position = f.tell()
            except FileNotFoundError:
                # If file doesn't exist yet, start from beginning when it's created
                last_position = 0
            
            # Variables to track download status
            current_file = ""
            file_size = 0
            total_size = 0  # Total size of all files
            file_sizes = {}  # Dictionary to store file sizes
            last_update_time = time.time()
            
            while True:
                try:
                    # Read new log content
                    with open(log_file, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()
                    
                    for line in new_lines:
                        line = line.strip()
                        
                        # Check for download start
                        if "Starting download of model:" in line:
                            self.logger.info("Model download started")
                            continue
                        
                        # Get file size information when a download starts
                        if "Starting download of file:" in line:
                            match = re.search(r"Starting download of file: (.+) \(Size: ([\d\.]+) MB\)", line)
                            if match:
                                current_file = match.group(1)
                                file_size = float(match.group(2))
                                file_sizes[current_file] = file_size
                                total_size = sum(file_sizes.values())
                                # self.logger.info(f"Starting download of {current_file} ({file_size} MB)")
                        
                        # Track file download progress
                        if "Downloaded" in line and "MB /" in line:
                            match = re.search(r"File (.+): Downloaded ([\d\.]+) MB / ([\d\.]+) MB \(([\d\.]+)%\)", line)
                            if match:
                                file_name = match.group(1)
                                downloaded_mb = float(match.group(2))
                                total_mb = float(match.group(3))
                                percentage = float(match.group(4))
                                
                                # Update file size if it was updated (especially for model.safetensors)
                                if total_mb > file_sizes.get(file_name, 0):
                                    file_sizes[file_name] = total_mb
                                    total_size = sum(file_sizes.values())
                                
                                # Calculate overall progress
                                if total_size > 0:
                                    # Sum up all downloaded data
                                    completed_files_size = sum([file_sizes.get(f, 0) for f in file_sizes if f != file_name])
                                    current_file_downloaded = (percentage / 100.0) * total_mb
                                    overall_downloaded = completed_files_size + current_file_downloaded
                                    current_progress = (overall_downloaded / total_size) * 100
                                    current_progress = min(99.0, current_progress)  # Cap at 99% until fully complete
                                    # Update progress at most once per second
                                    current_time = time.time()
                                    if current_time - last_update_time >= 3.0:

                                        self._update_progress(
                                            "downloading_the_base_model", 
                                            "model_download", 
                                            current_progress, 
                                            f"Overall: {current_progress:.1f}% - Downloading {file_name}: {percentage}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)"
                                        )
                                        last_update_time = current_time

                        if "Download completed." in line:
                            self.progress.mark_step_completed(ProcessStep.MODEL_DOWNLOAD)
                            self.logger.info("Model download completed")
                            return True
                    
                    # Briefly pause to avoid excessive CPU usage
                    time.sleep(0.1)
                    
                except IOError as e:
                    self.logger.error(f"Failed to read log file: {str(e)}")
                    time.sleep(0.1)
                    continue
                    
        except Exception as e:
            self.logger.error(f"Failed to monitor model download progress: {str(e)}")
            return False
            
    def merge_weights(self) -> bool:
        """Merge weights"""
        try:
            # Mark step as in progress
            self.progress.mark_step_in_progress(ProcessStep.MERGE_WEIGHTS)

            paths = self._get_model_paths(self.model_name)
            
            # Check if model exists
            if not os.path.exists(paths["base_path"]):
                self.logger.error(f"Model '{self.model_name}' does not exist, please download first")
                self.progress.mark_step_failed(ProcessStep.MERGE_WEIGHTS)
                return False
            
            # Check if training output exists
            if not os.path.exists(paths["personal_dir"]):
                return jsonify(APIResponse.error(
                    message=f"Model '{model_name}' training output does not exist, please train model first",
                    code=400
                ))

            # Ensure merged output directory exists
            os.makedirs(paths["merged_dir"], exist_ok=True)
                
            script_path = os.path.join(
                os.getcwd(), "lpm_kernel/L2/merge_weights_for_user.sh"
                )
            log_path = os.path.join(os.getcwd(), "logs", f"merge_weights_{self.model_name}.log")
            
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            # Use script executor to execute merge script
            script_executor = ScriptExecutor()
            result = script_executor.execute(
                script_path=script_path, script_type="merge_weights", log_file=log_path
            )
            
            self.logger.info(f"Weight merge task result: {result}")
            
            # Check if script execution was successful
            if result.get('returncode', 1) != 0:
                error_msg = f"Merge weights failed: {result.get('error', 'Unknown error')}"
                self.logger.error(error_msg)
                self.progress.mark_step_failed(ProcessStep.MERGE_WEIGHTS)
                return False
                
            # Check if merged model files exist
            config_path = os.path.join(paths["merged_dir"], "config.json")
            if not os.path.exists(config_path):
                error_msg = f"Merged model files not found in {paths['merged_dir']}"
                self.logger.error(error_msg)
                self.progress.mark_step_failed(ProcessStep.MERGE_WEIGHTS)
                return False
            
            self.logger.info("Weight merge completed successfully")
            self.progress.mark_step_completed(ProcessStep.MERGE_WEIGHTS)
            return True

        except Exception as e:
            self.progress.mark_step_failed(ProcessStep.MERGE_WEIGHTS)
            self.logger.error(f"Merge weights failed: {str(e)}")
            return False

    def convert_model(self) -> bool:
        """Convert model to GGUF format"""
        try:
            # Mark step as in progress
            self.progress.mark_step_in_progress(ProcessStep.CONVERT_MODEL)

            # Get paths for the model
            paths = self._get_model_paths(self.model_name)
            
            # Check if merged model exists
            merged_model_dir = paths["merged_dir"]
            self.logger.info(f"Merged model path: {merged_model_dir}")
            if not os.path.exists(merged_model_dir):
                self.logger.error(f"Model '{self.model_name}' merged output does not exist, please merge model first")
                self.progress.mark_step_failed(ProcessStep.CONVERT_MODEL)
                return False
            
            # Get GGUF output directory
            gguf_dir = paths["gguf_dir"]
            self.logger.info(f"GGUF output directory: {gguf_dir}")
            
            script_path = os.path.join(os.getcwd(), "lpm_kernel/L2/convert_hf_to_gguf.py")
            gguf_path = os.path.join(gguf_dir, "model.gguf")
            self.logger.info(f"GGUF output path: {gguf_path}")
            
            # Build parameters
            args = [
                merged_model_dir,
                "--outfile",
                gguf_path,
                "--outtype",
                "f16",
            ]
            self.logger.info(f"Parameters: {args}")
            
            
            # Ensure GGUF output directory exists
            os.makedirs(os.path.dirname(gguf_path), exist_ok=True)
            
            # Use script executor to execute conversion script
            script_executor = ScriptExecutor()
            result = script_executor.execute(
                script_path=script_path,
                script_type="convert_model",
                args=args
            )
            
            self.logger.info(f"Model conversion result: {result}")
            
            # Check if script execution was successful
            if result.get('returncode', 1) != 0:
                error_msg = f"Model conversion failed: {result.get('error', 'Unknown error')}"
                self.logger.error(error_msg)
                self.progress.mark_step_failed(ProcessStep.CONVERT_MODEL)
                return False
                
            # Check if GGUF model file exists
            if not os.path.exists(gguf_path):
                error_msg = f"GGUF model file not found at {gguf_path}"
                self.logger.error(error_msg)
                self.progress.mark_step_failed(ProcessStep.CONVERT_MODEL)
                return False
            
            self.logger.info("Model conversion completed successfully")
            self.progress.mark_step_completed(ProcessStep.CONVERT_MODEL)
            return True
            
        except Exception as e:
            self.progress.mark_step_failed(ProcessStep.CONVERT_MODEL)
            self.logger.error(f"Convert model failed: {str(e)}")
            return False

    def check_training_condition(self) -> bool:
        """
        Check if the conditions for training are met
        Returns:
            bool: True if conditions are met, False otherwise
        """
        try:
            # Check if there are any documents that need embedding
            if document_service.check_all_documents_embeding_status():
                self.logger.warning("Cannot start training: There are documents that need embedding process first")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error checking training conditions: {str(e)}", exc_info=True)
            if self.progress.progress.current_stage:
                current_step = self.progress.progress.stages[self.progress.progress.current_stage].current_step
                if current_step:
                    step = ProcessStep(current_step)
                    self.progress.mark_step_failed(step)

    def start_process(self) -> bool:
        """Start training process"""
        try:
            self.is_stopped = False
            # Store the current process PID
            self.current_pid = os.getpid()  # Store the PID
            self.logger.info(f"Training process started with PID: {self.current_pid}")
            # Get the ordered list of all steps
            ordered_steps = ProcessStep.get_ordered_steps()

            # Get the last successfully completed step
            last_successful_step = self.progress.get_last_successful_step()
            start_index = 0
            if last_successful_step:
                start_index = ordered_steps.index(last_successful_step) + 1

            # Start executing from the step after the last successful one
            for step in ordered_steps[start_index:]:
                self.current_step = step
                if self.is_stopped:
                    self.logger.info("Training process aborted during step")
                    self.progress.mark_step_failed(step)
                    break  # If stop is requested, exit the loop
            
                self.logger.info(f"Starting step: {step.value}")

                # Execute the corresponding method
                method_name = step.get_method_name()
                if not hasattr(self, method_name):
                    self.logger.error(f"Method {method_name} not found")
                    self.progress.mark_step_failed(step)
                    return False

                method = getattr(self, method_name)
                success = method()

                if not success:
                    self.logger.error(f"Step {step.value} failed")
                    self.logger.info(f'Marking step as failed: stage={step.value}, step={step.value}')
                    self.progress.mark_step_failed(step)
                    return False
                self.logger.info(f"Step {step.value} completed successfully")
                # self.progress.mark_step_completed(step)
            if self.is_stopped:
                self.logger.info("Training process was stopped during a step")
            else:
               self.logger.info("Training process completed...")

            return True
        except Exception as e:
            self.logger.error(f"Exception occurred: {str(e)}")
            self.progress.mark_step_failed(step)
            return False

    def set_retrian_progress(self):
        """Save current progress
        
        This method saves the current progress to the progress file and triggers the progress callback if available.
        """
        try:
            self.progress.reset_progress()
            for step_name in self.progress.progress.stages["downloading_the_base_model"].steps:
                self.progress.progress.update_progress("downloading_the_base_model", step_name, Status.COMPLETED)
            self.progress._save_progress()
            self.logger.info("Progress saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save progress: {str(e)}")
    
    def stop_process(self):
        """Stop training process
        
        Returns:
            bool: True if the process was stopped successfully, False otherwise
        """
        try:
            # Set the stop flag
            self.is_stopped = True
            self.logger.info("Training process has been requested to stop")
            # mark train stop
            if self.current_step == ProcessStep.TRAIN:
                self.progress.mark_step_failed(ProcessStep.TRAIN)
            
            # First check if we have the current process PID
            if not hasattr(self, 'current_pid') or not self.current_pid:
                self.logger.info("No active process PID found")
                if self.progress.progress.current_stage:
                    current_step = self.progress.progress.stages[self.progress.progress.current_stage].current_step
                    if current_step:
                        step = ProcessStep(current_step)
                        self.progress.mark_step_failed(step)
                return True
                
            import psutil
            try:
                self.logger.info(f"Attempting to terminate process with PID: {self.current_pid}")
                
                # Check if the process exists
                if psutil.pid_exists(self.current_pid):
                    # Get the process object
                    process = psutil.Process(self.current_pid)
                    
                    # Get all child processes
                    children = process.children(recursive=True)
                    
                    # Terminate all child processes first
                    for child in children:
                        self.logger.info(f"Terminating child process with PID: {child.pid}")
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    
                    # Wait for children to terminate
                    gone, still_alive = psutil.wait_procs(children, timeout=3)
                    
                    # Kill any remaining children
                    for child in still_alive:
                        self.logger.info(f"Killing child process with PID: {child.pid}")
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    
                    # Note: We don't terminate the main process as it's this process
                    self.logger.info(f"All child processes of {self.current_pid} have been terminated") 
                    gc.collect()
                    return True
                else:
                    self.logger.warning(f"Process with PID {self.current_pid} no longer exists")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                self.logger.error(f"Failed to terminate process: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error stopping training process: {str(e)}")
            return False
