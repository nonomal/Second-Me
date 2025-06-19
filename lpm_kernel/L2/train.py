from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from tqdm import tqdm
import functools
# Standard library imports
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import torch.amp
# Third-party imports
import datasets
import psutil
import torch.multiprocessing as mp
import transformers
from peft import LoraConfig
from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments, set_seed
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lpm_kernel.configs.logging import TRAIN_LOG_FILE
from datasets import load_dataset

# Local imports
from lpm_kernel.L2.utils import (
    create_and_prepare_model,
    formatting_prompts_func,
    create_chat_data,
    release_ollama_models_early,
)

from lpm_kernel.configs.logging import LOGGING_CONFIG
import logging.config
from lpm_kernel.configs.logging import get_train_process_logger
from lpm_kernel.L2.memory_manager import get_memory_manager

logger = get_train_process_logger()


# Configure how tqdm displays in logs
class LogTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mininterval", 1.0)
        kwargs.setdefault("ascii", True)
        super().__init__(*args, **kwargs)

# Replace the default tqdm
sys.modules["tqdm"].tqdm = LogTqdm

# Debug callback for logging training progress
class DebugCallback(transformers.TrainerCallback):
    def __init__(self):
        self.total_time = 0
        self.last_time = time.time()
        self.total_steps = 0
        self.current_step = 0
        self.progress_percentage = 0.0
        self.progress_file = os.path.join(os.path.dirname(TRAIN_LOG_FILE), "train_progress.json")

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps
        progress_data = {
            "percentage": 0.0,
            "current_step": 0,
            "total_steps": self.total_steps
        }
        self._write_progress_to_file(progress_data)
        logger.info(f"Training started. Total steps: {self.total_steps}")

    def _write_progress_to_file(self, progress_data):
        try:
            import json
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json_str = json.dumps(progress_data)
                f.write(json_str)
        except Exception as e:
            logger.error(f"Error writing progress to file: {str(e)}")

    def on_step_end(self, args, state, control, **kwargs):

        current_time = time.time()
        step_time = current_time - self.last_time
        self.total_time += step_time
        self.last_time = current_time
        self.current_step = state.global_step
            
        # Log step time and training progress
        logger.info(f"Step {state.global_step}: {step_time:.2f}s - Total training time: {self.total_time:.2f}s")

        self.progress_percentage = min(100.0, (self.current_step / self.total_steps) * 100)
        progress_data = {
            "percentage": self.progress_percentage,
            "current_step": self.current_step,
            "total_steps": self.total_steps
        }
        self._write_progress_to_file(progress_data)
        
        logger.info(
            f"Updated train_progress: percentage={self.progress_percentage}, current_step={self.current_step}, total_steps={self.total_steps}")

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"Epoch {state.epoch} completed")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="float32",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )
    use_cuda: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables CUDA GPU acceleration for training and inference when available."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """
    dataset_name: Optional[str] = field(
        default="resources/L2/data/merged.json",
        metadata={"help": "The preference dataset to use."},
    )
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, appends `eos_token_id` at the end of each sample being packed."
        },
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, tokenizers adds special tokens to each sample being packed."
        },
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    is_sequential: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, the dataset is sequential."},
    )

    is_cot: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, the dataset is COT dataset."},
    )
    user_name: Optional[str] = field(
        default="User",
        metadata={"help": "The name of the user."},
    )

class MindSFTTrainer(SFTTrainer):
    def get_train_dataloader(self) -> DataLoader:
        """
                Returns the training [`~torch.utils.data.DataLoader`].

                Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
                training if necessary) otherwise.

                Subclass and override this method if you want to inject some custom behavior.
                """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = SequentialSampler(self.train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

def seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def main(model_args, data_args, training_args):
    logger.info(f"Python version--------------------: {sys.version}")

    # Configure logging
    logging.config.dictConfig(LOGGING_CONFIG)

    logger.info("Begin training...")

    # Ensure logs are flushed immediately
    for handler in logging.getLogger().handlers:
        handler.flush()

    # Get memory manager for optimization
    memory_manager = get_memory_manager()
    memory_manager.cleanup_memory(force=True)
    
    # Release Ollama models if they exist to free up VRAM
    if torch.cuda.is_available() and model_args.use_cuda:
        release_ollama_models_early()
    
    logger.info("Initializing training with memory optimizations")
    set_seed(training_args.seed)
    
    # Apply PyTorch memory optimizations to training arguments
    logger.info("Applying memory optimizations to training configuration")
    training_args = memory_manager.optimize_training_args(training_args)

    # 禁用MPS内存上限
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    logger.info("Setting PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable memory upper limit")

    # --- Accelerate optimizer state offloading logic ---
    # Enable optimizer state offload to CPU if VRAM is low
    vram_total = memory_manager.get_memory_info().get("vram_total_gb", 0)
    use_accelerate_offload = False
    if torch.cuda.is_available() and model_args.use_cuda and vram_total > 0 and vram_total < 16:
        logger.info("Enabling Hugging Face Accelerate optimizer state offload to CPU for low VRAM GPUs")
        accelerate_config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "NO",
            "downcast_bf16": False,
            "fsdp_config": {},
            "main_training_function": "main",
            "mixed_precision": "no",
            "num_machines": 1,
            "num_processes": 1,
            "use_cpu": False,
            "zero3_init_flag": False,
            "offload_optimizer_device": "cpu",
            "offload_param_device": "none"
        }
        training_args.accelerate_config = accelerate_config
        use_accelerate_offload = True

    # Model loading with device_map="auto" for automatic offloading
    logger.info(f"Loading model with automatic memory management from {model_args.model_name_or_path}")
    
    # Create model arguments dict with automatic offloading
    model_kwargs = {
        # Don't use "auto" device_map initially to avoid meta tensor issues
        "device_map": None,
        "trust_remote_code": True
    }
    
    # Configure quantization if requested
    if model_args.use_4bit_quantization:
        from transformers import BitsAndBytesConfig
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, model_args.bnb_4bit_quant_storage_dtype)
        
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit_quantization,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )
        # For 4-bit models, we can use device_map="auto"
        model_kwargs["device_map"] = "auto"
        logger.info("Using 4-bit quantization for memory efficiency")
    elif model_args.use_8bit_quantization:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=model_args.use_8bit_quantization
        )
        # For 8-bit models, we can use device_map="auto"
        model_kwargs["device_map"] = "auto"
        logger.info("Using 8-bit quantization for memory efficiency")
    
    # Flash attention for memory efficiency when supported
    if model_args.use_flash_attn and torch.cuda.is_available() and model_args.use_cuda:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2 for memory efficiency")

    # Set default device map if not already set by quantization
    if "device_map" not in model_kwargs or model_kwargs["device_map"] is None:
        model_kwargs["device_map"] = {"": int(os.getenv("LOCAL_RANK", 0))}
    
    # Set default torch dtype if using CUDA
    if torch.cuda.is_available() and model_args.use_cuda:
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    # Add gradient checkpointing related settings
    model_kwargs["use_cache"] = not training_args.gradient_checkpointing
    
    logger.info(f"Loading model with settings: {model_kwargs}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )
    except Exception as load_err:
        logger.error(f"Error while loading model: {load_err}")
        logger.error(traceback.format_exc())
        raise

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="right")

    if model_args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            r=model_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=model_args.lora_target_modules.split(",")
            if model_args.lora_target_modules != "all-linear"
            else model_args.lora_target_modules,
        )
    else:
        peft_config = None
    
    # If model has meta tensors, handle them properly
    if hasattr(model, "is_meta") and model.is_meta:
        logger.info("Model has meta tensors, using to_empty() to properly initialize")
        device = "cuda" if torch.cuda.is_available() and model_args.use_cuda else "cpu"
        model = model.to_empty(device=device)
    
    # Apply gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        logger.info("Enabling gradient checkpointing for memory efficiency")
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    # Allow only one full forward/backward pass at a time (if needed for memory)
    if torch.cuda.is_available() and memory_manager.get_memory_info().get("vram_total_gb", 0) < 8:
        torch.cuda.set_per_process_memory_fraction(0.9)
        logger.info("Setting memory fraction limit to avoid OOM errors")

    # datasets
    dataset = load_dataset("json", data_files=data_args.dataset_name, split="train")
    train_dataset = dataset.map(create_chat_data, batched=True, remove_columns=dataset.column_names)

    response_template = "\n<|im_start|>assistant\n"
    
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    training_args.dataset_kwargs = {
        "append_concat_token": data_args.append_concat_token,
        "add_special_tokens": data_args.add_special_tokens,
    }


    trainer = MindSFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        data_collator=collator,
    )
    
    # Print model details
    trainer.accelerator.print(f"{trainer.model}")
    
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    
    # Memory usage tracking callback
    class MemoryMonitorCallback(transformers.TrainerCallback):
        def __init__(self):
            self.memory_manager = get_memory_manager()
        
        def on_step_end(self, args, state, control, **kwargs):
            # Check memory every 5 steps
            if state.global_step % 5 == 0 and torch.cuda.is_available():
                info = self.memory_manager.get_memory_info()
                vram_usage_pct = info.get("vram_used_gb", 0) / info.get("vram_total_gb", 1) * 100
                
                if vram_usage_pct > 90:
                    logger.info(f"VRAM usage high ({vram_usage_pct:.1f}%), cleaning cache")
                    self.memory_manager.cleanup_memory()
        
        def on_save(self, args, state, control, **kwargs):
            # Free up memory before saving
            self.memory_manager.cleanup_memory(force=True)

    # Add memory monitoring
    trainer.add_callback(MemoryMonitorCallback())
    
    # Add existing debug callback
    trainer.add_callback(DebugCallback())

    # Resume from checkpoint if specified
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    # Training with automatic memory management
    try:
        logger.info("Starting training with memory-optimized configuration")
        trainer.train(resume_from_checkpoint=checkpoint)
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

    # Save the model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    
    # Clean up before saving
    memory_manager.cleanup_memory(force=True)
    
    trainer.save_model()
    logger.info("Training completed successfully")


# Create a patch to handle autocast compatibility
def get_autocast():
    if hasattr(torch.cpu, "amp") and hasattr(torch.cpu.amp, "autocast"):
        # Old version
        return torch.cpu.amp.autocast
    else:
        # New version
        return lambda **kwargs: torch.amp.autocast("cpu", **kwargs)


# Replace the original torch.cpu.amp.autocast with our compatible function
torch.cpu.amp.autocast = get_autocast()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SFTConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
