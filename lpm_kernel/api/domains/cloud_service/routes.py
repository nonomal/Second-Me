from flask import Blueprint, jsonify, request, Response
from typing import Dict, Any, Optional, List

from sqlalchemy import table
from ...common.responses import APIResponse
from ...common.errors import APIError, ErrorCodes
import logging
import gc
import os
import time
import json
import psutil
import tempfile
import threading
import signal
import sys
from multiprocessing import Process
from datetime import datetime
from pathlib import Path
from ....configs.config import Config
from .service import CloudService
from .cloud_trainprocess_service import CloudTrainProcessService
from .cloud_progress_holder import CloudProgressHolder
from ...services.user_llm_config_service import UserLLMConfigService
from ...dto.user_llm_config_dto import UpdateUserLLMConfigDTO
from lpm_kernel.api.domains.cloud_service.dto.cloud_inference_dto import CloudInferenceRequest
from lpm_kernel.api.services.local_llm_service import local_llm_service

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()
cloud_bp = Blueprint("cloud_service", __name__, url_prefix="/api/cloud_service")

@cloud_bp.route("/set_api_key", methods=["POST"])
def set_api_key():
    """
    Request: JSON object, containing:
    - api_key: str, Cloud Service API Key
    """
    try:
        data = request.json
        api_key = data.get('api_key')
        
        if not api_key:
            return jsonify(APIResponse.error("API key is required"))
        
        user_llm_config_service = UserLLMConfigService()
        
        config = user_llm_config_service.get_available_llm()
        
        update_data = {}
        if config:
            update_data = config.dict()
        
        update_data['cloud_service_api_key'] = api_key
        
        dto = UpdateUserLLMConfigDTO(**update_data)
        user_llm_config_service.update_config(1, dto)
            
        return jsonify(APIResponse.success(message="API key setting successful and saved to database"))
        
    except Exception as e:
        logger.error(f"Failed to set API key: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to set API key: {str(e)}"))


@cloud_bp.route("/get_api_key", methods=["GET"])
def get_api_key():
    try:
        user_llm_config_service = UserLLMConfigService()
        config = user_llm_config_service.get_available_llm()
        
        api_key = ""
        if config and hasattr(config, 'cloud_service_api_key'):
            api_key = config.cloud_service_api_key or ""
            
            if api_key:
                return jsonify(APIResponse.success(data={
                    "api_key": api_key
                }))
        
        return jsonify(APIResponse.success(data={
            "api_key": ""
        }))
    except Exception as e:
        logger.error(f"Failed to get API key: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to get API key: {str(e)}"))

@cloud_bp.route("/list_available_models", methods=["GET"])
def list_available_models():
    try:
        cloud_service = CloudService()
        
        models = cloud_service.list_available_models()
        
        if not models:
            logger.warning("No models available for fine-tuning")
        
        return jsonify(APIResponse.success(data=models))
    
    except Exception as e:
        logger.error(f"Failed to list available models: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to list available models: {str(e)}"))


# ============= Cloud Training Process Routes =============

@cloud_bp.route("/train/resume", methods=["POST"])
def resume_cloud_training():
    """Resume cloud training process from the last checkpoint
    
    Request: JSON object, containing:
    - model_name: str, optional, the model name to resume training for
    - base_model: str, optional, the base model to use for fine-tuning
    - training_type: str, optional, the training type to use
    - hyper_parameters: dict, optional, hyperparameters for training
    
    If model_name is not provided, the system will attempt to resume the most recent training process.
    """
    try:

        cloud_train_service = CloudTrainProcessService.get_instance()

        if cloud_train_service is None:
            logger.warning("No training parameters found in file")
            return jsonify(APIResponse.error("No training parameters found. Please use /train/start endpoint for initial training."))

        thread = threading.Thread(target=cloud_train_service.start_process)
        thread.daemon = True
        thread.start()

        return jsonify(APIResponse.success("Cloud resume training process started successfully"))
    
    except Exception as e:
        logger.error(f"Failed to resume cloud training: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to resume cloud training: {str(e)}"))

@cloud_bp.route("/train/stop", methods=["POST"])
def stop_cloud_training():
    """Stop cloud training process
    
    Request: JSON object, containing:
    - model_name: str, optional, the model name to stop training for
    
    If model_name is not provided, the system will attempt to stop the most recent training process.
    """
    try:

        train_service = CloudTrainProcessService.get_instance()
        
        if not train_service:
            return jsonify(APIResponse.error("No training parameters found. Please use /train/start endpoint for initial training."))
        
        # 停止训练进程
        success = train_service.stop_process()
        
        if success:
            return jsonify(APIResponse.success(message=f"Cloud training process  stopped successfully"))
        else:
            return jsonify(APIResponse.error(f"Failed to stop cloud training process"))
    
    except Exception as e:
        logger.error(f"Failed to stop cloud training process: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to stop cloud training process: {str(e)}"))

@cloud_bp.route("/train/start", methods=["POST"])
def start_cloud_training():
    """Start cloud training process"""
    try:
        params_dir = Path("data/cloud_progress")
        if params_dir.exists():
            import os
            logger.info("Cleaning cloud_progress directory...")
            for file_path in params_dir.glob("*"):
                if file_path.is_file():
                    os.remove(file_path)
                    logger.info(f"Removed file: {file_path}")
            logger.info("Cloud_progress directory cleaned successfully")
            
        data = request.json
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = timestamp
        
        base_model = data.get("base_model")
        training_type = data.get("training_type", "efficient_sft")
        hyper_parameters = data.get("hyper_parameters", {})
        
        training_params = {
            "model_name": model_name,
            "base_model": base_model,
            "training_type": training_type,
            "hyper_parameters": hyper_parameters,
            "created_at": datetime.now().isoformat()
        }
        

        params_dir = Path("data/cloud_progress")
        params_dir.mkdir(parents=True, exist_ok=True)
        
        params_file = params_dir / "cloud_training_params.json"
        with open(params_file, "w", encoding="utf-8") as f:
            json.dump(training_params, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training parameters saved to {params_file}")
        
        train_service = CloudTrainProcessService(current_model_name=model_name, base_model=base_model, training_type=training_type, hyper_parameters=hyper_parameters)
        
        def async_train_process():
            try:
                logger.info(f"Starting async cloud training process for model: {model_name}")
                success = train_service.start_process()
                if not success:
                    # 检查是否是由于用户停止导致的
                    if train_service.is_stopped:
                        logger.info("Async cloud training process was stopped by user")
                    else:
                        logger.error("Async cloud training process failed")
                else:
                    logger.info(f"Async cloud training process completed successfully with job_id: {train_service.job_id}")
            except Exception as e:
                logger.error(f"Async cloud training process failed with error: {str(e)}", exc_info=True)
        
        thread = threading.Thread(target=async_train_process, daemon=True)
        thread.start()
        logger.info(f"Started async thread for cloud training process")
        
        return jsonify(APIResponse.success("Cloud training process started successfully"))
    except Exception as e:
        logger.error(f"Start cloud training failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to start cloud training: {str(e)}"))


@cloud_bp.route("/train/status/training/<job_id>", methods=["GET"])
def get_cloud_training_status(job_id):
    """Get cloud training status"""
    try:
        cloud_service = CloudService()
        model_id = None
        status = cloud_service.check_fine_tune_status(job_id)

        if not status:
            return jsonify(APIResponse.error("Failed to get training status"))

        if status == "SUCCEEDED":
            model_id = cloud_service.model_id
        
        return jsonify(APIResponse.success(data=model_id, message=status))
    except Exception as e:
        logger.error(f"Get cloud training status failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to get cloud training status: {str(e)}"))

@cloud_bp.route("/train/progress", methods=["GET"])
def get_cloud_training_progress():
    """获取云端训练的详细进度信息"""
    try:
        # 获取进度数据
        progress_holder = None
        job_id = None
        
        # 先检查进度文件是否存在
        progress_file = Path("data/cloud_progress/cloud_progress.json")
        if progress_file.exists():
            # 尝试从现有的训练服务实例获取进度
            train_service = CloudTrainProcessService.get_instance()
            if train_service:
                progress_holder = train_service.progress
                job_id = train_service.job_id
            else:
                try:
                    # 直接读取文件内容
                    with open(progress_file, "r", encoding="utf-8") as f:
                        progress_data = json.load(f)
                    
                    # 创建一个新的进度持有者
                    progress_holder = CloudProgressHolder(model_name=progress_data.get("model_name"), job_id=progress_data.get("job_id"))
                    progress_holder.progress.data = progress_data
                    progress_holder._rebuild_mappings()
                    
                    logger.info(f"Loaded progress data directly from file: {progress_file}")
                except Exception as e:
                    logger.error(f"Error reading progress file directly: {str(e)}")
                    # 如果直接读取失败，尝试使用get_latest_progress
                    progress_holder, job_id = CloudProgressHolder.get_latest_progress()
        
        # 如果还是没有找到进度数据，创建一个新的空进度
        if not progress_holder:
            progress_holder = CloudProgressHolder()
            job_id = None
            logger.info("Created new empty progress holder as no existing progress was found")
        
        # 获取进度数据
        progress_data = progress_holder.get_progress()
        
        # 添加一些额外的元数据
        response_data = {
            "progress": progress_data,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "job_id": job_id
        }
        
        return jsonify(APIResponse.success(data=response_data))
    except Exception as e:
        logger.error(f"Get cloud training progress failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to get cloud training progress: {str(e)}"))

@cloud_bp.route("/train/progress/reset", methods=["POST"])
def reset_cloud_training_progress():
    """重置云端训练的进度信息并强制初始化cloud_progress.json文件"""
    try:
        # 获取当前的CloudTrainProcessService实例
        train_service = CloudTrainProcessService.get_instance()
        
        # 如果有训练服务实例，立即终止它
        if train_service:
            logger.info(f"Found running CloudTrainProcessService instance, forcibly terminating it...")
            
            # 设置停止标志
            train_service.is_stopped = True
            
            # 终止进程的通用函数
            def kill_process_with_children(pid, process_name):
                try:
                    if not psutil.pid_exists(pid):
                        logger.warning(f"{process_name} with PID {pid} no longer exists")
                        return
                        
                    logger.info(f"Forcibly terminating {process_name} with PID: {pid}")
                    process = psutil.Process(pid)
                    
                    # 获取并终止所有子进程
                    children = process.children(recursive=True)
                    for child in children:
                        try:
                            logger.info(f"Killing child process with PID: {child.pid}")
                            child.kill()  # 直接使用 kill 而不是 terminate，确保立即终止
                        except Exception as e:
                            logger.error(f"Error killing child process: {str(e)}")
                    
                    # 终止主进程
                    try:
                        process.kill()  # 直接使用 kill 而不是 terminate
                        logger.info(f"Successfully killed {process_name} with PID: {pid}")
                    except Exception as e:
                        logger.error(f"Error killing main process: {str(e)}")
                except Exception as e:
                    logger.error(f"Error in kill_process_with_children for {process_name}: {str(e)}", exc_info=True)
            
            # 终止数据处理进程
            if hasattr(train_service, '_data_processing_process') and train_service._data_processing_process and train_service._data_processing_pid:
                kill_process_with_children(train_service._data_processing_pid, "data processing process")
                
            # 终止等待任务完成进程
            if hasattr(train_service, '_wait_completion_process') and train_service._wait_completion_process and train_service._wait_completion_pid:
                kill_process_with_children(train_service._wait_completion_pid, "wait completion process")
            
            # 如果有任务ID，记录下来但不等待取消结果
            job_id = None
            if hasattr(train_service, 'job_id') and train_service.job_id:
                job_id = train_service.job_id
                logger.info(f"Found running job: {job_id}, will be cancelled separately")
            
            # 强制重置实例变量，不等待任何进程或线程完成
            CloudTrainProcessService._instance = None
            CloudTrainProcessService._initialized = False
            logger.info("Forcibly reset CloudTrainProcessService instance variables")
            
            # 如果有任务ID，在后台发送取消请求
            if job_id:
                try:
                    # 创建新的CloudService实例取消任务，避免使用已重置的train_service
                    cloud_service = CloudService()
                    
                    # 定义取消任务的进程函数
                    def cancel_job_process(job_id):
                        try:
                            # 注册信号处理程序
                            def handle_sigterm(signum, frame):
                                logger.info(f"Cancel job process received SIGTERM signal, exiting...")
                                sys.exit(0)
                                
                            signal.signal(signal.SIGTERM, handle_sigterm)
                            
                            # 发送取消请求
                            cloud_service.cancel_fine_tune_job(job_id)
                        except Exception as e:
                            logger.error(f"Error in cancel job process: {str(e)}", exc_info=True)
                    
                    # 使用进程发送取消请求，不阻塞主进程
                    cancel_process = Process(
                        target=cancel_job_process,
                        args=(job_id,),
                        name="CancelJobProcess",
                        daemon=True
                    )
                    cancel_process.start()
                    logger.info(f"Started background process to cancel job {job_id} with PID: {cancel_process.pid}")
                except Exception as e:
                    logger.error(f"Error starting job cancellation thread: {str(e)}")
        
        # 清除现有的CloudTrainProcessService实例相关文件
        params_dir = Path("data/cloud_progress")
        params_dir.mkdir(parents=True, exist_ok=True)
        
        # 删除所有相关文件
        files_to_delete = [
            "cloud_training_params.json",  # 训练参数文件
            "cloud_progress.json",         # 进度文件
            "job_id.json"                 # 任务ID文件
        ]
        
        for file_name in files_to_delete:
            file_path = params_dir / file_name
            if file_path.exists():
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")
        
        # 强制重置CloudTrainProcessService的实例变量
        CloudTrainProcessService._instance = None
        CloudTrainProcessService._initialized = False  # 重置初始化标志
        logger.info("Reset CloudTrainProcessService instance variables")
        
        # 创建全新的进度持有者
        new_progress_holder = CloudProgressHolder()
        
        # 重置进度
        new_progress_holder.progress.reset()
        
        gc.collect()
        logger.info("Forced garbage collection to clean up any lingering references")
        
        # 定义进度文件路径
        progress_file_path = params_dir / "cloud_progress.json"
        
        # 保存初始化的进度（只保存一次）
        new_progress_holder.save_progress()
        logger.info(f"Created new progress file with completely fresh state: {progress_file_path}")
        
        # 创建一个空的训练参数文件，确保下次不会加载旧的训练服务
        empty_params = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reset": True
        }
        params_file = params_dir / "cloud_training_params.json"
        with open(params_file, "w", encoding="utf-8") as f:
            json.dump(empty_params, f, indent=2, ensure_ascii=False)
        logger.info(f"Created empty training params file to prevent loading old service: {params_file}")
        
        return jsonify(APIResponse.success(message=f"Cloud training progress has been completely reset with a fresh state"))
    except Exception as e:
        logger.error(f"Reset cloud training progress failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to reset cloud training progress: {str(e)}"))

# ... (其他代码保持不变)
def search_job_info():
    try:
        # 使用data/cloud_progress文件夹存储job_id.json
        params_dir = Path("data/cloud_progress")
        params_dir.mkdir(parents=True, exist_ok=True)
        job_file_path = params_dir / "job_id.json"
        
        if not job_file_path.exists():
            return jsonify(APIResponse.success(data={
                "exists": False,
                "message": "Job information file not found"
            }))
        
        try:
            with open(job_file_path, "r") as f:
                job_info = json.load(f)
                
            return jsonify(APIResponse.success(data=job_info))
        except json.JSONDecodeError:
            return jsonify(APIResponse.error("Invalid JSON format in job information file"))
    except Exception as e:
        logger.error(f"Get job info failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to get job information: {str(e)}"))

@cloud_bp.route("/train/deploy", methods=["POST"])
def deploy_cloud_model():
    """Deploy fine-tuned model"""
    try:
        data = request.json
        model_id = data.get("model_id")
        capacity = data.get("capacity", 2)
        
        if not model_id:
            return jsonify(APIResponse.error("model_id is required"))
        
        cloud_service = CloudService()
        
        model_id = cloud_service.deploy_model( capacity=capacity)
        
        if not model_id:
            return jsonify(APIResponse.error("Failed to deploy model"))
        
        return jsonify(APIResponse.success(message="Model deployment started", data={
            "model_id": model_id
        }))
    except Exception as e:
        logger.error(f"Deploy model failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to deploy model: {str(e)}"))

@cloud_bp.route("/train/deployment_status/<model_id>", methods=["GET"])
def check_cloud_deployment_status(model_id):
    """Check deployment status"""
    try:
        cloud_service = CloudService()

        status = cloud_service.check_deployment_status(model_id)
        
        if status is None:
            return jsonify(APIResponse.error("Failed to check deployment status"))
        
        return jsonify(APIResponse.success(data=status))
    except Exception as e:
        logger.error(f"Check deployment status failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to check deployment status: {str(e)}"))

@cloud_bp.route("/train/inference", methods=["POST"])
def run_cloud_inference():
    """Run inference using deployed model with local knowledge retrieval
    
    This endpoint accepts a request in OpenAI-compatible format and returns a response
    in the same format. It supports both streaming and non-streaming responses.
    It also supports local knowledge retrieval before sending to cloud inference.
    
    Request format:
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "model_id": "your-model-id",
        "temperature": 0.1,
        "max_tokens": 2000,
        "stream": true,
        "enable_l0_retrieval": false,
        "enable_l1_retrieval": false,
        "role_id": "optional-role-id"
    }
    """
    try:
        
        try:
            body = CloudInferenceRequest(**request.json)
        except Exception as e:
            logger.error(f"Invalid request format: {str(e)}")
            return jsonify(APIResponse.error(f"Invalid request format: {str(e)}"))
        
        # 1. 检查必要参数
        if not body.messages:
            return jsonify(APIResponse.error("messages are required"))
        if not body.model_id:
            return jsonify(APIResponse.error("model_id is required"))

        # 2. 执行本地知识检索（如果启用）
        enhanced_messages = body.messages.copy()
        
        if body.enable_l0_retrieval or body.enable_l1_retrieval:
            logger.info("Performing local knowledge retrieval before cloud inference")
            
            # 从本地 ChatService 获取知识增强的消息
            from lpm_kernel.api.domains.kernel2.dto.chat_dto import ChatRequest
            from lpm_kernel.api.domains.kernel2.services.chat_service import chat_service
            
            # 构造临时的 ChatRequest 对象用于知识检索
            temp_chat_request = ChatRequest(
                message="",  # 将通过 messages 字段传递
                messages=body.messages,
                model="",  # 云端推理不需要本地模型
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                metadata={
                    'enable_l0_retrieval': body.enable_l0_retrieval,
                    'enable_l1_retrieval': body.enable_l1_retrieval,
                    'role_id': body.role_id
                }
            )
            
            # 使用 ChatService 构建增强的消息（仅用于知识检索和prompt构建）
            try:
                enhanced_messages = chat_service._build_messages(temp_chat_request)
                logger.info(f"Enhanced messages with local knowledge: {len(enhanced_messages)} messages")
            except Exception as e:
                logger.error(f"Local knowledge retrieval failed: {str(e)}")
                # 如果知识检索失败，继续使用原始消息
                enhanced_messages = body.messages

        # 3. 创建CloudService实例
        cloud_service = CloudService()

        try:
            # 4. 调用run_inference方法，使用增强后的消息
            response = cloud_service.run_inference(
                messages=enhanced_messages,
                model_id=body.model_id,
                stream=body.stream,
                temperature=body.temperature,
                max_tokens=body.max_tokens
            )
            
            # 5. 处理流式或非流式响应
            if body.stream:
                return response
            else:
                # 对于非流式响应，返回完整的JSON响应
                if not response:
                    return jsonify(APIResponse.error("Failed to run inference"))
                return jsonify(APIResponse.success(data=response))
                
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Inference error: {error_msg}")
            error_response = {
                "error": {
                    "message": error_msg,
                    "type": "server_error",
                    "code": "inference_error"
                }
            }
            
            # 根据请求类型返回错误响应
            if body.stream:
                return local_llm_service.handle_stream_response(iter([error_response]))
            else:
                return jsonify(APIResponse.error(message=error_msg))
                
    except Exception as e:
        logger.error(f"Run inference failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to run inference: {str(e)}"))

@cloud_bp.route("/train/list_deployments", methods=["GET"])
def list_deployments():
    """获取所有已部署的模型列表"""
    try:
        cloud_service = CloudService()
        deployments = cloud_service.list_deployments()
        
        if deployments is None:
            return jsonify(APIResponse.error("Failed to list deployments"))
        
        return jsonify(APIResponse.success(data={
            "deployments": deployments,
            "count": len(deployments)
        }))
    except Exception as e:
        logger.error(f"List deployments failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to list deployments: {str(e)}"))


@cloud_bp.route("/train/delete_deployment", methods=["POST"])
def delete_cloud_deployment():
    """Delete deployed model"""
    try:
        data = request.json
        model_id = data.get("model_id")
        
        if not model_id:
            return jsonify(APIResponse.error("model_id is required"))
        
        cloud_service = CloudService()
        
        status = cloud_service.delete_deployment(model_id)
        
        if status is None:
            return jsonify(APIResponse.error("Failed to delete deployment"))
        
        return jsonify(APIResponse.success(message=f"Deployment deleted successfully", data=status))
    except Exception as e:
        logger.error(f"Delete deployment failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to delete deployment: {str(e)}"))

@cloud_bp.route("/train/delete_fine_tune_job", methods=["POST"])
def delete_fine_tune_job():
    """Delete fine-tune job"""
    try:
        data = request.json
        job_id = data.get("job_id")
        
        if not job_id:
            return jsonify(APIResponse.error("job_id is required"))
        
        cloud_service = CloudService()
        
        success = cloud_service.delete_fine_tune_job(job_id)
        
        if not success:
            return jsonify(APIResponse.error("Failed to delete fine-tune job"))
        
        return jsonify(APIResponse.success(message=f"Fine-tune job deleted successfully"))
    except Exception as e:
        logger.error(f"Delete fine-tune job failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to delete fine-tune job: {str(e)}"))
