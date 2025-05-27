from flask import Blueprint, jsonify, request, Response
from typing import Dict, Any, Optional, List

from sqlalchemy import table
from ...common.responses import APIResponse
from ...common.errors import APIError, ErrorCodes
import logging
import os
import time
import json
import tempfile
import threading
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

@cloud_bp.route("/train/stop", methods=["POST"])
def stop_cloud_training():
    """Stop cloud training process
    
    Request: JSON object, containing:
    - model_name: str, optional, the model name to stop training for
    
    If model_name is not provided, the system will attempt to stop the most recent training process.
    """
    try:
        data = request.json or {}
        model_name = data.get("model_name")
        
        # 如果没有提供model_name，尝试从job_id.json文件中获取最近的训练任务
        if not model_name:
            try:
                current_dir = Path(__file__).parent
                job_file_path = current_dir / "job_id.json"
                
                if job_file_path.exists():
                    with open(job_file_path, "r") as f:
                        job_info = json.load(f)
                        job_id = job_info.get("job_id")
                        if job_id:
                            logger.info(f"Found job_id {job_id} from job_id.json")
                            # 使用时间戳作为模型名称
                            model_name = time.strftime("%Y%m%d_%H%M%S")
                else:
                    logger.warning("No job_id.json file found")
            except Exception as e:
                logger.error(f"Failed to read job ID from file: {str(e)}", exc_info=True)
        
        if not model_name:
            return jsonify(APIResponse.error("No model_name provided and no active training job found"))
        
        # 获取CloudTrainProcessService实例
        train_service = CloudTrainProcessService.get_instance(current_model_name=model_name)
        
        if not train_service:
            return jsonify(APIResponse.error(f"No training service found for model: {model_name}"))
        
        # 停止训练进程
        success = train_service.stop_process()
        
        if success:
            return jsonify(APIResponse.success(message=f"Cloud training process for model {model_name} stopped successfully"))
        else:
            return jsonify(APIResponse.error(f"Failed to stop cloud training process for model {model_name}"))
    
    except Exception as e:
        logger.error(f"Failed to stop cloud training process: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to stop cloud training process: {str(e)}"))

@cloud_bp.route("/train/start", methods=["POST"])
def start_cloud_training():
    """Start cloud training process"""
    try:
        data = request.json
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = timestamp
        
        base_model = data.get("base_model")
        training_type = data.get("training_type", "efficient_sft")
        hyper_parameters = data.get("hyper_parameters", {})
        
        train_service = CloudTrainProcessService(current_model_name=model_name, base_model=base_model, training_type=training_type, hyper_parameters=hyper_parameters)
        
        def async_train_process():
            try:
                logger.info(f"Starting async cloud training process for model: {model_name}")
                success = train_service.start_process()
                if not success:
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
        
        # 尝试从现有的训练服务实例获取进度
        train_service = CloudTrainProcessService.get_instance()
        if train_service:
            # 如果有正在运行的训练服务，使用其进度
            progress_holder = train_service.progress
            job_id = train_service.job_id
        else:
            # 如果没有正在运行的训练服务，尝试加载最新的进度文件
            progress_holder, job_id = CloudProgressHolder.get_latest_progress()
            if not progress_holder:
                # 如果没有找到进度文件，创建一个新的空进度
                progress_holder = CloudProgressHolder()
                job_id = None
        
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
    """重置云端训练的进度信息"""
    try:
        # 先检查是否有正在运行的训练服务
        train_service = CloudTrainProcessService.get_instance()
        job_id = None
        model_name = None
        
        if train_service:
            # 如果有正在运行的训练服务，使用其job_id和model_name
            job_id = train_service.job_id
            model_name = train_service.current_model_name
        else:
            # 如果没有正在运行的训练服务，尝试加载最新的进度文件
            progress_holder, job_id = CloudProgressHolder.get_latest_progress()
            if progress_holder:
                model_name = progress_holder.model_name
        
        # 创建一个新的进度持有者
        progress_holder = CloudProgressHolder(model_name=model_name, job_id=job_id)
        
        # 重置进度
        progress_holder.progress.reset()
        
        # 设置job_id和model_name
        if job_id:
            progress_holder.progress.data["job_id"] = job_id
        if model_name:
            progress_holder.progress.data["model_name"] = model_name
        
        # 保存重置后的进度
        progress_holder.save_progress()
        
        return jsonify(APIResponse.success(message=f"Cloud training progress for job {job_id} has been reset"))
    except Exception as e:
        logger.error(f"Reset cloud training progress failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to reset cloud training progress: {str(e)}"))

# ... (其他代码保持不变)
def search_job_info():
    try:
        current_dir = Path(__file__).parent
        job_file_path = current_dir / "job_id.json"
        
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
    """Run inference using deployed model
    
    This endpoint accepts a request in OpenAI-compatible format and returns a response
    in the same format. It supports both streaming and non-streaming responses.
    
    Request format:
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "model_id": "your-model-id",
        "temperature": 0.1,
        "max_tokens": 2000,
        "stream": true
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

        # 2. 创建CloudService实例
        cloud_service = CloudService()

        try:
            # 3. 调用run_inference方法，传递流式参数
            response = cloud_service.run_inference(
                messages=body.messages,
                model_id=body.model_id,
                stream=body.stream,
                temperature=body.temperature,
                max_tokens=body.max_tokens
            )
            
            # 4. 处理流式或非流式响应
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
