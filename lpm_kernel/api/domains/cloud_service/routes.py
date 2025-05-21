from flask import Blueprint, jsonify, request, Response
from ...common.responses import APIResponse
from ...common.errors import APIError, ErrorCodes
import logging
import os
import time
import json
import tempfile
from pathlib import Path
from ....configs.config import Config
from .service import CloudService

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()
cloud_bp = Blueprint("cloud_service", __name__)

# 全局变量，用于存储用户提供的API密钥创建的CloudService实例
service = None

@cloud_bp.route("/create_fine_tune_job", methods=["POST"])
def create_fine_tune_job():
    """
    Create model tuning tasks (one-stop interface)
    
    Request: JSON object, containing:
    - api_key: str, Cloud Service API Key
    - file_path: str, Training data file path
    - base_model: str
    - training_type: str, Training type, default is' efficient_ft '
    - Hyperparameters: Object, optional hyperparameters
    - description: str, Optional file description
    """
    try:
        # 获取请求数据
        data = request.json
            
        # 提取参数
        api_key = data.get('api_key')
        file_path = data.get('file_path')
        base_model = data.get('base_model')
        training_type = data.get('training_type', "efficient_sft")
        description = data.get('description', "")
        hyper_parameters = data.get('hyper_parameters', {})
        
        # 检查必要参数
        if not api_key:
            return jsonify(APIResponse.error("需要提供API密钥"))
        
        if not file_path:
            return jsonify(APIResponse.error("需要提供训练文件路径"))
            
        # 检查文件是否存在
        file_path = Path(file_path)
        if not file_path.exists():
            return jsonify(APIResponse.error(f"训练文件不存在: {file_path}"))
        
        service = CloudService(api_key=api_key)
        
        # 上传文件
        upload_result = service.upload_training_file(
            file_path=str(file_path),
            description=description
        )
        
        if not upload_result:
            return jsonify(APIResponse.error("上传训练文件失败"))
        
        # 创建调优任务
        result = service.create_fine_tune_job(
            base_model=base_model,
            training_type=training_type,
            hyper_parameters=hyper_parameters
        )
        
        if not result:
            return jsonify(APIResponse.error("创建调优任务失败"))
        
        return jsonify(APIResponse.success(data={
            "job_id": service.job_id,
            "file_id": service.file_id
        }))
    
    except Exception as e:
        logger.error(f"创建调优任务失败: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"创建调优任务失败: {str(e)}"))


@cloud_bp.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload training file to cloud service
    
    Request: UploadFileRequest JSON object containing:
    - file_path: str, path to the training file
    - description: str, optional description of the file
    """
    try:
        # Check if API key is configured
        if not api_key:
            return jsonify(APIResponse.error("Cloud service API key not configured"))
        
        # Get request data
        data = request.json
        file_path = data.get('file_path')
        description = data.get('description')
        
        if not file_path:
            return jsonify(APIResponse.error("file_path is required"))
            
        # Upload file
        result = cloud_service.upload_training_file(
            file_path=file_path,
            description=description
        )
        
        if not result:
            return jsonify(APIResponse.error("Failed to upload training file"))
        
        return jsonify(APIResponse.success(data={"file_id": cloud_service.file_id}))
    
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"File upload failed: {str(e)}"))


@cloud_bp.route("/fine-tune", methods=["POST"])
def fine_tune():
    """
    Create and start a fine-tuning job
    
    Request: FineTuneRequest JSON object containing:
    - base_model: str, base model to fine-tune, default is "qwen1.5-72b-chat"
    - training_type: str, type of training, default is "sft"
    - hyper_parameters: Dict, optional hyper parameters
    - file_path: str, optional path to training file (if not already uploaded)
    - description: str, optional description of the file (if uploading)
    """
    try:
        # Check if API key is configured
        if not api_key:
            return jsonify(APIResponse.error("Cloud service API key not configured"))
        
        # Get request data
        data = request.json
        file_path = data.get('file_path')
        description = data.get('description')
        base_model = data.get('base_model', "qwen1.5-72b-chat")
        training_type = data.get('training_type', "sft")
        hyper_parameters = data.get('hyper_parameters', {})
        
        # If file_path is provided, upload the file first
        if file_path:
            upload_result = cloud_service.upload_training_file(
                file_path=file_path,
                description=description
            )
            
            if not upload_result:
                return jsonify(APIResponse.error("Failed to upload training file"))
        
        # Create fine-tuning job
        result = cloud_service.create_fine_tune_job(
            base_model=base_model,
            training_type=training_type,
            hyper_parameters=hyper_parameters
        )
        
        if not result:
            return jsonify(APIResponse.error("Failed to create fine-tuning job"))
        
        return jsonify(APIResponse.success(data={
            "job_id": cloud_service.job_id,
            "file_id": cloud_service.file_id
        }))
    
    except Exception as e:
        logger.error(f"Fine-tuning job creation failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Fine-tuning job creation failed: {str(e)}"))


@cloud_bp.route("/fine-tune/status/<job_id>", methods=["GET"])
def check_fine_tune_status(job_id):
    """
    Check the status of a fine-tuning job
    
    Path parameter:
    - job_id: str, ID of the fine-tuning job
    """
    try:
        # Check if API key is configured
        if not api_key:
            return jsonify(APIResponse.error("Cloud service API key not configured"))
        
        # Set job_id in service
        cloud_service.job_id = job_id
        
        # Check status
        status = cloud_service.check_fine_tune_status()
        
        if status is None:
            return jsonify(APIResponse.error("Failed to check fine-tuning job status"))
        
        response_data = {
            "status": status
        }
        
        # If job completed successfully, include model_id
        if status == "SUCCEEDED" and cloud_service.model_id:
            response_data["model_id"] = cloud_service.model_id
        
        return jsonify(APIResponse.success(data=response_data))
    
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Status check failed: {str(e)}"))


@cloud_bp.route("/fine-tune/logs/<job_id>", methods=["GET"])
def get_fine_tune_logs(job_id):
    """
    Get logs for a fine-tuning job
    
    Path parameter:
    - job_id: str, ID of the fine-tuning job
    
    Query parameters:
    - offset: int, offset for logs pagination
    - line: int, number of lines to return
    """
    try:
        # Check if API key is configured
        if not api_key:
            return jsonify(APIResponse.error("Cloud service API key not configured"))
        
        # Get query parameters
        offset = request.args.get('offset', default=0, type=int)
        line = request.args.get('line', default=1000, type=int)
        
        # Set job_id in service
        cloud_service.job_id = job_id
        
        # Get logs
        logs = cloud_service.get_fine_tune_logs(offset=offset, line=line)
        
        if logs is None:
            return jsonify(APIResponse.error("Failed to get fine-tuning logs"))
        
        return jsonify(APIResponse.success(data={"logs": logs}))
    
    except Exception as e:
        logger.error(f"Log retrieval failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Log retrieval failed: {str(e)}"))


@cloud_bp.route("/deploy", methods=["POST"])
def deploy_model():
    """
    Deploy a fine-tuned model
    
    Request: DeployModelRequest JSON object containing:
    - capacity: int, capacity for the deployment, default is 2
    """
    try:
        # Check if API key is configured
        if not api_key:
            return jsonify(APIResponse.error("Cloud service API key not configured"))
        
        # Get request data
        data = request.json
        capacity = data.get('capacity', 2)
        
        # Deploy model
        result = cloud_service.deploy_model(capacity=capacity)
        
        if not result:
            return jsonify(APIResponse.error("Failed to deploy model"))
        
        return jsonify(APIResponse.success(data={
            "deployment_id": cloud_service.deployment_id,
            "model_id": cloud_service.model_id
        }))
    
    except Exception as e:
        logger.error(f"Model deployment failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Model deployment failed: {str(e)}"))


@cloud_bp.route("/deploy/status/<deployment_id>", methods=["GET"])
def check_deployment_status(deployment_id):
    """
    Check the status of a model deployment
    
    Path parameter:
    - deployment_id: str, ID of the deployment
    """
    try:
        # Check if API key is configured
        if not api_key:
            return jsonify(APIResponse.error("Cloud service API key not configured"))
        
        # Set deployment_id in service
        cloud_service.deployment_id = deployment_id
        
        # Check status
        status = cloud_service.check_deployment_status()
        
        if status is None:
            return jsonify(APIResponse.error("Failed to check deployment status"))
        
        return jsonify(APIResponse.success(data={"status": status}))
    
    except Exception as e:
        logger.error(f"Deployment status check failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Deployment status check failed: {str(e)}"))


@cloud_bp.route("/inference", methods=["POST"])
def run_inference():
    """
    Run inference with a deployed model
    
    Request: InferenceRequest JSON object containing:
    - user_input: str, input text for the model
    """
    try:
        # Check if API key is configured
        if not api_key:
            return jsonify(APIResponse.error("Cloud service API key not configured"))
        
        # Get request data
        data = request.json
        user_input = data.get('user_input')
        
        if not user_input:
            return jsonify(APIResponse.error("user_input is required"))
            
        # Run inference
        result = cloud_service.run_inference(user_input=user_input)
        
        if result is None:
            return jsonify(APIResponse.error("Inference failed"))
        
        return jsonify(APIResponse.success(data={"output": result}))
    
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Inference failed: {str(e)}"))


@cloud_bp.route("/models", methods=["GET"])
def list_models():
    """
    List available models that support fine-tuning
    """
    try:
        # Check if API key is configured
        if not api_key:
            return jsonify(APIResponse.error("Cloud service API key not configured"))
        
        # List models
        models = cloud_service.list_available_models()
        
        return jsonify(APIResponse.success(data={"models": models}))
    
    except Exception as e:
        logger.error(f"Listing models failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Listing models failed: {str(e)}"))
