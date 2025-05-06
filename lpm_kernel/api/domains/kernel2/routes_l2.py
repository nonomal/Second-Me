import logging
import os
import time
import sys
import torch  # Add torch import for CUDA detection
import traceback
from dataclasses import asdict

from flask import Blueprint, jsonify, request
from flask_pydantic import validate

from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.api.domains.kernel2.dto.chat_dto import (
    ChatRequest,
)
from lpm_kernel.api.domains.kernel2.services.chat_service import chat_service
from lpm_kernel.api.domains.kernel2.services.prompt_builder import (
    BasePromptStrategy,
    RoleBasedStrategy,
    KnowledgeEnhancedStrategy,
)
from lpm_kernel.api.domains.loads.services import LoadService
from lpm_kernel.api.services.local_llm_service import local_llm_service

from ...common.script_executor import ScriptExecutor
from ....configs.config import Config

logger = logging.getLogger(__name__)

kernel2_bp = Blueprint("kernel2", __name__, url_prefix="/api/kernel2")

# Create script executor instance
script_executor = ScriptExecutor()


@kernel2_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    config = Config.from_env()
    app_name = config.app_name or "Service"  # Add default value to prevent None

    status = local_llm_service.get_server_status()
    if status.is_running and status.process_info:
        return jsonify(
            APIResponse.success(
                data={
                    "status": "running",
                    "pid": status.process_info.pid,
                    "cpu_percent": status.process_info.cpu_percent,
                    "memory_percent": status.process_info.memory_percent,
                    "uptime": time.time() - status.process_info.create_time,
                }
            )
        )
    else:
        return jsonify(APIResponse.success(data={"status": "stopped"}))


@kernel2_bp.route("/username", methods=["GET"])
def username():
    return jsonify(APIResponse.success(data={"username": LoadService.get_current_upload_name()}))

# read IN_DOCKER_ENV and output
@kernel2_bp.route("/docker/env", methods=["GET"])
def docker_env():
    return jsonify(APIResponse.success(data={"in_docker_env": os.getenv("IN_DOCKER_ENV")}))

@kernel2_bp.route("/llama/start", methods=["POST"])
def start_llama_server():
    """Start llama-server service"""
    try:
        # Get request parameters
        data = request.get_json()
        if not data or "model_name" not in data:
            return jsonify(APIResponse.error(message="Missing required parameter: model_name", code=400))

        model_name = data["model_name"]
        # Get optional use_gpu parameter with default value of True
        use_gpu = data.get("use_gpu", True)
        base_dir = os.getcwd()
        model_dir = os.path.join(base_dir, "resources/model/output/gguf", model_name)
        gguf_path = os.path.join(model_dir, "model.gguf")

        server_path = os.path.join(os.getcwd(), "llama.cpp/build/bin")
        if os.path.exists(os.path.join(os.getcwd(), "llama.cpp/build/bin/Release")):
            server_path = os.path.join(os.getcwd(), "llama.cpp/build/bin/Release")
            
        # Determine the executable name based on platform (.exe for Windows)
        if sys.platform.startswith("win"):
            server_executable = "llama-server.exe"
        else:
            server_executable = "llama-server"
        server_path = os.path.join(server_path, server_executable)

        # Check if model file exists
        if not os.path.exists(gguf_path):
            return jsonify(APIResponse.error(
                message=f"Model '{model_name}' GGUF file does not exist, please convert model first",
                code=400
            ))

        # Start the server using the LocalLLMService with GPU acceleration if requested
        success = local_llm_service.start_server(gguf_path, use_gpu=use_gpu)
        
        if not success:
            return jsonify(APIResponse.error(message="Failed to start llama-server", code=500))
            
        # Get updated service status
        status = local_llm_service.get_server_status()
        
        # Return success response with GPU info
        gpu_info = "with GPU acceleration" if use_gpu and torch.cuda.is_available() else "with CPU only"
        return jsonify(
            APIResponse.success(
                data={
                    "model_name": model_name,
                    "gguf_path": gguf_path,
                    "status": "running" if status.is_running else "starting",
                    "use_gpu": use_gpu and torch.cuda.is_available(),
                    "gpu_info": gpu_info
                },
                message=f"llama-server service started {gpu_info}"
            )
        )

    except Exception as e:
        error_msg = f"Failed to start llama-server: {str(e)}"
        logger.error(error_msg)
        return jsonify(APIResponse.error(message=error_msg, code=500))


# Flag to track if service is stopping
_stopping_server = False

@kernel2_bp.route("/llama/stop", methods=["POST"])
def stop_llama_server():
    """Stop llama-server service - Force immediate termination of the process"""
    global _stopping_server

    try:
        # If service is already stopping, return notification
        if _stopping_server:
            return jsonify(APIResponse.success(message="llama-server service is stopping"))

        _stopping_server = True  # Set stopping flag

        try:
            # use improved local_llm_service.stop_server() to stop all llama-server process
            status = local_llm_service.stop_server()

            # check if there are still processes running
            if status.is_running and status.process_info:
                pid = status.process_info.pid
                logger.warning(f"llama-server process still running: {pid}")
                return jsonify(APIResponse.success(
                    message="llama-server service could not be fully stopped. Please try again.",
                    data={"running_pid": pid}
                ))
            else:
                return jsonify(APIResponse.success(message="llama-server service has been stopped successfully"))

        except Exception as e:
            logger.error(f"Error while stopping llama-server: {str(e)}")
            return jsonify(APIResponse.error(message=f"Error stopping llama-server: {str(e)}", code=500))
        finally:
            _stopping_server = False

    except Exception as e:
        _stopping_server = False
        logger.error(f"Failed to stop llama-server: {str(e)}")
        return jsonify(APIResponse.error(message=f"Failed to stop llama-server: {str(e)}", code=500))


@kernel2_bp.route("/llama/status", methods=["GET"])
@validate()
def get_llama_server_status():
    """Get llama-server service status"""
    try:
        status = local_llm_service.get_server_status()
        return APIResponse.success(asdict(status))

    except Exception as e:
        logger.error(f"Error getting llama-server status: {str(e)}", exc_info=True)
        return APIResponse.error(f"Error getting llama-server status: {str(e)}")

@kernel2_bp.route("/chat", methods=["POST"])
@validate()
def chat(body: ChatRequest):
    """
    Chat interface - Stream response (OpenAI API compatible)

    Request parameters: Compatible with OpenAI Chat Completions API format
    - messages: List[Dict[str, str]], standard OpenAI message list with format:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I am a helpful assistant."},
            {"role": "user", "content": "What can you do for me?"}  
        ]
    - metadata: Dict[str, Any], additional parameters for request processing (optional):
        {
            "enable_l0_retrieval": true,  // whether to enable knowledge retrieval
            "enable_l1_retrieval": false, // whether to enable advanced knowledge retrieval
            "role_id": "uuid-string"      // optional role UUID for system customization
        }
    - stream: bool, whether to stream the response (default: True)
    - model: str, model identifier (optional, default uses configured model)
    - temperature: float, controls randomness (default: 0.1)
    - max_tokens: int, maximum tokens to generate (default: 2000)

    Response: Standard OpenAI Chat Completions API format
    For stream=true (Server-Sent Events):
    - id: str, response unique identifier
    - object: "chat.completion.chunk"
    - created: int, timestamp
    - model: str, model identifier
    - system_fingerprint: str, system fingerprint
    - choices: [
        {
          "index": 0,
          "delta": {"content": str},
          "finish_reason": null or "stop"
        }
      ]
    
    The last event will be: data: [DONE]
    
    For stream=false:
    - Complete response object with full message content
    """
    try:
        logger.info(f"Starting chat request: {body}")
        # 1. Check service status
        status = local_llm_service.get_server_status()
        if not status.is_running:
            # Format error response in OpenAI-compatible format
            error_msg = "LLama server is not running"
            logger.error(error_msg)
            error_response = {
                "error": {
                    "message": error_msg,
                    "type": "server_error",
                    "code": "service_unavailable"
                }
            }
            # Return as regular JSON response for non-stream or stream-compatible error
            if not body.stream:
                return APIResponse.error(message="Service temporarily unavailable", code=503), 503
            return local_llm_service.handle_stream_response(iter([error_response]))

        try:
            # Use chat_service to process request with OpenAI-compatible format
            response = chat_service.chat(
                request=body,
                stream=body.stream,  # Respect the stream parameter from request
                json_response=False,
                strategy_chain=[BasePromptStrategy, RoleBasedStrategy, KnowledgeEnhancedStrategy]
            )
            
            # Handle streaming or non-streaming response appropriately
            if body.stream:
                return local_llm_service.handle_stream_response(response)
            else:
                # For non-streaming, return the complete response as JSON
                return jsonify(response)

        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Value error: {error_msg}")
            error_response = {
                "error": {
                    "message": error_msg,
                    "type": "invalid_request_error",
                    "code": "bad_request"
                }
            }
            if not body.stream:
                return jsonify(error_response), 400
            return local_llm_service.handle_stream_response(iter([error_response]))

    except Exception as e:
        error_msg = f"Request processing failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        error_response = {
            "error": {
                "message": error_msg,
                "type": "server_error",
                "code": "internal_server_error"
            }
        }
        if not getattr(body, 'stream', True):  # Default to stream if attribute missing
            return jsonify(error_response), 500
        return local_llm_service.handle_stream_response(iter([error_response]))


@kernel2_bp.route("/cuda/available", methods=["GET"])
def check_cuda_available():
    """Check if CUDA is available for model training/inference"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_info = {}
        
        if cuda_available:
            cuda_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0)
            }
        
        return jsonify(APIResponse.success(
            data={
                "cuda_available": cuda_available,
                "cuda_info": cuda_info
            },
            message="CUDA availability check completed"
        ))
    except Exception as e:
        error_msg = f"Error checking CUDA availability: {str(e)}"
        logger.error(error_msg)
        return jsonify(APIResponse.error(message=error_msg, code=500))
