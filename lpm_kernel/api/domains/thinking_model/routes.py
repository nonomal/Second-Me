from flask import Blueprint, jsonify, request
from http import HTTPStatus
from typing import Dict, Any

from lpm_kernel.api.dto.thinking_model_dto import ThinkingModelDTO, UpdateThinkingModelDTO
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.common.logging import logger

thinking_model_bp = Blueprint("thinking_model", __name__, url_prefix="/api/thinking-models")
user_llm_config_service = UserLLMConfigService()

def validate_thinking_model(data: Dict[Any, Any]) -> Dict[str, str]:
    """Validate thinking model configuration
    
    Args:
        data: Configuration data
        
    Returns:
        Dictionary with error messages if validation fails, empty dict if validation passes
    """
    errors = {}
    
    # Validate required fields
    if not data.get('thinking_model_name'):
        errors['thinking_model_name'] = 'Thinking model name is required'
    
    if not data.get('thinking_endpoint'):
        errors['thinking_endpoint'] = 'Thinking endpoint is required'
    
    return errors


@thinking_model_bp.route("", methods=["GET"])
def get_all_thinking_models():
    """Get all thinking models"""
    try:
        models = user_llm_config_service.get_all_thinking_models()
        return jsonify(
            APIResponse.success(
                data=[model.dict() for model in models],
                message="Successfully retrieved thinking models"
            )
        ), HTTPStatus.OK
    except Exception as e:
        logger.error(f"Failed to retrieve thinking models: {str(e)}", exc_info=True)
        return jsonify(
            APIResponse.error(f"Failed to retrieve thinking models: {str(e)}")
        ), HTTPStatus.INTERNAL_SERVER_ERROR


@thinking_model_bp.route("/<int:model_id>", methods=["GET"])
def get_thinking_model(model_id: int):
    """Get thinking model by ID"""
    try:
        model = user_llm_config_service.get_thinking_model(model_id)
        if not model:
            return jsonify(
                APIResponse.error("Thinking model not found")
            ), HTTPStatus.NOT_FOUND
        
        return jsonify(
            APIResponse.success(
                data=model.dict(),
                message="Successfully retrieved thinking model"
            )
        ), HTTPStatus.OK
    except Exception as e:
        logger.error(f"Failed to retrieve thinking model: {str(e)}", exc_info=True)
        return jsonify(
            APIResponse.error(f"Failed to retrieve thinking model: {str(e)}")
        ), HTTPStatus.INTERNAL_SERVER_ERROR


@thinking_model_bp.route("", methods=["POST"])
def create_thinking_model():
    """Create a new thinking model"""
    try:
        # Validate request data
        request_data = request.json
        validation_errors = validate_thinking_model(request_data)
        
        if validation_errors:
            error_message = "; ".join([f"{k}: {v}" for k, v in validation_errors.items()])
            return jsonify(
                APIResponse.error(f"Validation failed: {error_message}")
            ), HTTPStatus.BAD_REQUEST
        
        # Create thinking model
        data = ThinkingModelDTO(**request_data)
        model = user_llm_config_service.create_thinking_model(data)
        
        return jsonify(
            APIResponse.success(
                data=model.dict(),
                message="Thinking model created successfully"
            )
        ), HTTPStatus.CREATED
    
    except Exception as e:
        logger.error(f"Failed to create thinking model: {str(e)}", exc_info=True)
        return jsonify(
            APIResponse.error(f"Failed to create thinking model: {str(e)}")
        ), HTTPStatus.INTERNAL_SERVER_ERROR


@thinking_model_bp.route("/<int:model_id>", methods=["PUT"])
def update_thinking_model(model_id: int):
    """Update thinking model"""
    try:
        # Validate request data
        request_data = request.json
        validation_errors = validate_thinking_model(request_data)
        
        if validation_errors:
            error_message = "; ".join([f"{k}: {v}" for k, v in validation_errors.items()])
            return jsonify(
                APIResponse.error(f"Validation failed: {error_message}")
            ), HTTPStatus.BAD_REQUEST
        
        # Update thinking model
        data = UpdateThinkingModelDTO(**request_data)
        model = user_llm_config_service.update_thinking_model(model_id, data)
        
        return jsonify(
            APIResponse.success(
                data=model.dict(),
                message="Thinking model updated successfully"
            )
        ), HTTPStatus.OK
    
    except Exception as e:
        logger.error(f"Failed to update thinking model: {str(e)}", exc_info=True)
        return jsonify(
            APIResponse.error(f"Failed to update thinking model: {str(e)}")
        ), HTTPStatus.INTERNAL_SERVER_ERROR


@thinking_model_bp.route("/<int:model_id>", methods=["DELETE"])
def delete_thinking_model(model_id: int):
    """Delete thinking model"""
    try:
        model = user_llm_config_service.delete_thinking_model(model_id)
        if not model:
            return jsonify(
                APIResponse.error("Thinking model not found")
            ), HTTPStatus.NOT_FOUND
        
        return jsonify(
            APIResponse.success(
                data=model.dict(),
                message="Thinking model deleted successfully"
            )
        ), HTTPStatus.OK
    
    except Exception as e:
        logger.error(f"Failed to delete thinking model: {str(e)}", exc_info=True)
        return jsonify(
            APIResponse.error(f"Failed to delete thinking model: {str(e)}")
        ), HTTPStatus.INTERNAL_SERVER_ERROR
