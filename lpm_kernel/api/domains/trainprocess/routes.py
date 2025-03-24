import json
import logging
import time
from pathlib import Path

from flask import Blueprint, jsonify, Response, request

from lpm_kernel.file_data.trainprocess_service import TrainProcessService
from .progress import Status
from ...common.responses import APIResponse
from threading import Thread

trainprocess_bp = Blueprint("trainprocess", __name__, url_prefix="/api/trainprocess")


def progress_callback(progress_update):
    """Progress callback function to update training progress"""
    if not isinstance(progress_update, dict):
        return

    try:
        # Get current progress status
        train_service = TrainProcessService()
        progress = train_service.progress.progress

        # Update progress
        stage = progress_update.get("stage")
        step = progress_update.get("step")
        status = Status[progress_update.get("status", "IN_PROGRESS").upper()]
        prog = progress_update.get("progress")

        if stage and step:
            progress.update_progress(stage, step, status, prog)
    except Exception as e:
        logging.error(f"Progress callback error: {str(e)}")


def clear_specific_logs():
    """Clear specific log files (backend.log and train.log)"""
    log_dir = Path(__file__).parent.parent.parent.parent.parent / 'logs'
    specific_logs = ['backend.log', 'train.log']
    
    for log_name in specific_logs:
        log_file = log_dir / log_name
        if log_file.exists():
            try:
                with open(log_file, 'w') as f:
                    f.truncate(0)
            except Exception as e:
                logging.error(f"Failed to clear log file {log_file}: {e}")


@trainprocess_bp.route("/start", methods=["POST"])
def start_process():
    """
    Start training process, returns progress stream ID
    
    Request parameters:
        model_name: Model name
    
    Includes the following steps:
    1. Health check
    2. Generate L0
    3. Generate document embeddings
    4. Process document chunks
    5. Generate chunk embeddings
    6. Analyze documents
    7. Generate L1
    8. Download model
    9. Prepare data
    10. Train model
    11. Merge weights
    12. Convert model

    Returns:
        Response: JSON response
        {
            "code": 0 for success, non-zero for failure,
            "message": "Error message",
            "data": {
                "progress_id": "Progress stream ID",
                "model_name": "Model name"
            }
        }
    """
    logging.info("Training process starting...")  # Log the startup
    try:
        data = request.get_json()
        if not data or "model_name" not in data:
            return jsonify(APIResponse.error(message="Missing required parameters"))

        model_name = data["model_name"]

        # Create service instance, pass in progress callback and model name
        train_service = TrainProcessService(
            progress_callback=progress_callback,
            model_name=model_name
        )
        if not train_service.check_training_condition():
            # Reset progress
            clear_specific_logs()
            train_service.set_retrian_progress()

        thread = Thread(target=train_service.start_process)
        thread.daemon = True
        thread.start()

        return jsonify(
            APIResponse.success(
                data={
                    "model_name": model_name
                }
            )
        )
    
    except Exception as e:
        logging.error(f"Training process failed: {str(e)}")
        return jsonify(APIResponse.error(message=f"Training process error: {str(e)}"))


@trainprocess_bp.route("/logs", methods=["GET"])
def stream_logs():
    """Get training logs in real-time"""
    log_file_path = "logs/backend.log"  # Log file path
    last_position = 0
    
    # Define keywords to exclude
    exclude_keywords = [
        "GET /api"
    ]

    def generate_logs():
        nonlocal last_position
        while True:
            try:
                with open(log_file_path, 'r') as log_file:
                    log_file.seek(last_position)
                    new_lines = log_file.readlines()  # Read new lines

                    for line in new_lines:
                        # Skip empty lines
                        if not line.strip():
                            continue
                            
                        # Check if the line contains any of the exclude keywords
                        if any(exclude_word in line for exclude_word in exclude_keywords):
                            continue
                            
                        yield f"data: {line.strip()}\n\n"
                            
                    last_position = log_file.tell()
            except Exception as e:
                # If file reading fails, record error and continue
                yield f"data: Error reading log file: {str(e)}\n\n"
                
            time.sleep(1)  # Check for new logs every second

    return Response(
        generate_logs(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache, no-transform',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
            'Transfer-Encoding': 'chunked'
        }
    )


@trainprocess_bp.route("/progress/<model_name>", methods=["GET"])
def get_progress(model_name):
    """Get current progress (non-real-time)"""
    progress_name = f'trainprocess_progress_{model_name}.json'  # Build filename based on the provided model_name
    try:
        train_service = TrainProcessService(progress_file=progress_name, model_name=model_name)  # Pass in specific progress file
        progress = train_service.progress.progress

        return jsonify(
            APIResponse.success(
                data=progress.to_dict()  # Return progress data
            )
        )
    except Exception as e:
        return jsonify(APIResponse.error(message=str(e)))

@trainprocess_bp.route("/progress/reset", methods=["POST"])
def reset_progress():
    """
    Reset progress

    Returns:
        Response: JSON response
        {
            "code": 0 for success, non-zero for failure,
            "message": "Error message",
            "data": null
        }
    """
    try:
        train_service = TrainProcessService()
        train_service.progress.reset_progress()

        return jsonify(APIResponse.success(message="Progress reset successfully"))
    except Exception as e:
        logging.error(f"Reset progress failed: {str(e)}")
        return jsonify(APIResponse.error(message=f"Failed to reset progress: {str(e)}"))


@trainprocess_bp.route("/stop", methods=["POST"])
def stop_training():
    """Stop training process and wait until status is failed"""
    try:
        # Get the TrainProcessService instance
        train_service = TrainProcessService()  # Need to get instance based on your implementation
        
        # Stop the process
        train_service.stop_process()
        
        # Wait for the status to change to FAILED
        wait_interval = 5  # Check interval in seconds
        
        while True:
            # Get the current progress
            progress = train_service.progress.progress
            
            # Check if status is FAILED
            if progress.status == Status.FAILED:
                return jsonify(APIResponse.success(
                    message="Training process has been stopped and status is confirmed as failed",
                    data={"status": "failed"}
                ))
            
            # Wait before checking again
            time.sleep(wait_interval)

    except Exception as e:
        logging.error(f"Error stopping training process: {str(e)}")
        return jsonify(APIResponse.error(message=f"Error stopping training process: {str(e)}"))


@trainprocess_bp.route("/model_name", methods=["GET"])
def get_model_name():
    """
    Get the model name currently used by the training service
    
    Returns:
        Response: JSON response
        {
            "code": 0 for success, non-zero for failure,
            "message": "Error message",
            "data": {
                "model_name": "Model name"
            }
        }
    """
    try:
        # Get TrainProcessService instance
        train_service = TrainProcessService()
        model_name = train_service.model_name
        
        return jsonify(APIResponse.success(data={"model_name": model_name}))
    except Exception as e:
        logging.error(f"Failed to get model name: {str(e)}")
        return jsonify(APIResponse.error(message=f"Failed to get model name: {str(e)}"))


@trainprocess_bp.route("/retrain", methods=["POST"])
def retrain():
    """
    Reset progress to data processing stage (data_processing not started) and automatically start the training process
    
    Request parameters:
        model_name: Model name (required)
    
    Returns:
        Response: JSON response
        {
            "code": 0 for success, non-zero for failure,
            "message": "Error message",
            "data": {
                "progress_id": "Progress stream ID",
                "model_name": "Model name"
            }
        }
    """
    try:
        # Clear log files first
        clear_specific_logs()
        
        # get request parameters
        data = request.get_json() or {}
        model_name = data.get("model_name")
        
        if not model_name:
            return jsonify(APIResponse.error(message="missing necessary parameter: model_name", code=400))
        
        # Create training service instance
        train_service = TrainProcessService(
            progress_callback=progress_callback,
            model_name=model_name
        )
        train_service.set_retrian_progress()

        thread = Thread(target=train_service.start_process)
        thread.daemon = True
        thread.start()
        
        return jsonify(
            APIResponse.success(
                message="Successfully reset progress to data processing stage and started training process",
                data={
                    "model_name": model_name
                }
            )
        )
    except Exception as e:
        logging.error(f"Retrain reset failed: {str(e)}")
        return jsonify(APIResponse.error(message=f"Failed to reset progress to data processing stage: {str(e)}"))
