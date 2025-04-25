"""
Training parameters management module.
This module provides functions for managing and accessing training parameters.
"""

import logging
import json
import os

# Configure logger
logger = logging.getLogger(__name__)


class TrainingParamsManager:
    """
    Training parameters manager class.
    """
    
    # Default training parameters
    _default_training_params = {
        "model_name": "Qwen2.5-0.5B-Instruct",
        "learning_rate": 1e-4,
        "number_of_epochs": 3,
        "concurrency_threads": 2,
        "data_synthesis_mode": "low",
        "use_cuda": False,  # Default to using CUDA when available
        "is_cot": False
    }
    
    # Parameters file path
    _params_file_path = None
    
    @classmethod
    def _get_params_file_path(cls):
        """
        Get the training parameters file path
        """
        if cls._params_file_path is None:
            # Set the parameters file path
            progress_dir = os.path.join(os.getcwd(), "data", "progress")
            if not os.path.exists(progress_dir):
                os.makedirs(progress_dir)
            cls._params_file_path = os.path.join(progress_dir, "training_params.json")
        
        return cls._params_file_path
    
    @classmethod
    def update_training_params(cls, params, preserve_previous=True):
        """
        Update the latest training parameters and save to file
        
        Args:
            params: Training parameters, either a TrainingParams object or a dictionary
            preserve_previous: If True, preserve previous parameters and update with new ones.
                              If False, use only the new parameters, discarding previous ones.
        """
        # Convert params to dict if it's a TrainingParams object
        if hasattr(params, 'to_dict') and callable(getattr(params, 'to_dict')):
            params_dict = params.to_dict()
        else:
            params_dict = params
        
        # Get current parameters based on preserve_previous flag
        if preserve_previous:
            # First try to load existing parameters
            current_params = cls.get_latest_training_params()
            
            # Update parameters
            for key, value in params_dict.items():
                if key in cls._default_training_params:
                    current_params[key] = value
                    logger.debug(f"Updated training parameter {key} to {value}")
                else:
                    logger.warning(f"Ignoring unknown parameter: {key}")
        else:
            # Use only the new parameters, with defaults for missing ones
            current_params = cls._default_training_params.copy()
            for key, value in params_dict.items():
                if key in cls._default_training_params:
                    current_params[key] = value
                    logger.debug(f"Set training parameter {key} to {value}")
                else:
                    logger.warning(f"Ignoring unknown parameter: {key}")
        
        # Save to file
        params_file = cls._get_params_file_path()
        try:
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(current_params, f, indent=2)
            logger.info(f"Training parameters saved to {params_file}")
        except Exception as e:
            logger.error(f"Failed to save training parameters to file: {str(e)}", exc_info=True)
    
    @classmethod
    def get_latest_training_params(cls):
        """
        Get the latest training parameters from file
        
        Returns:
            dict: Dictionary containing the latest training parameters
        """
        params_file = cls._get_params_file_path()
        
        # If file exists, read from file
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                
                # Replace null values with default values
                default_params = cls._default_training_params.copy()
                for key, value in default_params.items():
                    if key not in params or params[key] is None:
                        params[key] = value
                
                logger.debug(f"Loaded training parameters from {params_file}")
                return params
            except Exception as e:
                logger.error(f"Failed to load training parameters from file: {str(e)}", exc_info=True)
                # If reading fails, return default parameters
                return cls._default_training_params.copy()
        else:
            # If file does not exist, return default parameters
            return cls._default_training_params.copy()