"""
Training parameters management module.
This module provides functions for managing and accessing training parameters.
"""

import logging

# Configure logger
logger = logging.getLogger(__name__)


class TrainingParamsManager:
    """
    Training parameters manager class.
    """
    
    # Default training parameters
    _latest_training_params = {
        "model_name": "Qwen2.5-0.5B-Instruct",
        "learning_rate": 1e-4,
        "number_of_epochs": 3,
        "concurrency_threads": 2,
        "data_synthesis_mode": "low"
    }
    
    @classmethod
    def update_training_params(cls, params):
        """
        Update the latest training parameters
        
        Args:
            params: Dictionary containing training parameters
        """
        for key, value in params.items():
            if key in cls._latest_training_params:
                cls._latest_training_params[key] = value
                logger.debug(f"Updated training parameter {key} to {value}")
            else:
                logger.warning(f"Ignoring unknown parameter: {key}")
    
    @classmethod
    def get_latest_training_params(cls):
        """
        Get the latest training parameters
        
        Returns:
            dict: Dictionary containing the latest training parameters
        """
        return cls._latest_training_params.copy()