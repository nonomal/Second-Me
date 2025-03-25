from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.config import Config
from typing import List, Union
import requests
import numpy as np
from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()


class LLMClient:
    """LLM client utility class"""

    def __init__(self):
        self.config = Config.from_env()
        self.user_llm_config_service = UserLLMConfigService()
        # self.user_llm_config = self.user_llm_config_service.get_available_llm()

        # self.chat_api_key = self.user_llm_config.chat_api_key
        # self.chat_base_url = self.user_llm_config.chat_endpoint
        # self.chat_model = self.user_llm_config.chat_model_name
        # self.embedding_api_key = self.user_llm_config.embedding_api_key
        # self.embedding_base_url = self.user_llm_config.embedding_endpoint
        # self.embedding_model = self.user_llm_config.embedding_model_name


    def get_embedding(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Calculate text embedding

        Args:
            texts (str or list): Input text or list of texts

        Returns:
            numpy.ndarray: Embedding vector of the text
        """
        # Ensure texts is in list format
        if isinstance(texts, str):
            texts = [texts]

        user_llm_config = self.user_llm_config_service.get_available_llm()
        if not user_llm_config:
            raise Exception("No LLM configuration found")
        # Prepare request data
        headers = {
            "Authorization": f"Bearer {user_llm_config.embedding_api_key}",
            "Content-Type": "application/json",
        }

        data = {"input": texts, "model": user_llm_config.embedding_model_name}

        logger.info(f"Getting embedding for {data}")
        try:
            # Send request to embedding endpoint
            response = requests.post(
                f"{user_llm_config.embedding_endpoint}/embeddings", headers=headers, json=data
            )

            # Check response status
            response.raise_for_status()
            result = response.json()

            # Extract embedding vectors
            embeddings = [item["embedding"] for item in result["data"]]
            return np.array(embeddings)

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get embeddings: {str(e)}")

    @property
    def chat_credentials(self):
        """Get LLM authentication information"""
        return {"api_key": self.chat_api_key, "base_url": self.chat_base_url}
