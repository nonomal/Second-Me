import os
import logging
import requests
import time
from typing import Dict, Any, Iterator, Optional
from pydantic import BaseModel
from pathlib import Path
from ....configs.config import Config

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()


class CloudServiceRequest(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = None
    model: Optional[str] = "default"
    stream: Optional[bool] = False


class CloudService:
    def __init__(self, api_key=None):
        self.api_key = api_key

        self.base_url = os.environ.get('AL_Base_URL')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        self.file_id = None
        self.job_id = None
        self.model_id = None
        self.deployment_id = None

    def upload_training_file(self, file_path, description=None):
        """上传训练数据文件"""

        url = f"{self.base_url}/files"

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"The training data file does not exist: {file_path}")

        files = {
            'files': (file_path.name, open(file_path, 'rb'))
        }

        data = {}
        if description:
            data['descriptions'] = description

        response = requests.post(url, headers=self.headers, files=files, data=data)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"Upload failed! Error: {response_data}")
            return False

        self.file_id = response_data.get('data', {}).get('uploaded_files', [])[0].get('file_id')
        logger.info(f"Upload successful! File ID: {self.file_id}")
        return True

    def create_fine_tune_job(self, base_model="qwen1.5-72b-chat", training_type="efficient_sft", hyper_parameters=None):
        """创建模型调优任务"""
        if not self.file_id:
            raise ValueError("Please upload the training data file first!")

        logger.info(f"Creating {training_type} fine-tuning job, base model: {base_model}...")

        url = f"{self.base_url}/fine-tunes"
        headers = {**self.headers, "Content-Type": "application/json"}

        payload = {
            "model": base_model,
            "training_file_ids": [self.file_id],
            "hyper_parameters": hyper_parameters or {},
            "training_type": training_type
        }

        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()

        if response.status_code != 200:
            error_msg = response_data.get('message', 'Unknown error')
            error_code = response_data.get('code', 'Unknown error code')
            logger.error(f"Failed to create fine-tuning job! Error code: {error_code}, Error message: {error_msg}")

            if "unsupported model" in error_msg:
                logger.warning("\nModel does not support fine-tuning, please use --list-models parameter to view the list of models that support fine-tuning")

            return False

        self.job_id = response_data.get('data', {}).get('job_id')
        logger.info(f"Fine-tuning job created successfully! Job ID: {self.job_id}")
        return True

    def check_fine_tune_status(self):
        """检查模型调优任务状态"""
        if not self.job_id:
            raise ValueError("Please create a tuning task first!")

        url = f"{self.base_url}/fine-tunes/{self.job_id}"
        headers = {**self.headers, "Content-Type": "application/json"}

        response = requests.get(url, headers=headers)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"Failed to query fine-tuning job status! Error: {response_data}")
            return None

        status = response_data.get('data', {}).get('status')
        logger.info(f"Fine-tuning job status: {status}")

        # 如果调优成功，保存模型ID
        if status == "SUCCEEDED":
            self.model_id = response_data.get('data', {}).get('finetuned_output')
            logger.info(f"Fine-tuning successful! Model ID: {self.model_id}")

        return status

    def get_fine_tune_logs(self, offset=0, line=1000):
        """获取模型调优过程日志"""
        if not self.job_id:
            raise ValueError("Please create a tuning task first!")

        url = f"{self.base_url}/fine-tunes/{self.job_id}/logs?offset={offset}&line={line}"
        headers = {**self.headers, "Content-Type": "application/json"}

        response = requests.get(url, headers=headers)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"Failed to get fine-tuning logs! Error: {response_data}")
            return None

        logs = response_data.get('data', {}).get('logs', "")
        logger.info("Fine-tuning logs:")
        logger.info(logs)

        return logs

    def deploy_model(self, capacity=2):
        """部署调优后的模型"""
        if not self.model_id:
            raise ValueError("Please complete the model tuning first")

        logger.info(f"Deploying model: {self.model_id}...")

        url = f"{self.base_url}/deployments"
        headers = {**self.headers, "Content-Type": "application/json"}

        payload = {
            "model_name": self.model_id,
            "capacity": capacity
        }

        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"Model deployment failed! Error: {response_data}")
            return False

        self.deployment_id = response_data.get('data', {}).get('deployment_id')
        logger.info(f"Model deployment created successfully! Deployment ID: {self.deployment_id}")
        return True

    def check_deployment_status(self):
        """检查模型部署状态"""
        if not self.deployment_id:
            raise ValueError("Please deploy the model first!")

        url = f"{self.base_url}/deployments/{self.deployment_id}"
        headers = {**self.headers, "Content-Type": "application/json"}

        response = requests.get(url, headers=headers)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"Failed to query deployment status! Error: {response_data}")
            return None

        status = response_data.get('data', {}).get('status')
        logger.info(f"Model deployment status: {status}")
        return status

    def run_inference(self, user_input):
        """使用调优后的模型进行推理"""
        if not self.deployment_id:
            raise ValueError("Please deploy the model first!")

        url = f"{self.base_url}/services/aigc/text-generation/generation"
        headers = {**self.headers, "Content-Type": "application/json"}

        payload = {
            "model": self.deployment_id,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            },
            "parameters": {
                "result_format": "message"
            }
        }

        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"Model inference failed! Error: {response_data}")
            return None

        result = response_data.get('data', {}).get('output')
        logger.info(f"Model output: {result}")
        return result

    def wait_for_job_completion(self, timeout=3600, check_interval=60):
        """等待调优任务完成"""
        logger.info(f"Waiting for fine-tuning job to complete, maximum wait time: {timeout} seconds...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.check_fine_tune_status()

            if status == "SUCCEEDED":
                logger.info("Fine-tuning job completed successfully!")
                return True
            elif status in ["FAILED", "CANCELLED"]:
                logger.error(f"Fine-tuning job failed, status: {status}")
                return False

            logger.info(f"Fine-tuning job in progress, checking again in {check_interval} seconds...")
            time.sleep(check_interval)

        logger.warning(f"Wait timeout! Waited {timeout} seconds, job still not completed.")
        return False

    def wait_for_deployment(self, timeout=900, check_interval=30):
        """等待模型部署完成"""
        logger.info(f"Waiting for model deployment to complete, maximum wait time: {timeout} seconds...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.check_deployment_status()

            if status == "RUNNING":
                logger.info("Model deployment successful, now ready to use!")
                return True
            elif status in ["FAILED", "CANCELLED"]:
                logger.error(f"Model deployment failed, status: {status}")
                return False

            logger.info(f"Model deployment in progress, checking again in {check_interval} seconds...")
            time.sleep(check_interval)

        logger.warning(f"Wait timeout! Waited {timeout} seconds, deployment still not completed.")
        return False

    def list_available_models(self):
        """列出可用于调优的模型"""
        logger.info("Getting list of models that support fine-tuning...")

        url = f"{self.base_url}/models"
        headers = {**self.headers, "Content-Type": "application/json"}

        response = requests.get(url, headers=headers)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"Failed to get model list! Error: {response_data}")
            return []

        models = []
        try:
            for model in response_data.get('data', {}).get('models', []):
                model_id = model.get('model', '')
                capabilities = model.get('capabilities', [])

                # 检查模型是否支持调优
                if 'fine-tuning' in capabilities:
                    models.append({
                        'id': model_id,
                        'capabilities': capabilities
                    })
                    logger.info(f"- {model_id} (supports fine-tuning)")
        except Exception as e:
            logger.error(f"Error parsing model list: {e}")

        if not models:
            logger.warning("No models found that support fine-tuning")

        return models

