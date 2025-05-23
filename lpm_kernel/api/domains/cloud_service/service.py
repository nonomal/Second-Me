import os
import requests
import time
from typing import Dict, Any, Iterator, Optional
from pathlib import Path
import json

from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()


class CloudService:
    def __init__(self, api_key=None):
        if api_key:
            self.api_key = api_key
        else:
            try:
                user_llm_config_service = UserLLMConfigService()
                config = user_llm_config_service.get_available_llm()
                if config and hasattr(config, 'cloud_service_api_key'):
                    self.api_key = config.cloud_service_api_key
            except Exception as e:
                logger.error(f"Failed to get API key from database: {str(e)}")    

        self.base_url = os.environ.get('AL_Base_URL')

        if self.api_key:
            self.headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
        else:
            self.headers = {}
            logger.warning("API key is not set. API calls may fail.")

        self.file_id = None
        self.job_id = None
        self.model_id = None
        self.deployment_id = None

    def upload_training_file(self, file_path = "resources/L2/data/merged.json", description=None):
        """上传训练数据文件"""
        url = f"{self.base_url}/files"

        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        file_path = project_root / file_path
        file_path = self.convert_to_jsonl(str(file_path))
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

            return None

        self.job_id = response_data.get('output', {}).get('job_id')
        logger.info(f"Fine-tuning job created successfully! Job ID: {self.job_id}")
        return self.job_id

    def check_fine_tune_status(self, job_id):
        """检查模型调优任务状态"""

        url = f"{self.base_url}/fine-tunes/{job_id}"
        headers = {**self.headers, "Content-Type": "application/json"}

        response = requests.get(url, headers=headers)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"Failed to query fine-tuning job status! Error: {response_data}")
            return None

        status = response_data.get('output', {}).get('status')
        logger.info(f"Fine-tuning job status: {status}")

        if status == "SUCCEEDED":
            self.model_id = response_data.get('output', {}).get('finetuned_output')
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

        logs = response_data.get('output', {}).get('logs', "")
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

        self.deployment_id = response_data.get('output', {}).get('deployment_id')
        logger.info(f"Model deployment created successfully! Deployment ID: {self.deployment_id}")
        return True

    def check_deployment_status(self, deployment_id):
        """检查模型部署状态"""

        url = f"{self.base_url}/deployments/{deployment_id}"
        headers = {**self.headers, "Content-Type": "application/json"}

        response = requests.get(url, headers=headers)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"Failed to query deployment status! Error: {response_data}")
            return None

        status = response_data.get('output', {}).get('status')
        logger.info(f"Model deployment status: {status}")
        return status

    def delete_deployment(self, model_id):
        url = f"{self.base_url}/deployments/{model_id}"
        headers = {**self.headers, "Content-Type": "application/json"}

        response = requests.delete(url, headers=headers)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"Failed to delete deployment model! Error: {response_data}")
            return None

        status = response_data.get('output', {}).get('status')
        logger.info(f"Model delete deployment status: {status}")
        return status
        
    def delete_fine_tune_job(self, job_id):
        """删除微调任务
        
        Args:
            job_id: 微调任务ID
            
        Returns:
            bool: 是否成功删除
        """
        url = f"{self.base_url}/fine-tunes/{job_id}"
        headers = {**self.headers, "Content-Type": "application/json"}
        
        try:
            response = requests.delete(url, headers=headers)
            response_data = response.json()
            
            if response.status_code != 200:
                logger.error(f"Failed to delete fine-tune job! Error: {response_data}")
                return False
                
            logger.info(f"Fine-tune job deleted successfully! Job ID: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Exception during fine-tune job deletion: {str(e)}")
            return False

    def run_inference(self, user_input, deployment_id):
        """使用调优后的模型进行推理"""

        url = f"{self.base_url}/services/aigc/text-generation/generation"
        headers = {**self.headers, "Content-Type": "application/json"}

        payload = {
            "model": deployment_id,
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

        result = response_data.get('text')
        logger.info(f"Model output: {result}")
        return result

    def wait_for_job_completion(self, job_id, check_interval=5, log_interval=10):

        logger.info(f"Waiting for fine-tuning job to complete...")
        
        log_offset = 0
        counter = 0
        
        while True:

            status = self.check_fine_tune_status(job_id=job_id)
            
            if status == "SUCCEEDED":
                self._fetch_and_print_logs(log_offset)
                logger.info("Fine-tuning job completed successfully!")
                return True
            elif status in ["FAILED", "CANCELLED"]:
                self._fetch_and_print_logs(log_offset)
                logger.error(f"Fine-tuning job failed, status: {status}")
                return False
            
            if counter % (log_interval // check_interval) == 0:
                new_offset = self._fetch_and_print_logs(log_offset)
                if new_offset > log_offset:
                    log_offset = new_offset
            
            logger.info(f"Fine-tuning job in progress (status: {status}), checking again in {check_interval} seconds...")
            time.sleep(check_interval)
            counter += 1
    
    def _fetch_and_print_logs(self, offset=0, line=1000):

        try:
            logs = self.get_fine_tune_logs(offset=offset, line=line)
            if logs:
                if isinstance(logs, list):
                    logs_text = '\n'.join(str(log) for log in logs) if logs else ''
                else:
                    logs_text = logs
                
                if logs_text and logs_text.strip():
                    new_lines = logs_text.count('\n') + (0 if logs_text.endswith('\n') else 1)
                    new_offset = offset + new_lines
                    
                    for log_line in logs_text.splitlines():
                        logger.info(f"[Fine-tune Log] {log_line}")
                    
                    return new_offset
            
            return offset
        except Exception as e:
            logger.error(f"Error fetching logs: {str(e)}")
            return offset

    def wait_for_deployment(self, check_interval=5):

        while True:
            status = self.check_deployment_status()

            if status == "RUNNING":
                logger.info("Model deployment successful, now ready to use!")
                return True
            elif status in ["FAILED", "CANCELLED"]:
                logger.error(f"Model deployment failed, status: {status}")
                return False

            logger.info(f"Model deployment in progress, checking again in {check_interval} seconds...")
            time.sleep(check_interval)

    def list_available_models(self):
        """列出可用于调优的模型"""
        logger.info("Returning hardcoded list of models that support fine-tuning...")
        
        # 直接返回图片中显示的模型列表
        models = [
            {"id": "qwen2.5-72b-instruct"},
            {"id": "qwen2.5-32b-instruct"},
            {"id": "qwen2.5-14b-instruct"},
            {"id": "qwen2.5-7b-instruct"},
            {"id": "qwen2-72b-instruct"},
            {"id": "qwen2-7b-instruct"},
            {"id": "qwen1.5-72b-chat"},
            {"id": "qwen1.5-14b-chat"},
            {"id": "qwen1.5-7b-chat"},
            {"id": "qwen-turbo"},
            {"id": "qwen-turbo-0624"},
            {"id": "qwen-plus-0723"},
            {"id": "qwen-vl-max-0201"},
            {"id": "qwen-vl-plus"}
        ]
        
        logger.info(f"Returning {len(models)} models")
        return models

    @staticmethod
    def convert_to_jsonl(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            output_file = file_path + "l"
            conversations = []
            for item in data:
                if 'user' not in item or 'assistant' not in item:
                    logger.warning(f"跳过缺少user或assistant字段的项: {item}")
                    continue

                conversation = {
                    "messages": [
                        {"role": "user", "content": item['user']},
                        {"role": "assistant", "content": item['assistant']}
                    ]
                }

                for key, value in item.items():
                    if key not in ['user', 'assistant']:
                        conversation[key] = value

                conversations.append(conversation)

            original_count = len(conversations)
            if original_count == 0:
                logger.error("There is no valid dialogue data to convert")


            if original_count < 40:
                copies_needed = (40 + original_count - 1) // original_count
                conversations = conversations * copies_needed


            with open(output_file, 'w', encoding='utf-8') as f:
                for conversation in conversations:
                    f.write(json.dumps(conversation, ensure_ascii=False) + '\n')

            final_count = len(conversations)
            logger.info(f'Converted {original_count} to {final_count} conversations')
            return output_file

        except Exception as e:
            logger.error(f"Failed to convert to JSONL: {str(e)}")
            return None

