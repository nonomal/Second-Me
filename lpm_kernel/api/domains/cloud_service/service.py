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
        self.model_id = None  # 微调生成的模型的ID

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

    def create_fine_tune_job(self, base_model="qwen2.5-7b-instruct", training_type="efficient_sft", hyper_parameters=None):
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

        self.model_id = response_data.get('output', {}).get('deployment_id')
        logger.info(f"Model deployment created successfully! Deployment ID: {self.model_id}")
        return True

    def check_deployment_status(self, model_id):
        """检查模型部署状态"""

        url = f"{self.base_url}/deployments/{model_id}"
        headers = {**self.headers, "Content-Type": "application/json"}

        response = requests.get(url, headers=headers)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"Failed to query deployment status! Error: {response_data}")
            return None

        status = response_data.get('output', {}).get('status')
        logger.info(f"Model deployment status: {status}")
        return status
    
    def list_deployments(self):
        """获取所有已部署的模型列表"""
        
        url = f"{self.base_url}/deployments"
        headers = {**self.headers, "Content-Type": "application/json"}
        
        try:
            response = requests.get(url, headers=headers)
            response_data = response.json()
            
            if response.status_code != 200:
                logger.error(f"Failed to list deployments! Error: {response_data}")
                return None
            
            deployments = response_data.get('output', {}).get('deployments', [])
            logger.info(f"Found {len(deployments)} deployments")
            
            result = []
            for deployment in deployments:
                result.append({
                    "name": deployment.get("name"),
                    "deployed_model": deployment.get("deployed_model"),
                    "status": deployment.get("status"),
                    "base_model": deployment.get("base_model")
                })
            
            return result
        except Exception as e:
            logger.error(f"Exception when listing deployments: {str(e)}", exc_info=True)
            return None

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

    def run_inference(self, user_input, model_id):
        """使用调优后的模型进行推理"""

        url = f"{self.base_url}/services/aigc/text-generation/generation"
        headers = {**self.headers, "Content-Type": "application/json"}

        payload = {
            "model": model_id,
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

    def cancel_fine_tune_job(self, job_id):
        """取消正在进行的模型调优任务
        
        根据API文档: POST /api/v1/fine-tunes/<job_id>/cancel
        """
        url = f"{self.base_url}/fine-tunes/{job_id}/cancel"
        headers = {**self.headers, "Content-Type": "application/json"}
        
        try:
            response = requests.post(url, headers=headers)
            response_data = response.json()
            
            if response.status_code != 200:
                logger.error(f"Failed to cancel fine-tune job! Error: {response_data}")
                return False
                
            logger.info(f"Fine-tune job cancelled successfully! Job ID: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Exception during fine-tune job cancellation: {str(e)}")
            return False
            
    def wait_for_job_completion(self, job_id, check_interval=10, log_interval=10, progress_callback=None):
        """
        等待微调任务完成，并通过回调函数更新进度
        
        Args:
            job_id: 微调任务ID
            check_interval: 检查间隔（秒）
            log_interval: 日志获取间隔（秒）
            progress_callback: 进度回调函数，接收参数(status, progress, message)
        
        Returns:
            bool: 任务是否成功完成
        """
        logger.info(f"Waiting for fine-tuning job to complete...")
        
        log_offset = 0
        counter = 0
        start_time = time.time()
        estimated_total_minutes = None  # 从日志中解析的估计总时间
        queuing_phase = True  # 初始阶段为排队阶段
        fine_tune_start_time = None  # 微调开始的时间
        
        # 初始化进度
        if progress_callback:
            progress_callback("IN_PROGRESS", 0, "Fine-tuning job started, waiting in queue...")
        
        while True:
            status = self.check_fine_tune_status(job_id=job_id)
            elapsed_minutes = (time.time() - start_time) / 60
            
            # 定期获取日志并解析估计时间
            if counter % (log_interval // check_interval) == 0:
                new_offset, parsed_minutes = self._fetch_and_print_logs(log_offset)
                
                # 更新日志偏移量
                if new_offset > log_offset:
                    log_offset = new_offset
                
                # 检查是否找到估计时间
                if parsed_minutes is not None and estimated_total_minutes is None:
                    # 当找到估计时间时，表示数据上传和微调任务创建已完成
                    # 并且微调即将开始
                    estimated_total_minutes = parsed_minutes
                    logger.info(f"Using estimated fine-tune time from logs: {estimated_total_minutes} minutes")
                    
                    # 标记上传和创建任务已完成，微调开始
                    if queuing_phase:
                        queuing_phase = False
                        fine_tune_start_time = time.time()
                        logger.info("Fine-tuning phase started")
                        
                        # 通知回调函数上传和创建任务已完成
                        if progress_callback:
                            progress_callback("IN_PROGRESS", 10, f"Data uploaded and fine-tune job created. Estimated time: {estimated_total_minutes} minutes")
                
                # 如果还没有找到估计时间，但发现微调开始的日志
                logs_content = self.get_fine_tune_logs(offset=0, line=1000)
                logs_str = str(logs_content) if logs_content else ""
                if queuing_phase and ("start to fine-tune" in logs_str or "Fine-tune started" in logs_str):
                    queuing_phase = False
                    fine_tune_start_time = time.time()
                    logger.info("Fine-tuning phase started")
            
            # 根据状态和阶段计算进度
            if status == "SUCCEEDED":
                # 微调完成
                new_offset, _ = self._fetch_and_print_logs(log_offset)
                logger.info("Fine-tuning job completed successfully!")
                
                self.model_id = self._get_model_id_from_job(job_id)
                
                if progress_callback:
                    progress_callback("COMPLETED", 100, f"Fine-tuning job completed successfully! Model ID: {self.model_id}")
                
                return True
            elif status in ["FAILED", "CANCELLED"]:
                # 微调失败
                new_offset, _ = self._fetch_and_print_logs(log_offset)
                error_message = f"Fine-tuning job failed, status: {status}"
                logger.error(error_message)
                
                if progress_callback:
                    progress_callback("FAILED", 0, error_message)
                
                return False
            
            # 计算进度
            if queuing_phase:
                # 在排队阶段，这包括数据上传和创建微调任务
                # 根据经过的时间估计进度，但最高只能达到10%
                queue_progress = min(10, elapsed_minutes * 2)  # 每分钟增加2%，最高10%
                estimated_progress = queue_progress
                progress_message = f"Preparing data and creating fine-tune job. Elapsed time: {int(elapsed_minutes)} minutes"
            else:
                # 微调已经开始，计算实际微调进度
                if estimated_total_minutes is not None and fine_tune_start_time is not None:
                    # 使用从日志解析的估计时间计算进度
                    fine_tune_elapsed = (time.time() - fine_tune_start_time) / 60
                    # 微调阶段占总进度的90%（从10%到95%）
                    progress_percent = (fine_tune_elapsed / estimated_total_minutes) * 85
                    estimated_progress = min(95, 10 + progress_percent)
                    
                    # 根据进度百分比定制消息
                    if estimated_progress < 30:
                        stage_desc = "Initializing training environment"
                    elif estimated_progress < 60:
                        stage_desc = "Training model with your data"
                    else:
                        stage_desc = "Finalizing model training"
                        
                    progress_message = f"{stage_desc}. Elapsed: {int(fine_tune_elapsed)}/{estimated_total_minutes} minutes (~{int(estimated_progress)}%)"
                else:
                    # 如果没有估计时间，保持在微调初始阶段
                    estimated_progress = 15  # 固定在初始阶段的进度
                    fine_tune_elapsed = (time.time() - fine_tune_start_time) / 60 if fine_tune_start_time else elapsed_minutes
                    progress_message = f"Waiting for training progress information. Elapsed time: {int(fine_tune_elapsed)} minutes"
            
            if progress_callback:
                progress_callback("IN_PROGRESS", estimated_progress, progress_message)
            
            logger.info(f"Fine-tuning job in progress (status: {status}), checking again in {check_interval} seconds...")
            time.sleep(check_interval)
            counter += 1
    
    def _fetch_and_print_logs(self, offset=0, line=1000):
        """
        获取日志并解析估计时间信息
        
        Returns:
            tuple: (new_offset, estimated_minutes)
                new_offset: 新的日志偏移量
                estimated_minutes: 从日志中解析的估计时间（分钟），如果没有找到则为None
        """
        estimated_minutes = None
        try:
            logs = self.get_fine_tune_logs(offset=offset, line=line)
            
            if not logs:
                return offset, estimated_minutes
                
            if isinstance(logs, list):
                logs_text = '\n'.join(str(log) for log in logs) if logs else ''
            else:
                logs_text = str(logs)
                
            if not logs_text or not logs_text.strip():
                return offset, estimated_minutes
                
            log_lines = logs_text.splitlines()
            new_offset = offset + len(log_lines)
            
            # 解析日志中的估计时间信息
            for log_line in log_lines:
                if log_line.strip():
                    logger.info(f"[Fine-tune Log] {log_line}")
                    
                    # 查找估计时间信息
                    if "Fine-tune estimated time:" in log_line:
                        try:
                            # 提取时间值，格式如 "Fine-tune estimated time: 3.60 mins!"
                            time_str = log_line.split("Fine-tune estimated time:")[1].split("mins")[0].strip()
                            estimated_minutes = float(time_str)
                            logger.info(f"Parsed estimated fine-tune time: {estimated_minutes} minutes")
                        except Exception as parse_error:
                            logger.error(f"Error parsing estimated time: {str(parse_error)}")
            
            return new_offset, estimated_minutes
        except Exception as e:
            logger.error(f"Error fetching logs: {str(e)}", exc_info=True)
            return offset, estimated_minutes

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

    def _get_model_id_from_job(self, job_id):
        """
        从微调任务中获取模型ID
        
        Args:
            job_id: 微调任务ID
            
        Returns:
            str: 模型ID或None
        """
        try:
            # 获取任务详情
            url = f"{self.base_url}/fine-tunes/{job_id}"
            headers = {**self.headers, "Content-Type": "application/json"}
            
            response = requests.get(url, headers=headers)
            response_data = response.json()
            
            if response.status_code != 200:
                logger.error(f"Failed to get fine-tune job details! Error: {response_data}")
                return None
                
            # 从响应中提取模型ID
            model_id = response_data.get('output', {}).get('model_id')
            if model_id:
                logger.info(f"Retrieved model ID {model_id} from job {job_id}")
                return model_id
            else:
                logger.warning(f"No model ID found in job {job_id}")
                return None
        except Exception as e:
            logger.error(f"Exception when getting model ID from job: {str(e)}")
            return None
    
    def list_available_models(self):
        """列出可用于调优的模型"""
        logger.info("Returning hardcoded list of models that support fine-tuning...")
        
        models = [

            {"id": "qwen2.5-7b-instruct"},
            {"id": "qwen2.5-14b-instruct"},
            {"id": "qwen2.5-32b-instruct"},
            {"id": "qwen2.5-72b-instruct"},
            

            {"id": "qwen2-7b-instruct"},
            {"id": "qwen2-72b-instruct"},
            

            {"id": "qwen1.5-7b-chat"},
            {"id": "qwen1.5-14b-chat"},
            {"id": "qwen1.5-72b-chat"},
            

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

