import os
import requests
import time
from typing import Dict, Any, Iterator, Optional, Generator, List
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
        self.model_id = None  # ID of the fine-tuned model

    def upload_training_file(self, file_path = "resources/L2/data/merged.json", description=None):
        """Upload training data file"""
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

    def create_fine_tune_job(self, base_model, training_type="efficient_sft", hyper_parameters=None):
        """Create fine-tuning job"""
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
        """Check fine-tuning job status"""

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
            output = response_data.get('output', {})
            self.model_id = output.get('finetuned_output') or output.get('model_id')
            logger.info(f"Fine-tuning successful! Model ID: {self.model_id}")

        return status

    def get_fine_tune_logs(self, offset=0, line=1000):
        """Get fine-tuning logs"""
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

    def check_deployment_status(self, model_id):
        """Check model deployment status"""

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
        """Get list of all deployed models"""

        url_deployments = f"{self.base_url}/deployments"
        url_fine_tunes = f"{self.base_url}/fine-tunes"
        headers = {**self.headers, "Content-Type": "application/json"}
        
        try:
            response = requests.get(url_deployments, headers=headers)
            response_data = response.json()
            
            if response.status_code != 200:
                logger.error(f"Failed to list deployments! Error: {response_data}")
                return None
            
            deployments = response_data.get('output', {}).get('deployments', [])
            # 筛选状态为RUNNING的部署模型
            running_deployments = [d for d in deployments if d.get("status") == "RUNNING"]
            logger.info(f"Found {len(running_deployments)} running deployments out of {len(deployments)} total")

            # 获取微调的模型
            response = requests.get(url_fine_tunes, headers=headers)
            response_data = response.json()

            if response.status_code != 200:
                logger.error(f"Failed to list fine-tunes! Error: {response_data}")
                return None

            fine_tunes = response_data.get('output', {}).get('jobs', [])
            # 筛选状态为SUCCEEDED的微调模型
            succeeded_fine_tunes = [ft for ft in fine_tunes if ft.get("status") == "SUCCEEDED"]
            logger.info(f"Found {len(succeeded_fine_tunes)} succeeded fine-tunes out of {len(fine_tunes)} total")

            result = []
            for ft in succeeded_fine_tunes:
                deployed_model = None
                is_deployed = False
                name = None
                
                # 提取微调模型ID的后缀
                ft_id_suffix = ft.get("job_id", "").split("-")[-1] if ft.get("job_id") else ""
                
                # 查找匹配的部署模型
                for d in running_deployments:
                    d_model_suffix = d.get("model_name", "").split("-")[-1] if d.get("model_name") else ""
                    if ft_id_suffix and d_model_suffix and ft_id_suffix == d_model_suffix:
                        is_deployed = True
                        deployed_model = d.get("deployed_model")
                        name = d.get("name")
                        break
                
                result.append({
                    "job_id": ft.get("job_id"),
                    "base_model": ft.get("base_model"),
                    "is_deployed": is_deployed,
                    "name": name,
                    "deployed_model": deployed_model,
                    "hyper_parameters": ft.get("hyper_parameters"),
                    "training_type": ft.get("training_type"),
                    "usage": ft.get("usage"),
                })


            return result
        except Exception as e:
            logger.error(f"Error listing deployments: {str(e)}")
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

    def handle_cloud_stream_response(self, response_iter):
        """
        Process cloud service streaming response, converting Alibaba Cloud DashScope format to a format consistent with the local interface

        Args:
            response_iter: Response generator

        Returns:
            Flask Response object containing streaming response
        """
        from flask import Response
        import uuid
        from datetime import datetime

        def generate():
            logger.info("=== Starting cloud stream response handler (exact local format match mode) ===")

            try:
                chunk_count = 0
                response_id = f"chatcmpl-{uuid.uuid4().hex[:22]}"
                system_fingerprint = f"b170-{uuid.uuid4().hex[:8]}"

                for chunk in response_iter:
                    chunk_count += 1
                    logger.debug(f"Processing chunk #{chunk_count}: {type(chunk)}")

                    openai_format = None

                    if isinstance(chunk, dict) and "error" in chunk:
                        error_msg = chunk.get("error", "Unknown error")
                        logger.error(f"Error in response: {error_msg}")
                        error_data = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(datetime.now().timestamp()),
                            "model": "models/lpm",
                            "system_fingerprint": system_fingerprint,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": f"Error: {error_msg}"
                                    },
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        data_str = json.dumps(error_data)
                        yield f"data: {data_str}\n\n".encode('utf-8')
                        yield b"data: [DONE]\n\n"
                        return

                    if isinstance(chunk, dict) and "output" in chunk and "choices" in chunk["output"]:
                        choices = chunk["output"]["choices"]
                        if choices and len(choices) > 0:
                            choice = choices[0]
                            content = ""
                            finish_reason = None

                            if "message" in choice and "content" in choice["message"]:
                                current_content = choice["message"]["content"]

                                if chunk_count == 1:
                                    self._full_content = current_content
                                    content = current_content
                                else:
                                    if not hasattr(self, '_full_content'):
                                        self._full_content = ""

                                    if current_content.startswith(self._full_content):
                                        content = current_content[len(self._full_content):]
                                        self._full_content = current_content
                                    else:
                                        content = current_content
                                        self._full_content = current_content
                                        logger.warning(f"Unable to determine incremental content, using full content")
                            else:
                                content = ""

                            if "finish_reason" in choice:
                                finish_reason = choice["finish_reason"]


                            openai_format = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": int(datetime.now().timestamp()),
                                "model": "models/lpm",
                                "system_fingerprint": system_fingerprint,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": content
                                        },
                                        "finish_reason": finish_reason
                                    }
                                ]
                            }

                    if not openai_format and isinstance(chunk, dict):
                        logger.warning(f"Unable to parse chunk into OpenAI format: {chunk}")

                        current_content = ""

                        if "content" in chunk:
                            current_content = chunk["content"]
                        elif "text" in chunk:
                            current_content = chunk["text"]
                        elif "message" in chunk:
                            current_content = chunk.get("message", "")
                        else:
                            current_content = str(chunk)

                        if chunk_count == 1 or not hasattr(self, '_raw_full_content'):
                            self._raw_full_content = current_content
                            content = current_content
                        else:
                            if current_content.startswith(self._raw_full_content):
                                content = current_content[len(self._raw_full_content):]
                                self._raw_full_content = current_content
                            else:
                                content = current_content
                                self._raw_full_content = current_content
                                logger.warning(f"Unable to determine incremental content for raw data, using full content")

                        openai_format = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(datetime.now().timestamp()),
                            "model": "models/lpm",
                            "system_fingerprint": system_fingerprint,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": content
                                    },
                                    "finish_reason": None
                                }
                            ]
                        }

                    if openai_format:
                        data_str = json.dumps(openai_format)
                        logger.debug(f"Yielding exact local format match: {data_str[:100]}...")
                        yield f"data: {data_str}\n\n".encode('utf-8')

                empty_content = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": "models/lpm",
                    "system_fingerprint": system_fingerprint,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": ""
                            },
                            "finish_reason": None
                        }
                    ]
                }
                empty_str = json.dumps(empty_content)
                yield f"data: {empty_str}\n\n".encode('utf-8')

                final_message = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": "models/lpm",
                    "system_fingerprint": system_fingerprint,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": None
                            },
                            "finish_reason": "stop"
                        }
                    ]
                }
                final_str = json.dumps(final_message)
                yield f"data: {final_str}\n\n".encode('utf-8')

                logger.info(f"Stream complete, processed {chunk_count} chunks")
                yield b"data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Error in stream response handler: {str(e)}", exc_info=True)
                response_id = f"chatcmpl-{uuid.uuid4().hex[:22]}"
                system_fingerprint = f"b170-{uuid.uuid4().hex[:8]}"
                error_response = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": "models/lpm",
                    "system_fingerprint": system_fingerprint,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": f"Error: {str(e)}"
                            },
                            "finish_reason": "stop"
                        }
                    ]
                }
                error_str = json.dumps(error_response)
                yield f"data: {error_str}\n\n".encode('utf-8')
                yield b"data: [DONE]\n\n"

        headers = {
            'Cache-Control': 'no-cache, no-transform',
            'X-Accel-Buffering': 'no',
            'Content-Type': 'text/event-stream',
            'Connection': 'keep-alive',
            'Transfer-Encoding': 'chunked'
        }

        logger.info(f"Creating Response with headers: {headers}")
        return Response(generate(), mimetype='text/event-stream', headers=headers, direct_passthrough=True)

    def run_inference(self, messages, model_id, stream=False, temperature=0.7, max_tokens=2048):
        """Run inference using fine-tuned model

        Args:
            messages: Message list in OpenAI format
            model_id: Model ID
            stream: Whether to use streaming output
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate

        Returns:
            If stream=True, returns a Flask Response object, otherwise returns complete response
        """
        # Record request parameters
        logger.info(f"Cloud inference request - model_id: {model_id}, stream: {stream}, temperature: {temperature}, max_tokens: {max_tokens}")
        logger.debug(f"Messages: {messages}")

        url = f"{self.base_url}/services/aigc/text-generation/generation"
        logger.info(f"API URL: {url}")

        # Create basic headers
        headers = {**self.headers, "Content-Type": "application/json"}

        # If stream is True, add specific header
        if stream:
            headers["X-DashScope-SSE"] = "enable"

        payload = {
            "model": model_id,
            "input": {
                "messages": messages
            },
            "parameters": {
                "result_format": "message",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
        }

        if stream:
            def generate():
                try:
                    logger.info(f"Sending stream request to {url}")
                    logger.debug(f"Request payload: {payload}")

                    with requests.post(url, headers=headers, json=payload, stream=True) as response:
                        logger.info(f"Received response with status code: {response.status_code}")

                        if response.status_code != 200:
                            error_msg = f"Model inference failed! Status code: {response.status_code}"
                            logger.error(error_msg)
                            error_response = {
                                "output": {
                                    "choices": [
                                        {
                                            "message": {
                                                "content": f"API请求失败: {error_msg}",
                                                "role": "assistant"
                                            },
                                            "finish_reason": "stop"
                                        }
                                    ]
                                }
                            }
                            yield error_response
                            return

                        response_received = False
                        chunk_count = 0

                        for line in response.iter_lines():
                            if not line:
                                continue

                            line = line.decode('utf-8')
                            response_received = True

                            if line.startswith('data:'):
                                data = line[5:].strip()

                                if data == '[DONE]':
                                    logger.info("Received [DONE] marker")
                                    break

                                try:
                                    chunk_data = json.loads(data)
                                    chunk_count += 1
                                    logger.debug(f"Processing chunk #{chunk_count}: {json.dumps(chunk_data)[:100]}...")

                                    if 'output' in chunk_data and 'choices' in chunk_data['output'] and len(chunk_data['output']['choices']) > 0:
                                        choice = chunk_data['output']['choices'][0]
                                        if 'message' in choice and 'content' in choice['message']:
                                            content = choice['message']['content']
                                            logger.info(f"Chunk #{chunk_count} content: {content[:50]}...")

                                    yield chunk_data
                                except json.JSONDecodeError as e:
                                    logger.error(f"Failed to decode JSON: {e} - Raw data: {data}")
                                except Exception as e:
                                    logger.error(f"Error processing SSE data: {str(e)}", exc_info=True)

                        logger.info(f"Processed {chunk_count} chunks from SSE response")

                        # If no response was received or no chunks were processed, return an empty response
                        if not response_received or chunk_count == 0:
                            logger.warning("No response received from API")
                            empty_data = {
                                "output": {
                                    "choices": [
                                        {
                                            "message": {
                                                "content": "No response received from API",
                                                "role": "assistant"
                                            },
                                            "finish_reason": "stop"
                                        }
                                    ]
                                }
                            }
                            yield empty_data

                except Exception as e:
                    error_msg = f"Exception during cloud inference: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    error_response = {
                        "output": {
                            "choices": [
                                {
                                    "message": {
                                        "content": f"Error: {str(e)}",
                                        "role": "assistant"
                                    },
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                    }
                    yield error_response

            logger.debug("Passing generator directly to handle_cloud_stream_response")
            return self.handle_cloud_stream_response(generate())
        else:
            try:
                response = requests.post(url, headers=headers, json=payload)
                response_data = response.json()

                if response.status_code != 200:
                    logger.error(f"Model inference failed! Error: {response_data}")
                    return None

                logger.info(f"Received non-streaming response: {json.dumps(response_data)[:200]}...")
                return response_data
            except Exception as e:
                logger.error(f"Exception during cloud inference: {str(e)}", exc_info=True)
                return None

    def cancel_fine_tune_job(self, job_id):
        """
        Cancel the fine-tuning job

        Args:
            job_id: The ID of the fine-tuning job to cancel

        Returns:
            bool: True if the job was successfully canceled, False otherwise
        """
        url = f"{self.base_url}/fine-tunes/{job_id}/cancel"
        headers = {**self.headers, "Content-Type": "application/json"}

        try:
            response = requests.post(url, headers=headers)
            response_data = response.json()

            if response.status_code != 200:
                logger.error(f"Failed to cancel fine-tune job! Error: {response_data}")
                return False

            logger.info(f"Fine-tune job canceled successfully! Job ID: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Exception during fine-tune job cancellation: {str(e)}")
            return False

    def wait_for_job_completion(self, job_id, check_interval=10, log_interval=10, progress_callback=None):
        """

        Wait for the fine-tuning job to complete and update progress via the callback function

        Args:
            job_id: Fine-tuning job ID
            check_interval: Check interval in seconds
            log_interval: Log retrieval interval in seconds
            progress_callback: Progress callback function, receives parameters (status, progress, message)

        Returns:
            bool: Whether the job completed successfully

        """
        logger.info(f"Waiting for fine-tuning job to complete...")

        log_offset = 0
        counter = 0
        start_time = time.time()
        estimated_total_minutes = None
        queuing_phase = True
        fine_tune_start_time = None

        if progress_callback:
            progress_callback("IN_PROGRESS", 0, "Fine-tuning job started, waiting in queue...")

        while True:
            status = self.check_fine_tune_status(job_id=job_id)
            elapsed_minutes = (time.time() - start_time) / 60

            if counter % (log_interval // check_interval) == 0:
                new_offset, parsed_minutes = self._fetch_and_print_logs(log_offset)

                if new_offset > log_offset:
                    log_offset = new_offset

                if parsed_minutes is not None and estimated_total_minutes is None:
                    estimated_total_minutes = parsed_minutes
                    logger.info(f"Using estimated fine-tune time from logs: {estimated_total_minutes} minutes")

                    if queuing_phase:
                        queuing_phase = False
                        fine_tune_start_time = time.time()
                        logger.info("Fine-tuning phase started")

                        if progress_callback:
                            # Start fine-tuning phase at 66% progress
                            progress_callback("IN_PROGRESS", 66, f"Data uploaded and fine-tune job created. Estimated time: {estimated_total_minutes} minutes")

                logs_content = self.get_fine_tune_logs(offset=0, line=1000)
                logs_str = str(logs_content) if logs_content else ""
                if queuing_phase and ("start to fine-tune" in logs_str or "Fine-tune started" in logs_str):
                    queuing_phase = False
                    fine_tune_start_time = time.time()
                    logger.info("Fine-tuning phase started")

                    if progress_callback:
                        # Start fine-tuning phase at 66% progress
                        progress_callback("IN_PROGRESS", 66, f"Fine-tuning phase started. Estimated time: {estimated_total_minutes if estimated_total_minutes else 'unknown'} minutes")

            if status == "SUCCEEDED":
                new_offset, _ = self._fetch_and_print_logs(log_offset)
                logger.info("Fine-tuning job completed successfully!")

                if progress_callback:
                    progress_callback("COMPLETED", 100, f"Fine-tuning job completed successfully! Model ID: {self.model_id}")

                return True
            elif status in ["FAILED","CANCELED"]:
                new_offset, _ = self._fetch_and_print_logs(log_offset)
                error_message = f"Fine-tuning job failed or canceled, status: {status}"
                logger.error(error_message)

                if progress_callback:
                    progress_callback("FAILED", 0, error_message)

                return False

            if queuing_phase:
                # Set queue phase progress directly to 66%
                estimated_progress = 66
                progress_message = f"Preparing data and initializing fine-tune job. Elapsed time: {int(elapsed_minutes)} minutes"
            else:
                if estimated_total_minutes is not None and fine_tune_start_time is not None:
                    fine_tune_elapsed = (time.time() - fine_tune_start_time) / 60
                    # Calculate raw progress as a percentage of elapsed time vs total estimated time
                    raw_progress_percent = (fine_tune_elapsed / estimated_total_minutes) * 100

                    adjusted_progress_percent = 66 + (raw_progress_percent / 3)

                    # Cap at 99% (100% is reserved for completion)
                    estimated_progress = min(99, adjusted_progress_percent)

                    # Determine stage description based on adjusted progress
                    if estimated_progress < 75:  # First third of the final stage
                        stage_desc = "Initializing training environment"
                    elif estimated_progress < 90:  # Second third of the final stage
                        stage_desc = "Training model with your data"
                    else:  # Final third of the final stage
                        stage_desc = "Finalizing model training"

                    progress_message = f"{stage_desc}. Elapsed: {int(fine_tune_elapsed)}/{estimated_total_minutes} minutes (~{int(estimated_progress)}%)"
                else:
                    # If we can't calculate exact progress, default to 66% (start of fine-tuning phase)
                    estimated_progress = 66
                    fine_tune_elapsed = (time.time() - fine_tune_start_time) / 60 if fine_tune_start_time else elapsed_minutes
                    progress_message = f"Waiting for training progress information. Elapsed time: {int(fine_tune_elapsed)} minutes"

            if progress_callback:
                progress_callback("IN_PROGRESS", estimated_progress, progress_message)

            logger.info(f"Fine-tuning job in progress (status: {status}), checking again in {check_interval} seconds...")
            time.sleep(check_interval)
            counter += 1

    def _fetch_and_print_logs(self, offset=0, line=1000):
        """
        Fetch logs and parse estimated time information

        Returns:
            tuple: (new_offset, estimated_minutes)
                new_offset: The new log offset
                estimated_minutes: Estimated time (in minutes) parsed from the logs, or None if not found
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

            for log_line in log_lines:
                if log_line.strip():
                    logger.info(f"[Fine-tune Log] {log_line}")

                    if "Fine-tune estimated time:" in log_line:
                        try:
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
            elif status in ["FAILED", "CANCELED"]:
                logger.error(f"Model deployment failed or canceled, status: {status}")
                return False

            logger.info(f"Model deployment in progress, checking again in {check_interval} seconds...")
            time.sleep(check_interval)


    def list_available_models(self):
        logger.info("Returning hardcoded list of models that support fine-tuning...")

        models = [
            {"id": "qwen3-8b"},
            {"id": "qwen3-14b"},
            {"id": "qwen3-32b"},
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
                    logger.warning(f"Skip items that are missing user or assistant fields: {item}")
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
