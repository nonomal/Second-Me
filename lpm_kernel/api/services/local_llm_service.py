import os
import json
import logging
import psutil
import time
import subprocess
from typing import Iterator, Any, Optional, Generator, Dict
from datetime import datetime
from flask import Response
from openai import OpenAI
from lpm_kernel.api.domains.kernel2.dto.server_dto import ServerStatus, ProcessInfo
from lpm_kernel.configs.config import Config
import uuid

logger = logging.getLogger(__name__)

class LocalLLMService:
    """Service for managing local LLM client and server"""
    
    def __init__(self):
        self._client = None
        self._stopping_server = False
        
    @property
    def client(self) -> OpenAI:
        config = Config.from_env()
        """Get the OpenAI client for local LLM server"""
        if self._client is None:
            base_url = config.get("LOCAL_LLM_SERVICE_URL")
            if not base_url:
                raise ValueError("LOCAL_LLM_SERVICE_URL environment variable is not set")
                
            self._client = OpenAI(
                base_url=base_url,
                api_key="sk-no-key-required"
            )
        return self._client

    def start_server(self, model_path: str) -> bool:
        """
        Start the llama-server service
        """
        try:
            # Check if server is already running
            status = self.get_server_status()
            if status.is_running:
                logger.info("LLama server is already running")
                return True

            # Start server
            cmd = [
                "llama-server",
                "-m", model_path,
                "--host", "0.0.0.0",
                "--port", "8000"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for server to start
            time.sleep(2)
            
            # Check if process started successfully
            if process.poll() is None:
                logger.info("LLama server started successfully")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"Failed to start llama-server: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting llama-server: {str(e)}")
            return False

    def stop_server(self) -> ServerStatus:
        """
        Stop the llama-server service.
        Find and forcibly terminate all llama-server processes
        
        Returns:
            ServerStatus: Service status object containing information about whether processes are still running
        """
        try:
            if self._stopping_server:
                logger.info("Server is already in the process of stopping")
                return self.get_server_status()
            
            self._stopping_server = True
        
            try:
                # Find all possible llama-server processes and forcibly terminate them
                terminated_pids = []
                for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                    try:
                        cmdline = proc.cmdline()
                        if any("llama-server" in cmd for cmd in cmdline):
                            pid = proc.pid
                            logger.info(f"Force terminating llama-server process, PID: {pid}")
                            
                            # Directly use kill signal to forcibly terminate
                            proc.kill()
                            
                            # Ensure the process has been terminated
                            try:
                                proc.wait(timeout=0.2)  # Slightly increase wait time to ensure process termination
                                terminated_pids.append(pid)
                                logger.info(f"Successfully terminated llama-server process {pid}")
                            except psutil.TimeoutExpired:
                                # If timeout, try to terminate again
                                logger.warning(f"Process {pid} still running, sending SIGKILL again")
                                try:
                                    import os
                                    import signal
                                    os.kill(pid, signal.SIGKILL)  # Use system-level SIGKILL signal
                                    terminated_pids.append(pid)
                                    logger.info(f"Successfully force killed llama-server process {pid} with SIGKILL")
                                except ProcessLookupError:
                                    # Process no longer exists
                                    terminated_pids.append(pid)
                                    logger.info(f"Process {pid} no longer exists after kill attempt")
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                
                if terminated_pids:
                    logger.info(f"Terminated llama-server processes: {terminated_pids}")
                else:
                    logger.info("No running llama-server process found")
                
                # Check again if any llama-server processes are still running
                return self.get_server_status()
            
            finally:
                self._stopping_server = False
            
        except Exception as e:
            logger.error(f"Error stopping llama-server: {str(e)}")
            self._stopping_server = False
            return ServerStatus.not_running()

    def get_server_status(self) -> ServerStatus:
        """
        Get the current status of llama-server
        Returns: ServerStatus object
        """
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = proc.cmdline()
                    if any("llama-server" in cmd for cmd in cmdline):
                        with proc.oneshot():
                            process_info = ProcessInfo(
                                pid=proc.pid,
                                cpu_percent=proc.cpu_percent(),
                                memory_percent=proc.memory_percent(),
                                create_time=proc.create_time(),
                                cmdline=cmdline,
                            )
                            return ServerStatus.running(process_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
            return ServerStatus.not_running()
            
        except Exception as e:
            logger.error(f"Error checking llama-server status: {str(e)}")
            return ServerStatus.not_running()

    def _parse_response_chunk(self, chunk):
        """Parse different response chunk formats into a standardized format."""
        try:
            if chunk is None:
                logger.warning("Received None chunk")
                return None
                
            # logger.info(f"Parsing response chunk: {chunk}")
            # Handle custom format
            if isinstance(chunk, dict) and "type" in chunk and chunk["type"] == "chat_response":
                logger.info(f"Processing custom format response: {chunk}")
                return {
                    "id": str(uuid.uuid4()),  # Generate a unique ID
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": "models/lpm",
                    "system_fingerprint": None,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk.get("content", "")
                            },
                            "finish_reason": "stop" if chunk.get("done", False) else None
                        }
                    ]
                }
            
            # Handle OpenAI format
            if not hasattr(chunk, 'choices'):
                logger.warning(f"Chunk has no choices attribute: {chunk}")
                return None
                
            choices = getattr(chunk, 'choices', [])
            if not choices:
                logger.warning("Chunk has empty choices")
                return None
                
            # logger.info(f"Processing OpenAI format response: choices={choices}")
            delta = choices[0].delta
            
            # Create standard response structure
            response_data = {
                "id": chunk.id,
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": "models/lpm",
                "system_fingerprint": chunk.system_fingerprint if hasattr(chunk, 'system_fingerprint') else None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            # Keep even if content is None, let the client handle it
                            "content": delta.content if hasattr(delta, 'content') else ""
                        },
                        "finish_reason": choices[0].finish_reason
                    }
                ]
            }
            
            # If there is neither content nor finish_reason, skip
            if not (hasattr(delta, 'content') or choices[0].finish_reason):
                logger.debug("Skipping chunk with no content and no finish_reason")
                return None
                
            return response_data
            
        except Exception as e:
            logger.error(f"Error parsing response chunk: {e}, chunk: {chunk}")
            return None

    def handle_stream_response(self, response_iter: Iterator[Any]) -> Response:
        """Handle streaming response from the LLM server"""
        def generate():
            chunk = None  # Initialize chunk variable
            start_time = time.time()
            chunk_count = 0
            last_chunk_time = start_time
            last_heartbeat_time = start_time
            heartbeat_interval = 10  # Interval for sending heartbeat (seconds)
            
            logger.info(f"[STREAM_DEBUG] Starting stream response at {start_time}")
            
            try:
                for chunk in response_iter:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    chunk_interval = current_time - last_chunk_time
                    
                    # Check if heartbeat needs to be sent
                    if current_time - last_heartbeat_time >= heartbeat_interval:
                        logger.info(f"[STREAM_DEBUG] Sending heartbeat after {current_time - last_heartbeat_time:.2f}s inactivity")
                        yield b": heartbeat\n\n"  # SSE comment line as heartbeat, frontend will ignore this line
                        last_heartbeat_time = current_time
                    
                    chunk_count += 1
                    
                    logger.info(f"[STREAM_DEBUG] Received chunk #{chunk_count} after {elapsed_time:.2f}s (interval: {chunk_interval:.2f}s)")
                    last_chunk_time = current_time
                    
                    if chunk is None:
                        logger.warning("Received None chunk in stream, skipping")
                        continue
                        
                    # Check if this is the done marker for custom format
                    if chunk == "[DONE]":
                        logger.info(f"[STREAM_DEBUG] Received [DONE] marker after {elapsed_time:.2f}s")
                        yield b"data: [DONE]\n\n"
                        return  # Use return instead of break to ensure [DONE] in finally won't be executed
                    
                    # Handle OpenAI error format directly
                    if isinstance(chunk, dict) and "error" in chunk:
                        logger.warning(f"[STREAM_DEBUG] Received error response after {elapsed_time:.2f}s: {chunk}")
                        data_str = json.dumps(chunk)
                        yield f"data: {data_str}\n\n".encode('utf-8')
                        # After sending error, send [DONE] marker to close the stream properly
                        yield b"data: [DONE]\n\n"
                        return
                    
                    response_data = self._parse_response_chunk(chunk)
                    if response_data:
                        data_str = json.dumps(response_data)
                        content = response_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        content_length = len(content) if content else 0
                        logger.info(f"[STREAM_DEBUG] Sending chunk #{chunk_count}, content length: {content_length}, elapsed: {elapsed_time:.2f}s")
                        yield f"data: {data_str}\n\n".encode('utf-8')
                        last_heartbeat_time = current_time  # Reset heartbeat time
                    else:
                        logger.warning(f"[STREAM_DEBUG] Parsed response data is None, skipping chunk #{chunk_count}")
                
                # Handle the case where the iterator is empty, ensure a thinking message is sent before completion
                current_time = time.time()
                if chunk_count == 0 and current_time - start_time > heartbeat_interval:
                    logger.info("[STREAM_DEBUG] No chunks received yet, sending thinking message")
                    thinking_message = {
                        "id": str(uuid.uuid4()),
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now().timestamp()),
                        "model": "models/lpm",
                        "system_fingerprint": None,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": ""  # Empty content won't affect frontend display
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                    data_str = json.dumps(thinking_message)
                    yield f"data: {data_str}\n\n".encode('utf-8')
                    
            except Exception as e:
                current_time = time.time()
                elapsed_time = current_time - start_time
                logger.error(f"[STREAM_DEBUG] Stream error after {elapsed_time:.2f}s with {chunk_count} chunks: {str(e)}", exc_info=True)
                
                # Check if it's a BrokenPipeError specifically
                if isinstance(e, BrokenPipeError):
                    logger.error(f"[STREAM_DEBUG] BrokenPipeError detected, likely client disconnect at {elapsed_time:.2f}s")
                
                error_msg = json.dumps({'error': str(e)})
                try:
                    yield f"data: {error_msg}\n\n".encode('utf-8')
                except Exception as yield_error:
                    logger.error(f"[STREAM_DEBUG] Failed to yield error message: {str(yield_error)}")
            finally:
                current_time = time.time()
                total_time = current_time - start_time
                logger.info(f"[STREAM_DEBUG] Stream completed after {total_time:.2f}s with {chunk_count} chunks")
                
                if chunk != "[DONE]":  # Only send if [DONE] marker was not received
                    logger.info(f"[STREAM_DEBUG] Sending final [DONE] marker at {total_time:.2f}s")
                    yield b"data: [DONE]\n\n"
                
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache, no-transform',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
                'Transfer-Encoding': 'chunked'
            }
        )


# Global instance
local_llm_service = LocalLLMService()
