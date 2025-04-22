import os
import json
import logging
import psutil
import time
import subprocess
import threading
import queue
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
        # Create a queue for thread communication
        message_queue = queue.Queue()
        # Create an event flag to notify when model processing is complete
        completion_event = threading.Event()
        # Create a variable to track if heartbeat is needed after first response
        first_response_received = False
        
        def heartbeat_thread():
            """Thread function for sending heartbeats"""
            start_time = time.time()
            heartbeat_interval = 10  # Send heartbeat every 10 seconds
            heartbeat_count = 0
            
            logger.info("[STREAM_DEBUG] Heartbeat thread started")
            
            try:
                # Send initial heartbeat
                message_queue.put((b": initial heartbeat\n\n", "[INITIAL_HEARTBEAT]"))
                last_heartbeat_time = time.time()
                
                while not completion_event.is_set():
                    current_time = time.time()
                    
                    # Check if we need to send a heartbeat
                    if current_time - last_heartbeat_time >= heartbeat_interval:
                        heartbeat_count += 1
                        elapsed = current_time - start_time
                        logger.info(f"[STREAM_DEBUG] Sending heartbeat #{heartbeat_count} at {elapsed:.2f}s")
                        message_queue.put((f": heartbeat #{heartbeat_count}\n\n".encode('utf-8'), "[HEARTBEAT]"))
                        last_heartbeat_time = current_time
                    
                    # Short sleep to prevent CPU spinning
                    time.sleep(0.1)
                
                logger.info(f"[STREAM_DEBUG] Heartbeat thread stopping after {heartbeat_count} heartbeats")
            except Exception as e:
                logger.error(f"[STREAM_DEBUG] Error in heartbeat thread: {str(e)}", exc_info=True)
                message_queue.put((f"data: {{\"error\": \"Heartbeat error: {str(e)}\"}}\n\n".encode('utf-8'), "[ERROR]"))
        
        def model_response_thread():
            """Thread function for processing model responses"""
            chunk = None
            start_time = time.time()
            chunk_count = 0
            
            try:
                logger.info("[STREAM_DEBUG] Model response thread started")
                
                # Process model responses
                for chunk in response_iter:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    chunk_count += 1
                    
                    logger.info(f"[STREAM_DEBUG] Received chunk #{chunk_count} after {elapsed_time:.2f}s")
                    
                    if chunk is None:
                        logger.warning("[STREAM_DEBUG] Received None chunk, skipping")
                        continue
                    
                    # Check if it's an end marker
                    if chunk == "[DONE]":
                        logger.info(f"[STREAM_DEBUG] Received [DONE] marker after {elapsed_time:.2f}s")
                        message_queue.put((b"data: [DONE]\n\n", "[DONE]"))
                        break
                    
                    # Handle error responses
                    if isinstance(chunk, dict) and "error" in chunk:
                        logger.warning(f"[STREAM_DEBUG] Received error response: {chunk}")
                        data_str = json.dumps(chunk)
                        message_queue.put((f"data: {data_str}\n\n".encode('utf-8'), "[ERROR]"))
                        message_queue.put((b"data: [DONE]\n\n", "[DONE]"))
                        break
                    
                    # Handle normal responses
                    response_data = self._parse_response_chunk(chunk)
                    if response_data:
                        data_str = json.dumps(response_data)
                        content = response_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        content_length = len(content) if content else 0
                        logger.info(f"[STREAM_DEBUG] Sending chunk #{chunk_count}, content length: {content_length}, elapsed: {elapsed_time:.2f}s")
                        message_queue.put((f"data: {data_str}\n\n".encode('utf-8'), "[CONTENT]"))
                    else:
                        logger.warning(f"[STREAM_DEBUG] Parsed response data is None for chunk #{chunk_count}")
                
                # Handle the case where no responses were received
                if chunk_count == 0:
                    logger.info("[STREAM_DEBUG] No chunks received, sending empty message")
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
                    message_queue.put((f"data: {data_str}\n\n".encode('utf-8'), "[THINKING]"))
                
                # Model processing is complete, send end marker
                if chunk != "[DONE]":
                    logger.info(f"[STREAM_DEBUG] Sending final [DONE] marker after {elapsed_time:.2f}s")
                    message_queue.put((b"data: [DONE]\n\n", "[DONE]"))
                
            except Exception as e:
                logger.error(f"[STREAM_DEBUG] Error processing model response: {str(e)}", exc_info=True)
                message_queue.put((f"data: {{\"error\": \"{str(e)}\"}}\n\n".encode('utf-8'), "[ERROR]"))
                message_queue.put((b"data: [DONE]\n\n", "[DONE]"))
            finally:
                # Set completion event to notify heartbeat thread to stop
                completion_event.set()
                logger.info(f"[STREAM_DEBUG] Model response thread completed with {chunk_count} chunks")
        
        def generate():
            """Main generator function for generating responses"""
            # Start heartbeat thread
            heart_thread = threading.Thread(target=heartbeat_thread, daemon=True)
            heart_thread.start()
            
            # Start model response processing thread
            model_thread = threading.Thread(target=model_response_thread, daemon=True)
            model_thread.start()
            
            try:
                # Get messages from queue and return to client
                while True:
                    try:
                        # Use short timeout to get message, prevent blocking
                        message, message_type = message_queue.get(timeout=0.1)
                        logger.debug(f"[STREAM_DEBUG] Yielding message type: {message_type}")
                        yield message
                        
                        # If end marker is received, exit loop
                        if message_type == "[DONE]":
                            logger.info("[STREAM_DEBUG] Received [DONE] marker, ending generator")
                            break
                    except queue.Empty:
                        # Queue is empty, continue trying to get message
                        # Check if model thread has completed but didn't send [DONE]
                        if completion_event.is_set() and not model_thread.is_alive():
                            logger.warning("[STREAM_DEBUG] Model thread completed without [DONE], ending generator")
                            yield b"data: [DONE]\n\n"
                            break
                        pass
            except GeneratorExit:
                # Client closed connection
                logger.info("[STREAM_DEBUG] Client closed connection (GeneratorExit)")
                completion_event.set()
            except Exception as e:
                logger.error(f"[STREAM_DEBUG] Error in generator: {str(e)}", exc_info=True)
                try:
                    yield f"data: {{\"error\": \"Generator error: {str(e)}\"}}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"
                except:
                    pass
                completion_event.set()
            finally:
                # Ensure completion event is set
                completion_event.set()
                # Wait for threads to complete
                if heart_thread.is_alive():
                    heart_thread.join(timeout=1.0)
                if model_thread.is_alive():
                    model_thread.join(timeout=1.0)
                logger.info("[STREAM_DEBUG] Generator completed")
        
        # Return response
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
