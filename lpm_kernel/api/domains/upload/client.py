import aiohttp
import logging
from lpm_kernel.api.domains.upload.TrainingTags import TrainingTags
from lpm_kernel.configs import config
import websockets
import json
import asyncio
from lpm_kernel.configs.config import Config
import time
import requests
from lpm_kernel.api.common.responses import ResponseHandler
from lpm_kernel.api.domains.loads.load_service import LoadService
from typing import Optional, List, Dict
import os  # 添加用于文件路径操作
from pathlib import Path  # 添加用于路径处理

logger = logging.getLogger(__name__)

class HeartbeatConfig:
    """Heartbeat Configuration Class"""
    def __init__(
        self,
        interval: int = 30,  # Heartbeat interval (seconds)
        timeout: int = 10,   # Heartbeat timeout (seconds)
        max_retries: int = 3,  # Maximum retry count
        retry_interval: int = 5  # Retry interval (seconds)
    ):
        self.interval = interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_interval = retry_interval

class RegistryClient:
    def __init__(self, heartbeat_config: HeartbeatConfig = None):
        config = Config.from_env()
        self.server_url = config.get("REGISTRY_SERVICE_URL")
        # Convert HTTP URL to WebSocket URL
        self.ws_url = self.server_url.replace('http://', 'ws://').replace('https://', 'wss://')
        # Store all active WebSocket connections
        self.active_connections = {}
        # Heartbeat configuration
        self.heartbeat_config = heartbeat_config or HeartbeatConfig()

    def _get_auth_header(self):
        """
        Get the Authorization header for authenticated requests
        
        Returns:
            dict: Authorization header or empty dict if no credentials
        """
        current_load, error, _ = LoadService.get_current_load(with_password=True)
        if not current_load or not current_load.instance_id or not current_load.instance_password:
            logger.info("No credentials found for auth")
            return {}
        instance_id = current_load.instance_id
        instance_password = current_load.instance_password
        
        logger.info(f"Using credentials for auth: {instance_id}:{instance_password}")
        return {
            "Authorization": f"Bearer {instance_id}:{instance_password}"
        }

    def _get_service_status_file_path(self):
        """Get the path for service status file"""
        return os.path.join(os.getcwd(), "data", "service_status.json")

    def _get_current_service_type(self):
        """
        Get current active service type from service status file
        
        Returns:
            tuple: (service_type, model_data) where service_type is 'local' or 'cloud'
        """
        try:
            status_file_path = self._get_service_status_file_path()
            if not os.path.exists(status_file_path):
                logger.warning("Service status file not found, defaulting to local service")
                return "local", {}
            
            with open(status_file_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
            
            service_type = status_data.get("service_type", "local")
            model_data = status_data.get("model_data", {})
            status = status_data.get("status", "inactive")
            
            if status != "active":
                logger.warning(f"Service status is {status}, defaulting to local service")
                return "local", {}
            
            logger.info(f"Current active service type: {service_type}")
            return service_type, model_data
            
        except Exception as e:
            logger.error(f"Failed to read service status file: {str(e)}")
            return "local", {}

    async def _check_local_service_status(self):
        """
        Check if local service is available
        
        Returns:
            bool: True if local service is available
        """
        try:
            config = Config.from_env()
            status_url = f"{config.KERNEL2_SERVICE_URL}/api/kernel2/llama/status"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(status_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        status_data = await response.json()
                        return status_data.get("data", {}).get("is_loaded", False)
                    
        except Exception as e:
            logger.error(f"Failed to check local service status: {str(e)}")
        
        return False

    async def _check_cloud_service_status(self):
        """
        Check if cloud service is available
        
        Returns:
            bool: True if cloud service is available
        """
        try:
            config = Config.from_env()
            status_url = f"{config.KERNEL2_SERVICE_URL}/api/cloud_service/service/status"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(status_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        status_data = await response.json()
                        return status_data.get("data", {}).get("status") == "active"
                    
        except Exception as e:
            logger.error(f"Failed to check cloud service status: {str(e)}")
        
        return False

    def _transform_request_for_cloud(self, request_data: dict, model_data: dict):
        """
        Transform OpenAI-compatible request for cloud inference API
        
        Args:
            request_data: Original request data
            model_data: Model data from service status
            
        Returns:
            dict: Transformed request for cloud inference
        """
        cloud_request = {
            "messages": request_data.get("messages", []),
            "model_id": model_data.get("model_id", ""),
            "temperature": request_data.get("temperature", 0.1),
            "max_tokens": request_data.get("max_tokens", 2000),
            "stream": request_data.get("stream", True),
            "enable_l0_retrieval": request_data.get("enable_l0_retrieval", False),
            "enable_l1_retrieval": request_data.get("enable_l1_retrieval", False),
            "role_id": request_data.get("role_id")
        }
        
        # 从metadata中提取知识检索参数（如果存在）
        metadata = request_data.get("metadata", {})
        if metadata:
            cloud_request["enable_l0_retrieval"] = metadata.get("enable_l0_retrieval", False)
            cloud_request["enable_l1_retrieval"] = metadata.get("enable_l1_retrieval", False)
            cloud_request["role_id"] = metadata.get("role_id")
        
        return cloud_request

    def get_ws_url(self, instance_id: str, instance_password: str) -> str:
        """
        Generate WebSocket URL for the specified instance
        
        Args:
            instance_id: Instance ID
            instance_password: Instance password
            
        Returns:
            str: WebSocket URL
        """
        return f"{self.ws_url}/api/ws/{instance_id}?password={instance_password}"

    def register_upload(self, upload_name: str, instance_id: str = None, description: str = None, email: str = None, tags: TrainingTags = None):
        """
        Register Upload instance with the registry center
        
        Args:
            upload_name: Upload name
            instance_id: Instance ID (optional)
            description: Description (optional)
            email: User email (optional)
            
        Returns:
            Registration data
        """
        headers = self._get_auth_header()
        tags_dict = tags.model_dump() if tags else None
        response = requests.post(
            f"{self.server_url}/api/upload/register",
            headers=headers,
            json={
                "upload_name": upload_name,
                "instance_id": instance_id,
                "description": description,
                "email": email,
                "tags": tags_dict
            }
        )
        return ResponseHandler.handle_response(
            response,
            success_log=f"Upload {upload_name} registered successfully in registry center, instance ID: {instance_id}",
            error_prefix="Registration"
        )

    def unregister_upload(self, instance_id: str):
        """Unregister Upload instance from registry center
        
        Args:
            instance_id: Instance ID
            
        Returns:
            dict: Unregistration result
        """
        headers = self._get_auth_header()
        response = requests.delete(
            f"{self.server_url}/api/upload/{instance_id}",
            headers=headers
        )
        return ResponseHandler.handle_response(
            response,
            success_log=f"Upload instance {instance_id} unregistered successfully from registry center",
            error_prefix="Unregistration"
        )

    async def connect_websocket(self, instance_id: str, instance_password: str):
        """Connect to registry center WebSocket and start keep-alive
        
        Args:
            instance_id: Instance ID
            instance_password: Instance password
            
        Returns:
            websockets.WebSocketClientProtocol: WebSocket connection
        """
        # Check if connection already exists and is active
        connection_key = f"{instance_id}"
        if connection_key in self.active_connections:
            existing_ws = self.active_connections[connection_key]
            try:
                # Check if connection is still active and send heartbeat
                if await self.send_heartbeat(existing_ws):
                    logger.info(f"Using existing WebSocket connection: {connection_key}")
                    return existing_ws
                raise Exception("Heartbeat failed")
            except Exception:
                # If heartbeat fails, connection is disconnected, remove from active connections
                logger.warning(f"Existing WebSocket connection is disconnected, creating new connection: {connection_key}")
                del self.active_connections[connection_key]

        # Create new connection
        ws_uri = self.get_ws_url(instance_id, instance_password)
        try:
            logger.info(f"Connecting to WebSocket: {ws_uri}")
            websocket = await websockets.connect(ws_uri)
            logger.info(f"WebSocket connection established: {ws_uri}")
            
            # Add additional attributes to WebSocket connection
            websocket.instance_id = instance_id
            websocket.connection_key = connection_key
            
            # Store new connection
            self.active_connections[connection_key] = websocket
            
            # Add lock to prevent concurrent message reception
            websocket.recv_lock = asyncio.Lock()
            
            # Start heartbeat task
            websocket.heartbeat_task = asyncio.create_task(
                self._keep_alive_with_ping(websocket, instance_id),
                name=f"heartbeat_{connection_key}"
            )
            await self.handle_messages(websocket)
            
            return websocket
        except Exception as e:
            logger.error(f"WebSocket connection failed: {str(e)}", exc_info=True)
            raise
            
    async def _keep_alive(self, websocket, instance_id: str):
        """Keep WebSocket connection alive
        
        Args:
            websocket: WebSocket connection
            instance_id: Instance ID
        """
        connection_key = f"{instance_id}"
        logger.info(f"Starting heartbeat task: {connection_key}")
        
        retry_count = 0
        last_success_time = time.time()
        
        try:
            while True:
                try:
                    # Send heartbeat at configured interval
                    await asyncio.sleep(self.heartbeat_config.interval)
                    
                    # Check last successful heartbeat time
                    if time.time() - last_success_time > self.heartbeat_config.interval * 2:
                        logger.warning(f"Upload (ID: {instance_id}) heartbeat timeout")
                        raise websockets.exceptions.ConnectionClosed(1006, "Heartbeat timeout")
                    
                    success = await self.send_heartbeat(websocket)
                    if success:
                        retry_count = 0  # Reset retry count
                        last_success_time = time.time()
                        # logger.info(f"Upload (ID: {instance_id}) heartbeat sent")
                    else:
                        retry_count += 1
                        if retry_count >= self.heartbeat_config.max_retries:
                            logger.error(f"Upload (ID: {instance_id}) heartbeat retry count exceeded")
                            raise websockets.exceptions.ConnectionClosed(1006, "Heartbeat retry count exceeded")
                        logger.warning(f"Upload (ID: {instance_id}) heartbeat send failed, retrying {retry_count} times")
                        await asyncio.sleep(self.heartbeat_config.retry_interval)
                        continue
                        
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(f"Upload (ID: {instance_id}) WebSocket connection closed: {str(e)}")
                    # Clean up connection
                    if connection_key in self.active_connections:
                        del self.active_connections[connection_key]
                    # Cancel related tasks
                    if hasattr(websocket, 'message_task'):
                        websocket.message_task.cancel()
                    break
                    
                except Exception as e:
                    logger.error(f"Upload (ID: {instance_id}) send heartbeat failed: {str(e)}", exc_info=True)
                    retry_count += 1
                    if retry_count >= self.heartbeat_config.max_retries:
                        logger.error(f"Upload (ID: {instance_id}) heartbeat retry count exceeded")
                        raise
                    await asyncio.sleep(self.heartbeat_config.retry_interval)
                    
        except asyncio.CancelledError:
            logger.info(f"Heartbeat task cancelled: {connection_key}")
            raise
        except Exception as e:
            logger.error(f"Upload (ID: {instance_id}) keep alive task failed: {str(e)}")
            # Clean up connection
            if connection_key in self.active_connections:
                del self.active_connections[connection_key]
            raise

    async def _keep_alive_with_ping(self, websocket, instance_id: str):
        """Keep WebSocket connection alive using native ping/pong
        
        Args:
            websocket: WebSocket connection
            instance_id: Instance ID
        """
        connection_key = f"{instance_id}"
        logger.info(f"Starting ping task: {connection_key}")
        
        try:
            while True:
                try:
                    await asyncio.sleep(self.heartbeat_config.interval)
                    await websocket.ping()
                    # logger.debug(f"Ping sent successfully for {instance_id}")
                    
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(f"Upload (ID: {instance_id}) WebSocket connection closed: {str(e)}")
                    if connection_key in self.active_connections:
                        del self.active_connections[connection_key]
                    if hasattr(websocket, 'message_task'):
                        websocket.message_task.cancel()
                    break
                    
                except Exception as e:
                    logger.error(f"Upload (ID: {instance_id}) ping failed: {str(e)}")
                    if connection_key in self.active_connections:
                        del self.active_connections[connection_key]
                    raise
                    
        except asyncio.CancelledError:
            logger.info(f"Ping task cancelled: {connection_key}")
            raise
        except Exception as e:
            logger.error(f"Upload (ID: {instance_id}) keep alive task failed: {str(e)}")
            if connection_key in self.active_connections:
                del self.active_connections[connection_key]
            raise

    async def send_heartbeat(self, websocket):
        """Send heartbeat message
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            bool: Whether heartbeat was sent successfully
        """
        try:
            heartbeat_message = json.dumps({
                "type": "heartbeat",
                "data": {
                    "timestamp": int(time.time()),
                    "instance_id": websocket.instance_id if hasattr(websocket, 'instance_id') else 'unknown',
                    "status": "alive"
                },
                "version": "1.0"
            })
            # logger.info(f"Preparing to send heartbeat message: {heartbeat_message}")
            
            # Set send timeout
            async with asyncio.timeout(self.heartbeat_config.timeout):
                await websocket.send(heartbeat_message)
                # logger.info("Heartbeat message sent successfully")
                return True
                
        except asyncio.TimeoutError:
            logger.error("Sending heartbeat message timed out")
            return False
        except Exception as e:
            logger.error(f"Sending heartbeat failed: {str(e)}", exc_info=True)
            return False

    async def handle_messages(self, websocket):
        """Handle received WebSocket messages with intelligent routing"""
        try:
            while True:
                try:
                    # Use lock to ensure only one coroutine calls recv at a time
                    async with websocket.recv_lock:
                        message = await websocket.recv()
                        data = json.loads(message)
                        message_type = data.get("type")

                    if message_type == "heartbeat_ack":
                        continue
                    elif message_type == "chat":
                        # Handle chat request with intelligent routing
                        try:
                            request_data = data.get("request", {})
                            logger.info(f"[Request details: {json.dumps(request_data, ensure_ascii=False)}")
                            
                            # 1. 检查当前服务状态
                            service_type, model_data = self._get_current_service_type()
                            logger.info(f"Current service type: {service_type}")
                            
                            # 2. 根据服务类型选择不同的处理路径
                            if service_type == "cloud":
                                # 使用云服务推理接口
                                await self._handle_cloud_inference(websocket, data, request_data, model_data)
                            else:
                                # 使用本地聊天接口（默认）
                                await self._handle_local_chat(websocket, data, request_data)

                        except Exception as e:
                            logger.error(f"Failed to process chat request: {str(e)}")
                            await websocket.send(json.dumps({
                                "type": "chat_response",
                                "request_id": data.get("request_id"),
                                "error": f"Error processing chat request: {str(e)}"
                            }))
                    else:
                        logger.debug(f"Received unknown message type: {message}")
                except websockets.exceptions.ConnectionClosed:
                    logger.error("WebSocket connection closed")
                    break
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON message: {message}")
                except Exception as e:
                    logger.error(f"Failed to process message: {str(e)}")
        except Exception as e:
            logger.error(f"Message processing loop failed: {str(e)}")
            raise

    async def _handle_local_chat(self, websocket, data, request_data):
        """Handle chat request using local chat interface"""
        async with aiohttp.ClientSession() as session:
            logger.info(f"Routing to local chat interface")
            config = Config.from_env()
            kernel2_url = f"{config.KERNEL2_SERVICE_URL}/api/kernel2/chat"
            
            async with session.post(
                kernel2_url,
                json=request_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                },
                timeout=aiohttp.ClientTimeout(total=None),
                chunked=True
            ) as response:
                # Check response status
                logger.info(f"Local chat response status: {response.status}")
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"[request_id: {data.get('request_id')}] Failed to call local chat interface: {error_text}")
                    await websocket.send(json.dumps({
                        "type": "chat_response",
                        "request_id": data.get("request_id"),
                        "error": f"Failed to call local chat interface: {error_text}"
                    }))
                    return

                logger.debug(f"Starting to read local chat streaming response")
                message_count = 0
                
                # Direct forwarding of streaming response
                async for line in response.content:
                    if line:
                        try:
                            # Convert bytes to string
                            decoded_line = line.decode('utf-8')
                            logger.debug(f"[request_id: {data.get('request_id')}] Received raw data: {decoded_line.strip()}")
                            
                            # Check if it's SSE format data
                            if decoded_line.startswith("data: "):
                                message_count += 1
                                data_content = decoded_line[6:].strip()
                                
                                # Check if it's a completion marker
                                if data_content == "[DONE]":
                                    logger.info(f"[request_id: {data.get('request_id')}] Local chat completed, processed {message_count} messages")
                                    await websocket.send(json.dumps({
                                        "type": "chat_response",
                                        "request_id": data.get("request_id"),
                                        "done": True
                                    }))
                                    continue
                                
                                # Directly forward original SSE data
                                await websocket.send(json.dumps({
                                    "type": "chat_response",
                                    "request_id": data.get("request_id"),
                                    "raw_sse": data_content,
                                    "done": False
                                }))
                                logger.debug(f"[requestId: {data.get('request_id')}] Forwarded local chat SSE message #{message_count}")
                        except UnicodeDecodeError as e:
                            logger.error(f"[requestId: {data.get('request_id')}] Failed to decode local chat response: {str(e)}")
                        except Exception as e:
                            logger.error(f"[requestId: {data.get('request_id')}] Error processing local chat response: {str(e)}")

    async def _handle_cloud_inference(self, websocket, data, request_data, model_data):
        """Handle chat request using cloud inference interface"""
        # Transform request for cloud inference API
        cloud_request = self._transform_request_for_cloud(request_data, model_data)
        
        async with aiohttp.ClientSession() as session:
            logger.info(f"Routing to cloud inference interface with model: {model_data.get('model_id', 'unknown')}")
            config = Config.from_env()
            cloud_url = f"{config.KERNEL2_SERVICE_URL}/api/cloud_service/train/inference"
            
            async with session.post(
                cloud_url,
                json=cloud_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                },
                timeout=aiohttp.ClientTimeout(total=None),
                chunked=True
            ) as response:
                # Check response status
                logger.info(f"Cloud inference response status: {response.status}")
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"[request_id: {data.get('request_id')}] Failed to call cloud inference interface: {error_text}")
                    await websocket.send(json.dumps({
                        "type": "chat_response",
                        "request_id": data.get("request_id"),
                        "error": f"Failed to call cloud inference interface: {error_text}"
                    }))
                    return

                logger.debug(f"Starting to read cloud inference streaming response")
                message_count = 0
                
                # Direct forwarding of streaming response
                async for line in response.content:
                    if line:
                        try:
                            # Convert bytes to string
                            decoded_line = line.decode('utf-8')
                            logger.debug(f"[request_id: {data.get('request_id')}] Received cloud raw data: {decoded_line.strip()}")
                            
                            # Check if it's SSE format data
                            if decoded_line.startswith("data: "):
                                message_count += 1
                                data_content = decoded_line[6:].strip()
                                
                                # Check if it's a completion marker
                                if data_content == "[DONE]":
                                    logger.info(f"[request_id: {data.get('request_id')}] Cloud inference completed, processed {message_count} messages")
                                    await websocket.send(json.dumps({
                                        "type": "chat_response",
                                        "request_id": data.get("request_id"),
                                        "done": True
                                    }))
                                    continue
                                
                                # Directly forward original SSE data
                                await websocket.send(json.dumps({
                                    "type": "chat_response",
                                    "request_id": data.get("request_id"),
                                    "raw_sse": data_content,
                                    "done": False
                                }))
                                logger.debug(f"[requestId: {data.get('request_id')}] Forwarded cloud inference SSE message #{message_count}")
                        except UnicodeDecodeError as e:
                            logger.error(f"[requestId: {data.get('request_id')}] Failed to decode cloud inference response: {str(e)}")
                        except Exception as e:
                            logger.error(f"[requestId: {data.get('request_id')}] Error processing cloud inference response: {str(e)}")

    def list_uploads(self, page_no: int = 1, page_size: int = 10, status: Optional[List[str]] = None):
        """Get list of registered Upload instances with pagination and status filter
        
        Args:
            page_no (int): Page number, starting from 1
            page_size (int): Number of items per page
            status (Optional[List[str]]): List of status to filter by
            
        Returns:
            dict: Dictionary containing information about Upload instances
        """
        # headers = self._get_auth_header()
        params = {
            "page_no": page_no,
            "page_size": page_size
        }
        if status:
            params["status"] = status
            
        response = requests.get(
            f"{self.server_url}/api/upload/list",
            # headers=headers,
            params=params
        )
        return ResponseHandler.handle_response(
            response,
            error_prefix="Failed to retrieve list"
        )

    def count_uploads(self):
        """Get count of all registered Upload instances
        
        Returns:
            dict: Dictionary containing count of Upload instances
        """
        response = requests.get(
            f"{self.server_url}/api/upload/count",
        )
        return ResponseHandler.handle_response(
            response,
            error_prefix="Failed to retrieve count"
        )

    def get_upload_detail(self, instance_id: str) -> Dict:
        """Get detailed information of an Upload instance
        
        Args:
            instance_id (str): Instance ID of the Upload
            
        Returns:
            dict: Dictionary containing instance information with the following fields:
                upload_name (str): Name of the upload
                instance_id (str): Instance ID
                status (str): Current status of the upload
                description (str, optional): Description of the upload
                email (str, optional): Associated email address
                registration_time (datetime): Time when the instance was registered
                last_heartbeat (datetime, optional): Time of the last heartbeat
                is_connected (bool, optional): Connection status, defaults to False
                instance_password (str, optional): Password for instance registration
        """
        headers = self._get_auth_header()
        response = requests.get(
            f"{self.server_url}/api/upload/{instance_id}",
            headers=headers
        )
        return ResponseHandler.handle_response(
            response,
            error_prefix="Failed to retrieve upload details"
        )

    def update_upload(self, instance_id: str, upload_name: str = None, capabilities: dict = None, email: str = None):
        """Update Upload instance information in the registry center
        
        Args:
            instance_id: Instance ID
            upload_name: New upload name (optional)
            capabilities: New capability set (optional)
            email: New user email (optional)
            
        Returns:
            dict: Update result
        """
        update_data = {}
        if upload_name is not None:
            update_data["upload_name"] = upload_name
        if capabilities is not None:
            update_data["capabilities"] = capabilities
        if email is not None:
            update_data["email"] = email
            
        if not update_data:
            logger.warning("No update data provided for update_upload")
            return {"message": "No update data provided"}
        
        headers = self._get_auth_header()
        response = requests.put(
            f"{self.server_url}/api/upload/{instance_id}",
            headers=headers,
            json=update_data
        )
        return ResponseHandler.handle_response(
            response,
            success_log=f"Upload instance {instance_id} updated successfully",
            error_prefix="Update"
        )

    def create_role(self, role_id, name, description, system_prompt, icon, instance_id, is_active=True,
                   enable_l0_retrieval=True, enable_l1_retrieval=True):
        """Create a new role in the registry center
        
        Args:
            role_id: Role UUID
            name: Role name
            description: Role description
            system_prompt: System prompt
            icon: Icon URL
            instance_id: Instance ID
            enable_l0_retrieval: Enable L0 retrieval
            enable_l1_retrieval: Enable L1 retrieval
            
        Returns:
            dict: Created role data
        """
        headers = self._get_auth_header()
        response = requests.post(
            f"{self.server_url}/api/roles",
            headers=headers,
            json={
                "role_id": role_id,
                "instance_id": instance_id,
                "name": name,
                "description": description,
                "system_prompt": system_prompt,
                "is_active": is_active,
                "icon": icon,
                "enable_l0_retrieval": enable_l0_retrieval,
                "enable_l1_retrieval": enable_l1_retrieval
            }
        )
        return ResponseHandler.handle_response(
            response,
            success_log=f"Role {name} created successfully in registry center",
            error_prefix="Role creation"
        )