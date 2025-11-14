"""
ComfyUI Client Module - WebSocket & REST API Integration
Handles communication with ComfyUI server for workflow execution
"""

import json
import uuid
import time
import base64
import asyncio
import websockets
import requests
from io import BytesIO
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class ComfyUIConfig:
    """Configuration for ComfyUI connection"""
    host: str = "127.0.0.1"
    port: int = 8188
    client_id: str = None
    timeout: int = 300

    def __post_init__(self):
        if self.client_id is None:
            self.client_id = str(uuid.uuid4())

    @property
    def server_address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def http_url(self) -> str:
        return f"http://{self.server_address}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.server_address}/ws?clientId={self.client_id}"


class ComfyUIClient:
    """
    Complete ComfyUI client with WebSocket and REST API support
    Handles workflow execution, progress tracking, and result retrieval
    """

    def __init__(self, config: Optional[ComfyUIConfig] = None):
        self.config = config or ComfyUIConfig()
        self.websocket = None
        self.execution_callbacks: Dict[str, Callable] = {}
        self.is_connected = False

    async def connect(self):
        """Establish WebSocket connection to ComfyUI server"""
        try:
            self.websocket = await websockets.connect(
                self.config.ws_url,
                ping_interval=30,
                ping_timeout=10,
                max_size=10**8  # 100MB max message size
            )
            self.is_connected = True
            print(f"✓ Connected to ComfyUI at {self.config.server_address}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to ComfyUI: {e}")
            self.is_connected = False
            return False

    async def disconnect(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            print("✓ Disconnected from ComfyUI")

    def queue_prompt(self, workflow: Dict[str, Any], images: Optional[Dict[str, np.ndarray]] = None) -> Optional[str]:
        """
        Queue a workflow for execution on ComfyUI server

        Args:
            workflow: ComfyUI workflow JSON
            images: Optional dict of image name -> numpy array for input images

        Returns:
            prompt_id if successful, None otherwise
        """
        try:
            # Upload input images if provided
            if images:
                for name, image_data in images.items():
                    self._upload_image(name, image_data)

            # Prepare prompt request
            prompt_data = {
                "prompt": workflow,
                "client_id": self.config.client_id
            }

            # Send POST request to queue prompt
            response = requests.post(
                f"{self.config.http_url}/prompt",
                json=prompt_data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                prompt_id = result.get("prompt_id")
                print(f"✓ Workflow queued with ID: {prompt_id}")
                return prompt_id
            else:
                error_data = response.json()
                print(f"✗ Failed to queue workflow: {error_data}")
                return None

        except Exception as e:
            print(f"✗ Error queuing workflow: {e}")
            return None

    def _upload_image(self, name: str, image_data: np.ndarray):
        """Upload an image to ComfyUI server"""
        try:
            # Convert numpy array to bytes
            is_success, buffer = cv2.imencode('.png', image_data)
            if not is_success:
                raise ValueError("Failed to encode image")

            # Prepare multipart form data
            files = {
                'image': (f'{name}.png', buffer.tobytes(), 'image/png')
            }
            data = {
                'overwrite': 'true'
            }

            response = requests.post(
                f"{self.config.http_url}/upload/image",
                files=files,
                data=data,
                timeout=30
            )

            if response.status_code == 200:
                print(f"✓ Image '{name}' uploaded successfully")
            else:
                print(f"✗ Failed to upload image '{name}': {response.text}")

        except Exception as e:
            print(f"✗ Error uploading image '{name}': {e}")

    async def wait_for_execution(
        self,
        prompt_id: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        node_callback: Optional[Callable[[str, Dict], None]] = None
    ) -> bool:
        """
        Wait for workflow execution to complete via WebSocket

        Args:
            prompt_id: The prompt ID to wait for
            progress_callback: Optional callback(status, current, total)
            node_callback: Optional callback(node_name, output_data)

        Returns:
            True if execution completed successfully
        """
        if not self.is_connected:
            print("✗ Not connected to ComfyUI WebSocket")
            return False

        try:
            start_time = time.time()

            while True:
                # Check timeout
                if time.time() - start_time > self.config.timeout:
                    print(f"✗ Execution timeout after {self.config.timeout}s")
                    return False

                # Receive WebSocket message
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=30
                )

                data = json.loads(message)
                msg_type = data.get('type')

                # Handle different message types
                if msg_type == 'execution_start':
                    print(f"⚙ Execution started for prompt {prompt_id}")
                    if progress_callback:
                        progress_callback("started", 0, 100)

                elif msg_type == 'executing':
                    node_data = data.get('data', {})
                    node_name = node_data.get('node')

                    if node_name:
                        print(f"⚙ Executing node: {node_name}")
                        if node_callback:
                            node_callback(node_name, node_data)
                    else:
                        # Execution completed
                        print(f"✓ Execution completed for prompt {prompt_id}")
                        if progress_callback:
                            progress_callback("completed", 100, 100)
                        return True

                elif msg_type == 'progress':
                    prog_data = data.get('data', {})
                    current = prog_data.get('value', 0)
                    total = prog_data.get('max', 100)

                    if progress_callback:
                        progress_callback("progress", current, total)

                elif msg_type == 'execution_error':
                    error_data = data.get('data', {})
                    print(f"✗ Execution error: {error_data}")
                    if progress_callback:
                        progress_callback("error", 0, 100)
                    return False

                elif msg_type == 'execution_cached':
                    print(f"⚡ Some nodes were cached")

        except asyncio.TimeoutError:
            print(f"✗ WebSocket timeout waiting for execution")
            return False
        except Exception as e:
            print(f"✗ Error during execution: {e}")
            return False

    def get_history(self, prompt_id: str) -> Optional[Dict]:
        """Retrieve execution history and results for a prompt"""
        try:
            response = requests.get(
                f"{self.config.http_url}/history/{prompt_id}",
                timeout=30
            )

            if response.status_code == 200:
                history = response.json()
                return history.get(prompt_id, {})
            else:
                print(f"✗ Failed to get history: {response.status_code}")
                return None

        except Exception as e:
            print(f"✗ Error getting history: {e}")
            return None

    def get_images(self, prompt_id: str) -> List[np.ndarray]:
        """
        Retrieve all output images from execution

        Returns:
            List of images as numpy arrays (BGR format for OpenCV)
        """
        history = self.get_history(prompt_id)
        if not history:
            return []

        images = []
        outputs = history.get('outputs', {})

        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                for image_info in node_output['images']:
                    filename = image_info['filename']
                    subfolder = image_info.get('subfolder', '')
                    folder_type = image_info.get('type', 'output')

                    # Download image
                    image_data = self._download_image(filename, subfolder, folder_type)
                    if image_data is not None:
                        images.append(image_data)

        return images

    def _download_image(self, filename: str, subfolder: str = '', folder_type: str = 'output') -> Optional[np.ndarray]:
        """Download an image from ComfyUI server"""
        try:
            params = {
                'filename': filename,
                'subfolder': subfolder,
                'type': folder_type
            }

            response = requests.get(
                f"{self.config.http_url}/view",
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                # Convert bytes to numpy array
                image_array = np.frombuffer(response.content, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                return image
            else:
                print(f"✗ Failed to download image '{filename}': {response.status_code}")
                return None

        except Exception as e:
            print(f"✗ Error downloading image '{filename}': {e}")
            return None

    async def execute_workflow(
        self,
        workflow: Dict[str, Any],
        images: Optional[Dict[str, np.ndarray]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> List[np.ndarray]:
        """
        Complete workflow execution: queue, wait, and retrieve results

        Args:
            workflow: ComfyUI workflow JSON
            images: Optional input images
            progress_callback: Optional progress callback

        Returns:
            List of output images as numpy arrays
        """
        # Ensure connection
        if not self.is_connected:
            await self.connect()

        # Queue workflow
        prompt_id = self.queue_prompt(workflow, images)
        if not prompt_id:
            return []

        # Wait for execution
        success = await self.wait_for_execution(prompt_id, progress_callback)
        if not success:
            return []

        # Retrieve results
        return self.get_images(prompt_id)

    def load_workflow(self, workflow_path: str) -> Optional[Dict]:
        """Load a workflow JSON file"""
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            print(f"✓ Loaded workflow from {workflow_path}")
            return workflow
        except Exception as e:
            print(f"✗ Error loading workflow: {e}")
            return None

    def update_workflow_params(self, workflow: Dict, params: Dict[str, Any]) -> Dict:
        """
        Update workflow parameters

        Args:
            workflow: Original workflow
            params: Dictionary of node_id -> {param_name: value}

        Returns:
            Updated workflow
        """
        workflow_copy = json.loads(json.dumps(workflow))  # Deep copy

        for node_id, node_params in params.items():
            if node_id in workflow_copy:
                for param_name, param_value in node_params.items():
                    if 'inputs' in workflow_copy[node_id]:
                        workflow_copy[node_id]['inputs'][param_name] = param_value

        return workflow_copy

    def get_system_stats(self) -> Optional[Dict]:
        """Get ComfyUI system statistics"""
        try:
            response = requests.get(
                f"{self.config.http_url}/system_stats",
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None

    def interrupt_execution(self):
        """Interrupt current execution"""
        try:
            response = requests.post(
                f"{self.config.http_url}/interrupt",
                timeout=10
            )
            return response.status_code == 200
        except:
            return False


# Synchronous wrapper for easier use
class ComfyUIClientSync:
    """Synchronous wrapper around ComfyUIClient for easier integration"""

    def __init__(self, config: Optional[ComfyUIConfig] = None):
        self.client = ComfyUIClient(config)
        self.loop = None

    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.client.connect())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.loop:
            self.loop.run_until_complete(self.client.disconnect())
            self.loop.close()

    def execute_workflow(
        self,
        workflow: Dict[str, Any],
        images: Optional[Dict[str, np.ndarray]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> List[np.ndarray]:
        """Execute workflow synchronously"""
        if not self.loop:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        return self.loop.run_until_complete(
            self.client.execute_workflow(workflow, images, progress_callback)
        )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create client
        config = ComfyUIConfig(host="127.0.0.1", port=8188)
        client = ComfyUIClient(config)

        # Connect
        await client.connect()

        # Load workflow
        workflow = client.load_workflow("workflows/upscale_4k.json")

        if workflow:
            # Execute
            def progress_cb(status, current, total):
                print(f"Progress: {status} - {current}/{total}")

            results = await client.execute_workflow(
                workflow,
                progress_callback=progress_cb
            )

            print(f"Received {len(results)} output images")

        # Disconnect
        await client.disconnect()

    # Run example
    asyncio.run(main())
