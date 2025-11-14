"""
Autodesk Flame PyBox Handler for ComfyUI Integration
Processes Flame clips through ComfyUI workflows via WebSocket
"""

import os
import sys
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

# Flame PyBox imports (these will be available in Flame environment)
try:
    import flame
    FLAME_AVAILABLE = True
except ImportError:
    FLAME_AVAILABLE = False
    print("⚠ Flame module not available - running in standalone mode")

# Import our ComfyUI client
from comfyui_client import ComfyUIClient, ComfyUIConfig


class FlameComfyUIProcessor:
    """
    Main processor class for Flame <-> ComfyUI integration
    Handles frame extraction, workflow execution, and result integration
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Flame-ComfyUI processor

        Args:
            config_path: Path to configuration JSON file
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # ComfyUI client
        self.comfy_config = ComfyUIConfig(
            host=self.config.get('comfyui_host', '127.0.0.1'),
            port=self.config.get('comfyui_port', 8188)
        )
        self.client = ComfyUIClient(self.comfy_config)

        # Workflow library
        self.workflows_dir = Path(self.config.get('workflows_dir', './workflows'))
        self.available_workflows = self._scan_workflows()

        # Processing state
        self.current_clip = None
        self.current_workflow = None
        self.processing = False
        self.progress = 0.0

        print(f"✓ FlameComfyUIProcessor initialized")
        print(f"  - ComfyUI: {self.comfy_config.server_address}")
        print(f"  - Workflows: {len(self.available_workflows)} available")

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            'comfyui_host': '127.0.0.1',
            'comfyui_port': 8188,
            'workflows_dir': './workflows',
            'temp_dir': '/tmp/flame_comfyui',
            'default_resolution': [1920, 1080],
            'color_space': 'sRGB',
            'bit_depth': 8
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                print(f"✓ Loaded config from {config_path}")
            except Exception as e:
                print(f"⚠ Error loading config: {e}, using defaults")

        return default_config

    def _scan_workflows(self) -> Dict[str, str]:
        """Scan workflows directory for available workflows"""
        workflows = {}

        if not self.workflows_dir.exists():
            print(f"⚠ Workflows directory not found: {self.workflows_dir}")
            return workflows

        for workflow_file in self.workflows_dir.glob('*.json'):
            workflow_name = workflow_file.stem
            workflows[workflow_name] = str(workflow_file)
            print(f"  ✓ Found workflow: {workflow_name}")

        return workflows

    def get_workflow_list(self) -> List[str]:
        """Get list of available workflow names"""
        return list(self.available_workflows.keys())

    def load_workflow(self, workflow_name: str) -> Optional[Dict]:
        """Load a workflow by name"""
        if workflow_name not in self.available_workflows:
            print(f"✗ Workflow '{workflow_name}' not found")
            return None

        workflow_path = self.available_workflows[workflow_name]
        return self.client.load_workflow(workflow_path)

    def extract_frame_from_clip(self, clip, frame_number: int) -> Optional[np.ndarray]:
        """
        Extract a single frame from Flame clip

        Args:
            clip: Flame clip object
            frame_number: Frame number to extract

        Returns:
            Frame as numpy array (BGR format) or None
        """
        if not FLAME_AVAILABLE:
            print("✗ Flame not available")
            return None

        try:
            # Get frame dimensions
            width = clip.width
            height = clip.height

            # Read frame from clip
            # Flame uses RGBA format, need to convert to BGR
            frame_data = clip.get_frame(frame_number)

            # Convert to numpy array
            # Flame returns flat array, reshape to (height, width, channels)
            frame_rgba = np.frombuffer(frame_data, dtype=np.uint8)
            frame_rgba = frame_rgba.reshape((height, width, 4))

            # Convert RGBA to BGR (OpenCV format)
            frame_bgr = frame_rgba[:, :, [2, 1, 0]]  # Swap R and B channels

            return frame_bgr

        except Exception as e:
            print(f"✗ Error extracting frame: {e}")
            return None

    def extract_frames_from_clip(self, clip, frame_range: Optional[tuple] = None) -> List[np.ndarray]:
        """
        Extract multiple frames from Flame clip

        Args:
            clip: Flame clip object
            frame_range: Optional (start, end) tuple, defaults to full clip

        Returns:
            List of frames as numpy arrays
        """
        if not FLAME_AVAILABLE:
            print("✗ Flame not available")
            return []

        try:
            if frame_range is None:
                start_frame = clip.start_frame
                end_frame = clip.end_frame
            else:
                start_frame, end_frame = frame_range

            frames = []
            total_frames = end_frame - start_frame + 1

            for i, frame_num in enumerate(range(start_frame, end_frame + 1)):
                frame = self.extract_frame_from_clip(clip, frame_num)
                if frame is not None:
                    frames.append(frame)

                # Progress update
                progress = (i + 1) / total_frames * 100
                print(f"⚙ Extracting frames: {progress:.1f}%", end='\r')

            print()  # New line after progress
            print(f"✓ Extracted {len(frames)} frames")
            return frames

        except Exception as e:
            print(f"✗ Error extracting frames: {e}")
            return []

    def inject_frame_to_clip(self, clip, frame_data: np.ndarray, frame_number: int) -> bool:
        """
        Inject processed frame back to Flame clip

        Args:
            clip: Flame clip object
            frame_data: Processed frame (BGR format)
            frame_number: Target frame number

        Returns:
            True if successful
        """
        if not FLAME_AVAILABLE:
            print("✗ Flame not available")
            return False

        try:
            # Convert BGR to RGBA for Flame
            height, width = frame_data.shape[:2]

            # Add alpha channel
            frame_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            frame_rgba[:, :, [2, 1, 0]] = frame_data  # Swap BGR to RGB
            frame_rgba[:, :, 3] = 255  # Full opacity

            # Write frame to clip
            clip.set_frame(frame_number, frame_rgba.tobytes())

            return True

        except Exception as e:
            print(f"✗ Error injecting frame: {e}")
            return False

    async def process_clip_async(
        self,
        clip,
        workflow_name: str,
        frame_range: Optional[tuple] = None,
        workflow_params: Optional[Dict] = None,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Process Flame clip through ComfyUI workflow (async)

        Args:
            clip: Flame clip object
            workflow_name: Name of workflow to use
            frame_range: Optional frame range (start, end)
            workflow_params: Optional parameters to update in workflow
            progress_callback: Optional progress callback function

        Returns:
            True if processing successful
        """
        self.processing = True
        self.current_clip = clip
        self.progress = 0.0

        try:
            # Load workflow
            workflow = self.load_workflow(workflow_name)
            if not workflow:
                return False

            # Update workflow parameters if provided
            if workflow_params:
                workflow = self.client.update_workflow_params(workflow, workflow_params)

            # Connect to ComfyUI
            if not self.client.is_connected:
                await self.client.connect()

            # Extract frames
            print(f"⚙ Extracting frames from clip...")
            frames = self.extract_frames_from_clip(clip, frame_range)
            if not frames:
                return False

            self.progress = 10.0
            if progress_callback:
                progress_callback(self.progress)

            # Process each frame
            total_frames = len(frames)
            processed_frames = []

            for i, frame in enumerate(frames):
                print(f"\n⚙ Processing frame {i+1}/{total_frames}")

                # Prepare input images
                input_images = {"input_image": frame}

                # Execute workflow
                def exec_progress(status, current, total):
                    frame_progress = (current / total) * 80  # 80% of progress bar
                    total_progress = 10 + (i / total_frames) * 80 + (frame_progress / total_frames)
                    self.progress = total_progress

                    if progress_callback:
                        progress_callback(self.progress)

                    print(f"  {status}: {current}/{total}")

                results = await self.client.execute_workflow(
                    workflow,
                    images=input_images,
                    progress_callback=exec_progress
                )

                if results:
                    processed_frames.append(results[0])  # Take first output
                    print(f"✓ Frame {i+1} processed")
                else:
                    print(f"✗ Frame {i+1} failed")
                    processed_frames.append(frame)  # Keep original on failure

            # Inject processed frames back to clip
            print(f"\n⚙ Injecting processed frames...")
            if frame_range:
                start_frame = frame_range[0]
            else:
                start_frame = clip.start_frame if FLAME_AVAILABLE else 0

            for i, processed_frame in enumerate(processed_frames):
                frame_num = start_frame + i
                if FLAME_AVAILABLE:
                    success = self.inject_frame_to_clip(clip, processed_frame, frame_num)
                    if not success:
                        print(f"⚠ Failed to inject frame {frame_num}")

            self.progress = 100.0
            if progress_callback:
                progress_callback(self.progress)

            print(f"\n✓ Processing complete!")
            return True

        except Exception as e:
            print(f"✗ Error during processing: {e}")
            return False

        finally:
            self.processing = False

    def process_clip(
        self,
        clip,
        workflow_name: str,
        frame_range: Optional[tuple] = None,
        workflow_params: Optional[Dict] = None,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Process Flame clip through ComfyUI workflow (synchronous wrapper)
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                self.process_clip_async(
                    clip,
                    workflow_name,
                    frame_range,
                    workflow_params,
                    progress_callback
                )
            )
            return result
        finally:
            loop.close()

    def process_single_frame(
        self,
        frame: np.ndarray,
        workflow_name: str,
        workflow_params: Optional[Dict] = None
    ) -> Optional[np.ndarray]:
        """
        Process a single frame through ComfyUI workflow

        Args:
            frame: Input frame as numpy array
            workflow_name: Workflow to use
            workflow_params: Optional workflow parameters

        Returns:
            Processed frame or None
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def process():
            # Load workflow
            workflow = self.load_workflow(workflow_name)
            if not workflow:
                return None

            # Update parameters
            if workflow_params:
                workflow = self.client.update_workflow_params(workflow, workflow_params)

            # Connect
            if not self.client.is_connected:
                await self.client.connect()

            # Execute
            results = await self.client.execute_workflow(
                workflow,
                images={"input_image": frame}
            )

            return results[0] if results else None

        try:
            result = loop.run_until_complete(process())
            return result
        finally:
            loop.close()


# PyBox Handler Functions (called by Flame)
def setup_handler(handler):
    """
    Setup function called by Flame when PyBox is initialized

    Args:
        handler: Flame PyBox handler object
    """
    print("=" * 60)
    print("ComfyUI PyBox Handler - Initializing")
    print("=" * 60)

    # Create processor instance
    config_path = os.path.join(os.path.dirname(__file__), 'comfyui_config.json')
    handler.processor = FlameComfyUIProcessor(config_path)

    # Setup handler parameters
    handler.add_parameter({
        'name': 'workflow',
        'type': 'menu',
        'label': 'Workflow',
        'values': handler.processor.get_workflow_list(),
        'default': handler.processor.get_workflow_list()[0] if handler.processor.get_workflow_list() else 'none'
    })

    handler.add_parameter({
        'name': 'process_mode',
        'type': 'menu',
        'label': 'Process Mode',
        'values': ['Current Frame', 'Frame Range', 'All Frames'],
        'default': 'Current Frame'
    })

    handler.add_parameter({
        'name': 'start_frame',
        'type': 'int',
        'label': 'Start Frame',
        'default': 1,
        'min': 1
    })

    handler.add_parameter({
        'name': 'end_frame',
        'type': 'int',
        'label': 'End Frame',
        'default': 100,
        'min': 1
    })

    print("✓ Handler setup complete")


def process_frame(handler, frame, current_frame_num):
    """
    Process function called by Flame for each frame

    Args:
        handler: Flame PyBox handler object
        frame: Input frame data
        current_frame_num: Current frame number

    Returns:
        Processed frame data
    """
    # Get parameters
    workflow_name = handler.get_parameter_value('workflow')
    process_mode = handler.get_parameter_value('process_mode')

    # Convert frame to numpy array
    height, width = handler.height, handler.width
    frame_array = np.frombuffer(frame, dtype=np.float32)
    frame_array = frame_array.reshape((height, width, 4))

    # Convert to uint8 BGR
    frame_bgr = (frame_array[:, :, [2, 1, 0]] * 255).astype(np.uint8)

    # Process based on mode
    if process_mode == 'Current Frame':
        # Process single frame
        result = handler.processor.process_single_frame(frame_bgr, workflow_name)

        if result is not None:
            # Convert back to RGBA float32
            result_rgba = np.zeros((height, width, 4), dtype=np.float32)
            result_rgba[:, :, [2, 1, 0]] = result.astype(np.float32) / 255.0
            result_rgba[:, :, 3] = 1.0

            return result_rgba.tobytes()

    # Return original frame if processing failed
    return frame


def cleanup_handler(handler):
    """Cleanup function called when PyBox is destroyed"""
    print("=" * 60)
    print("ComfyUI PyBox Handler - Cleanup")
    print("=" * 60)


# Standalone test mode
if __name__ == "__main__":
    print("=" * 60)
    print("Flame ComfyUI PyBox - Standalone Test Mode")
    print("=" * 60)

    # Create test processor
    processor = FlameComfyUIProcessor()

    # List available workflows
    print(f"\nAvailable workflows: {processor.get_workflow_list()}")

    # Test with a sample image if available
    import cv2

    test_image_path = "test_image.jpg"
    if os.path.exists(test_image_path):
        print(f"\nTesting with {test_image_path}")

        # Load test image
        test_frame = cv2.imread(test_image_path)

        if test_frame is not None and processor.get_workflow_list():
            # Process with first available workflow
            workflow_name = processor.get_workflow_list()[0]
            print(f"Processing with workflow: {workflow_name}")

            result = processor.process_single_frame(test_frame, workflow_name)

            if result is not None:
                # Save result
                output_path = "test_output.jpg"
                cv2.imwrite(output_path, result)
                print(f"✓ Result saved to {output_path}")
            else:
                print("✗ Processing failed")
    else:
        print(f"\n⚠ Test image not found: {test_image_path}")
        print("  Create a test image to run standalone test")
