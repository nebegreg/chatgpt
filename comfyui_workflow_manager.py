"""
ComfyUI Workflow Manager - GUI for selecting and managing workflows
Integrates with Autodesk Flame and the Magic Wand application
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QListWidget, QTextEdit,
    QProgressBar, QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFileDialog, QMessageBox, QTabWidget, QSplitter, QLineEdit,
    QScrollArea, QListWidgetItem
)

from comfyui_client import ComfyUIClient, ComfyUIConfig


class WorkflowExecutionThread(QThread):
    """Thread for executing ComfyUI workflows asynchronously"""

    progress_updated = pyqtSignal(str, int, int)
    execution_finished = pyqtSignal(bool, list)
    log_message = pyqtSignal(str)

    def __init__(self, client: ComfyUIClient, workflow: Dict, images: Optional[Dict] = None):
        super().__init__()
        self.client = client
        self.workflow = workflow
        self.images = images
        self.results = []

    def run(self):
        """Execute workflow in separate thread"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            def progress_callback(status, current, total):
                self.progress_updated.emit(status, current, total)
                self.log_message.emit(f"Progress: {status} - {current}/{total}")

            # Ensure connection
            if not self.client.is_connected:
                loop.run_until_complete(self.client.connect())

            # Execute workflow
            self.results = loop.run_until_complete(
                self.client.execute_workflow(
                    self.workflow,
                    self.images,
                    progress_callback
                )
            )

            self.execution_finished.emit(True, self.results)

        except Exception as e:
            self.log_message.emit(f"Error: {str(e)}")
            self.execution_finished.emit(False, [])

        finally:
            loop.close()


class WorkflowLibraryWidget(QWidget):
    """Widget for browsing and selecting workflows"""

    workflow_selected = pyqtSignal(str, dict)

    def __init__(self, workflows_dir: str = './workflows'):
        super().__init__()
        self.workflows_dir = Path(workflows_dir)
        self.workflows = {}
        self.init_ui()
        self.load_workflows()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("ComfyUI Workflow Library")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Search box
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter workflows...")
        self.search_box.textChanged.connect(self.filter_workflows)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_box)
        layout.addLayout(search_layout)

        # Workflow list
        self.workflow_list = QListWidget()
        self.workflow_list.itemClicked.connect(self.on_workflow_clicked)
        self.workflow_list.itemDoubleClicked.connect(self.on_workflow_double_clicked)
        layout.addWidget(self.workflow_list)

        # Workflow info
        info_group = QGroupBox("Workflow Info")
        info_layout = QVBoxLayout()

        self.info_name = QLabel("Name: -")
        self.info_description = QLabel("Description: -")
        self.info_nodes = QLabel("Nodes: -")
        self.info_category = QLabel("Category: -")

        info_layout.addWidget(self.info_name)
        info_layout.addWidget(self.info_description)
        info_layout.addWidget(self.info_category)
        info_layout.addWidget(self.info_nodes)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_workflows)

        self.select_btn = QPushButton("Select Workflow")
        self.select_btn.clicked.connect(self.select_workflow)
        self.select_btn.setEnabled(False)

        button_layout.addWidget(self.refresh_btn)
        button_layout.addWidget(self.select_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def load_workflows(self):
        """Load workflows from directory"""
        self.workflow_list.clear()
        self.workflows.clear()

        if not self.workflows_dir.exists():
            self.workflows_dir.mkdir(parents=True, exist_ok=True)
            QMessageBox.warning(
                self,
                "Workflows Directory",
                f"Workflows directory created at:\n{self.workflows_dir}\n\nAdd workflow JSON files there."
            )
            return

        # Scan for workflow files
        for workflow_file in self.workflows_dir.glob('*.json'):
            try:
                with open(workflow_file, 'r', encoding='utf-8') as f:
                    workflow_data = json.load(f)

                # Extract metadata
                metadata = self.extract_workflow_metadata(workflow_data)
                workflow_name = workflow_file.stem

                self.workflows[workflow_name] = {
                    'path': str(workflow_file),
                    'data': workflow_data,
                    'metadata': metadata
                }

                # Add to list
                item = QListWidgetItem(workflow_name)
                item.setData(Qt.ItemDataRole.UserRole, workflow_name)

                # Set icon based on category
                category = metadata.get('category', 'general')
                # Could add category-specific icons here

                self.workflow_list.addItem(item)

            except Exception as e:
                print(f"Error loading workflow {workflow_file}: {e}")

        self.info_name.setText(f"Loaded {len(self.workflows)} workflows")

    def extract_workflow_metadata(self, workflow_data: Dict) -> Dict:
        """Extract metadata from workflow"""
        metadata = {
            'nodes': len(workflow_data),
            'description': 'No description',
            'category': 'general'
        }

        # Try to extract description from workflow
        # Some workflows have metadata in specific nodes
        for node_id, node_data in workflow_data.items():
            if 'title' in node_data and 'description' in node_data.get('title', '').lower():
                metadata['description'] = node_data.get('title', '')
                break

        # Detect category based on node types
        node_types = [node.get('class_type', '') for node in workflow_data.values()]

        if any('upscale' in nt.lower() for nt in node_types):
            metadata['category'] = 'upscaling'
        elif any('denoise' in nt.lower() for nt in node_types):
            metadata['category'] = 'denoising'
        elif any('controlnet' in nt.lower() for nt in node_types):
            metadata['category'] = 'vfx'
        elif any('video' in nt.lower() or 'frame' in nt.lower() for nt in node_types):
            metadata['category'] = 'video'

        return metadata

    def filter_workflows(self, text: str):
        """Filter workflows by search text"""
        for i in range(self.workflow_list.count()):
            item = self.workflow_list.item(i)
            workflow_name = item.data(Qt.ItemDataRole.UserRole)

            # Show item if search matches
            visible = text.lower() in workflow_name.lower()
            item.setHidden(not visible)

    def on_workflow_clicked(self, item: QListWidgetItem):
        """Handle workflow click"""
        workflow_name = item.data(Qt.ItemDataRole.UserRole)
        workflow_info = self.workflows.get(workflow_name)

        if workflow_info:
            metadata = workflow_info['metadata']

            self.info_name.setText(f"Name: {workflow_name}")
            self.info_description.setText(f"Description: {metadata.get('description', 'N/A')}")
            self.info_category.setText(f"Category: {metadata.get('category', 'N/A')}")
            self.info_nodes.setText(f"Nodes: {metadata.get('nodes', 0)}")

            self.select_btn.setEnabled(True)

    def on_workflow_double_clicked(self, item: QListWidgetItem):
        """Handle workflow double click - auto select"""
        self.select_workflow()

    def select_workflow(self):
        """Emit signal for selected workflow"""
        current_item = self.workflow_list.currentItem()
        if current_item:
            workflow_name = current_item.data(Qt.ItemDataRole.UserRole)
            workflow_info = self.workflows.get(workflow_name)

            if workflow_info:
                self.workflow_selected.emit(workflow_name, workflow_info['data'])


class WorkflowExecutionWidget(QWidget):
    """Widget for executing workflows and monitoring progress"""

    def __init__(self, client: ComfyUIClient):
        super().__init__()
        self.client = client
        self.current_workflow = None
        self.current_workflow_name = ""
        self.execution_thread = None
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Workflow Execution")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Current workflow
        workflow_group = QGroupBox("Current Workflow")
        workflow_layout = QVBoxLayout()

        self.workflow_label = QLabel("No workflow selected")
        self.workflow_label.setStyleSheet("font-weight: bold;")
        workflow_layout.addWidget(self.workflow_label)

        workflow_group.setLayout(workflow_layout)
        layout.addWidget(workflow_group)

        # Input image
        input_group = QGroupBox("Input Image")
        input_layout = QVBoxLayout()

        self.input_image_label = QLabel("No image loaded")
        self.input_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_image_label.setMinimumHeight(200)
        self.input_image_label.setStyleSheet("border: 2px dashed #888; background: #222;")

        input_btn_layout = QHBoxLayout()
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_input_image)
        input_btn_layout.addWidget(self.load_image_btn)

        input_layout.addWidget(self.input_image_label)
        input_layout.addLayout(input_btn_layout)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Parameters (dynamic based on workflow)
        self.params_group = QGroupBox("Parameters")
        self.params_layout = QVBoxLayout()
        self.params_group.setLayout(self.params_layout)
        layout.addWidget(self.params_group)

        # Execution controls
        exec_layout = QHBoxLayout()

        self.execute_btn = QPushButton("Execute Workflow")
        self.execute_btn.clicked.connect(self.execute_workflow)
        self.execute_btn.setEnabled(False)
        self.execute_btn.setStyleSheet("background: #28a745; color: white; font-weight: bold; padding: 10px;")

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_execution)
        self.cancel_btn.setEnabled(False)

        exec_layout.addWidget(self.execute_btn)
        exec_layout.addWidget(self.cancel_btn)

        layout.addLayout(exec_layout)

        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.status_label = QLabel("Ready")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Log
        log_group = QGroupBox("Execution Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)

        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        self.setLayout(layout)

        # Input image data
        self.input_image = None

    def set_workflow(self, workflow_name: str, workflow_data: Dict):
        """Set current workflow"""
        self.current_workflow_name = workflow_name
        self.current_workflow = workflow_data

        self.workflow_label.setText(f"Selected: {workflow_name}")
        self.log(f"Workflow loaded: {workflow_name}")

        # Update execute button state
        self.update_execute_button_state()

        # Build parameter UI
        self.build_parameters_ui()

    def build_parameters_ui(self):
        """Build UI for workflow parameters"""
        # Clear existing parameters
        for i in reversed(range(self.params_layout.count())):
            self.params_layout.itemAt(i).widget().setParent(None)

        if not self.current_workflow:
            return

        # Add common parameters
        # (In a real implementation, parse workflow to extract configurable params)

        info_label = QLabel("Configure workflow parameters:")
        self.params_layout.addWidget(info_label)

        # Example: Denoise strength
        denoise_layout = QHBoxLayout()
        denoise_label = QLabel("Denoise Strength:")
        self.denoise_spin = QDoubleSpinBox()
        self.denoise_spin.setRange(0.0, 1.0)
        self.denoise_spin.setSingleStep(0.05)
        self.denoise_spin.setValue(0.35)
        denoise_layout.addWidget(denoise_label)
        denoise_layout.addWidget(self.denoise_spin)
        self.params_layout.addLayout(denoise_layout)

        # Example: Upscale factor
        upscale_layout = QHBoxLayout()
        upscale_label = QLabel("Upscale Factor:")
        self.upscale_spin = QSpinBox()
        self.upscale_spin.setRange(1, 8)
        self.upscale_spin.setValue(2)
        upscale_layout.addWidget(upscale_label)
        upscale_layout.addWidget(self.upscale_spin)
        self.params_layout.addLayout(upscale_layout)

    def load_input_image(self):
        """Load input image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )

        if file_path:
            # Load image with OpenCV
            self.input_image = cv2.imread(file_path)

            if self.input_image is not None:
                # Display thumbnail
                self.display_image_thumbnail(self.input_image, self.input_image_label)
                self.log(f"Loaded image: {file_path}")

                # Update execute button state
                self.update_execute_button_state()
            else:
                QMessageBox.warning(self, "Error", "Failed to load image")

    def display_image_thumbnail(self, image: np.ndarray, label: QLabel, max_size: int = 300):
        """Display image as thumbnail in label"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to fit
        h, w = image_rgb.shape[:2]
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image_rgb, (new_w, new_h))

        # Convert to QPixmap
        qimage = QImage(resized.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        label.setPixmap(pixmap)

    def update_execute_button_state(self):
        """Update execute button enabled state"""
        can_execute = (
            self.current_workflow is not None and
            self.input_image is not None
        )
        self.execute_btn.setEnabled(can_execute)

    def execute_workflow(self):
        """Execute current workflow"""
        if not self.current_workflow or self.input_image is None:
            return

        self.log("=" * 50)
        self.log(f"Executing workflow: {self.current_workflow_name}")

        # Prepare input images
        input_images = {"input_image": self.input_image}

        # Update workflow parameters
        # (In real implementation, apply parameter values to workflow)

        # Start execution thread
        self.execution_thread = WorkflowExecutionThread(
            self.client,
            self.current_workflow,
            input_images
        )

        self.execution_thread.progress_updated.connect(self.on_progress_updated)
        self.execution_thread.execution_finished.connect(self.on_execution_finished)
        self.execution_thread.log_message.connect(self.log)

        self.execution_thread.start()

        # Update UI state
        self.execute_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("Executing...")

    def cancel_execution(self):
        """Cancel current execution"""
        if self.execution_thread and self.execution_thread.isRunning():
            self.execution_thread.terminate()
            self.log("Execution cancelled")
            self.status_label.setText("Cancelled")
            self.reset_execution_state()

    def on_progress_updated(self, status: str, current: int, total: int):
        """Handle progress update"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)

        self.status_label.setText(f"{status}: {current}/{total}")

    def on_execution_finished(self, success: bool, results: List[np.ndarray]):
        """Handle execution completion"""
        if success and results:
            self.log(f"✓ Execution successful! Received {len(results)} output(s)")
            self.status_label.setText("Completed successfully")
            self.progress_bar.setValue(100)

            # Display first result
            if results:
                self.display_result(results[0])

        else:
            self.log("✗ Execution failed")
            self.status_label.setText("Failed")

        self.reset_execution_state()

    def reset_execution_state(self):
        """Reset UI after execution"""
        self.execute_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def display_result(self, result_image: np.ndarray):
        """Display execution result"""
        # Save result
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result",
            f"result_{self.current_workflow_name}.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)"
        )

        if output_path:
            cv2.imwrite(output_path, result_image)
            self.log(f"✓ Result saved to: {output_path}")

            # Show result
            QMessageBox.information(
                self,
                "Success",
                f"Workflow executed successfully!\n\nResult saved to:\n{output_path}"
            )

    def log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )


class ComfyUIWorkflowManager(QMainWindow):
    """Main window for ComfyUI workflow management"""

    def __init__(self, config: Optional[ComfyUIConfig] = None):
        super().__init__()

        # Configuration
        self.config = config or ComfyUIConfig()

        # ComfyUI client
        self.client = ComfyUIClient(self.config)

        # Connection status
        self.connected = False

        self.init_ui()
        self.check_connection()

    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("ComfyUI Workflow Manager - Flame Integration")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()

        # Connection status bar
        status_layout = QHBoxLayout()

        self.connection_status = QLabel("⚫ Disconnected")
        self.connection_status.setStyleSheet("font-weight: bold;")

        self.server_label = QLabel(f"Server: {self.config.server_address}")

        self.reconnect_btn = QPushButton("Reconnect")
        self.reconnect_btn.clicked.connect(self.check_connection)

        status_layout.addWidget(self.connection_status)
        status_layout.addWidget(self.server_label)
        status_layout.addStretch()
        status_layout.addWidget(self.reconnect_btn)

        main_layout.addLayout(status_layout)

        # Splitter for library and execution
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Workflow library
        self.library_widget = WorkflowLibraryWidget(
            workflows_dir=self.config.host  # Using config, should be workflows_dir param
        )
        self.library_widget.workflow_selected.connect(self.on_workflow_selected)

        # Execution widget
        self.execution_widget = WorkflowExecutionWidget(self.client)

        splitter.addWidget(self.library_widget)
        splitter.addWidget(self.execution_widget)
        splitter.setSizes([400, 800])

        main_layout.addWidget(splitter)

        central_widget.setLayout(main_layout)

        # Status bar
        self.statusBar().showMessage("Ready")

    def check_connection(self):
        """Check connection to ComfyUI server"""
        self.statusBar().showMessage("Checking connection...")

        # Try to get system stats
        stats = self.client.get_system_stats()

        if stats:
            self.connected = True
            self.connection_status.setText("🟢 Connected")
            self.connection_status.setStyleSheet("color: #28a745; font-weight: bold;")
            self.statusBar().showMessage("Connected to ComfyUI")
        else:
            self.connected = False
            self.connection_status.setText("🔴 Disconnected")
            self.connection_status.setStyleSheet("color: #dc3545; font-weight: bold;")
            self.statusBar().showMessage("Failed to connect to ComfyUI server")

            QMessageBox.warning(
                self,
                "Connection Failed",
                f"Could not connect to ComfyUI server at {self.config.server_address}\n\n"
                f"Please ensure ComfyUI is running."
            )

    def on_workflow_selected(self, workflow_name: str, workflow_data: Dict):
        """Handle workflow selection"""
        self.execution_widget.set_workflow(workflow_name, workflow_data)
        self.statusBar().showMessage(f"Workflow selected: {workflow_name}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Apply dark theme
    app.setStyle("Fusion")

    # Create and show window
    config = ComfyUIConfig(host="127.0.0.1", port=8188)
    window = ComfyUIWorkflowManager(config)
    window.show()

    sys.exit(app.exec())
