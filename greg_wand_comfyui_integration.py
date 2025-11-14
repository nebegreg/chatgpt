"""
Integration module for Greg Wand Magic Wand with ComfyUI
Adds ComfyUI workflow processing to the Magic Wand application
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QGroupBox, QProgressBar, QTextEdit, QMessageBox,
    QTabWidget, QSpinBox, QDoubleSpinBox
)

from comfyui_client import ComfyUIClient, ComfyUIConfig


class ComfyUIProcessingThread(QThread):
    """Thread for processing images through ComfyUI"""

    progress_updated = pyqtSignal(str, int, int)
    processing_finished = pyqtSignal(bool, np.ndarray)
    log_message = pyqtSignal(str)

    def __init__(
        self,
        client: ComfyUIClient,
        workflow: Dict,
        input_image: np.ndarray,
        params: Optional[Dict] = None
    ):
        super().__init__()
        self.client = client
        self.workflow = workflow
        self.input_image = input_image
        self.params = params or {}

    def run(self):
        """Execute workflow in separate thread"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            def progress_callback(status, current, total):
                self.progress_updated.emit(status, current, total)
                self.log_message.emit(f"  {status}: {current}/{total}")

            # Update workflow with parameters
            workflow = self.workflow.copy()
            if self.params:
                workflow = self.client.update_workflow_params(workflow, self.params)

            # Ensure connection
            if not self.client.is_connected:
                self.log_message.emit("Connecting to ComfyUI...")
                loop.run_until_complete(self.client.connect())

            # Execute workflow
            self.log_message.emit("Executing ComfyUI workflow...")
            results = loop.run_until_complete(
                self.client.execute_workflow(
                    workflow,
                    images={"input_image": self.input_image},
                    progress_callback=progress_callback
                )
            )

            if results and len(results) > 0:
                self.log_message.emit("✓ Processing complete!")
                self.processing_finished.emit(True, results[0])
            else:
                self.log_message.emit("✗ Processing failed - no results")
                self.processing_finished.emit(False, self.input_image)

        except Exception as e:
            self.log_message.emit(f"✗ Error: {str(e)}")
            self.processing_finished.emit(False, self.input_image)

        finally:
            loop.close()


class ComfyUIIntegrationWidget(QWidget):
    """
    Widget that integrates ComfyUI workflows into the Magic Wand application
    Can be added as a tab or panel to the main window
    """

    image_processed = pyqtSignal(np.ndarray)

    def __init__(self, magic_wand, config_path: str = "comfyui_config.json"):
        super().__init__()

        self.magic_wand = magic_wand
        self.processing_thread = None

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize ComfyUI client
        comfy_config = ComfyUIConfig(
            host=self.config.get('comfyui_host', '127.0.0.1'),
            port=self.config.get('comfyui_port', 8188)
        )
        self.client = ComfyUIClient(comfy_config)

        # Workflow library
        self.workflows_dir = Path(self.config.get('workflows_dir', './workflows'))
        self.workflows = {}

        self.init_ui()
        self.load_workflows()
        self.check_connection()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠ Failed to load config: {e}, using defaults")
            return {
                'comfyui_host': '127.0.0.1',
                'comfyui_port': 8188,
                'workflows_dir': './workflows'
            }

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("ComfyUI AI Processing")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Connection status
        status_group = QGroupBox("Connection Status")
        status_layout = QHBoxLayout()

        self.connection_label = QLabel("⚫ Disconnected")
        self.connection_label.setStyleSheet("font-weight: bold;")

        self.reconnect_btn = QPushButton("Reconnect")
        self.reconnect_btn.clicked.connect(self.check_connection)

        status_layout.addWidget(self.connection_label)
        status_layout.addStretch()
        status_layout.addWidget(self.reconnect_btn)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Workflow selection
        workflow_group = QGroupBox("Select Workflow")
        workflow_layout = QVBoxLayout()

        self.workflow_combo = QComboBox()
        self.workflow_combo.currentIndexChanged.connect(self.on_workflow_changed)

        self.workflow_info = QLabel("Select a workflow to see details")
        self.workflow_info.setWordWrap(True)

        workflow_layout.addWidget(self.workflow_combo)
        workflow_layout.addWidget(self.workflow_info)

        workflow_group.setLayout(workflow_layout)
        layout.addWidget(workflow_group)

        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()

        # Denoise strength
        denoise_layout = QHBoxLayout()
        denoise_label = QLabel("Denoise Strength:")
        self.denoise_spin = QDoubleSpinBox()
        self.denoise_spin.setRange(0.0, 1.0)
        self.denoise_spin.setSingleStep(0.05)
        self.denoise_spin.setValue(0.35)
        denoise_layout.addWidget(denoise_label)
        denoise_layout.addWidget(self.denoise_spin)
        denoise_layout.addStretch()
        params_layout.addLayout(denoise_layout)

        # CFG Scale
        cfg_layout = QHBoxLayout()
        cfg_label = QLabel("CFG Scale:")
        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(1.0, 20.0)
        self.cfg_spin.setSingleStep(0.5)
        self.cfg_spin.setValue(7.5)
        cfg_layout.addWidget(cfg_label)
        cfg_layout.addWidget(self.cfg_spin)
        cfg_layout.addStretch()
        params_layout.addLayout(cfg_layout)

        # Steps
        steps_layout = QHBoxLayout()
        steps_label = QLabel("Steps:")
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(5, 100)
        self.steps_spin.setValue(20)
        steps_layout.addWidget(steps_label)
        steps_layout.addWidget(self.steps_spin)
        steps_layout.addStretch()
        params_layout.addLayout(steps_layout)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Process buttons
        button_layout = QHBoxLayout()

        self.process_current_btn = QPushButton("Process Current Layer")
        self.process_current_btn.clicked.connect(self.process_current_layer)
        self.process_current_btn.setStyleSheet("background: #28a745; color: white; font-weight: bold;")

        self.process_all_btn = QPushButton("Process All Layers")
        self.process_all_btn.clicked.connect(self.process_all_layers)

        button_layout.addWidget(self.process_current_btn)
        button_layout.addWidget(self.process_all_btn)

        layout.addLayout(button_layout)

        # Progress
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)

        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()
        self.setLayout(layout)

        # Disable processing initially
        self.set_processing_enabled(False)

    def load_workflows(self):
        """Load available workflows"""
        self.workflow_combo.clear()
        self.workflows.clear()

        if not self.workflows_dir.exists():
            self.log("⚠ Workflows directory not found")
            return

        for workflow_file in self.workflows_dir.glob('*.json'):
            try:
                with open(workflow_file, 'r', encoding='utf-8') as f:
                    workflow_data = json.load(f)

                workflow_name = workflow_file.stem
                self.workflows[workflow_name] = {
                    'path': str(workflow_file),
                    'data': workflow_data
                }

                # Get preset info if available
                preset_info = self.config.get('workflow_presets', {}).get(workflow_name, {})
                display_name = preset_info.get('description', workflow_name)

                self.workflow_combo.addItem(display_name, workflow_name)

            except Exception as e:
                self.log(f"✗ Error loading {workflow_file.name}: {e}")

        self.log(f"✓ Loaded {len(self.workflows)} workflows")

    def check_connection(self):
        """Check connection to ComfyUI"""
        stats = self.client.get_system_stats()

        if stats:
            self.connection_label.setText("🟢 Connected")
            self.connection_label.setStyleSheet("color: #28a745; font-weight: bold;")
            self.log("✓ Connected to ComfyUI")
            self.set_processing_enabled(True)
        else:
            self.connection_label.setText("🔴 Disconnected")
            self.connection_label.setStyleSheet("color: #dc3545; font-weight: bold;")
            self.log("✗ Failed to connect to ComfyUI")
            self.set_processing_enabled(False)

    def set_processing_enabled(self, enabled: bool):
        """Enable/disable processing controls"""
        self.process_current_btn.setEnabled(enabled and self.workflow_combo.count() > 0)
        self.process_all_btn.setEnabled(enabled and self.workflow_combo.count() > 0)

    def on_workflow_changed(self, index: int):
        """Handle workflow selection change"""
        workflow_name = self.workflow_combo.itemData(index)

        if workflow_name:
            preset_info = self.config.get('workflow_presets', {}).get(workflow_name, {})

            description = preset_info.get('description', 'No description')
            category = preset_info.get('category', 'general')

            self.workflow_info.setText(f"Category: {category}\n{description}")

            # Update parameters from preset
            params = preset_info.get('parameters', {})
            self.denoise_spin.setValue(params.get('denoise_strength', 0.35))
            self.cfg_spin.setValue(params.get('cfg_scale', 7.5))
            self.steps_spin.setValue(params.get('steps', 20))

    def get_current_workflow(self) -> Optional[Dict]:
        """Get currently selected workflow"""
        workflow_name = self.workflow_combo.currentData()
        if workflow_name and workflow_name in self.workflows:
            return self.workflows[workflow_name]['data']
        return None

    def get_current_layer_image(self) -> Optional[np.ndarray]:
        """Get current layer image from Magic Wand"""
        if not self.magic_wand.layers:
            return None

        current_layer = self.magic_wand.layers[self.magic_wand.current_layer_index]
        return current_layer.image.copy()

    def process_current_layer(self):
        """Process the current layer through ComfyUI"""
        workflow = self.get_current_workflow()
        if not workflow:
            QMessageBox.warning(self, "No Workflow", "Please select a workflow first")
            return

        input_image = self.get_current_layer_image()
        if input_image is None:
            QMessageBox.warning(self, "No Image", "No layer image available")
            return

        # Prepare parameters
        params = {}  # In production, map UI params to workflow nodes

        # Start processing
        self.log("=" * 50)
        self.log(f"Processing: {self.workflow_combo.currentText()}")

        self.processing_thread = ComfyUIProcessingThread(
            self.client,
            workflow,
            input_image,
            params
        )

        self.processing_thread.progress_updated.connect(self.on_progress_updated)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.log_message.connect(self.log)

        self.processing_thread.start()

        # Disable controls during processing
        self.process_current_btn.setEnabled(False)
        self.process_all_btn.setEnabled(False)

    def process_all_layers(self):
        """Process all visible layers"""
        # For simplicity, just process current layer
        # In production, iterate through all layers
        self.process_current_layer()

    def on_progress_updated(self, status: str, current: int, total: int):
        """Handle progress update"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)

        self.progress_label.setText(f"{status}: {current}/{total}")

    def on_processing_finished(self, success: bool, result_image: np.ndarray):
        """Handle processing completion"""
        if success:
            self.log("✓ Processing complete!")

            # Update current layer with result
            if self.magic_wand.layers:
                current_layer = self.magic_wand.layers[self.magic_wand.current_layer_index]
                current_layer.image = result_image
                self.magic_wand.update_image()

                # Emit signal
                self.image_processed.emit(result_image)

            self.progress_bar.setValue(100)
            self.progress_label.setText("Complete")

        else:
            self.log("✗ Processing failed")
            self.progress_label.setText("Failed")

        # Re-enable controls
        self.set_processing_enabled(True)

    def log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )


def integrate_comfyui_into_magic_wand(main_window):
    """
    Function to integrate ComfyUI widget into the Magic Wand main window

    Args:
        main_window: The MainWindow instance from greg_wand.py

    Returns:
        The ComfyUI integration widget
    """
    # Create ComfyUI widget
    comfyui_widget = ComfyUIIntegrationWidget(main_window.magic_wand)

    # Add as new tab to right panel or as separate panel
    # This depends on how you want to integrate it

    # Option 1: Add to existing right panel
    main_window.right_panel.addWidget(comfyui_widget)

    # Option 2: Create tabbed interface (more organized)
    # if not hasattr(main_window, 'tab_widget'):
    #     # Convert right panel to tabbed interface
    #     tab_widget = QTabWidget()
    #
    #     # Move existing widgets to tabs
    #     original_widget = QWidget()
    #     original_widget.setLayout(main_window.right_panel)
    #     tab_widget.addTab(original_widget, "Magic Wand")
    #
    #     # Add ComfyUI tab
    #     tab_widget.addTab(comfyui_widget, "ComfyUI AI")
    #
    #     # Replace right panel with tabs
    #     main_window.layout.addWidget(tab_widget)

    return comfyui_widget


if __name__ == "__main__":
    # Standalone test
    from PyQt6.QtWidgets import QApplication
    from greg_wand import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()

    # Integrate ComfyUI
    comfyui_widget = integrate_comfyui_into_magic_wand(window)

    window.show()
    sys.exit(app.exec())
