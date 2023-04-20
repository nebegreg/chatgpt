import cv2
import sys
import numpy as np
from PyQt6 import QtCore


from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QFile, QDataStream, QIODevice, pyqtSignal
from PyQt6.QtGui import (QImage, QPixmap, QIcon, QKeySequence, QGuiApplication, QColor, QAction, QShortcut, QTransform,)
from PyQt6.QtWidgets import (QApplication, QGraphicsScene, QGraphicsView, QMainWindow, QVBoxLayout, QWidget,QRadioButton,
                             QSlider, QCheckBox, QMenu, QToolBar, QPushButton, QHBoxLayout,
                             QFileDialog, QMessageBox, QProgressBar, QColorDialog, QLabel, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,QListWidgetItem)
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtCore import Qt
from PyQt6 import QtWidgets, QtGui, QtCore
composition_modes = {
    'normal': QPainter.CompositionMode.CompositionMode_SourceOver,
    'add': QPainter.CompositionMode.CompositionMode_Plus,
    'subtract': QPainter.CompositionMode.CompositionMode_Difference,
    # Add other blend modes if needed
}

class BrushTool:
    def __init__(self, parent=None):
        self.parent = parent
        self.brush_size = 5
        self.color = (0, 0, 255)
        self.drawing = False
        self.last_point = None

    def set_brush_size(self, size):
        self.brush_size = size

    def set_color(self, color):
        self.color = color

    def start_drawing(self, x, y):
        self.drawing = True
        self.last_point = (x, y)

    def stop_drawing(self):
        self.drawing = False
        self.last_point = None

    def draw(self, x, y):
        if not self.drawing:
            return
        cv2.line(self.parent.layers[self.parent.current_layer_index].mask, self.last_point, (x, y), self.color, self.brush_size)
        self.last_point = (x, y)
        self.parent.show_image()
                             


class Layer:
    def __init__(self, image, name, blend_mode="normal", visible=True, opacity=1, mask=None, mask_color=(0, 0, 255)):
        self.image = image
        self.name = name
        self.blend_mode = blend_mode
        self.visible = visible
        self.opacity = opacity
        self.mask = mask
        self.mask_color = mask_color
   


    def set_blend_mode(self, blend_mode):
        self.blend_mode = blend_mode

    def apply_blend_mode(self, base_image):
        if self.blend_mode == 'normal':
            return self.image
        elif self.blend_mode == 'add':
            return cv2.add(base_image, self.image)
        elif self.blend_mode == 'subtract':
            return cv2.subtract(base_image, self.image)
        else:
            return self.image

class UndoRedoStack:
    def __init__(self, max_size=10):
        self.stack = []
        self.max_size = max_size

    def push(self, item):
        self.stack.append(item)
        if len(self.stack) > self.max_size:
            self.stack.pop(0)

    def pop(self):
        return self.stack.pop() if self.stack else None

    def top(self):
        return self.stack[-1] if self.stack else None

    def clear(self):
        self.stack.clear()

    def __len__(self):
        return len(self.stack)

class MagicWand(QGraphicsView):
    update_layers_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.brush_size = 5
        self.color = (0, 0, 255)
        self.active_tool = 'brush'
        
        self.drawing = False
        self.last_point = None

        self.tolerance = 30
        self.sensitivity = 50  # ajout de l'attribut sensitivity avec une valeur par défaut
        self.blur = 0
        self.contiguous = True
        
        self.mask = None
        self.operation = 'replace'
        self.anti_aliasing = False
        self.edge_detection = False
        self.low_threshold = 0
        self.high_threshold = 100
        self.original_image = None
        self.image = None
        self.parent = parent
        if self.original_image is not None:
           self.layers = [Layer(self.original_image.copy(), "Base Layer", blend_mode="normal")]

        else:
            self.layers = []
        self.action_history = []
        self.current_layer_index = 0
        self.anti_aliasing = True
        self.edge_detection = True
        self.low_threshold = 100
        self.high_threshold = 200
        self.zoom_level = 1
        self.selection_mode = "magic_wand"
        self.mask = None
        self.contiguous = True
        self.tolerance = 30
        self.operation = 'replace'
        self.color = (0, 0, 255)
        self.setScene(QGraphicsScene())
        self.base_layer = self.scene().addPixmap(QPixmap())
        self.selection_layer = self.scene().addPixmap(QPixmap())
        self.undo_stack = UndoRedoStack()
        self.redo_stack = UndoRedoStack()
        self.active_layer = None
    def set_brush_size(self, size):
        self.brush_size = size

    def set_color(self, color):
        self.color = color

    def start_drawing(self, x, y):
        self.drawing = True
        self.last_point = (x, y)

    def stop_drawing(self):
        self.drawing = False
        self.last_point = None

    def draw(self, x, y):
        if not self.drawing:
            return
        cv2.line(self.magic_wand.layers[self.magic_wand.current_layer_index].mask, self.last_point, (x, y), self.color, self.brush_size)
        self.last_point = (x, y)
        self.magic_wand.show_image()

    def set_tool(self, tool):
        self.active_tool = tool
    
    def add_action_to_history(self, action):
        self.action_history.append(action)

    
           
    def set_selection_mode(self, mode):
        self.selection_mode = mode

    def update_image(self):
        result_image = np.zeros_like(self.original_image)
        final_image = self.layers[0].image.copy()
        qimage = QImage(final_image.data, final_image.shape[1], final_image.shape[0], final_image.strides[0], QImage.Format.Format_RGB888)

        qpixmap = QPixmap.fromImage(qimage)

        painter = QPainter(qpixmap)
        for i, layer in enumerate(self.layers):
            if layer.visible:
                painter.setCompositionMode(composition_modes[layer.blend_mode])
                layer_qimage = QImage(layer.image.data, layer.image.shape[1], layer.image.shape[0], layer.image.strides[0], QImage.Format.Format_RGB888)
                result_image = self.blend_layers(result_image, layer.image, layer.blend_mode, layer.opacity, layer.mask)
                painter.drawImage(0, 0, layer_qimage)

        painter.end()
        self.image = cv2.bitwise_and(final_image, final_image, mask=self.get_global_mask())
        self.show_image()

    def show_image(self):
        if self.image is not None:
            qimage = QImage(self.image.data, self.image.shape[1], self.image.shape[0], QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(qimage)
            self.base_layer.setPixmap(pixmap)
            
            
            self.update_selection_layer()
            self.get_global_selection()
            
    def update_selection_layer(self, operation, mask):
        current_layer = self.layers[self.current_layer_index]
        if current_layer.mask is None:
            current_layer.mask = np.zeros_like(self.original_image[:, :, 0])
        
        if operation == 'add':
            current_layer.mask = cv2.bitwise_or(current_layer.mask, mask)
        elif operation == 'subtract':
            current_layer.mask = cv2.bitwise_and(current_layer.mask, cv2.bitwise_not(mask))
        else:  # 'replace' operation
            current_layer.mask = mask

    def set_active_layer(self, index):
        self.active_layer = index
        self.update_selection_layer()

    def undo(self):
        if len(self.undo_stack) > 1:
            self.redo_stack.push(self.undo_stack.pop())
            self.image, self.mask = self.undo_stack.top()
            self.show_image()
            self.update_selection_layer()
    def redo(self):
        if self.redo_stack:
            self.undo_stack.push(self.redo_stack.pop())
            self.image, self.mask = self.undo_stack.top()
            self.show_image()
            self.update_selection_layer()
            
    def add_layer(self):
        new_layer = Layer(self.original_image.copy(), f"Layer {len(self.layers)}",blend_mode="normal", visible=True, opacity=1, mask=None, mask_color=(0, 0, 255))
        self.layers.append(new_layer)
        self.current_layer_index = len(self.layers) - 1
        self.update_layers_updated.emit()  # Emit the signal
        self.add_action_to_history(("add_layer", new_layer))
        return new_layer

    def remove_layer(self, index):
        if 0 <= index < len(self.layers) and len(self.layers) > 1:
            self.layers.pop(index)
            self.current_layer_index = max(0, index - 1)
            self.update_image()
            self.update_layers_updated.emit() 
            self.add_action_to_history(("remove_layer", index))
            self.get_global_selection() # Emit the signal

    def get_global_selection(self):
            selection = np.zeros(self.original_image.shape[:2], np.uint8)
            for layer in self.layers:
                if layer.image is not None:
                    mask = self.get_layer_mask(layer)
                    selection = cv2.bitwise_or(selection, mask)
            return selection
    def get_global_mask(self):
        global_mask = np.zeros(self.original_image.shape[:2], np.uint8)
        for layer in self.layers:
            if layer.visible:
                if layer.blend_mode == 'normal':
                    global_mask = cv2.bitwise_or(global_mask, layer.selection)
                elif layer.blend_mode == 'add':
                    global_mask = cv2.add(global_mask, layer.selection)
                elif layer.blend_mode == 'subtract':
                    global_mask = cv2.subtract(global_mask, layer.selection)
        return global_mask

    
    def get_layer_mask(self, layer):
        if layer.image is not None:
            mask = layer.mask
            return mask
        return None   
    def reset(self):
        self.image = self.original_image.copy()
        self.mask = np.zeros(self.image.shape[:2], np.uint8)
        self.show_image()
        self.update_selection_layer()
    def save_state(self):
        if self.image is not None and self.mask is not None:
            state = (self.image.copy(), self.mask.copy())
            self.undo_stack.push(state)
            self.redo_stack.clear()

            
    def load_state(self, file_name):
        with QFile(file_name) as file:
            if file.open(QIODevice.ReadOnly):
                stream = QDataStream(file)
                magic_number = stream.readUInt32()
                if magic_number != 1:
                    raise ValueError("Invalid magic number in state file")
                state = {}
                stream >> state
                self.image = state['image']
                self.mask = state['mask']
                self.contiguous = state['contiguous']
                self.tolerance = state['tolerance']
                self.operation = state['operation']
                self.color = state['color']
                self.show_image()
                self.update_selection_layer()
                self.update_layers_updated.emit()

    
    def convert_to_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   

    def get_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def magic_wand(self, x, y, tolerance, contiguous, operation):
        # Convert the input image to HSV color space
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        
        # Get the HSV color at the specified pixel
        hsv_color = hsv[y, x]
        
        # Cast sensitivity and blur to uint8 before subtraction
        sensitivity = np.uint8(self.sensitivity)
        blur = np.uint8(self.blur)
        
        # Define the lower and upper bounds of the color range to select
        lower_bound = np.array([max(hsv_color[0] - tolerance, 0),
                          max(hsv_color[1].astype(np.int32) - sensitivity, 0),  # Change this line
                          max(hsv_color[2].astype(np.int32) - sensitivity, 0)], dtype=np.uint8)
        upper_bound = np.array([min(hsv_color[0] + tolerance, 180),
                          min(hsv_color[1].astype(np.int32) + sensitivity, 255),  # Change this line
                          min(hsv_color[2].astype(np.int32) + sensitivity, 255)], dtype=np.uint8)
        
        # Create a mask based on the color range
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Apply the mask to the image
        if operation == 'replace':
            self.image[mask != 0] = self.color
        elif operation == 'subtract':
            self.image[mask == 255] = [0, 0, 0]
        else:
            return
            
        # Apply a filter to remove small gaps between the selected area
        if contiguous:
            self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
        self.save_state()
        self.show_image()
        self.update_selection_layer('replace', mask)
        self.get_global_selection()




    def set_blend_mode(self, index, mode):
        if 0 <= index < len(self.layers):
            self.layers[index].blend_mode = mode
            self.update_image()
    def set_layer_visibility(self, index, state):
        if 0 <= index < len(self.layers):
            self.layers[index].visible = state
            self.update_image()
           
            
    def set_layer_opacity(self, index, opacity):
        if 0 <= index < len(self.layers):
            self.layers[index].opacity = opacity
            self.update_image()
          
   
    def set_anti_aliasing(self, state):
        self.anti_aliasing = state

    def set_edge_detection(self, state):
        self.edge_detection = state
    def set_low_threshold(self, value):
        self.low_threshold = value
    def set_high_threshold(self, value):
        self.high_threshold = value
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            
            modifiers = QGuiApplication.queryKeyboardModifiers()
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                self.operation = 'add'

            elif modifiers & Qt.KeyboardModifier.AltModifier:
                self.operation = 'subtract'
            else:
                self.operation = 'replace'
            self.magic_wand(x, y, self.tolerance, self.contiguous, self.operation)
 

    



    def open_video(self):
        options = QFileDialog.Option(QFileDialog.Option.ReadOnly)
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mov *.mp4);;All Files (*)", options=options)
        if file_name:
            self.video_path = file_name
            self.load_frame_from_video(file_name)

    def save_video(self):
        output_video_path, _ = QFileDialog.getSaveFileName(None, "Save Video", "", "Video Files (*.mp4 *.avi)")
        if not output_video_path:
            return
        self.magic_wand.apply_to_video(output_video_path, self.update_progress_bar)
        QMessageBox.information(None, "Save Video", "Video saved successfully!", QMessageBox.StandardButton.Ok)

    def update_progress_bar(self, progress):
        self.progress_bar.setValue(int(progress * 100))
        QtWidgets.QApplication.processEvents()
    def erode_mask(self, kernel_size=3, iterations=1):
        layer = self.layers[self.current_layer_index]
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size), (1, 1))
        layer.mask = cv2.erode(layer.mask, kernel, iterations=iterations)
        self.show_image()

    def dilate_mask(self, kernel_size=3, iterations=1):
        layer = self.layers[self.current_layer_index]
        
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size), (1, 1))

        layer.mask = cv2.dilate(layer.mask, kernel, iterations=iterations)
        self.show_image()
    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setZoomLevel(self.zoom_level * zoom_factor)
    def setZoomLevel(self, zoom_level):
        self.zoom_level = max(min(zoom_level, 4), 0.25)
        self.setTransform(QTransform().scale(zoom_level, zoom_level))
    def load_frame_from_video(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        if video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret:
                self.image = frame
                self.original_image = self.image.copy()
                self.mask = np.zeros(self.image.shape[:2], np.uint8)
                self.show_image()
                self.update_selection_layer()
            video_capture.release()
        else:
            print("Error: Unable to open the video file.")

    def apply_to_video(self, output_video_path, progress_callback):
        video_capture = cv2.VideoCapture(self.video_path)
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'ap4h')

        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height), isColor=True)

        confirmation = QMessageBox.question(None, "Apply to Video", "Are you sure you want to apply the effect to the entire video?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirmation == QMessageBox.StandardButton.No:
            return

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            self.original_image = frame.copy()
            self.magic_wand(0, 0)  # Apply the magic wand on the image
            video_writer.write(self.image)  # Write the image to the output video
            current_frame += 1
            progress_callback(current_frame / total_frames)

        video_capture.release()
        video_writer.release()



    
    def set_video_path(self, video_path):
        self.video_path = video_path

    def set_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color = (color.red(), color.green(), color.blue())

class LayerListWidget(QWidget):
    active_layer_changed = pyqtSignal(int)
    def __init__(self, magic_wand):
        super().__init__()
        self.magic_wand = magic_wand
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layer_layouts = []
        self.layer_checkboxes = []
        self.layer_radio_buttons = []
        self.layer_states = []
        self.selected_layers = set()  # Add this line
        self.add_layer_button = QPushButton("Add Layer")
        self.add_layer_button.clicked.connect(self.add_layer)
        self.layout.addWidget(self.add_layer_button)

        self.active_layer_index = 0
        self.update_layers()

    def add_layer(self):
        self.magic_wand.add_layer()
        self.update_layers()

    def update_layers(self):
        selected_layer_indexes = [i for i, checkbox in enumerate(self.layer_checkboxes) if checkbox.isChecked()]

        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)

        self.layout.addWidget(self.add_layer_button)

        self.layer_checkboxes = []
        self.layer_radio_buttons = []
        self.blend_mode_widgets = []

        for i, layer in enumerate(self.magic_wand.layers):
            layer_widget = QWidget()
            layer_layout = QHBoxLayout()
            self.layer_layouts.append(layer_layout)

            layer_widget.setLayout(layer_layout)

            layer_checkbox = QCheckBox(layer.name)
            layer_checkbox.setChecked(i in selected_layer_indexes)
            layer_checkbox.stateChanged.connect(lambda state, index=i: self.set_layer_visibility(index, state))
            self.layer_checkboxes.append(layer_checkbox)
            layer_layout.addWidget(layer_checkbox)

            layer_radio_button = QRadioButton()
            layer_radio_button.setChecked(i == self.active_layer_index)
            layer_radio_button.clicked.connect(lambda _, index=i: self.set_active_layer(index))
            self.layer_radio_buttons.append(layer_radio_button)
            layer_layout.addWidget(layer_radio_button)

            blend_mode_combobox = QComboBox()
            blend_mode_combobox.addItems(["Normal", "Add", "Subtract"])
            blend_mode_combobox.setCurrentIndex(["normal", "add", "subtract"].index(layer.blend_mode))
            blend_mode_combobox.currentIndexChanged.connect(lambda index, layer_index=i: self.set_blend_mode(layer_index, index))
            self.blend_mode_widgets.append(blend_mode_combobox)
            layer_layout.addWidget(blend_mode_combobox)

            remove_button = QPushButton("Remove")
            remove_button.clicked.connect(lambda _, index=i: self.remove_layer(index))
            layer_layout.addWidget(remove_button)

            self.layout.addWidget(layer_widget)

        self.layer_states = [checkbox.isChecked() for checkbox in self.layer_checkboxes]

    def set_active_layer(self, index):
        self.active_layer_index = index
        self.active_layer_changed.emit(index)


    def set_layer_visibility(self, index, state):
        if state == Qt.CheckState.Checked:
            self.selected_layers.add(index)
        else:
            self.selected_layers.discard(index)
        self.magic_wand.set_layer_visibility(index, state == Qt.CheckState.Checked)
     


   



    def set_blend_mode(self, layer_index, index):
        blend_mode = {0: "normal", 1: "add", 2: "subtract"}[index]
        self.magic_wand.set_blend_mode(layer_index, blend_mode)

    def remove_layer(self, index):
        self.magic_wand.remove_layer(index)
        self.update_layers()

    def mute_unmute_layers(self):
        for i, checkbox in enumerate(self.layer_checkboxes):
            if self.layer_states[i]:
                checkbox.setChecked(True)
                self.magic_wand.set_layer_visibility(i, True)
            else:
                checkbox.setChecked(False)
                self.magic_wand.set_layer_visibility(i, False)


        

class MagicWandWidgets(QWidget):
    def __init__(self, magic_wand):
        super().__init__()

        self.magic_wand = magic_wand

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.image_label = QLabel()

        self.layout.addWidget(self.image_label)
        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setRange(1, 50)
        self.brush_size_slider.setValue(self.magic_wand.brush_size)
        self.brush_size_slider.valueChanged.connect(self.magic_wand.set_brush_size)
        self.layout.addWidget(QLabel("Brush Size"))
        self.layout.addWidget(self.brush_size_slider)
        self.tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self.tolerance_slider.setRange(0, 100)
        self.tolerance_slider.setValue(self.magic_wand.tolerance)
        self.tolerance_slider.valueChanged.connect(self.set_tolerance)
        self.layout.addWidget(QLabel("Tolerance"))
        self.layout.addWidget(self.tolerance_slider)
        self.erode_button = QPushButton("Erode")
        self.erode_button.clicked.connect(self.magic_wand.erode_mask)
        self.layout.addWidget(self.erode_button)

        self.dilate_button = QPushButton("Dilate")
        self.dilate_button.clicked.connect(self.magic_wand.dilate_mask)
        self.layout.addWidget(self.dilate_button)


        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(0, 100)
        self.sensitivity_slider.setValue(self.magic_wand.sensitivity)
        self.sensitivity_slider.valueChanged.connect(self.set_sensitivity)
        self.layout.addWidget(QLabel("Sensitivity"))
        self.layout.addWidget(self.sensitivity_slider)

        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(0, 10)
        self.blur_slider.setValue(self.magic_wand.blur)
        self.blur_slider.valueChanged.connect(self.set_blur)
        self.layout.addWidget(QLabel("Blur"))
        self.layout.addWidget(self.blur_slider)

        self.contiguous_checkbox = QCheckBox("Contiguous")
        self.contiguous_checkbox.setChecked(self.magic_wand.contiguous)
        self.contiguous_checkbox.stateChanged.connect(self.set_contiguous)
        self.layout.addWidget(self.contiguous_checkbox)

        self.color_button = QPushButton("Set Color")
        self.color_button.clicked.connect(self.magic_wand.set_color)
        self.layout.addWidget(self.color_button)

    def set_active_layer(self, index):
        self.magic_wand.selection_layer = index
        
    def set_tolerance(self, value):
        self.magic_wand.tolerance = value

    def set_sensitivity(self, value):
        self.magic_wand.sensitivity = value

    def set_blur(self, value):
        self.magic_wand.blur = value

    def set_contiguous(self, state):
        self.magic_wand.contiguous = state == Qt.CheckState.Checked

    def set_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color = color
    def set_layer_visibility(self, index, visible):
        layer = self.layers[index]
        layer.visible = visible
        self.update_image()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Magic Wand")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.magic_wand = MagicWand()

        self.layout.addWidget(self.magic_wand)

        self.right_panel = QVBoxLayout()

        self.magic_wand_widgets = MagicWandWidgets(self.magic_wand)
        self.right_panel.addWidget(self.magic_wand_widgets)

        self.layer_list_widget = LayerListWidget(self.magic_wand)
        self.right_panel.addWidget(self.layer_list_widget)
        self.layer_list_widget.active_layer_changed.connect(self.magic_wand.set_active_layer)

        self.layout.addLayout(self.right_panel)

        self.magic_wand.update_layers_updated.connect(self.layer_list_widget.update_layers)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(30, 40, 200, 25)  # position et taille de la barre de progression
        self.progress_bar.setValue(0)  # valeur initiale de la barre de progression
        self.init_ui()
    def init_ui(self):
        self.init_toolbar()
        self.init_statusbar()

    def init_toolbar(self):
        toolbar = QToolBar("Tools")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        open_action = QAction(QIcon.fromTheme("document-open"), "Open", self)
        open_action.triggered.connect(self.magic_wand.open_video)
        toolbar.addAction(open_action)

        save_action = QAction(QIcon.fromTheme("document-save"), "Save", self)
        save_action.triggered.connect(self.save_video)
        toolbar.addAction(save_action)

        undo_action = QAction(QIcon.fromTheme("edit-undo"), "Undo", self)
        
        undo_action.triggered.connect(self.magic_wand.undo)
        toolbar.addAction(undo_action)

        redo_action = QAction(QIcon.fromTheme("edit-redo"), "Redo", self)
        redo_action.triggered.connect(self.magic_wand.redo)
        toolbar.addAction(redo_action)

        color_action = QAction(QIcon.fromTheme("color-picker"), "Set Color", self)
        color_action.triggered.connect(self.magic_wand.set_color)
        toolbar.addAction(color_action)

        reset_action = QAction(QIcon.fromTheme("view-refresh"), "Reset", self)
        reset_action.triggered.connect(self.magic_wand.reset)
        toolbar.addAction(reset_action)

        quit_action = QAction(QIcon.fromTheme("application-exit"), "Quit", self)
        quit_action.triggered.connect(self.close)
        toolbar.addAction(quit_action)

    def init_statusbar(self):
        statusbar = self.statusBar()

    def save_video(self):
        output_video_path, _ = QFileDialog.getSaveFileName(None, "Save Video", "", "Video Files (*.mp4 *.avi)")
        if not output_video_path:
            return
        self.magic_wand.apply_to_video(output_video_path, self.update_progress_bar)
        QMessageBox.information(None, "Save Video", "Video saved successfully!", QMessageBox.StandardButton.Ok)

    def update_progress_bar(self, progress):
        self.progress_bar.setValue(int(progress * 100))
        QtWidgets.QApplication.processEvents()



    
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
