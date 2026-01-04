import sys
import numpy as np
import tifffile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QLabel,
                             QVBoxLayout, QWidget, QSlider, QPushButton,
                             QHBoxLayout, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


class TiffViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Multi-Page TIFF Viewer")
        self.resize(800, 600)
        self.tiff_stack = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.scale_factor = 1.0

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        top_layout = QHBoxLayout()
        self.btn_open = QPushButton("Open TIFF File")
        self.btn_open.clicked.connect(self.open_file)
        top_layout.addWidget(self.btn_open)
        self.lbl_info = QLabel("No file loaded")
        top_layout.addWidget(self.lbl_info)
        main_layout.addLayout(top_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_image.setScaledContents(True)
        self.scroll_area.setWidget(self.lbl_image)
        main_layout.addWidget(self.scroll_area)

        controls_layout = QHBoxLayout()

        self.btn_back = QPushButton("<")
        self.btn_back.setFixedWidth(30)
        self.btn_back.setEnabled(False)
        self.btn_back.clicked.connect(self.step_backward)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.change_frame)

        self.btn_forward = QPushButton(">")
        self.btn_forward.setFixedWidth(30)
        self.btn_forward.setEnabled(False)
        self.btn_forward.clicked.connect(self.step_forward)

        controls_layout.addWidget(QLabel("Frame:"))
        controls_layout.addWidget(self.btn_back)
        controls_layout.addWidget(self.slider)
        controls_layout.addWidget(self.btn_forward)

        self.lbl_frame_count = QLabel("0/0")
        controls_layout.addWidget(self.lbl_frame_count)

        btn_zoom_in = QPushButton("+")
        btn_zoom_in.setFixedWidth(30)
        btn_zoom_in.clicked.connect(lambda: self.zoom_image(1.25))
        btn_zoom_out = QPushButton("-")
        btn_zoom_out.setFixedWidth(30)
        btn_zoom_out.clicked.connect(lambda: self.zoom_image(0.8))

        controls_layout.addWidget(QLabel("Zoom:"))
        controls_layout.addWidget(btn_zoom_out)
        controls_layout.addWidget(btn_zoom_in)
        main_layout.addLayout(controls_layout)

    def open_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open TIFF", "", "TIFF Files (*.tif *.tiff);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                self.tiff_stack = tifffile.imread(file_path)
                if self.tiff_stack.ndim == 2:
                    self.tiff_stack = self.tiff_stack[np.newaxis, ...]
                self.total_frames = self.tiff_stack.shape[0]
                self.current_frame_idx = 0
                self.slider.setMinimum(0)
                self.slider.setMaximum(self.total_frames - 1)
                self.slider.setValue(0)
                self.slider.setEnabled(True)
                self.btn_back.setEnabled(True)
                self.btn_forward.setEnabled(True)
                self.lbl_info.setText(f"Loaded: {file_path.split('/')[-1]}")
                self.scale_factor = 1.0
                self.display_frame()
            except Exception as e:
                self.lbl_info.setText(f"Error: {str(e)}")

    def step_backward(self):
        self.slider.setValue(self.slider.value() - 1)

    def step_forward(self):
        self.slider.setValue(self.slider.value() + 1)

    def change_frame(self):
        if self.tiff_stack is not None:
            self.current_frame_idx = self.slider.value()
            self.display_frame()

    def zoom_image(self, factor):
        if self.tiff_stack is not None:
            self.scale_factor *= factor
            self.display_frame()

    def display_frame(self):
        if self.tiff_stack is None:
            return
        raw_img = self.tiff_stack[self.current_frame_idx]
        norm_img = cv2_normalize(raw_img)
        height, width = norm_img.shape
        bytes_per_line = width
        q_img = QImage(norm_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        self.lbl_image.setPixmap(pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.FastTransformation))
        self.lbl_image.resize(new_width, new_height)
        self.lbl_frame_count.setText(f"{self.current_frame_idx + 1}/{self.total_frames}")


def cv2_normalize(img):
    img = img.astype(float)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    img = (img - min_val) / (max_val - min_val) * 255.0
    return img.astype(np.uint8)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = TiffViewer()
    viewer.show()
    sys.exit(app.exec_())