import sys
import tifffile
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QFileDialog, QLabel, QSlider, QHBoxLayout,
                             QSizePolicy, QGridLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


class DualCalciumViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Raw vs Denoised")
        self.resize(1200, 800)


        self.data_before = None
        self.data_after = None
        self.total_frames = 0


        self.gamma = 0.5
        self.contrast_cut = 99

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)


        buttons_layout = QHBoxLayout()

        self.btn_load_before = QPushButton("Before")
        self.btn_load_before.setMinimumHeight(40)
        self.btn_load_before.clicked.connect(lambda: self.open_file(is_before=True))

        self.btn_load_after = QPushButton("After")
        self.btn_load_after.setMinimumHeight(40)
        self.btn_load_after.clicked.connect(lambda: self.open_file(is_before=False))

        buttons_layout.addWidget(self.btn_load_before)
        buttons_layout.addWidget(self.btn_load_after)
        main_layout.addLayout(buttons_layout)


        display_container = QHBoxLayout()


        self.container_before = QVBoxLayout()
        self.lbl_before = QLabel("No Raw Data")
        self.lbl_before.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_before.setStyleSheet("background-color: black; color: white; border: 1px solid #444;")
        self.lbl_before.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.container_before.addWidget(QLabel("before"))
        self.container_before.addWidget(self.lbl_before, stretch=1)


        self.container_after = QVBoxLayout()
        self.lbl_after = QLabel("No Denoised Data")
        self.lbl_after.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_after.setStyleSheet("background-color: black; color: white; border: 1px solid #444;")
        self.lbl_after.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.container_after.addWidget(QLabel("After"))
        self.container_after.addWidget(self.lbl_after, stretch=1)

        display_container.addLayout(self.container_before)
        display_container.addLayout(self.container_after)
        main_layout.addLayout(display_container, stretch=1)


        controls_group = QVBoxLayout()


        controls_group.addWidget(QLabel("Frame Navigation:"))
        self.slider_frame = QSlider(Qt.Orientation.Horizontal)
        self.slider_frame.setEnabled(False)
        self.slider_frame.valueChanged.connect(self.update_displays)
        controls_group.addWidget(self.slider_frame)

        self.lbl_frame_info = QLabel("Frame: 0 / 0")
        self.lbl_frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_group.addWidget(self.lbl_frame_info)


        sliders_grid = QGridLayout()


        sliders_grid.addWidget(QLabel("Gamma (Background Brightness):"), 0, 0)
        self.slider_gamma = QSlider(Qt.Orientation.Horizontal)
        self.slider_gamma.setRange(1, 40)
        self.slider_gamma.setValue(10)  # 1.0
        self.slider_gamma.valueChanged.connect(self.update_displays)
        sliders_grid.addWidget(self.slider_gamma, 0, 1)


        sliders_grid.addWidget(QLabel("Contrast (Max Percentile):"), 1, 0)
        self.slider_contrast = QSlider(Qt.Orientation.Horizontal)
        self.slider_contrast.setRange(80, 100)
        self.slider_contrast.setValue(99)
        self.slider_contrast.valueChanged.connect(self.update_displays)
        sliders_grid.addWidget(self.slider_contrast, 1, 1)

        controls_group.addLayout(sliders_grid)
        main_layout.addLayout(controls_group)

    def open_file(self, is_before=True):
        path, _ = QFileDialog.getOpenFileName(self, "Open TIFF File", "", "TIFF Files (*.tif *.tiff)")
        if not path:
            return

        try:

            mapped_data = tifffile.memmap(path)

            if is_before:
                self.data_before = mapped_data
                print(f"Loaded Before: {path}")
            else:
                self.data_after = mapped_data
                print(f"Loaded After: {path}")


            if self.total_frames == 0:
                self.total_frames = mapped_data.shape[0]
                self.slider_frame.setRange(0, self.total_frames - 1)
                self.slider_frame.setEnabled(True)

            self.update_displays()

        except Exception as e:
            print(f"Error loading file: {e}")

    def process_frame(self, frame):
        if frame is None:
            return None


        frame = frame.astype(np.float32)

        gamma_val = self.slider_gamma.value() / 10.0
        contrast_percent = self.slider_contrast.value()

        vmin = np.percentile(frame, 1)
        vmax = np.percentile(frame, contrast_percent)

        img = np.clip(frame, vmin, vmax)
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img)

        img = np.power(img + 1e-6, gamma_val)
        return (img * 255).astype(np.uint8)

    def update_displays(self):
        if self.data_before is None and self.data_after is None:
            return

        idx = self.slider_frame.value()
        self.lbl_frame_info.setText(f"Frame: {idx + 1} / {self.total_frames}")


        if self.data_before is not None:
            processed = self.process_frame(self.data_before[idx])
            self.set_label_pixmap(self.lbl_before, processed)


        if self.data_after is not None:
            processed = self.process_frame(self.data_after[idx])
            self.set_label_pixmap(self.lbl_after, processed)

    def set_label_pixmap(self, label, frame_8bit):
        h, w = frame_8bit.shape
        qimg = QImage(frame_8bit.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        scaled_pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):

        super().resizeEvent(event)
        self.update_displays()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DualCalciumViewer()
    window.show()
    sys.exit(app.exec())