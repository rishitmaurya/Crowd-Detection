from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSpacerItem, QSizePolicy, QPushButton, QFileDialog, QLabel
from PyQt5.QtCore import Qt
import sys
import cv2
import os

# Import the detect_people function from crowd-detection.py
from crowd_detection import detect_people

class VideoUploaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.video_path = None  # To store the path of the uploaded videoq
    
    def initUI(self):
        self.setWindowTitle("Video Uploader")
        self.setGeometry(100, 100, 800, 600)

        # Create header label
        self.header_label = QLabel("Upload Your Video", self)
        self.header_label.setStyleSheet("background-color: black; font-size: 18px; font-weight: bold; color: white;")
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setFixedHeight(40)

        # Create buttons with styles and fixed width
        self.upload_button = QPushButton("Upload Video", self)
        self.upload_button.setStyleSheet("background-color: #3498db; color: white; font-size: 14px; padding: 10px;")
        self.upload_button.setFixedWidth(250)
        self.upload_button.clicked.connect(self.upload_video)

        self.submit_button = QPushButton("Submit", self)
        self.submit_button.setStyleSheet("background-color: #2ecc71; color: white; font-size: 14px; padding: 10px;")
        self.submit_button.setFixedWidth(250)
        self.submit_button.clicked.connect(self.submit_action)

        # Label to show the uploaded video path
        self.video_label = QLabel("", self)
        self.video_label.setStyleSheet("font-size: 12px; color: black;")

        # Arrange widgets in layout
        layout = QVBoxLayout()
        layout.addWidget(self.header_label)
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addWidget(self.upload_button, alignment=Qt.AlignCenter)
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addWidget(self.submit_button, alignment=Qt.AlignCenter)
        layout.addSpacerItem(QSpacerItem(20, 50, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.setLayout(layout)
    
    def upload_video(self):
        # Open file dialog to select a video file
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        
        if video_path:
            self.video_path = video_path  # Store the path of the selected video
            self.video_label.setText(f"Selected video: {video_path}")

    def submit_action(self):
        # Ensure a video path has been set
        if not self.video_path:
            self.video_label.setText("Please upload a video first.")
            return

        # Call the detect_people function with the selected video path
        detect_people(self.video_path)  # This will display the detection output using OpenCV

def main():
    app = QApplication(sys.argv)
    window = VideoUploaderApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
