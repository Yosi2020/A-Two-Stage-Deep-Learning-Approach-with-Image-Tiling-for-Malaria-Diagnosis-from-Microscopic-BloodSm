import sys
import sqlite3
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QIcon, QFont, QPainter, QColor, QBrush, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QHBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog,
    QMessageBox, QMenu, QGraphicsOpacityEffect
)
import os
import cv2
import torch
from PIL import Image
import numpy as np
import tensorflow as tf
import pybboxes as pbx
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from PIL import ImageQt
import torch
from ultralytics import YOLO
yolo_utils_path = os.path.abspath(r'C:\Windows\System32\yolov9\utils')
if yolo_utils_path not in sys.path:
    sys.path.insert(0, yolo_utils_path)

from yolov9.models.yolo import Model
from yolov9.utils.augmentations import letterbox
from yolov9.utils.general import non_max_suppression, scale_boxes
import time

# Database setup
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users
             (email text unique, password text, hospital_name text, position text)''')
conn.commit()


class LoginWindow(QWidget):
    def __init__(self, parent=None):
        super(LoginWindow, self).__init__(parent)
        self.setupUI()

    def setupUI(self):
        self.setStyleSheet("background-color: #2c3e50;")
        layout = QVBoxLayout(self)

        title = QLabel("Login")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 32px; color: white; ")
        layout.addWidget(title)

        self.email = QLineEdit(self)
        self.email.setPlaceholderText("Email")
        self.email.setStyleSheet("QLineEdit { border: 2px solid gray; border-radius: 10px; padding: 0 8px; background: white; selection-background-color: darkgray; font-size: 20px; }")
        layout.addWidget(self.email)

        self.password = QLineEdit(self)
        self.password.setEchoMode(QLineEdit.Password)  # Hide password text
        self.password.setPlaceholderText("Password")
        self.password.setStyleSheet("QLineEdit { border: 2px solid gray; border-radius: 10px; padding: 0 8px; background: white; selection-background-color: darkgray; font-size: 20px; }")
        layout.addWidget(self.password)

        forgot_password_button = QPushButton('Forgot Password?', self)
        forgot_password_button.setStyleSheet("QPushButton { background-color: transparent; border: none; color: white; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; }")
        forgot_password_button.clicked.connect(self.handle_forgot_password)
        layout.addWidget(forgot_password_button)

        login_button = QPushButton('Login', self)
        login_button.setStyleSheet("QPushButton { background-color: #27ae60; border: none; color: white; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px; }")
        login_button.clicked.connect(self.handle_login)
        layout.addWidget(login_button)

        signup_button = QPushButton('Sign Up', self)
        signup_button.setStyleSheet("QPushButton { background-color: #3498db; border: none; color: white; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px; }")
        signup_button.clicked.connect(self.handle_signup)
        layout.addWidget(signup_button)

        self.setLayout(layout)

    def handle_login(self):
        email = self.email.text()
        password = self.password.text()

        c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
        result = c.fetchone()
        if result:
            #QMessageBox.information(self, 'Login Successful', 'You have successfully logged in.')
            # Pass user details to MainWindow
            self.parent().user_details = {
                'Email': result[0],
                'Hospital Name': result[2],
                'Position': result[3]
            }
            self.parent().show_selector()
            # Here you would switch to the main application window
        else:
            QMessageBox.warning(self, 'Login Failed', 'The email or password is incorrect.')

    def handle_signup(self):
        self.parent().show_signup()

    def handle_forgot_password(self):
        # Placeholder for forgot password logic
        QMessageBox.information(self, 'Reset Password', 'Password reset is not implemented.')


class SignUpWindow(QWidget):
    def __init__(self, parent=None):
        super(SignUpWindow, self).__init__(parent)
        self.setupUI()

    def setupUI(self):
        self.setStyleSheet("background-color: #34495e;")
        layout = QVBoxLayout(self)

        title = QLabel("Sign Up")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 32px; color: white;")
        layout.addWidget(title)

        self.name = QLineEdit(self)
        self.name.setPlaceholderText("Name")
        self.name.setStyleSheet("QLineEdit { border: 2px solid gray; border-radius: 10px; padding: 0 8px; background: white; selection-background-color: darkgray; font-size: 20px; }")
        layout.addWidget(self.name)

        self.email = QLineEdit(self)
        self.email.setPlaceholderText("Email")
        self.email.setStyleSheet("QLineEdit { border: 2px solid gray; border-radius: 10px; padding: 0 8px; background: white; selection-background-color: darkgray; font-size: 20px; }")
        layout.addWidget(self.email)

        self.hospital_name = QLineEdit(self)
        self.hospital_name.setPlaceholderText("Hospital Name")
        self.hospital_name.setStyleSheet("QLineEdit { border: 2px solid gray; border-radius: 10px; padding: 0 8px; background: white; selection-background-color: darkgray; font-size: 20px; }")
        layout.addWidget(self.hospital_name)

        self.position = QLineEdit(self)
        self.position.setPlaceholderText("Position")
        self.position.setStyleSheet("QLineEdit { border: 2px solid gray; border-radius: 10px; padding: 0 8px; background: white; selection-background-color: darkgray; font-size: 20px; }")
        layout.addWidget(self.position)

        self.password = QLineEdit(self)
        self.password.setEchoMode(QLineEdit.Password)  # Hide password text
        self.password.setPlaceholderText("Password")
        self.password.setStyleSheet("QLineEdit { border: 2px solid gray; border-radius: 10px; padding: 0 8px; background: white; selection-background-color: darkgray; font-size: 20px; }")
        layout.addWidget(self.password)

        self.password_confirm = QLineEdit(self)
        self.password_confirm.setEchoMode(QLineEdit.Password)  # Hide password text
        self.password_confirm.setPlaceholderText("Confirm Password")
        self.password_confirm.setStyleSheet("QLineEdit { border: 2px solid gray; border-radius: 10px; padding: 0 8px; background: white; selection-background-color: darkgray; font-size: 20px; }")
        layout.addWidget(self.password_confirm)

        signup_button = QPushButton('Sign Up', self)
        signup_button.setStyleSheet("QPushButton { background-color: #2980b9; border: none; color: white; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px; }")
        signup_button.clicked.connect(self.handle_signup)
        layout.addWidget(signup_button)

        back_button = QPushButton('Back to Login', self)
        back_button.setStyleSheet("QPushButton { background-color: #95a5a6; border: none; color: white; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px; }")
        back_button.clicked.connect(self.handle_back)
        layout.addWidget(back_button)

        self.setLayout(layout)

    def handle_signup(self):
        name = self.name.text()
        email = self.email.text()
        hospital_name = self.hospital_name.text()
        position = self.position.text()
        password = self.password.text()
        password_confirm = self.password_confirm.text()

        if password != password_confirm:
            QMessageBox.warning(self, 'Error', 'Passwords do not match.')
            return

        try:
            c.execute("INSERT INTO users (email, password, hospital_name, position) VALUES (?, ?, ?, ?)",
                      (email, password, hospital_name, position))
            conn.commit()
            QMessageBox.information(self, 'Success', 'You have successfully registered.')
            self.parent().show_login()
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, 'Error', 'Email already exists.')

    def handle_back(self):
        self.parent().show_login()

class AnimatedButton(QPushButton):
    def __init__(self, title, icon_path, parent=None):
        super().__init__(title, parent)
        self.setIcon(QIcon(icon_path))
        self.setIconSize(QtCore.QSize(64, 64))  # Icon size
        # Text will be under the icon by default for QPushButton with both text and icon set
        self.setFont(QFont('Arial', 12))
        self.setFixedSize(200, 100)  # Button size
        
        # Set initial style
        self.setStyleSheet("""
            AnimatedButton {
                border: 2px solid #3498db;
                border-radius: 50px;  # Rounded corners
                color: white;
                padding: 10px;
                text-align: center;
                font-weight: bold;
                font-size: 16px;
            }
            AnimatedButton:hover {
                background-color: #2980b9;
                border-color: #2980b9;
            }
        """)
        self.initUI()
        
    def initUI(self):
        # Fade-in effect
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.fade_animation = QtCore.QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setStartValue(0)
        self.fade_animation.setEndValue(1)
        self.fade_animation.setDuration(1500)

    def fadeIn(self):
        self.fade_animation.start()

class MainWidget(QWidget):
    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)
        self.image_path = None  # Initialize the image_path attribute
        self.setupUI()
        self.model_detection = self.load_detection_model('best.pt', 'gelan-c.yaml')
        self.model_classification = self.load_classify_model()
        self.filePath = None
        self.stitched_image = None

    def load_classify_model(self):
        # Load and return the Keras model from a .h5 file
        model = tf.keras.models.load_model('best_model_mobilevit.h5')
        return model

    def setupUI(self):
        # Set the background color to match the login page
        self.setStyleSheet("background-color: #2c3e50;")  # Replace with your login page's color
        
        self.layout = QVBoxLayout(self)
        
        # Header Layout
        self.header_layout = QHBoxLayout()
        self.header = QWidget()
        self.header.setStyleSheet("background-color: #2c3e50;")  # Set header background color
        self.header.setLayout(self.header_layout)
        icon_size = QtCore.QSize(40, 40)
        
        # Menu Button on the left corner
        self.menu_button = QPushButton('☰')
        self.menu_button.setIconSize(icon_size)  # Set the icon size
        self.menu_button.setFixedSize(icon_size) 
        self.menu_button.setStyleSheet("background-color: #2c3e50; color: white; border: none;")
        self.menu_button.clicked.connect(self.show_menu)
        self.header_layout.addWidget(self.menu_button, alignment=QtCore.Qt.AlignLeft)
        
        # Title at the center
        self.title = QLabel("Malaria Diagnosis Application")
        self.title.setStyleSheet("color: white;")
        self.title.setFont(QFont('Arial', 24))  # Setting font size to 24pt
        self.header_layout.addWidget(self.title, alignment=QtCore.Qt.AlignCenter)
        
        self.subtitle = QLabel("for Thick Blood Smear")
        self.subtitle.setStyleSheet("color: white;")  # Optionally, make the subtitle font smaller or different style
        self.subtitle.setFont(QFont('Arial', 14))  # Smaller font size for subtitle
        self.header_layout.addWidget(self.subtitle, alignment=QtCore.Qt.AlignCenter)
        
        # Detect if we are running in a bundle or live system
        if getattr(sys, 'frozen', False):
            # If the application is frozen (packaged by PyInstaller), use the system's _MEIPASS directory
            bundle_dir = sys._MEIPASS
        else:
            # If the application is running live, use the directory of the script
            bundle_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the icon path relative to the bundle_dir
        icon_path = os.path.join(bundle_dir, "R.png")

        # Profile Button on the right corner
        self.profile_button = QPushButton()
        self.profile_button.setIcon(QIcon(icon_path))
        #self.profile_button.setIcon(QIcon("R.png"))
        self.menu_button.setIconSize(icon_size)  # Set the icon size
        self.menu_button.setFixedSize(icon_size) 
        self.profile_button.setStyleSheet("background-color: #2c3e50; border: none;")
        self.profile_button.clicked.connect(self.show_user_profile)
        self.header_layout.addWidget(self.profile_button, alignment=QtCore.Qt.AlignRight)
        
        self.layout.addWidget(self.header)
        # Adjusting the layout height to 'x' pt (adjust 'x' as per requirement)
        self.layout.setContentsMargins(0, 0, 0, 40)  # Replace 'x' with the actual value
 
        # Set the button size for all buttons to be uniform
        button_size = QtCore.QSize(100, 40)

        # Body with blue background at the bottom
        '''
        self.description = QLabel("This application will work on detecting malaria.")
        self.description.setStyleSheet("color: black; font-size: 12pt; background-color: rgb(240, 240, 240);")
        self.layout.addWidget(self.description, alignment=QtCore.Qt.AlignCenter)
        '''
        # Set up a horizontal layout for the zoom buttons
        self.zoom_button_layout = QHBoxLayout()

        self.upload_button = QPushButton('Upload Image')
        self.upload_button.setStyleSheet("background-color: #3498db; color: white; padding: 10px;")
        self.upload_button.setFixedSize(button_size)
        self.upload_button.clicked.connect(self.upload_image)
        
        # Zoom In button setup
        self.zoom_in_button = QPushButton('Zoom In')
        self.zoom_in_button.setStyleSheet("background-color: #3498db; color: #FFFFFF; padding: 5px;")
        self.zoom_in_button.setFixedSize(button_size)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        
        # Zoom Out button setup
        self.zoom_out_button = QPushButton('Zoom Out')
        self.zoom_out_button.setStyleSheet("background-color: #3498db; color: #FFFFFF; padding: 5px;")
        self.zoom_out_button.setFixedSize(button_size)
        self.zoom_out_button.clicked.connect(self.zoom_out)

        # Save Image button setup
        self.save_button = QPushButton('Save Image')
        self.save_button.setStyleSheet("background-color: #3498db; color: #FFFFFF; padding: 5px;")
        self.save_button.setFixedSize(button_size)
        self.save_button.clicked.connect(self.save_image)

        
        # Add Zoom In and Zoom Out buttons to the horizontal layout
        self.zoom_button_layout.addWidget(self.upload_button)
        self.zoom_button_layout.addWidget(self.zoom_in_button)
        self.zoom_button_layout.addWidget(self.zoom_out_button)
        self.zoom_button_layout.addWidget(self.save_button)
        self.zoom_button_layout.setAlignment(QtCore.Qt.AlignCenter)  # Center the zoom buttons
        
        # Add the zoom buttons layout to the main vertical layout
        self.layout.addLayout(self.zoom_button_layout)


        # Image display area
        self.image_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        self.image_view.setScene(self.scene)
        self.layout.addWidget(self.image_view)
        
        # Predict button setup
        self.predict_button = QPushButton('Predict')
        self.predict_button.setStyleSheet("background-color: #e74c3c; color: white; padding: 5px;")
        self.predict_button.setFixedSize(button_size)
        self.predict_button.clicked.connect(self.predict)
        
        # Add the Predict button to the main layout and center it
        self.layout.addWidget(self.predict_button, alignment=QtCore.Qt.AlignCenter)

        self.detection_label = QLabel()
        self.detection_label.setText("Detections: 0")
        self.detection_label.setStyleSheet("color: white; font-size: 12pt; background-color: #2c3e50;")
        self.detection_label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.detection_label)

        # Thin Blood Smear Analysis Button Setup
        self.thin_smear_button = QPushButton('Go to Thin Blood Smear Analysis')
        self.thin_smear_button.setStyleSheet("background-color: #3498db; color: white; padding: 10px;")
        self.thin_smear_button.setMinimumSize(QtCore.QSize(200, 40))
        self.thin_smear_button.clicked.connect(self.handle_thin)
        self.thin_smear_button.hide()  # Initially hide the button
        self.layout.addWidget(self.thin_smear_button, alignment=QtCore.Qt.AlignCenter)

        # Initially hide the control buttons
        self.zoom_in_button.hide()
        self.zoom_out_button.hide()
        self.save_button.hide()
        self.predict_button.hide()
        self.detection_label.hide()

        self.setLayout(self.layout)

        back_button = QPushButton('Back to Selector', self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        back_button.setStyleSheet("QPushButton { background-color: #95a5a6; border: none; color: white; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px; }")
        back_button.clicked.connect(self.handle_back)
        self.layout.addWidget(back_button)

    def handle_back(self):
        self.parent().show_selector()

    def handle_thin(self):
        self.parent().show_thin()

    # Method implementations for button clicks
    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            cropped_file_name = self.crop_to_content(file_name)  # Crop the image
            self.image_path = cropped_file_name  # Update the image path to the path of the cropped image
            self.display_image(self.image_path)  # Display the cropped image
            self.zoom_in_button.show()
            self.zoom_out_button.show()
            self.predict_button.show()

            # Hide the detection label when a new image is uploaded
            if self.detection_label:
                self.detection_label.hide()

    def display_image(self, file_name):
        # Load and display the image in QGraphicsView
        pixmap = QPixmap(file_name)
        self.image_item.setPixmap(pixmap)
        self.scene.setSceneRect(QtCore.QRectF(pixmap.rect()))  # Correctly set scene rect with QRectF
        self.image_view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
    
    def crop_to_content(self, file_name):
        # Open the image file
        img = Image.open(file_name)
        # Convert to numpy array
        img_array = np.array(img)

        # Define a threshold to consider if a pixel is 'black'
        threshold = 100  

        # Find the rows and columns with any values above the threshold
        rows = np.any(img_array.max(axis=2) > threshold, axis=1)
        cols = np.any(img_array.max(axis=2) > threshold, axis=0)

        # Find the indices of the first and last rows and columns where we have pixels above the threshold
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        if not len(row_indices) or not len(col_indices):  # Check if the image is completely black
            return file_name  # Return the original file name if the image is completely black

        # Calculate the bounding box of the non-black pixels
        top_row, bottom_row = row_indices[0], row_indices[-1]
        left_col, right_col = col_indices[0], col_indices[-1]

        # Crop the image to that bounding box considering all sides
        cropped_img = img.crop((left_col, top_row, right_col + 1, bottom_row + 1))

        # Save the cropped image to a new file
        cropped_file_name = f"{file_name.rsplit('.', 1)[0]}_cropped.{file_name.rsplit('.', 1)[1]}"
        cropped_img.save(cropped_file_name)

        return cropped_file_name

    def zoom_in(self):
        # Functionality to zoom in on the image
        self.image_view.scale(1.25, 1.25)

    def zoom_out(self):
        # Functionality to zoom out on the image
        self.image_view.scale(0.8, 0.8)

    # Inside your MainWidget class:
    def predict(self):
        if not self.image_path:
            QMessageBox.warning(self, 'Error', 'Please upload an image first.')
            return

        load_start_time = time.time()
        image = cv2.imread(self.image_path)
        if image is None:
            QMessageBox.warning(self, 'Error', 'Failed to load image.')
            return

        load_time = time.time() - load_start_time

        preprocess_start_time = time.time()

        tiles = self.tile_image(image)
        preprocess_time = time.time() - preprocess_start_time

        all_detections = []

        inference_time_total = 0  # Initialize total inference time

        for x_offset, y_offset, tile in tiles:
            inference_start_time = time.time()
            detections = self.detect_objects(tile, self.model_detection)
            inference_time_total += time.time() - inference_start_time
            
            postprocess_start_time = time.time()

            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                score = det['confidence']
                class_id = det['class_id']
                cropped_image = self.crop_center(tile, (x1, y1, x2, y2))
                predicted_class_id = self.classify_image(cropped_image)
                if class_id == predicted_class_id:
                    global_box = (x_offset + x1, y_offset + y1, x_offset + x2, y_offset + y2)
                    all_detections.append((global_box, score, predicted_class_id))

            postprocess_time = time.time() - postprocess_start_time


        for (x1, y1, x2, y2), score, class_id in all_detections:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 10)
            label = f"Class: P, Score: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)

        self.update_image_display(image)
        self.stitched_image = image.copy()

        self.predict_button.hide()

        print(f"Detections: {len(all_detections)}")

        if len(all_detections) > 0:
            self.thin_smear_button.show()
            self.detection_label.setText(f"Detections: {len(all_detections)} Malaria Parasites")
        else:
            self.detection_label.setText("Malaria free thick blood smear")
            self.thin_smear_button.hide()

        self.detection_label.show()

        self.save_button.show()

        # Print out times
        print(f"Load Time: {load_time:.2f} seconds")
        print(f"Preprocess Time: {preprocess_time:.2f} seconds")
        print(f"Inference Time: {inference_time_total:.2f} seconds")
        print(f"Postprocess Time: {postprocess_time:.2f} seconds")
        print(f"Total Time: {time.time() - load_start_time:.2f} seconds")


    def update_image_display(self, stitched_image):
        # Convert the result image to QImage
        height, width, channel = stitched_image.shape
        bytes_per_line = 3 * width
        qImg = QImage(stitched_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)

        # Set the pixmap to the image item and update the scene
        self.image_item.setPixmap(pixmap)
        self.scene.setSceneRect(QtCore.QRectF(pixmap.rect()))  # Update scene rect
        self.image_view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    
    def tile_image(self, image, tile_size=640, overlap=0.3):
        tiles = []
        ih, iw = image.shape[:2]
        stride = int(tile_size * (1 - overlap))  # Compute stride as tile_size less overlap
        ti = 0

        # Iterate over the image with the computed stride to create tiles
        for y in range(0, ih - tile_size + 1, stride):
            for x in range(0, iw - tile_size + 1, stride):
                tile = image[y:y + tile_size, x:x + tile_size]
                # Store the tile with its coordinates (x, y, width, height)
                tiles.append((x, y, tile))
                ti += 1
        print(f"tile : {ti}")
        return tiles


    def load_detection_model(self, weights_path, cfg_path):
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        model = Model(cfg=cfg_path, ch=3, nc=1)  # Adjust `nc` for the actual number of classes
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'].float().state_dict())
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model

    def detect_objects(self, tile, model):
        #print(f"Type of model before calling .to(): {type(model)}")  # Debugging line

        if not isinstance(model, torch.nn.Module):
            raise ValueError("Model is not a PyTorch model instance. It is: {}".format(type(model)))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        # Continue with image processing and inference as before
        # Load and preprocess the image
        img = letterbox(tile, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float() / 255.0
        img = img.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            results = model(img)
        detections = non_max_suppression(results, 0.25, 0.45, classes=None, agnostic=False)

        det_list = []
        for det in detections[0]:  # Assuming the batch size is 1
            print("Detection tuple:", det)
            x1, y1, x2, y2, conf, cls = det
            if conf < 0.7:
                continue
            det_dict = {
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf),
                'class_id': int(cls)
            }
            det_list.append(det_dict)

        return det_list

    def crop_center(self, image, bbox):
        x_min, y_min, x_max, y_max = bbox
        print(x_min, y_min, x_max, y_max)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # Radius of 48 pixels results in a 96x96 crop
        radius = 48
        crop_width = crop_height = 2 * radius

        # Calculate the top-left corner of the cropping area
        start_x = max(center_x - radius, 0)
        start_y = max(center_y - radius, 0)

        # Ensure the crop dimensions do not exceed the image bounds
        end_x = min(start_x + crop_width, image.shape[1])
        end_y = min(start_y + crop_height, image.shape[0])

        # Adjust if the calculated area is out of bounds or smaller than desired
        if (end_x - start_x) < crop_width:
            if start_x == 0:
                end_x = min(crop_width, image.shape[1])
            else:
                start_x = max(end_x - crop_width, 0)
        if (end_y - start_y) < crop_height:
            if start_y == 0:
                end_y = min(crop_height, image.shape[0])
            else:
                start_y = max(end_y - crop_height, 0)

        # Crop the image and return
        cropped_image = image[start_y:end_y, start_x:end_x]
        print(cropped_image)
        return cropped_image


    def classify_image(self, image):
        resized_image = cv2.resize(image, (96, 96))
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = resized_image.astype('float32') / 255.0
        resized_image = np.expand_dims(resized_image, axis=0)
        prediction = self.model_classification.predict(resized_image)
        predicted_class_id = np.argmax(prediction, axis=1)[0]
        print(f'prediction is {predicted_class_id}')
        return predicted_class_id

    def save_image(self):
        if hasattr(self, 'stitched_image') and self.stitched_image is not None:
            filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg)")
            if filePath:
                # Convert the image from BGR to RGB since OpenCV uses BGR by default
                rgb_image = cv2.cvtColor(self.stitched_image, cv2.COLOR_BGR2RGB)
                # Save the image using OpenCV
                success = cv2.imwrite(filePath, rgb_image)
                if success:
                    print("Image saved successfully.")
                else:
                    print("Error saving image.")
        else:
            print("No stitched image to save.")



    def show_user_profile(self):
        # Display user profile details - retrieve details from the database
        QMessageBox.information(self, 'Profile', 'User profile details go here.')

    def show_menu(self):
        menu = QMenu()
        home_action = menu.addAction("Home")
        terms_action = menu.addAction("Term & Policies")
        developers_action = menu.addAction("Developers")
        # Connect actions to methods, for example:
        home_action.triggered.connect(self.go_home)
        terms_action.triggered.connect(self.show_terms)
        developers_action.triggered.connect(self.show_developers)
        
        # Position the menu to appear below the button
        menu.exec_(self.menu_button.mapToGlobal(QtCore.QPoint(0, self.menu_button.height())))

    def go_home(self):
        # Method to handle "Home" action
        pass

    def show_terms(self):
        # Method to handle "Term & Policies" action
        pass

    def show_developers(self):
        # Method to handle "Developers" action
        pass

    def show_user_profile(self):
        # Check if user details are set
        if not self.parent().user_details:
            QMessageBox.warning(self, 'Not logged in', 'No user is currently logged in.')
            return

        # Create a message with the user details
        user_info = self.parent().user_details
        message = "\n".join(f"{key}: {value}" for key, value in user_info.items())

        QMessageBox.information(self, 'Profile', message)


class MainWidget3(QWidget):
    def __init__(self, parent=None):
        super(MainWidget3, self).__init__(parent)
        self.image_path = None  # Initialize the image_path attribute
        self.setupUI()
        self.model_detection = self.load_detection_model('thinbest.pt', 'gelan-c.yaml')
        self.model_classification = self.load_classify_model()
        self.filePath = None
        self.stitched_image = None

    def load_classify_model(self):
        # Load and return the Keras model from a .h5 file
        model = tf.keras.models.load_model('thinbest_model_mobilevit.h5')
        return model

    def setupUI(self):
        # Set the background color to match the login page
        self.setStyleSheet("background-color: #2c3e50;")  # Replace with your login page's color
        
        self.layout = QVBoxLayout(self)
        
        # Header Layout
        self.header_layout = QHBoxLayout()
        self.header = QWidget()
        self.header.setStyleSheet("background-color: #2c3e50;")  # Set header background color
        self.header.setLayout(self.header_layout)
        icon_size = QtCore.QSize(40, 40)
        
        # Menu Button on the left corner
        self.menu_button = QPushButton('☰')
        self.menu_button.setIconSize(icon_size)  # Set the icon size
        self.menu_button.setFixedSize(icon_size) 
        self.menu_button.setStyleSheet("background-color: #2c3e50; color: white; border: none;")
        self.menu_button.clicked.connect(self.show_menu)
        self.header_layout.addWidget(self.menu_button, alignment=QtCore.Qt.AlignLeft)
        
        # Title at the center
        self.title = QLabel("Malaria Diagnosis Application")
        self.title.setStyleSheet("color: white;")
        self.title.setFont(QFont('Arial', 24))  # Setting font size to 24pt
        self.header_layout.addWidget(self.title, alignment=QtCore.Qt.AlignCenter)
        
        self.subtitle = QLabel("For Thin Blood Smear")
        self.subtitle.setStyleSheet("color: white;")  # Optionally, make the subtitle font smaller or different style
        self.subtitle.setFont(QFont('Arial', 14))  # Smaller font size for subtitle
        self.header_layout.addWidget(self.subtitle, alignment=QtCore.Qt.AlignCenter)

        # Detect if we are running in a bundle or live system
        if getattr(sys, 'frozen', False):
            # If the application is frozen (packaged by PyInstaller), use the system's _MEIPASS directory
            bundle_dir = sys._MEIPASS
        else:
            # If the application is running live, use the directory of the script
            bundle_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the icon path relative to the bundle_dir
        icon_path = os.path.join(bundle_dir, "R.png")

        # Profile Button on the right corner
        self.profile_button = QPushButton()
        self.profile_button.setIcon(QIcon(icon_path))
        #self.profile_button.setIcon(QIcon("R.png"))
        self.menu_button.setIconSize(icon_size)  # Set the icon size
        self.menu_button.setFixedSize(icon_size) 
        self.profile_button.setStyleSheet("background-color: #2c3e50; border: none;")
        self.profile_button.clicked.connect(self.show_user_profile)
        self.header_layout.addWidget(self.profile_button, alignment=QtCore.Qt.AlignRight)
        
        self.layout.addWidget(self.header)
        # Adjusting the layout height to 'x' pt (adjust 'x' as per requirement)
        self.layout.setContentsMargins(0, 0, 0, 40)  # Replace 'x' with the actual value
 
        # Set the button size for all buttons to be uniform
        button_size = QtCore.QSize(100, 40)

        # Body with blue background at the bottom
        '''
        self.description = QLabel("This application will work on detecting malaria.")
        self.description.setStyleSheet("color: black; font-size: 12pt; background-color: rgb(240, 240, 240);")
        self.layout.addWidget(self.description, alignment=QtCore.Qt.AlignCenter)
        '''
        # Set up a horizontal layout for the zoom buttons
        self.zoom_button_layout = QHBoxLayout()

        self.upload_button = QPushButton('Upload Image')
        self.upload_button.setStyleSheet("background-color: #3498db; color: white; padding: 10px;")
        self.upload_button.setFixedSize(button_size)
        self.upload_button.clicked.connect(self.upload_image)
        
        # Zoom In button setup
        self.zoom_in_button = QPushButton('Zoom In')
        self.zoom_in_button.setStyleSheet("background-color: #3498db; color: #FFFFFF; padding: 5px;")
        self.zoom_in_button.setFixedSize(button_size)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        
        # Zoom Out button setup
        self.zoom_out_button = QPushButton('Zoom Out')
        self.zoom_out_button.setStyleSheet("background-color: #3498db; color: #FFFFFF; padding: 5px;")
        self.zoom_out_button.setFixedSize(button_size)
        self.zoom_out_button.clicked.connect(self.zoom_out)

        # Save Image button setup
        self.save_button = QPushButton('Save Image')
        self.save_button.setStyleSheet("background-color: #3498db; color: #FFFFFF; padding: 5px;")
        self.save_button.setFixedSize(button_size)
        self.save_button.clicked.connect(self.save_image)

        
        # Add Zoom In and Zoom Out buttons to the horizontal layout
        self.zoom_button_layout.addWidget(self.upload_button)
        self.zoom_button_layout.addWidget(self.zoom_in_button)
        self.zoom_button_layout.addWidget(self.zoom_out_button)
        self.zoom_button_layout.addWidget(self.save_button)
        self.zoom_button_layout.setAlignment(QtCore.Qt.AlignCenter)  # Center the zoom buttons
        
        # Add the zoom buttons layout to the main vertical layout
        self.layout.addLayout(self.zoom_button_layout)


        # Image display area
        self.image_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        self.image_view.setScene(self.scene)
        self.layout.addWidget(self.image_view)
        
        # Predict button setup
        self.predict_button = QPushButton('Predict')
        self.predict_button.setStyleSheet("background-color: #e74c3c; color: white; padding: 5px;")
        self.predict_button.setFixedSize(button_size)
        self.predict_button.clicked.connect(self.predict)
        
        # Add the Predict button to the main layout and center it
        self.layout.addWidget(self.predict_button, alignment=QtCore.Qt.AlignCenter)

        self.detection_label = QLabel()
        self.detection_label.setText("Detections: 0")
        self.detection_label.setStyleSheet("color: white; font-size: 12pt; background-color: #2c3e50;")
        self.detection_label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.detection_label)

        # Initially hide the control buttons
        self.zoom_in_button.hide()
        self.zoom_out_button.hide()
        self.save_button.hide()
        self.predict_button.hide()
        self.detection_label.hide()

        self.setLayout(self.layout)

        back_button = QPushButton('Back to Selector', self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        back_button.setStyleSheet("QPushButton { background-color: #95a5a6; border: none; color: white; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px; }")
        back_button.clicked.connect(self.handle_back)
        self.layout.addWidget(back_button)

    def handle_back(self):
        self.parent().show_selector()

    # Method implementations for button clicks
    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            cropped_file_name = self.crop_to_content(file_name)  # Crop the image
            self.image_path = cropped_file_name  # Update the image path to the path of the cropped image
            self.display_image(self.image_path)  # Display the cropped image
            self.zoom_in_button.show()
            self.zoom_out_button.show()
            self.predict_button.show()

            # Hide the detection label when a new image is uploaded
            if self.detection_label:
                self.detection_label.hide()

    def display_image(self, file_name):
        # Load and display the image in QGraphicsView
        pixmap = QPixmap(file_name)
        self.image_item.setPixmap(pixmap)
        self.scene.setSceneRect(QtCore.QRectF(pixmap.rect()))  # Correctly set scene rect with QRectF
        self.image_view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
    
    def crop_to_content(self, file_name):
        # Open the image file
        img = Image.open(file_name)
        # Convert to numpy array
        img_array = np.array(img)

        # Define a threshold to consider if a pixel is 'black'
        threshold = 100  

        # Find the rows and columns with any values above the threshold
        rows = np.any(img_array.max(axis=2) > threshold, axis=1)
        cols = np.any(img_array.max(axis=2) > threshold, axis=0)

        # Find the indices of the first and last rows and columns where we have pixels above the threshold
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        if not len(row_indices) or not len(col_indices):  # Check if the image is completely black
            return file_name  # Return the original file name if the image is completely black

        # Calculate the bounding box of the non-black pixels
        top_row, bottom_row = row_indices[0], row_indices[-1]
        left_col, right_col = col_indices[0], col_indices[-1]

        # Crop the image to that bounding box considering all sides
        cropped_img = img.crop((left_col, top_row, right_col + 1, bottom_row + 1))

        # Save the cropped image to a new file
        cropped_file_name = f"{file_name.rsplit('.', 1)[0]}_cropped.{file_name.rsplit('.', 1)[1]}"
        cropped_img.save(cropped_file_name)

        return cropped_file_name

    def zoom_in(self):
        # Functionality to zoom in on the image
        self.image_view.scale(1.25, 1.25)

    def zoom_out(self):
        # Functionality to zoom out on the image
        self.image_view.scale(0.8, 0.8)

    # Inside your MainWidget class:
    def predict(self):
        if not self.image_path:
            QMessageBox.warning(self, 'Error', 'Please upload an image first.')
            return

        image = cv2.imread(self.image_path)
        if image is None:
            QMessageBox.warning(self, 'Error', 'Failed to load image.')
            return

        tiles = self.tile_image(image)
        all_detections = []
        class_ids = []

        for x_offset, y_offset, tile in tiles:
            detections = self.detect_objects(tile, self.model_detection)
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                score = det['confidence']
                class_id = det['class_id']
                #cropped_image = self.crop_center(tile, (x1, y1, x2, y2))
                #predicted_class_id = self.classify_image(cropped_image)
                #if class_id == predicted_class_id:
                global_box = (x_offset + x1, y_offset + y1, x_offset + x2, y_offset + y2)
                all_detections.append((global_box, score, class_id))
                class_ids.append(class_id)

        clas = ['PV', 'PF']

        for (x1, y1, x2, y2), score, class_id in all_detections:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 10)
            label = f"Class: {clas[class_id]}, Score: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print(f"Class: {clas[class_id]}, Score: {score:.2f}")

        self.update_image_display(image)
        self.stitched_image = image.copy()

        self.predict_button.hide()

        print(f"Detections: {len(all_detections)}")

        if not class_ids:
            self.detection_label.setText("No detections found.")
        if all(c == 1 for c in class_ids):
            self.detection_label.setText("Detections result are P.falciparum")
        elif all(c == 0 for c in class_ids):
            self.detection_label.setText("Detections result are P.vivax")
        else:
            self.detection_label.setText("Detections result are Mixed")
            
        self.detection_label.show()

        self.save_button.show()


    def update_image_display(self, stitched_image):
        # Convert the result image to QImage
        height, width, channel = stitched_image.shape
        bytes_per_line = 3 * width
        qImg = QImage(stitched_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)

        # Set the pixmap to the image item and update the scene
        self.image_item.setPixmap(pixmap)
        self.scene.setSceneRect(QtCore.QRectF(pixmap.rect()))  # Update scene rect
        self.image_view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    
    def tile_image(self, image, tile_size=640, overlap=0.3):
        tiles = []
        ih, iw = image.shape[:2]
        stride = int(tile_size * (1 - overlap))  # Compute stride as tile_size less overlap
        ti = 0

        # Iterate over the image with the computed stride to create tiles
        for y in range(0, ih - tile_size + 1, stride):
            for x in range(0, iw - tile_size + 1, stride):
                tile = image[y:y + tile_size, x:x + tile_size]
                # Store the tile with its coordinates (x, y, width, height)
                tiles.append((x, y, tile))
                ti += 1
        print(f"tile : {ti}")
        return tiles

    
    def crop_center(self, image, bbox):
        x_min, y_min, x_max, y_max = bbox
        print(x_min, y_min, x_max, y_max)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # Radius of 48 pixels results in a 96x96 crop
        radius = 48
        crop_width = crop_height = 2 * radius

        # Calculate the top-left corner of the cropping area
        start_x = max(center_x - radius, 0)
        start_y = max(center_y - radius, 0)

        # Ensure the crop dimensions do not exceed the image bounds
        end_x = min(start_x + crop_width, image.shape[1])
        end_y = min(start_y + crop_height, image.shape[0])

        # Adjust if the calculated area is out of bounds or smaller than desired
        if (end_x - start_x) < crop_width:
            if start_x == 0:
                end_x = min(crop_width, image.shape[1])
            else:
                start_x = max(end_x - crop_width, 0)
        if (end_y - start_y) < crop_height:
            if start_y == 0:
                end_y = min(crop_height, image.shape[0])
            else:
                start_y = max(end_y - crop_height, 0)

        # Crop the image and return
        cropped_image = image[start_y:end_y, start_x:end_x]
        print(cropped_image)
        return cropped_image

    def load_detection_model(self, weights_path, cfg_path):
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        model = Model(cfg=cfg_path, ch=3, nc=2)  # Adjust `nc` for the actual number of classes
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'].float().state_dict())
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model

    def detect_objects(self, tile, model):
        #print(f"Type of model before calling .to(): {type(model)}")  # Debugging line

        if not isinstance(model, torch.nn.Module):
            raise ValueError("Model is not a PyTorch model instance. It is: {}".format(type(model)))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        # Continue with image processing and inference as before
        # Load and preprocess the image
        img = letterbox(tile, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float() / 255.0
        img = img.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            results = model(img)
        detections = non_max_suppression(results, 0.70, 0.7, classes=None, agnostic=False)

        det_list = []
        for det in detections[0]:  # Assuming the batch size is 1
            print("Detection tuple:", det)
            x1, y1, x2, y2, conf, cls = det
            if conf < 0.7:
                continue
            det_dict = {
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf),
                'class_id': int(cls)
            }
            det_list.append(det_dict)

        return det_list


    def classify_image(self, image):
        resized_image = cv2.resize(image, (96, 96))
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = resized_image.astype('float32') / 255.0
        resized_image = np.expand_dims(resized_image, axis=0)
        prediction = self.model_classification.predict(resized_image)
        predicted_class_id = np.argmax(prediction, axis=1)[0]
        print(f'prediction is {predicted_class_id}')
        return predicted_class_id

    def save_image(self):
        if hasattr(self, 'stitched_image') and self.stitched_image is not None:
            filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg)")
            if filePath:
                # Convert the image from BGR to RGB since OpenCV uses BGR by default
                rgb_image = cv2.cvtColor(self.stitched_image, cv2.COLOR_BGR2RGB)
                # Save the image using OpenCV
                success = cv2.imwrite(filePath, rgb_image)
                if success:
                    print("Image saved successfully.")
                else:
                    print("Error saving image.")
        else:
            print("No stitched image to save.")



    def show_user_profile(self):
        # Display user profile details - retrieve details from the database
        QMessageBox.information(self, 'Profile', 'User profile details go here.')

    def show_menu(self):
        menu = QMenu()
        home_action = menu.addAction("Home")
        terms_action = menu.addAction("Term & Policies")
        developers_action = menu.addAction("Developers")
        # Connect actions to methods, for example:
        home_action.triggered.connect(self.go_home)
        terms_action.triggered.connect(self.show_terms)
        developers_action.triggered.connect(self.show_developers)
        
        # Position the menu to appear below the button
        menu.exec_(self.menu_button.mapToGlobal(QtCore.QPoint(0, self.menu_button.height())))

    def go_home(self):
        # Method to handle "Home" action
        pass

    def show_terms(self):
        # Method to handle "Term & Policies" action
        pass

    def show_developers(self):
        # Method to handle "Developers" action
        pass

    def show_user_profile(self):
        # Check if user details are set
        if not self.parent().user_details:
            QMessageBox.warning(self, 'Not logged in', 'No user is currently logged in.')
            return

        # Create a message with the user details
        user_info = self.parent().user_details
        message = "\n".join(f"{key}: {value}" for key, value in user_info.items())

        QMessageBox.information(self, 'Profile', message)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Malaria Diagnosis Application')
        self.setGeometry(100, 100, 800, 600)
        self.show_login()
        self.user_details = None

    def show_login(self):
        self.login_widget = LoginWindow(self)
        self.setCentralWidget(self.login_widget)

    def show_signup(self):
        self.signup_widget = SignUpWindow(self)
        self.setCentralWidget(self.signup_widget)

    def show_selector(self):
        self.sel_widget = MainWidget(self)
        self.setCentralWidget(self.sel_widget)

    def show_thin(self):
        # Assume MainWidget3 is your thin blood smear analysis widget
        self.thin_widget = MainWidget3(self)
        self.setCentralWidget(self.thin_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
