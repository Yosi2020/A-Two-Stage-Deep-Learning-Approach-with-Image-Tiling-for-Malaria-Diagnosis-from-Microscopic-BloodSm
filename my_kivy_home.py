from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.graphics import Color, Rectangle
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.graphics.transformation import Matrix
from kivy.graphics.context_instructions import MatrixInstruction
from kivy.graphics import Scale
from kivy.graphics.context_instructions import PushMatrix, PopMatrix
from kivy.core.window import Window
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
import numpy as np
import os
import cv2
import tensorflow as tf
import torch
from torchvision.transforms import functional as F
from yolov9.models.yolo import Model
from yolov9.utils.augmentations import letterbox
from yolov9.utils.general import non_max_suppression, scale_boxes

PROFILE_ICON_PATH = "R.png"

class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_path = None  
        self.matrix = Matrix()
        self.scale_factor = 1.0 
        self.model_detection = self.load_detection_model('best.pt', 'gelan-c.yaml')
        self.model_classification = self.load_classify_model()
        self.popup = None

        self.layout = BoxLayout(orientation='vertical')

        self.image_widget = Image(size_hint_y=0.7)

        with self.canvas.before:
            Color(35/255, 47/255, 62/255, 1)  # Background color
            self.rect = Rectangle(size=self.size, pos=self.pos)
            self.bind(size=self._update_rect, pos=self._update_rect)

        # Header Layout
        header_layout = BoxLayout(size_hint_y=None, height=60, padding=10)
        menu_button = Button(text='☰', size_hint_x=None, width=60)
        title = Label(text='Malaria Diagnosis Application', color=(1, 1, 1, 1))
        profile_button = Button(background_normal=PROFILE_ICON_PATH, size_hint_x=None, width=60)

        header_layout.add_widget(menu_button)
        header_layout.add_widget(title)
        header_layout.add_widget(profile_button)

        # Body
        description = Label(text='Thick Blood Smear Analysis', size_hint_y=None, height=50, color=(1, 1, 1, 1))
        
        # Button layout for upload and save
        button_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)
        
        # Upload Image Button
        upload_button = Button(text='Upload Image', background_color=(41/255, 128/255, 185/255, 1))
        upload_button.bind(on_release=self.upload_image)

        # Save Image Button
        self.save_button = Button(text='Save Image', background_color=(46/255, 204/255, 113/255, 1))
        self.save_button.bind(on_press=self.save_image)
        self.save_button.opacity = 0  # Initially invisible
        self.save_button.disabled = True  # Initially disabled
        
        button_layout.add_widget(upload_button)
        button_layout.add_widget(self.save_button)

        # Zoom buttons
        zoom_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)
        zoom_in_button = Button(text='Zoom In', background_color=(41/255, 128/255, 185/255, 1))
        zoom_out_button = Button(text='Zoom Out', background_color=(41/255, 128/255, 185/255, 1))
        zoom_layout.add_widget(zoom_in_button)
        zoom_layout.add_widget(zoom_out_button)
        zoom_in_button.bind(on_press=self.zoom_in)
        zoom_out_button.bind(on_press=self.zoom_out)

        # Predict button
        self.predict_button = Button(text='Predict', size_hint_y=None, height=50, background_color=(231/255, 76/255, 60/255, 1))
        self.predict_button.bind(on_press=self.predict)
        
        # Thin Smear Button setup
        self.thin_smear_button = Button(text='Thin Smear Analysis', size_hint_y=None, height=50, background_color=(41/255, 128/255, 185/255, 1))
        self.thin_smear_button.bind(on_press=self.thin_smear_analysis)  # Assuming you have a method for this action
        self.thin_smear_button.opacity = 0  # Initially hide the button
        self.thin_smear_button.disabled = True  # Initially disable the button

        self.detection_label = Label(text="Detection results will appear here", size_hint_y=None, height=30, color=(1, 1, 1, 1))

        self.layout.add_widget(header_layout)
        self.layout.add_widget(description)
        self.layout.add_widget(button_layout)
        self.layout.add_widget(self.image_widget)
        self.layout.add_widget(zoom_layout)
        self.layout.add_widget(self.predict_button)
        self.layout.add_widget(self.detection_label)
        self.layout.add_widget(self.thin_smear_button)

        self.add_widget(self.layout)


    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def upload_image(self, instance):
        filechooser = FileChooserListView(filters=['*.png', '*.jpg', '*.jpeg', '*.bmp'])
        self.popup = Popup(title="Upload Image", content=filechooser, size_hint=(0.9, 0.9))
        filechooser.bind(on_submit=self.select_image)
        self.popup.open()
        if self.predict_button.disabled == True:
            self.predict_button.opacity = 0
            self.predict_button.disabled = False

    def select_image(self, filechooser, selection, touch):
        if selection:
            cropped_file_name = self.crop_to_content(selection[0])
            self.image_path = cropped_file_name
            self.display_image(cropped_file_name)
            if self.popup:  # Check if the popup is initialized
                self.popup.dismiss()  # Dismiss the popup
            self.predict_button.opacity = 1
            self.predict_button.disabled = False


    def display_image(self, file_name):
        self.image_widget.source = file_name
        self.image_widget.reload()

    def crop_to_content(self, file_name):
        img = PILImage.open(file_name)
        img_array = np.array(img)

        threshold = 100
        rows = np.any(img_array.max(axis=2) > threshold, axis=1)
        cols = np.any(img_array.max(axis=2) > threshold, axis=0)

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        if not len(row_indices) or not len(col_indices):
            return file_name

        top_row, bottom_row = row_indices[0], row_indices[-1]
        left_col, right_col = col_indices[0], col_indices[-1]

        cropped_img = img.crop((left_col, top_row, right_col + 1, bottom_row + 1))
        cropped_file_name = f"{file_name.rsplit('.', 1)[0]}_cropped.{file_name.rsplit('.', 1)[1]}"
        cropped_img.save(cropped_file_name)
        return cropped_file_name

    def apply_scale(self, scale_factor):
        # Clear any previous transformations on the image's canvas to avoid stacking effects
        self.image_widget.canvas.before.clear()

        # Apply the scale transformation only to the image
        with self.image_widget.canvas.before:
            PushMatrix()
            Scale(scale_factor, scale_factor, 1, origin=self.image_widget.center)
        self.image_widget.canvas.after.clear()
        with self.image_widget.canvas.after:
            PopMatrix()


    def zoom_in(self, *args):
        self.scale_factor *= 1.25
        self.apply_scale(self.scale_factor)  # Corrected method name here

    def zoom_out(self, *args):
        self.scale_factor *= 0.8
        self.apply_scale(self.scale_factor)  # Corrected method name here

    def predict(self, instance):
        if not self.image_path:
            print('Error: Please upload an image first.')  # Using print for messages, adjust as needed for GUI feedback.
            return

        image = cv2.imread(self.image_path)
        if image is None:
            print('Error: Failed to load image.')
            return

        tiles = self.tile_image(image)
        all_detections = []

        for x_offset, y_offset, tile in tiles:
            detections = self.detect_objects(tile, self.model_detection)
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                score = det['confidence']
                class_id = det['class_id']
                cropped_image = self.crop_center(tile, (x1, y1, x2, y2))
                predicted_class_id = self.classify_image(cropped_image)
                if class_id == predicted_class_id:
                    global_box = (x_offset + x1, y_offset + y1, x_offset + x2, y_offset + y2)
                    all_detections.append((global_box, score, predicted_class_id))

        for (x1, y1, x2, y2), score, class_id in all_detections:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 10)
            label = f"Class: P, Score: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)

        self.update_image_display(image)
        self.stitched_image = image.copy()

        # Hide the predict button by setting its opacity to 0
        self.predict_button.opacity = 0
        self.predict_button.disabled = True  # Optionally disable the button to prevent clicks

        print(f"Detections: {len(all_detections)}")

        if len(all_detections) > 0:
            self.thin_smear_button.opacity = 1
            self.thin_smear_button.disabled = False
            self.detection_label.text = f"Detections: {len(all_detections)} Malaria Parasites"
        else:
            self.detection_label.text = "Malaria free thick blood smear"
            self.thin_smear_button.opacity = 0
            self.thin_smear_button.disabled = True

        self.detection_label.opacity = 1
        self.save_button.opacity = 1
        self.save_button.disabled = False


    def load_classify_model(self):
        # Load and return the Keras model from a .h5 file
        model = tf.keras.models.load_model('best_model_mobilevit.h5')
        return model

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


    def classify_image(self, image):
        resized_image = cv2.resize(image, (96, 96))
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = resized_image.astype('float32') / 255.0
        resized_image = np.expand_dims(resized_image, axis=0)
        prediction = self.model_classification.predict(resized_image)
        predicted_class_id = np.argmax(prediction, axis=1)[0]
        print(f'prediction is {predicted_class_id}')
        return predicted_class_id

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


    def update_image_display(self, image):
        # Convert the image to texture and display in the Kivy Image widget
        buffer = cv2.flip(image, 0).tostring()
        texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture

    def show_popup(self, title, message):
        popup_content = BoxLayout(orientation='vertical')
        popup_content.add_widget(Label(text=message))
        close_btn = Button(text='Close', size_hint_y=None, height=50)
        popup_content.add_widget(close_btn)
        popup = Popup(title=title, content=popup_content, size_hint=(None, None), size=(400, 400))
        close_btn.bind(on_press=popup.dismiss)
        popup.open()

    def thin_smear_analysis(self, instance):
        self.manager.transition.direction = 'left'  # Optional: set transition direction
        self.manager.current = 'thin_smear'  # Switch to the thin smear screen

    def save_image(self):
        if hasattr(self, 'stitched_image') and self.stitched_image is not None:
            # Create a layout for Popup
            content = BoxLayout(orientation='vertical', padding=10, spacing=10)
            text_input = TextInput(text='Enter filename', size_hint_y=None, height=30)
            save_btn = Button(text='Save', size_hint_y=None, height=30)
            cancel_btn = Button(text='Cancel', size_hint_y=None, height=30)
            content.add_widget(text_input)
            content.add_widget(save_btn)
            content.add_widget(cancel_btn)

            # Create the Popup
            popup = Popup(title='Save Image', content=content, size_hint=(None, None), size=(300, 150))

            # Bind the on_press event of the save button to actually save the image
            def save_btn_pressed(instance):
                file_name = text_input.text
                if not file_name.endswith('.png') and not file_name.endswith('.jpg'):
                    file_name += '.png'  # Default to PNG if no extension provided
                # Convert the image from BGR to RGB since OpenCV uses BGR by default
                rgb_image = cv2.cvtColor(self.stitched_image, cv2.COLOR_BGR2RGB)
                # Save the image using OpenCV
                success = cv2.imwrite(file_name, rgb_image)
                if success:
                    print("Image saved successfully.")
                else:
                    print("Error saving image.")
                popup.dismiss()

            save_btn.bind(on_press=save_btn_pressed)
            cancel_btn.bind(on_press=lambda x: popup.dismiss())
            
            popup.open()
        else:
            print("No stitched image to save.")


class HomeScreen_for_thin(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_path = None  
        self.matrix = Matrix()
        self.scale_factor = 1.0 
        self.model_detection = self.load_detection_model('thinbest.pt', 'gelan-c.yaml')
        self.model_classification = self.load_classify_model()
        self.popup = None

        self.layout = BoxLayout(orientation='vertical')

        self.image_widget = Image(size_hint_y=0.7)

        with self.canvas.before:
            Color(35/255, 47/255, 62/255, 1)  # Background color
            self.rect = Rectangle(size=self.size, pos=self.pos)
            self.bind(size=self._update_rect, pos=self._update_rect)

        # Header Layout
        header_layout = BoxLayout(size_hint_y=None, height=60, padding=10)
        menu_button = Button(text='☰', size_hint_x=None, width=60)
        title = Label(text='Malaria Diagnosis Application', color=(1, 1, 1, 1))
        profile_button = Button(background_normal=PROFILE_ICON_PATH, size_hint_x=None, width=60)

        header_layout.add_widget(menu_button)
        header_layout.add_widget(title)
        header_layout.add_widget(profile_button)

        # Body
        description = Label(text='Thin Blood Smear Analysis', size_hint_y=None, height=50, color=(1, 1, 1, 1))
        
        # Button layout for upload and save
        button_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)
        
        # Upload Image Button
        upload_button = Button(text='Upload Image', background_color=(41/255, 128/255, 185/255, 1))
        upload_button.bind(on_release=self.upload_image)

        # Save Image Button
        self.save_button = Button(text='Save Image', background_color=(46/255, 204/255, 113/255, 1))
        self.save_button.bind(on_press=self.save_image)
        self.save_button.opacity = 0  # Initially invisible
        self.save_button.disabled = True  # Initially disabled
        
        button_layout.add_widget(upload_button)
        button_layout.add_widget(self.save_button)

        # Zoom buttons
        zoom_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)
        zoom_in_button = Button(text='Zoom In', background_color=(41/255, 128/255, 185/255, 1))
        zoom_out_button = Button(text='Zoom Out', background_color=(41/255, 128/255, 185/255, 1))
        zoom_layout.add_widget(zoom_in_button)
        zoom_layout.add_widget(zoom_out_button)
        zoom_in_button.bind(on_press=self.zoom_in)
        zoom_out_button.bind(on_press=self.zoom_out)

        # Predict button
        self.predict_button = Button(text='Predict', size_hint_y=None, height=50, background_color=(231/255, 76/255, 60/255, 1))
        self.predict_button.bind(on_press=self.predict)

        self.detection_label = Label(text="Detection results will appear here", size_hint_y=None, height=50, color=(1, 1, 1, 1))

         # Back Button
        back_button = Button(text='Back to Home', size_hint_y=None, height=50)
        back_button.bind(on_press=self.go_back)

        self.layout.add_widget(header_layout)
        self.layout.add_widget(description)
        self.layout.add_widget(button_layout)
        self.layout.add_widget(self.image_widget)
        self.layout.add_widget(zoom_layout)
        self.layout.add_widget(self.predict_button)
        self.layout.add_widget(self.detection_label)
        self.layout.add_widget(back_button)

        self.add_widget(self.layout)


    def go_back(self, instance):
        self.manager.transition.direction = 'right'
        self.manager.current = 'home'


    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def upload_image(self, instance):
        filechooser = FileChooserListView(filters=['*.png', '*.jpg', '*.jpeg', '*.bmp'])
        self.popup = Popup(title="Upload Image", content=filechooser, size_hint=(0.9, 0.9))
        filechooser.bind(on_submit=self.select_image)
        self.popup.open()

    def select_image(self, filechooser, selection, touch):
        if selection:
            cropped_file_name = self.crop_to_content(selection[0])
            self.image_path = cropped_file_name
            self.display_image(cropped_file_name)
            if self.popup:  # Check if the popup is initialized
                self.popup.dismiss()  # Dismiss the popup
            self.predict_button.opacity = 1
            self.predict_button.disabled = False

    def display_image(self, file_name):
        self.image_widget.source = file_name
        self.image_widget.reload()

    def crop_to_content(self, file_name):
        img = PILImage.open(file_name)
        img_array = np.array(img)

        threshold = 100
        rows = np.any(img_array.max(axis=2) > threshold, axis=1)
        cols = np.any(img_array.max(axis=2) > threshold, axis=0)

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        if not len(row_indices) or not len(col_indices):
            return file_name

        top_row, bottom_row = row_indices[0], row_indices[-1]
        left_col, right_col = col_indices[0], col_indices[-1]

        cropped_img = img.crop((left_col, top_row, right_col + 1, bottom_row + 1))
        cropped_file_name = f"{file_name.rsplit('.', 1)[0]}_cropped.{file_name.rsplit('.', 1)[1]}"
        cropped_img.save(cropped_file_name)
        return cropped_file_name

    def apply_scale(self, scale_factor):
        # Clear any previous transformations on the image's canvas to avoid stacking effects
        self.image_widget.canvas.before.clear()

        # Apply the scale transformation only to the image
        with self.image_widget.canvas.before:
            PushMatrix()
            Scale(scale_factor, scale_factor, 1, origin=self.image_widget.center)
        self.image_widget.canvas.after.clear()
        with self.image_widget.canvas.after:
            PopMatrix()


    def zoom_in(self, *args):
        self.scale_factor *= 1.25
        self.apply_scale(self.scale_factor)  # Corrected method name here

    def zoom_out(self, *args):
        self.scale_factor *= 0.8
        self.apply_scale(self.scale_factor)  # Corrected method name here

    def predict(self, instance):
        if not self.image_path:
            print('Error: Please upload an image first.')  # Using print for messages, adjust as needed for GUI feedback.
            return

        image = cv2.imread(self.image_path)
        if image is None:
            print('Error: Failed to load image.')
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
                cropped_image = self.crop_center(tile, (x1, y1, x2, y2))
                predicted_class_id = self.classify_image(cropped_image)
                if class_id == predicted_class_id:
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

        # Hide the predict button by setting its opacity to 0
        self.predict_button.opacity = 0
        self.predict_button.disabled = True  # Optionally disable the button to prevent clicks

        print(f"Detections: {len(all_detections)}")

        if not class_ids:  # Check if the list is empty
            self.detection_label.text = "No detections found."
        elif all(c == 1 for c in class_ids):
            self.detection_label.text = "Detections result are P.falciparum"
        elif all(c == 0 for c in class_ids):
            self.detection_label.text = "Detections result are P.vivax"
        else:
            self.detection_label.text = "Detections result are Mixed"

        self.detection_label.opacity = 1
        self.save_button.opacity = 1
        self.save_button.disabled = False


    def load_classify_model(self):
        # Load and return the Keras model from a .h5 file
        model = tf.keras.models.load_model('thinbest_model_mobilevit.h5')
        return model

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


    def classify_image(self, image):
        resized_image = cv2.resize(image, (96, 96))
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = resized_image.astype('float32') / 255.0
        resized_image = np.expand_dims(resized_image, axis=0)
        prediction = self.model_classification.predict(resized_image)
        predicted_class_id = np.argmax(prediction, axis=1)[0]
        print(f'prediction is {predicted_class_id}')
        return predicted_class_id

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


    def update_image_display(self, image):
        # Convert the image to texture and display in the Kivy Image widget
        buffer = cv2.flip(image, 0).tostring()
        texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture

    def show_popup(self, title, message):
        popup_content = BoxLayout(orientation='vertical')
        popup_content.add_widget(Label(text=message))
        close_btn = Button(text='Close', size_hint_y=None, height=50)
        popup_content.add_widget(close_btn)
        popup = Popup(title=title, content=popup_content, size_hint=(None, None), size=(400, 400))
        close_btn.bind(on_press=popup.dismiss)
        popup.open()


    def save_image(self):
        if hasattr(self, 'stitched_image') and self.stitched_image is not None:
            # Create a layout for Popup
            content = BoxLayout(orientation='vertical', padding=10, spacing=10)
            text_input = TextInput(text='Enter filename', size_hint_y=None, height=30)
            save_btn = Button(text='Save', size_hint_y=None, height=30)
            cancel_btn = Button(text='Cancel', size_hint_y=None, height=30)
            content.add_widget(text_input)
            content.add_widget(save_btn)
            content.add_widget(cancel_btn)

            # Create the Popup
            popup = Popup(title='Save Image', content=content, size_hint=(None, None), size=(300, 150))

            # Bind the on_press event of the save button to actually save the image
            def save_btn_pressed(instance):
                file_name = text_input.text
                if not file_name.endswith('.png') and not file_name.endswith('.jpg'):
                    file_name += '.png'  # Default to PNG if no extension provided
                # Convert the image from BGR to RGB since OpenCV uses BGR by default
                rgb_image = cv2.cvtColor(self.stitched_image, cv2.COLOR_BGR2RGB)
                # Save the image using OpenCV
                success = cv2.imwrite(file_name, rgb_image)
                if success:
                    print("Image saved successfully.")
                else:
                    print("Error saving image.")
                popup.dismiss()

            save_btn.bind(on_press=save_btn_pressed)
            cancel_btn.bind(on_press=lambda x: popup.dismiss())
            
            popup.open()
        else:
            print("No stitched image to save.")


class MainApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(HomeScreen_for_thin(name='thin_smear'))
        return sm

if __name__ == '__main__':
    MainApp().run()
