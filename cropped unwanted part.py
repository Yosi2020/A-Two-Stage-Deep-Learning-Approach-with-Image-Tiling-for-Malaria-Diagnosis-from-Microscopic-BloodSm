import os
import numpy as np
from PIL import Image

# Function to load image and labels
def load_image_and_labels(image_path, label_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)

    # Load labels
    with open(label_path, 'r') as file:
        labels = file.readlines()
        labels = [list(map(float, line.strip().split())) for line in labels]
    
    return image, labels

def save_labels(labels, label_path):
    """Save adjusted labels to a file."""
    with open(label_path, 'w') as file:
        for label in labels:
            file.write(' '.join(map(str, label)) + '\n')

def adjust_labels(labels, orig_height, orig_width, crop_box, crop_size):
    """Adjust labels for the cropped image."""
    left, top, right, bottom = crop_box
    cropped_width, cropped_height, _ = crop_size
    adjusted_labels = []
    
    print(f"oringal width {orig_width}")
    print(f"oringal height {orig_height}")

    for class_id, x_center, y_center, width, height in labels:
        # Convert normalized coordinates to pixel coordinates relative to the original image
        box_x_center = int(x_center * orig_width)
        box_y_center = int(y_center * orig_height)
        box_width = int(width * orig_width)
        box_height = int(height * orig_height)

        # Convert center coordinates to top-left coordinates
        box_x1 = box_x_center - (box_width // 2)
        box_y1 = box_y_center - (box_height // 2)
        #print(f"box x1 {box_x1}")
        #print(f"left column {left}")
        #print(f"top column {top}")

        # Adjust coordinates to account for the crop
        box_x1 -= top - 1
        box_y1 -= left 

        # Re-normalize the adjusted box coordinates to the cropped image dimensions
        new_x_center = (box_x1 + (box_width // 2)) / cropped_width
        new_y_center = (box_y1 + (box_height // 2)) / cropped_height
        new_width = box_width / cropped_width
        new_height = box_height / cropped_height

        # Append the adjusted label
        adjusted_labels.append([class_id, new_x_center, new_y_center, new_width, new_height])

    return adjusted_labels


def crop_to_content(image_path, label_path, output_dir):
    """Crop an image to its content and adjust labels."""
    # Load image and labels
    img = Image.open(image_path)
    #labels = load_labels(label_path)
    
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
    
    # Rotate the cropped image by 90 degrees clockwise
    #cropped_img = cropped_img.rotate(-90, expand=True)

    cropped_img_path = os.path.join(output_dir, os.path.basename(image_path))
    cropped_img.save(cropped_img_path)
    img1 = cv2.imread(cropped_img_path)
    print(img1.shape)
    
    # Adjust labels
    crop_box = (left_col, top_row, right_col, bottom_row)
    adjusted_labels = adjust_labels(labels, img.size[0], img.size[1], crop_box, img1.shape)
    adjusted_label_path = os.path.join(output_dir, os.path.basename(label_path))
    save_labels(adjusted_labels, adjusted_label_path)
    
    return cropped_img_path, adjusted_label_path

folder_path = r"E:\MSc Thesis\Previous\Dr\Camera-Fetya"
# Output directory
output_dir = r'E:\MSc Thesis\Previous\Dr\Cropped_image'

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".jpg"):
        image_path = os.path.join(folder_path, filename)
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(folder_path, base_name + '.txt')
        if os.path.exists(label_path):
            image, labels = load_image_and_labels(image_path, label_path)
            results = crop_to_content(image_path, label_path, output_dir)
            if results:
                cropped_image_path, adjusted_label_path = results
                print(f"Cropped image saved to: {cropped_image_path}")
                print(f"Adjusted label file saved to: {adjusted_label_path}")