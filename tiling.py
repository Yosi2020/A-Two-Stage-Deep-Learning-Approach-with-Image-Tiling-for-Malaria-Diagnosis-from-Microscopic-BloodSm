import cv2
import numpy as np
import os

# Function to load image and labels
def load_image_and_labels(image_path, label_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)

    # Load labels
    with open(label_path, 'r') as file:
        labels = file.readlines()
        labels = [list(map(float, line.strip().split())) for line in labels]
    
    return image, labels

# Adjust labels for the current tile
def adjust_labels_for_tile(labels, start_x, start_y, end_x, end_y, tile_size, width, height):
    tile_labels = []
    for label in labels:
        class_id, x_center, y_center, w, h = label
        # Convert from normalized coordinates to absolute coordinates in the original image space
        x_center_abs = x_center * width
        y_center_abs = y_center * height
        w_abs = w * width
        h_abs = h * height

        # Calculate the bounding box's position on the tile
        box_start_x_abs = x_center_abs - w_abs / 2
        box_start_y_abs = y_center_abs - h_abs / 2

        # If the center of the bounding box is within the current tile
        if start_x <= x_center_abs < end_x and start_y <= y_center_abs < end_y:
            # Adjust the coordinates and size to the tile's coordinate space
            # Convert to coordinates relative to the tile
            x_center_rel_tile = (x_center_abs - start_x) / (end_x - start_x)
            y_center_rel_tile = (y_center_abs - start_y) / (end_y - start_y)
            # Make sure to calculate width and height relative to the tile dimensions, not the original image
            w_rel_tile = min(w_abs, end_x - start_x) / (end_x - start_x)
            h_rel_tile = min(h_abs, end_y - start_y) / (end_y - start_y)

            # If the bounding box extends beyond the tile, clip it to the tile's edges
            x_min_tile = max(box_start_x_abs - start_x, 0) / tile_size
            y_min_tile = max(box_start_y_abs - start_y, 0) / tile_size
            x_max_tile = min(box_start_x_abs + w_abs - start_x, tile_size) / tile_size
            y_max_tile = min(box_start_y_abs + h_abs - start_y, tile_size) / tile_size

            # Recalculate the center based on the clipped bounding box
            x_center_rel_tile = (x_min_tile + x_max_tile) / 2
            y_center_rel_tile = (y_min_tile + y_max_tile) / 2
            # Recalculate the width and height based on the clipped edges
            w_rel_tile = x_max_tile - x_min_tile
            h_rel_tile = y_max_tile - y_min_tile

            tile_labels.append([class_id, x_center_rel_tile, y_center_rel_tile, w_rel_tile, h_rel_tile])

    return tile_labels

# Function to generate fixed size tiles and adjust labels
def generate_fixed_size_tiles_and_labels(image, labels, tile_size, overlap_percentage, output_dir, tile_count):
    height, width = image.shape[:2]

    # Calculate overlap in pixels
    overlap_x = int(tile_size * overlap_percentage / 100)
    overlap_y = overlap_x  # Assuming square tiles for simplicity

    # Calculate step size for moving from one tile to the next
    step_x = tile_size - overlap_x
    step_y = step_x

    # Iterate over the image to create tiles
    for start_y in range(0, height - tile_size, step_y):
        for start_x in range(0, width - tile_size, step_x):
            end_x = start_x + tile_size
            end_y = start_y + tile_size
            tile = image[start_y:end_y, start_x:end_x]

            # Adjust labels for the current tile
            tile_labels = adjust_labels_for_tile(labels, start_x, start_y, end_x, end_y,tile_size, width, height)

            # Save the tile and labels if there are objects in the tile
            if tile_labels:
                tile_filename = os.path.join(output_dir, f"tile_{tile_count}.png")
                cv2.imwrite(tile_filename, tile)
                label_filename = os.path.join(output_dir, f"tile_{tile_count}.txt")
                with open(label_filename, 'w') as label_file:
                    for label in tile_labels:
                        label_file.write(' '.join(map(str, label)) + '\n')
                tile_count += 1

    return tile_count

output_dir = r'E:\MSc Thesis\Previous\Dr\tile_output'
os.makedirs(output_dir, exist_ok=True)

tile_count = 0

folder_path = "E:\MSc Thesis\Previous\Dr\Cropped_image"
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(folder_path, base_name + '.txt')
        if os.path.exists(label_path):
            image, labels = load_image_and_labels(image_path, label_path)
            tile_count = generate_fixed_size_tiles_and_labels(image, labels, tile_size=640, overlap_percentage=30, output_dir=output_dir, tile_count = tile_count)

print(f"Total tiles generated: {tile_count}")
