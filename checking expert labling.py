import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pandas as pd

# Define the function to calculate the intersection area between two rectangles
def calculate_intersection_area(rect_a, rect_b):
    x_left = max(rect_a[0], rect_b[0])
    y_bottom = max(rect_a[1], rect_b[1])
    x_right = min(rect_a[2], rect_b[2])
    y_top = min(rect_a[3], rect_b[3])

    if x_right < x_left or y_top < y_bottom:
        return 0.0

    return (x_right - x_left) * (y_top - y_bottom)

# Define the function to check if the rectangles are correct based on the intersection rules
def is_correct(rect_a, rect_b):
    intersection_area = calculate_intersection_area(rect_a, rect_b)
    area_a = (rect_a[2] - rect_a[0]) * (rect_a[3] - rect_a[1])
    area_b = (rect_b[2] - rect_b[0]) * (rect_b[3] - rect_b[1])

    if intersection_area == area_a or intersection_area == area_b:
        return 'correct'
    if intersection_area >= 0.3 * min(area_a, area_b):
        return 'correct'

    return 'please see this image'

# Function to convert normalized coordinates to pixel coordinates and return rectangles
def get_rectangles_from_normalized_txt(txt_path, img_size):
    rectangles = []
    width, height = img_size
    
    if not os.path.exists(txt_path):
        return rectangles

    with open(txt_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        _, x_center_norm, y_center_norm, width_norm, height_norm = map(float, line.split())
        box_width = width_norm * width
        box_height = height_norm * height
        x_center = x_center_norm * width
        y_center = y_center_norm * height
        xmin = x_center - (box_width / 2)
        ymin = y_center - (box_height / 2)
        rect = (xmin, ymin, xmin + box_width, ymin + box_height)
        rectangles.append(rect)

    return rectangles

# Function to draw rectangles on the image
def draw_rectangles_on_image(image_path, rectangles, save_dir):
    img = cv2.imread(image_path)
    
    # Debugging: Print number of rectangles to be drawn
    print(f"Drawing {len(rectangles)} rectangles on {os.path.basename(image_path)}")

    for rect in rectangles:
        xmin, ymin, xmax, ymax = map(int, rect)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)

    output_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    

def compare_images(image_path, txt_path1, txt_path2):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rectangles1 = get_rectangles_from_normalized_txt(txt_path1, img.shape[1::-1]) if os.path.exists(txt_path1) else []
    rectangles2 = get_rectangles_from_normalized_txt(txt_path2, img.shape[1::-1]) if os.path.exists(txt_path2) else []

    matched_rectangles = []
    for rect1 in rectangles1:
        for rect2 in rectangles2:
            if is_correct(rect1, rect2) == 'correct':
                matched_rectangles.append(rect1)
                break

    return matched_rectangles, rectangles1, rectangles2

def compare_all_images(image_dir, label_dir1, label_dir2, save_dir):
    mismatched_images = []
    records = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):  # Update as needed
            basename = os.path.splitext(filename)[0]
            image_path = os.path.join(image_dir, basename + '.jpg')
            txt_path1 = os.path.join(label_dir1, basename + '.txt')
            txt_path2 = os.path.join(label_dir2, basename + '.txt')
            
            if not os.path.exists(txt_path1) or not os.path.exists(txt_path2):
                print(f"Label file missing for {basename}")
                continue

            matched_rectangles, rectangles1, rectangles2 = compare_images(image_path, txt_path1, txt_path2)
            if matched_rectangles:
                draw_rectangles_on_image(image_path, matched_rectangles, save_dir)

            num_detections1 = len(rectangles1)
            num_detections2 = len(rectangles2)
            num_matches = len(matched_rectangles)
            records.append([basename, num_detections1, num_detections2, num_matches])
            
            print("record basename "+ str(basename) + " Tirusew "+str(num_detections1)+ " Fetya "+ str(num_detections2)+ " Match "+ str(num_matches))
            
    df = pd.DataFrame(records, columns=['Image', 'Number of detection by Tirusew', 'Number of detection by Fetya', 'Number of Matched'])
    df.to_csv('detection_summary.csv', index=False)

    return mismatched_images

image_dir = r'E:\MSc Thesis\Dr\Camera 1_Tirusew'
label_dir1 = r'E:\MSc Thesis\Dr\Camera-Fetya'
label_dir2 = r'E:\MSc Thesis\Dr\Camera 1_Tirusew'
save_dir = r'E:\MSc Thesis\Dr\Matched_Images'  
mismatched = compare_all_images(image_dir, label_dir1, label_dir2, save_dir)

print(mismatched)
