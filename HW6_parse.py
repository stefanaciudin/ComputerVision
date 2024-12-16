import os
import random
import xml.etree.ElementTree as ET
from shutil import copy2

import cv2

# Paths to images, annotations, and output directories
images_path = "lab6/images"
annotations_path = "lab6/annotations.xml"
output_train_images = "lab6/train/images"
output_train_labels = "lab6/train/labels"
output_test_images = "lab6/test/images"
output_test_labels = "lab6/test/labels"

# Ensure output directories exist
os.makedirs(output_train_images, exist_ok=True)
os.makedirs(output_train_labels, exist_ok=True)
os.makedirs(output_test_images, exist_ok=True)
os.makedirs(output_test_labels, exist_ok=True)


# Parse annotations.xml
def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = {}

    for track in root.findall("track"):
        label = track.get("label")
        for box in track.findall("box"):
            frame = int(box.get("frame"))
            xtl, ytl, xbr, ybr = map(float, (box.get("xtl"), box.get("ytl"), box.get("xbr"), box.get("ybr")))
            if frame not in annotations:
                annotations[frame] = []
            annotations[frame].append({
                "label": label,
                "bbox": (xtl, ytl, xbr, ybr)
            })
    return annotations


annotations = parse_annotations(annotations_path)

# Get all image files
image_files = sorted([f for f in os.listdir(images_path) if f.endswith(".PNG")])
total_images = len(image_files)

# Split into train and test
train_count = int(0.7 * total_images)
train_files = random.sample(image_files, train_count)
test_files = list(set(image_files) - set(train_files))


# Generate YOLO format annotations
def generate_yolo_annotation(image_file, annotations, output_label_dir, image_size):
    height, width = image_size
    frame_number = int(image_file.split("_")[-1].split(".")[0])  # Extract frame number
    label_file = os.path.join(output_label_dir, os.path.splitext(image_file)[0] + ".txt")

    with open(label_file, "w") as f:
        if frame_number in annotations:
            for obj in annotations[frame_number]:
                label_id = 0 if obj["label"] == "car" else 1  # Assign class IDs (car=0, minivan=1)
                xtl, ytl, xbr, ybr = obj["bbox"]
                x_center = (xtl + xbr) / 2 / width
                y_center = (ytl + ybr) / 2 / height
                box_width = (xbr - xtl) / width
                box_height = (ybr - ytl) / height
                f.write(f"{label_id} {x_center} {y_center} {box_width} {box_height}\n")


# Move and generate annotations for train and test sets
def prepare_dataset(files, annotations, output_image_dir, output_label_dir, images_path):
    for file in files:
        source_image = os.path.join(images_path, file)
        target_image = os.path.join(output_image_dir, file)
        copy2(source_image, target_image)  # Copy image
        image_size = cv2.imread(source_image).shape[:2]  # (height, width)
        generate_yolo_annotation(file, annotations, output_label_dir, image_size)


prepare_dataset(train_files, annotations, output_train_images, output_train_labels, images_path)
prepare_dataset(test_files, annotations, output_test_images, output_test_labels, images_path)
