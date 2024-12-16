from ultralytics import YOLO
import xml.etree.ElementTree as ET
import os


def load_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = {}

    for track in root.findall(".//track"):
        track_id = int(track.get("id"))

        # Ensure label is properly extracted from the track element or its children
        track_label = track.get("label")  # This gets the label from the track, e.g., "car", "minivan"
        if track_label is None:
            continue  # If no label is present in the track, skip this track

        # Now extract all bounding boxes from this track
        for box in track.findall(".//box"):
            frame_number = int(box.get("frame"))

            # Extract bounding box coordinates
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            if frame_number not in annotations:
                annotations[frame_number] = []

            annotations[frame_number].append({
                "bbox": [xtl, ytl, xbr, ybr],
                "label": track_label  # Assign the label from the track
            })

    return annotations


def calculate_iou(box1, box2):
    """
    Calculates the Intersection Over Union (IOU) between two bounding boxes.
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou


def evaluate_yolo_model(model, test_images_path, annotations, iou_threshold=0.5):
    """
    Evaluates the YOLO model by comparing predictions to ground truth annotations.
    Counts the number of correctly detected cars.
    """
    test_images = sorted(
        [os.path.join(test_images_path, f) for f in os.listdir(test_images_path) if f.endswith(".PNG")])
    correct_detections = 0
    total_annotations = 0

    for image_file in test_images:
        frame_number = int(os.path.basename(image_file).split("_")[-1].split(".")[0])
        if frame_number not in annotations:
            continue

        # Load image and get predictions
        results = model(image_file)
        predictions = results[0].boxes.xyxy.cpu().numpy()  # Predicted bounding boxes
        pred_classes = results[0].boxes.cls.cpu().numpy()  # Predicted class IDs

        # Compare predictions with ground truth
        for obj in annotations[frame_number]:
            total_annotations += 1
            gt_bbox = obj["bbox"]
            gt_class = 0 if obj["label"] == "car" else 1

            # Check for matching bounding boxes
            for pred_bbox, pred_class in zip(predictions, pred_classes):
                if pred_class == gt_class:
                    iou = calculate_iou(gt_bbox, pred_bbox)
                    if iou >= iou_threshold:
                        correct_detections += 1
                        break

    print(f"Correct Detections: {correct_detections}")
    print(f"Total Ground Truth Annotations: {total_annotations}")
    print(f"Model Accuracy: {correct_detections / total_annotations:.2%}")


# Load the YOLO model
model = YOLO("lab6/runs/detect/train5/weights/best.pt")

# Load annotations from XML
annotations = load_annotations("lab6/annotations.xml")

# Evaluate the model on the test set
evaluate_yolo_model(model, "lab6/test/images", annotations)
