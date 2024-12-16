import os
import cv2
import numpy as np


def extract_bounding_boxes(image):
    """
    Extracts bounding boxes from an image using color segmentation.
    Assumes bounding boxes are drawn in a specific color (blue in this case).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_color = np.array([100, 150, 50])  # Lower HSV range for blue
    upper_color = np.array([130, 255, 255])


    mask = cv2.inRange(hsv, lower_color, upper_color)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x + w, y + h))

    return bounding_boxes


def find_target_in_first_frame(image, bounding_boxes):
    """
    Identifies the target car based on its blue bounding box in the first frame.
    Returns the bounding box of the target car.
    """
    if bounding_boxes:
        return bounding_boxes[0]
    return None


def track_target_car(dataset_path, target_bbox, starting_frame):
    """
    Tracks a specific car across frames based on bounding box proximity.
    """
    tracking_history = []
    previous_bbox = target_bbox

    frame_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".PNG")])
    print(frame_files)

    for frame_number in range(starting_frame, len(frame_files)):
        frame_path = os.path.join(dataset_path, frame_files[frame_number])
        frame = cv2.imread(frame_path)

        bounding_boxes = extract_bounding_boxes(frame)

        # Find the closest bounding box to the previous one
        closest_bbox = None
        min_distance = float("inf")

        for bbox in bounding_boxes:
            distance = calculate_bbox_distance(previous_bbox, bbox)
            if distance < min_distance:
                closest_bbox = bbox
                min_distance = distance

        if closest_bbox:
            tracking_history.append((frame_number, closest_bbox))
            previous_bbox = closest_bbox

            # Draw the tracked car's bounding box
            x1, y1, x2, y2 = map(int, closest_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Tracked Car", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        # Optional: Display the frame
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return tracking_history


def calculate_bbox_distance(bbox1, bbox2):
    """
    Calculates the Euclidean distance between the centers of two bounding boxes.
    """
    x1, y1, x2, y2 = bbox1
    cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2

    x3, y3, x4, y4 = bbox2
    cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2

    return ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5


# Main script
dataset_path = "lab6/boxes"
starting_frame = 0

first_frame_path = os.path.join(dataset_path, f"frame_{starting_frame:06d}.PNG")
first_frame = cv2.imread(first_frame_path)

bounding_boxes = extract_bounding_boxes(first_frame)

target_bbox = find_target_in_first_frame(first_frame, bounding_boxes)

if target_bbox:
    print(f"Target car found in frame {starting_frame}: {target_bbox}")
    tracking_history = track_target_car(dataset_path, target_bbox, starting_frame)

    if tracking_history:
        print("\nTracking complete. History of tracked car:")
        for frame_number, bbox in tracking_history:
            print(f"Frame {frame_number}: {bbox}")
else:
    print(f"No target car found in frame {starting_frame}.")
