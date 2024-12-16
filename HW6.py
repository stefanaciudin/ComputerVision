import os
from collections import deque

import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov5su.pt")

def draw_bounding_boxes(image, results, save_path=None):
    """
    Draws bounding boxes and labels on the image for detected objects.
    Returns the modified image and a list of detected objects with confidence scores.
    """
    detected_objects = []

    for box in results[0].boxes:  # Access 'boxes' properly
        x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
        conf = box.conf[0].item()     # Confidence score as a float
        cls = box.cls[0].item()       # Class ID as an integer
        label = model.names[int(cls)]  # Retrieve class name

        # Append the detected object and its confidence to the list
        detected_objects.append((label, conf))

        # Draw bounding box and label on the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    # Save the annotated image if a save path is provided
    if save_path:
        cv2.imwrite(save_path, image)

    return image, detected_objects


def count_persons(detected_objects):
    """
    Counts the number of persons in the list of detected objects.
    Returns the count of persons.
    """
    person_count = sum(1 for label, _ in detected_objects if label.lower() == "person")
    return person_count


def process_video(video_path, output_path):
    """
    Processes a video to annotate detected objects and draw bounding boxes for the
    top 3-4 objects with the highest confidence scores.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        # Sort detections by confidence score and select top 4
        top_detections = sorted(results[0].boxes, key=lambda box: box.conf[0].item(), reverse=True)[:4]

        # Draw bounding boxes for the top detections
        for box in top_detections:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            label = model.names[int(cls)]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        # Write the frame to the output video
        out.write(frame)

        # Optional: Display the video frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def count_vehicles_in_frame(image, results):
    """
    Counts the number of cars and minivans in a single frame.
    Returns the counts and detected objects for tracking.
    """
    car_count = 0
    minivan_count = 0
    detected_objects = []

    for box in results[0].boxes:
        cls = box.cls[0].item()  # Class ID
        conf = box.conf[0].item()  # Confidence score
        label = model.names[int(cls)]  # Class label

        # Count cars and minivans
        if label.lower() == "car":
            car_count += 1
        elif label.lower() == "minivan":
            minivan_count += 1

        # Add detected objects to the list
        detected_objects.append((label, conf, box.xyxy[0]))

    return car_count, minivan_count, detected_objects


def track_vehicle(tracking_id, objects_in_frame, tracking_history, frame_number):
    """
    Tracks a specific vehicle by ID from the entering frame to the exiting frame.
    """
    for obj in objects_in_frame:
        label, conf, bbox = obj
        if label.lower() == tracking_id:
            tracking_history.append((frame_number, bbox))
            return True
    return False


def process_dataset(dataset_path, tracking_id):
    """
    Processes a dataset of frames to count cars and minivans in each frame
    and track a specific vehicle from its entering to exiting frame.
    """
    frame_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".PNG")])
    vehicle_tracking_history = deque()
    first_frame = None
    last_frame = None

    print("Processing frames for vehicle tracking and counting...")

    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(dataset_path, frame_file)
        frame = cv2.imread(frame_path)

        # Run YOLO object detection
        results = model(frame)

        # Count vehicles in the frame
        car_count, minivan_count, detected_objects = count_vehicles_in_frame(frame, results)

        # Attempt to track the vehicle
        tracked = track_vehicle(tracking_id, detected_objects, vehicle_tracking_history, i)
        if tracked:
            if first_frame is None:
                first_frame = i
            last_frame = i

        # Print counts for the frame
        print(f"Frame {i}: Cars: {car_count}, Minivans: {minivan_count}")

    # Output vehicle tracking results
    if vehicle_tracking_history:
        print(f"\nVehicle '{tracking_id}' was tracked:")
        print(f"First detected in frame: {first_frame}")
        print(f"Last detected in frame: {last_frame}")
        print(f"Tracking history (frame, bbox): {list(vehicle_tracking_history)}")
    else:
        print(f"No '{tracking_id}' vehicle detected in the dataset.")


# Load the image
image = cv2.imread("lab6/candy.jpg")
image_persons = cv2.imread("lab3/b/Pratheepan_Dataset/FamilyPhoto/Family_1.png")
# Run YOLO object detection
results = model(image)

save_path = "lab6/results/candy_annotated.jpg"
image_with_boxes, detected_objects = draw_bounding_boxes(image, results, save_path=save_path)

# Convert to RGB for display and show the image
image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis("off")
plt.show()

results = model(image_persons)


save_path = "lab6/results/family_annotated.png"
image_with_boxes, detected_objects = draw_bounding_boxes(image_persons, results, save_path=save_path)

# Convert to RGB for display and show the image
image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis("off")
plt.show()

person_count = count_persons(detected_objects)

# Print the detected objects and their confidence scores
print("Detected Objects and Confidence Scores:")
for obj, score in detected_objects:
    print(f"{obj}: {score:.2f}")

# Print the count of persons in the image
print(f"Number of persons detected: {person_count}")

#process_video("lab6/cars.mp4", "lab6/cars_annotated.mp4")

dataset_path = "lab6/images"
process_dataset(dataset_path, tracking_id="truck")