from google.colab import files

# Upload video file manually
uploaded = files.upload()
pip install ultralytics opencv-python numpy
import cv2
import numpy as np
import csv
from google.colab import files
from ultralytics import YOLO

# Define polygon regions for traffic light detection
RedLight = np.array([[998, 125], [998, 155], [972, 152], [970, 127]])
GreenLight = np.array([[971, 200], [996, 200], [1001, 228], [971, 230]])
ROI = np.array([[910, 372], [388, 365], [338, 428], [917, 441]])

# Load the YOLO model
model = YOLO("yolov8m.pt")  # Replace with the correct path if necessary

# COCO classes for the YOLO model
coco = model.model.names
TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]

# File to save vehicle violation data
data_file = "/content/vehicle_violations.csv"

# Initialize CSV file and write headers
with open(data_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Vehicle Type", "Confidence", "Bounding Box (x1, y1, x2, y2)"])

# Dictionary to track vehicles with red bounding boxes (violations)
violations = {}

# Function to check if a polygonal region in the frame is bright
def is_region_light(frame, region):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [region], 255)
    region_pixels = cv2.bitwise_and(frame, frame, mask=mask)
    gray_region = cv2.cvtColor(region_pixels, cv2.COLOR_BGR2GRAY)
    brightness = cv2.mean(gray_region, mask=mask)[0]
    return brightness > 100  # Adjust as needed

# Function to log vehicle details with red bounding box
def log_vehicle_violation(frame_number, vehicle_type, confidence, bounding_box):
    vehicle_id = tuple(bounding_box)
    if vehicle_id not in violations:
        violations[vehicle_id] = {
            "vehicle_type": vehicle_type,
            "confidence": confidence,
            "frame_first_detected": frame_number
        }
        with open(data_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_number, vehicle_type, f"{confidence*100:.2f}%", f"({bounding_box[0]}, {bounding_box[1]}, {bounding_box[2]}, {bounding_box[3]})"])

# Load the video
cap = cv2.VideoCapture("tr.mp4")
output_video = "/content/traffic_violation_output.mp4"
frame_width, frame_height = 1100, 700
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_number = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success or frame is None:
        print("Video processing complete.")
        break

    frame_number += 1
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Draw polygons for traffic light and ROI detection
    cv2.polylines(frame, [RedLight], True, [0, 0, 255], 1)
    cv2.polylines(frame, [GreenLight], True, [0, 255, 0], 1)
    cv2.polylines(frame, [ROI], True, [255, 0, 0], 2)

    # YOLO model predictions on the frame
    results = model.predict(frame, conf=0.75, verbose=False)  # Suppress output
    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        classes = result.boxes.cls

        for box, conf, cls in zip(boxes, confs, classes):
            if coco[int(cls)] in TargetLabels:
                x1, y1, x2, y2 = map(int, box)
                bounding_box = (x1, y1, x2, y2)

                # If the vehicle is already a violator, keep it red
                if bounding_box in violations:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)
                else:
                    # Otherwise, check if RedLight region is active and vehicle is in ROI
                    if is_region_light(frame, RedLight) and (
                        cv2.pointPolygonTest(ROI, (x1, y1), False) >= 0 or cv2.pointPolygonTest(ROI, (x2, y2), False) >= 0
                    ):
                        # Mark the object with a red bounding box and log violation
                        cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)
                        log_vehicle_violation(frame_number, coco[int(cls)], conf, bounding_box)

    out.write(frame)  # Write the processed frame

cap.release()
out.release()

# Download the output video and data file
files.download(output_video)
files.download(data_file)
