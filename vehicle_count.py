# Remove any existing YOLOv5 directory
!rm -rf yolov5

# Clone the latest YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5

# Install all the required dependencies
!pip install -qr yolov5/requirements.txt
# Install YOLOv5 and Ultralytics dependencies
!pip install -qr yolov5/requirements.txt  # Install YOLOv5 dependencies
!pip install ultralytics  # Install Ultralytics package
!pip install sort
import torch
import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # For displaying images
from ultralytics import YOLO
model = YOLO('yolov5s.pt')  # Loads YOLOv5s
from google.colab import files

def upload_video():
    uploaded = files.upload()
    for name in uploaded.keys():
        return name

video_path = upload_video()

# Function to process the video and save output with labeled vehicles
def detect_vehicles(video_path, output_path="output_video.mp4"):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video details (height, width, FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    vehicle_classes = ['car', 'bus', 'truck', 'motorbike']
    vehicle_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit when video ends

        # Perform inference with YOLOv5
        results = model(frame)

        # Extract detections (the results are now a list of results)
        detections = results[0].boxes  # Use boxes to get the bounding box results

        # Initialize current vehicle number for this frame
        current_vehicle_number = 1

        # Filter for vehicles and label them
        for box in detections:
            # Extract box coordinates, confidence, and class index
            x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates
            conf = box.conf[0]  # Get confidence
            cls = int(box.cls[0])  # Get class index

            class_name = model.model.names[cls]  # Get class name
            if class_name in vehicle_classes:
                vehicle_count += 1
                # Draw bounding box and label vehicle number
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                label = f"Vehicle {current_vehicle_number} ({class_name})"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                current_vehicle_number += 1

        # Write the processed frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    print(f"Processing completed. Output saved to {output_path}")

# Upload the video
video_path = upload_video()

# Run the detection and generate the output video
detect_vehicles(video_path)

# Download the output video
files.download("output_video.mp4")


