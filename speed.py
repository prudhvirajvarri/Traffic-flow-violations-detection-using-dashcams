!nvidia-smi
!pip install -q supervision ultralytics
import cv2

import numpy as np
import supervision as sv

from tqdm import tqdm
from ultralytics import YOLO
from supervision.assets import VideoAssets, download_assets
from collections import defaultdict, deque
download_assets(VideoAssets.VEHICLES)
SOURCE_VIDEO_PATH = "vehicles.mp4"
TARGET_VIDEO_PATH = "vehicles-result.mp4"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_NAME = "yolov8x.pt"
MODEL_RESOLUTION = 1280
SOURCE = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
frame_iterator = iter(frame_generator)
frame = next(frame_iterator)
import cv2
import supervision as sv  # Importing the supervision library

# Create an sv.Color object in red
color = sv.Color(r=255, g=0, b=0)  # Assuming sv.Color takes RGB values

# Copy the frame to avoid modifying the original
annotated_frame = frame.copy()

# Draw the polygon on the copied frame with the specified color
annotated_frame = sv.draw_polygon(
    scene=annotated_frame,
    polygon=SOURCE,
    color=color,
    thickness=4
)

# Display the annotated frame
sv.plot_image(annotated_frame)

class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
import numpy as np
from collections import defaultdict, deque
from tqdm import tqdm
import supervision as sv  # Ensure correct import for your environment

# Load the YOLO model
model = YOLO(MODEL_NAME)

# Get video information
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

# Initialize tracker
byte_track = sv.ByteTrack(frame_rate=video_info.fps)

# Configure annotators
# Manually set thickness and text scale values
thickness = 2
text_scale = 1

# Use BoxAnnotator instead of BoundingBoxAnnotator
bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    text_position=sv.Position.BOTTOM_CENTER
)
trace_annotator = sv.TraceAnnotator(
    thickness=thickness,
    trace_length=video_info.fps * 2,
    position=sv.Position.BOTTOM_CENTER
)

# Initialize PolygonZone without frame_resolution_wh argument
polygon_zone = sv.PolygonZone(polygon=SOURCE)

# Dictionary to track object coordinates
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

# Open the output video
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:

    # Loop over frames in the source video
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Filter detections by confidence
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        detections = detections[detections.class_id != 0]

        # Filter detections outside the polygon zone
        detections = detections[polygon_zone.trigger(detections)]

        # Apply non-max suppression to refine detections
        detections = detections.with_nms(IOU_THRESHOLD)

        # Pass detections through the tracker
        detections = byte_track.update_with_detections(detections=detections)

        # Get anchor coordinates for detections
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

        # Store detection positions
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        # Format labels
        labels = []
        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                # Calculate speed
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")

        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Write frame to the target video
        sink.write_frame(annotated_frame)

