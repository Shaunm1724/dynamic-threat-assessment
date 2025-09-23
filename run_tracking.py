import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import DeepOcSort
from pathlib import Path 

# --- Configuration ---
VIDEO_PATH = './test.mp4' 
YOLO_MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.5
# Classes we want to track (COCO dataset: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck)
TRACKING_CLASSES = [0, 2, 3, 5, 7] 

# --- Initialization ---
# Load YOLO model
model = YOLO(YOLO_MODEL_PATH)

# Initialize the tracker with a Path object for the weights
tracker = DeepOcSort(
    reid_weights=Path('osnet_x0_25_msmt17.pt'),  
    device='cuda:0',                      
    half=True                             
)

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

# --- Main Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 1. Run YOLO detection
    results = model(frame, classes=TRACKING_CLASSES, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # 2. Extract detection data
    if len(results[0]) > 0:
        detections = results[0].boxes.data.cpu().numpy()
    else:
        detections = np.empty((0, 6))

    # 3. Update the tracker
    tracks = tracker.update(detections, frame)

    # 4. Draw bounding boxes and track IDs
    if tracks.shape[0] > 0:
        for track in tracks:
            # Unpack: [x1, y1, x2, y2, track_id, confidence, class_id, index]
            x1, y1, x2, y2, track_id, conf, cls = track[:7]
            x1, y1, x2, y2, track_id = map(int, [x1, y1, x2, y2, track_id])
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw the track ID
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("DeepOcSort Tracking - Corrected", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
