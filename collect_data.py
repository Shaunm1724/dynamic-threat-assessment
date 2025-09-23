import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import DeepOcSort
from pathlib import Path
import csv 

# --- Configuration ---
VIDEO_PATH = 'videos/4ktraffic.mp4' # IMPORTANT: Update this path
OUTPUT_CSV_PATH = 'csvs/4ktraffic.mp4.csv' # File to save the data
YOLO_MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.5
TRACKING_CLASSES = [0, 2, 3, 5, 7] # person, car, motorcycle, bus, truck

# --- Initialization ---
model = YOLO(YOLO_MODEL_PATH)
tracker = DeepOcSort(
    reid_weights=Path('osnet_x0_25_msmt17.pt'),
    device='cuda:0',
    half=True
)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

# --- CSV File Setup ---
# Open the file in write mode
with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
    # Create a writer object
    csv_writer = csv.writer(csvfile)
    # Write the header row
    csv_writer.writerow(['frame_id', 'track_id', 'x_center', 'y_center', 'width', 'height'])

    frame_number = 0
    # --- Main Loop ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_number += 1

        # 1. Run YOLO detection
        results = model(frame, classes=TRACKING_CLASSES, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # 2. Extract detection data
        if len(results[0]) > 0:
            detections = results[0].boxes.data.cpu().numpy()
        else:
            detections = np.empty((0, 6))

        # 3. Update the tracker
        tracks = tracker.update(detections, frame)

        # --- Data Logging ---
        # List to hold data for the current frame before writing
        frame_data_to_log = []

        # 4. Process and Draw tracks
        if tracks.shape[0] > 0:
            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls = track[:7]
                
                # --- Calculate required data points ---
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # Add data for this track to our list
                frame_data_to_log.append([frame_number, int(track_id), x_center, y_center, width, height])

                # --- Drawing (optional, but good for visualization) ---
                x1, y1, x2, y2, track_id = map(int, [x1, y1, x2, y2, track_id])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write all logged data for this frame to the CSV file
        if frame_data_to_log:
            csv_writer.writerows(frame_data_to_log)

        # Display the frame
        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Data collection complete. Data saved to {OUTPUT_CSV_PATH}")
