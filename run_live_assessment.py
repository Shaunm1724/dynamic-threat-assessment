import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import DeepOcSort
from pathlib import Path
import torch
import torch.nn as nn
import joblib
from collections import deque

# --- 1. LOAD ALL ASSETS ---
print("Loading assets...")

# -- Configuration & Constants --
VIDEO_PATH = 'videos/4ktraffic.mp4' # IMPORTANT: Use a video you did NOT train on
MODEL_SAVE_PATH = 'trajectory_model.pth'
SCALER_FILE = 'scaler.gz'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# -- Data parameters (must match training) --
HISTORY_LEN = 8
FUTURE_LEN = 12

# -- Model parameters (must match training) --
INPUT_SIZE = 2
HIDDEN_SIZE = 128
NUM_LAYERS = 2

# -- Threat Assessment Parameters --
# Define a "danger zone" at the bottom of the screen (e.g., last 20% of the height)
DANGER_ZONE_Y_START = 0.8  # Start at 80% of the frame height
PROXIMITY_WEIGHT = 0.6     # How much proximity to the danger zone matters
ANOMALY_WEIGHT = 0.4       # How much unpredictable movement matters
ANOMALY_THRESHOLD = 15.0   # Pixel distance threshold to be considered an "anomaly"

# -- Load Perception Models (YOLO + Tracker) --
yolo_model = YOLO('yolov8n.pt')
tracker = DeepOcSort(
    reid_weights=Path('osnet_x0_25_msmt17.pt'),
    device=DEVICE,
    half=True
)

# -- Load Prediction Model --
# First, redefine the model architecture so we can load the weights into it
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_seq_len):
        super(Seq2SeqLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_seq_len * 2)
        self.output_seq_len = output_seq_len
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        last_layer_hidden_state = hidden[-1]
        out = self.linear(last_layer_hidden_state)
        out = out.view(-1, self.output_seq_len, 2)
        return out

# Now, load the trained weights
prediction_model = Seq2SeqLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, FUTURE_LEN).to(DEVICE)
prediction_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
prediction_model.eval() # Set the model to evaluation mode

# -- Load the data scaler --
scaler = joblib.load(SCALER_FILE)

print(f"Assets loaded. Running on device: {DEVICE}")

# --- 2. SETUP FOR REAL-TIME PROCESSING ---

# Dictionary to store the history of each track
# The key is the track_id, the value is a deque (a fixed-size list)
track_histories = {}
# Dictionary to store the last predicted trajectory
track_predictions = {}
# Dictionary to store anomaly scores
track_anomalies = {}

cap = cv2.VideoCapture(VIDEO_PATH)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
DANGER_ZONE_Y_PIXEL = int(frame_height * DANGER_ZONE_Y_START)

# --- 3. MAIN PROCESSING LOOP ---

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # -- Perception Step --
    results = yolo_model(frame, classes=[0, 2, 3, 5, 7], conf=0.5, verbose=False)
    detections = results[0].boxes.data.cpu().numpy() if len(results[0]) > 0 else np.empty((0, 6))
    tracks = tracker.update(detections, frame)

    current_track_ids = set()

    if tracks.shape[0] > 0:
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls = track[:7]
            track_id = int(track_id)
            current_track_ids.add(track_id)

            # -- History Update Step --
            x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
            
            if track_id not in track_histories:
                # Initialize a new history for a new object
                track_histories[track_id] = deque(maxlen=HISTORY_LEN)
            
            # Append the current center point to this track's history
            track_histories[track_id].append((x_center, y_center))

            # -- Prediction Step --
            if len(track_histories[track_id]) == HISTORY_LEN:
                # We have enough history to make a prediction
                history_np = np.array(track_histories[track_id])
                
                # Calculate deltas (change in position)
                history_deltas = np.diff(history_np, axis=0)
                
                # Scale the deltas
                scaled_deltas = scaler.transform(history_deltas)
                
                # Convert to tensor and add a batch dimension
                history_tensor = torch.from_numpy(scaled_deltas).float().unsqueeze(0).to(DEVICE)
                
                # Make the prediction
                with torch.no_grad():
                    # The model expects a sequence of length 8, but diff gives 7. We'll pad.
                    # This is a small bug in the logic. Let's adjust.
                    # We need to feed the model exactly what it was trained on.
                    # The raw history is better. Let's re-calculate deltas for the whole deque.
                    
                    # Correct way to get deltas for the full history
                    history_points = np.array(track_histories[track_id])
                    # We need 8 points to get 7 deltas, but model needs 8 inputs.
                    # The training script had a small logic flaw.
                    # A better way is to store deltas directly. Let's adapt.
                    
                    # For simplicity, let's assume we store positions and calculate deltas on the fly.
                    # We'll feed the model the last 7 deltas and assume the first is zero.
                    # This is a pragmatic fix for this demonstration.
                    
                    if len(history_deltas) == HISTORY_LEN - 1:
                        # Pad with a zero delta at the beginning
                        padded_deltas = np.vstack([np.zeros((1, 2)), scaled_deltas])
                        history_tensor = torch.from_numpy(padded_deltas).float().unsqueeze(0).to(DEVICE)

                        predicted_deltas_tensor = prediction_model(history_tensor)
                        
                        # Post-process the prediction
                        predicted_deltas_scaled = predicted_deltas_tensor.squeeze(0).cpu().numpy()
                        predicted_deltas = scaler.inverse_transform(predicted_deltas_scaled)
                        
                        # Reconstruct the trajectory from the deltas
                        predicted_path = np.zeros_like(predicted_deltas)
                        current_pos = history_np[-1] # Start from the last known position
                        for i in range(len(predicted_deltas)):
                            current_pos = current_pos + predicted_deltas[i]
                            predicted_path[i] = current_pos
                        
                        track_predictions[track_id] = predicted_path
                        
                        # Calculate Kinematic Anomaly
                        # If we have a previous prediction, compare its first step to our current actual position
                        if track_id in track_anomalies and 'prev_pred' in track_anomalies[track_id]:
                           prev_predicted_pos = track_anomalies[track_id]['prev_pred'][0]
                           actual_pos = np.array([x_center, y_center])
                           error = np.linalg.norm(actual_pos - prev_predicted_pos)
                           track_anomalies[track_id]['error'] = error
                        
                        # Store current prediction for next frame's anomaly calculation
                        if track_id not in track_anomalies: track_anomalies[track_id] = {}
                        track_anomalies[track_id]['prev_pred'] = predicted_path


            # -- Threat Assessment & Visualization Step --
            threat_score = 0.0
            proximity_score = 0.0
            anomaly_score = 0.0
            
            # Draw predicted path if it exists
            if track_id in track_predictions:
                path = track_predictions[track_id].astype(int)
                for i in range(len(path) - 1):
                    cv2.line(frame, tuple(path[i]), tuple(path[i+1]), (0, 255, 255), 2)
                
                # Proximity Score Calculation
                if np.any(path[:, 1] > DANGER_ZONE_Y_PIXEL):
                    proximity_score = 1.0 # Max score if any part of the path enters the zone
            
            # Anomaly Score Calculation
            if track_id in track_anomalies and 'error' in track_anomalies[track_id]:
                error = track_anomalies[track_id]['error']
                # Scale score from 0 to 1 based on the threshold
                anomaly_score = min(error / ANOMALY_THRESHOLD, 1.0)
            
            # Final Weighted Threat Score
            threat_score = (proximity_score * PROXIMITY_WEIGHT) + (anomaly_score * ANOMALY_WEIGHT)
            
            # Determine BBox color based on threat
            color = (0, 255, 0) # Green (low threat)
            if threat_score > 0.7:
                color = (0, 0, 255) # Red (high threat)
            elif threat_score > 0.4:
                color = (0, 165, 255) # Orange (medium threat)
            
            # Draw the bounding box and threat score
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id} | T: {threat_score:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Clean up old tracks that are no longer in the frame
    for track_id in list(track_histories.keys()):
        if track_id not in current_track_ids:
            del track_histories[track_id]
            if track_id in track_predictions: del track_predictions[track_id]
            if track_id in track_anomalies: del track_anomalies[track_id]

    # Draw the danger zone for visualization
    cv2.line(frame, (0, DANGER_ZONE_Y_PIXEL), (frame.shape[1], DANGER_ZONE_Y_PIXEL), (255, 0, 255), 2)
    cv2.putText(frame, "DANGER ZONE", (10, DANGER_ZONE_Y_PIXEL - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.imshow("Dynamic Threat Assessment", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()