import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
import joblib # For saving the scaler

# --- Configuration ---
# Use a glob pattern to find all your data files
CSV_FILES_PATTERN = 'csvs/*.csv' 
OUTPUT_FILE = 'processed_trajectories.npz'
SCALER_FILE = 'scaler.gz'

# Sequence lengths: We'll use 8 past frames to predict 12 future frames
HISTORY_LEN = 8
FUTURE_LEN = 12
TOTAL_LEN = HISTORY_LEN + FUTURE_LEN

# --- Main Script ---
print("Starting data preprocessing...")

# 1. Load and concatenate all CSV files
csv_files = glob.glob(CSV_FILES_PATTERN)
print(f"Found {len(csv_files)} data files.")
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
print(f"Total rows loaded: {len(df)}")

# 2. Pre-calculate deltas (change in position)
# Sorting is crucial to ensure deltas are calculated correctly
df.sort_values(by=['track_id', 'frame_id'], inplace=True)
df['delta_x'] = df.groupby('track_id')['x_center'].diff().fillna(0)
df['delta_y'] = df.groupby('track_id')['y_center'].diff().fillna(0)

# 3. Scale the delta features
# Scaling helps the neural network learn more effectively
scaler = StandardScaler()
df[['delta_x', 'delta_y']] = scaler.fit_transform(df[['delta_x', 'delta_y']])
print("Feature scaling complete.")

# 4. Create sequences using a sliding window approach
sequences_X = [] # History (input)
sequences_y = [] # Future (target)

# Group by track_id to process each object's trajectory individually
grouped = df.groupby('track_id')
total_tracks = len(grouped)
print(f"Processing {total_tracks} unique tracks...")

for i, (track_id, track_df) in enumerate(grouped):
    # Get the relevant data for this track as a NumPy array
    track_data = track_df[['delta_x', 'delta_y']].to_numpy()
    
    # A track needs to be long enough to create at least one full sequence
    if len(track_data) >= TOTAL_LEN:
        # Slide a window across the track data
        for j in range(len(track_data) - TOTAL_LEN + 1):
            # The history part is the input (X)
            history = track_data[j : j + HISTORY_LEN]
            sequences_X.append(history)
            
            # The future part is the target (y)
            future = track_data[j + HISTORY_LEN : j + TOTAL_LEN]
            sequences_y.append(future)
    
    if (i + 1) % 500 == 0:
        print(f"  Processed {i+1}/{total_tracks} tracks...")

print(f"Created {len(sequences_X)} sequences.")

# 5. Convert lists to NumPy arrays
X = np.array(sequences_X)
y = np.array(sequences_y)

# 6. Save the processed data and the scaler
print(f"Saving processed data to {OUTPUT_FILE}...")
# We save X and y into a single compressed file
np.savez_compressed(OUTPUT_FILE, X=X, y=y)
# We also save the scaler object, as we'll need it later for predictions
joblib.dump(scaler, SCALER_FILE)

print("Preprocessing complete!")
print(f"Input shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")