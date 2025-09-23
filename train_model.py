import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

# --- Configuration & Hyperparameters ---
DATA_FILE = 'processed_trajectories.npz'
MODEL_SAVE_PATH = 'trajectory_model.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data parameters from preprocess_data.py
HISTORY_LEN = 8
FUTURE_LEN = 12

# Model parameters
INPUT_SIZE = 2  # delta_x, delta_y
HIDDEN_SIZE = 128 # Increased memory size
NUM_LAYERS = 2
OUTPUT_SEQ_LEN = FUTURE_LEN

# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 25 # Increased for better training

# --- 1. Corrected Model Definition ---
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_seq_len):
        super(Seq2SeqLSTM, self).__init__()
        # --- Encoder ---
        # The LSTM processes the input sequence and captures its meaning in the hidden state.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # --- Decoder ---
        # A linear layer maps the final hidden state of the LSTM to the desired output sequence length.
        # It will output a flat vector of (output_seq_len * 2 features).
        self.linear = nn.Linear(hidden_size, output_seq_len * 2) # 2 for delta_x and delta_y
        self.output_seq_len = output_seq_len

    def forward(self, x):
        # The LSTM's forward pass returns the full output sequence and the final hidden/cell states.
        # We only care about the final hidden state, as it summarizes the entire input sequence.
        _, (hidden, _) = self.lstm(x)
        
        # We take the hidden state from the last layer.
        # The shape is [num_layers, batch_size, hidden_size]. We want the last layer's state.
        last_layer_hidden_state = hidden[-1]
        
        # Pass this summary through the linear layer to get the prediction.
        out = self.linear(last_layer_hidden_state)
        
        # Reshape the output from a flat vector to the desired sequence shape: [batch_size, 12, 2]
        out = out.view(-1, self.output_seq_len, 2)
        return out

# --- 2. Data Loading and Preparation ---
print(f"Using device: {DEVICE}")
print("Loading and preparing data...")

with np.load(DATA_FILE) as data:
    X_train = data['X']
    y_train = data['y']

X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Data loaded. Training with {len(X_train)} samples.")

# --- 3. Model Training ---
model = Seq2SeqLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SEQ_LEN).to(DEVICE)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting model training...")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for i, (history, future_target) in enumerate(train_loader):
        history = history.to(DEVICE)
        future_target = future_target.to(DEVICE)
        
        # --- Forward pass ---
        predicted_future = model(history)
        loss = loss_function(predicted_future, future_target)
        
        # --- Backward pass and optimization ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    # Print loss statistics for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_epoch_loss:.6f}')

end_time = time.time()
print(f"Training finished in {(end_time - start_time):.2f} seconds.")

# --- 4. Save the Trained Model ---
# We save the model's "state dictionary" - its learned weights and biases
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")