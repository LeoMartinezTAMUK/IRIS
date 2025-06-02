import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, 64)          # Fully connected layer 2
        self.fc3 = nn.Linear(64, 1)            # Output layer for binary classification
        self.relu = nn.ReLU()                  # Activation function
        self.sigmoid = nn.Sigmoid()            # Sigmoid for binary output

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load trained deep learning model
model = torch.load("IDS_DL-model.pth")
model = model.to(device)  # Move model to GPU or CPU based on availability
model.eval()  # Set to evaluation mode

# File paths
log_file = "network_log.txt"
detection_log = "detections.log"

#%%

# Function to preprocess a single log entryS
def preprocess_log_entry(entry):
    """Convert raw log data to match model training format"""
    values = entry.strip().split(",")  

    # Convert to float (adjust if needed)
    features = np.array(values, dtype=np.float32)

    # Convert to tensor for model input
    tensor_input = torch.tensor(features).unsqueeze(0)  # Add batch dimension
    
    # Move the tensor to the same device as the model
    return tensor_input.to(device)

#%%

# Function to monitor the log file for new entries
def monitor_log_file(log_file, detection_log, duration=30):
    start_time = time.time()
    last_position = 0  # Track last read position
    intrusions_count = 0  # Count of intrusions
    normal_count = 0  # Count of normal traffic

    with open(detection_log, "w") as log:
        log.write("Timestamp, Line Number, Prediction\n")  # CSV Header

    line_number = 0  # Track line number for logging

    while time.time() - start_time < duration:
        with open(log_file, "r") as f:
            f.seek(last_position)  # Move to last read position
            new_lines = f.readlines()
            last_position = f.tell()  # Update position

        for line in new_lines:
            line_number += 1  # Increment line number
            
            if line.strip():  # Skip empty lines
                try:
                    input_data = preprocess_log_entry(line)
                    
                    with torch.no_grad():
                        prediction = model(input_data)
                    
                    # threshold for binary classification (not at 0.5 due to heavy imbalance in training)
                    predicted_label = (prediction > 0.89975).float().item()  
                    label_str = "Intrusion Detected!" if predicted_label == 1 else "Normal Traffic"
                    
                    # Update counters
                    if predicted_label == 1:
                        intrusions_count += 1
                        
                    else:
                        normal_count += 1

                    # Print real-time detection
                    print(f"[{time.strftime('%H:%M:%S')}] Line {line_number}: {label_str}")

                    # Log detection to file
                    with open(detection_log, "a") as log:
                        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{line_number},{label_str}\n")

                except Exception as e:
                    print(f"Error processing line {line_number}: {e}")

        # Print the current counts at each iteration
        print(f"[{time.strftime('%H:%M:%S')}] Total Intrusions: {intrusions_count}, Total Normal: {normal_count}")

        time.sleep(1)  # Wait before checking for new entries

# Run the real-time monitoring system
monitor_log_file(log_file, detection_log, duration=30)
