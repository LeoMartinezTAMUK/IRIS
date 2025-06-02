# Binary Classification UNSW-NB15 Network Intrusion Dataset

# Ensure GPU is properly used (Pytorch):
import torch
print(torch.cuda.is_available())
use_cuda = torch.cuda.is_available()

# Check CUDA properties
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:', torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)

# Set device to GPU
#device = torch.device("cuda")
device = torch.device("cpu") # Set device to CPU explicitly
print("Device: ", device)

#%%

# Binary Classification UNSW-NB15 Network Intrusion Dataset

# Imports
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef

# Load dataset
UNSWTrain = pd.read_csv("data/UNSW_NB15_training-set.csv")
UNSWTest = pd.read_csv("data/UNSW_NB15_testing-set.csv")
print("Length of training set: ", len(UNSWTrain))
print("Length of testing set: ", len(UNSWTest))

# Drop unnecessary columns
UNSWTrain.drop('id', axis=1, inplace=True)
UNSWTrain.drop('attack_cat', axis=1, inplace=True)
UNSWTest.drop('id', axis=1, inplace=True)
UNSWTest.drop('attack_cat', axis=1, inplace=True)

# Replace values for binary classification
UNSWTrain['is_ftp_login'] = UNSWTrain['is_ftp_login'].replace(2, 1).replace(4, 1)
UNSWTest['is_ftp_login'] = UNSWTest['is_ftp_login'].replace(2, 1)

# Define categorical columns and label encode them
categorical_columns = ['proto', 'service', 'state']
label_encoder = preprocessing.LabelEncoder()

for column in categorical_columns:
    UNSWTrain[column] = UNSWTrain[column].astype(str)
    UNSWTest[column] = UNSWTest[column].astype(str)
    label_encoder.fit(pd.concat([UNSWTrain[column], UNSWTest[column]], axis=0))
    UNSWTrain[column] = label_encoder.transform(UNSWTrain[column])
    UNSWTest[column] = label_encoder.transform(UNSWTest[column])

# Scale numerical columns using StandardScaler
columns_to_scale = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload',
    'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin',
    'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
    'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst'
]

scaler = StandardScaler()
for column in columns_to_scale:
    UNSWTrain[column] = scaler.fit_transform(UNSWTrain[[column]])
    UNSWTest[column] = scaler.transform(UNSWTest[[column]])

# Prepare feature and target variables
X_train = UNSWTrain.iloc[:, :-1].values.astype('float32')
y_train = UNSWTrain.iloc[:, -1].values.astype('float32')
X_test = UNSWTest.iloc[:, :-1].values.astype('float32')
y_test = UNSWTest.iloc[:, -1].values.astype('float32')

# Convert data to PyTorch tensors and move to the appropriate device
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Print shapes of the datasets
print(X_train.shape, X_test.shape)

#%%

# Print the class distribution in both the training and testing datasets
train_class_distribution = pd.Series(y_train.cpu().numpy().flatten()).value_counts()
test_class_distribution = pd.Series(y_test.cpu().numpy().flatten()).value_counts()

print("Class distribution in training dataset:")
print(train_class_distribution)

print("Class distribution in testing dataset:")
print(test_class_distribution)

#%%

# Save the processed data to CSV files
# Convert features and labels back to DataFrame
train_data = pd.DataFrame(X_train.numpy())  # Convert tensor back to numpy for DataFrame
train_data['label'] = y_train.numpy()

test_data = pd.DataFrame(X_test.numpy())  # Convert tensor back to numpy for DataFrame
test_data['label'] = y_test.numpy()

# Save the DataFrames to CSV files
train_data.to_csv('processed_train_data.csv', index=False)
test_data.to_csv('processed_test_data.csv', index=False)

print("Training and testing datasets have been saved as 'processed_train_data.csv' and 'processed_test_data.csv'.")

#%%

import torch
import torch.nn as nn
import torch.optim as optim

# Define a neural network model
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

# Initialize the model
input_size = X_train.shape[1]  # Number of input features
model = NeuralNet(input_size).to(device)  # Move the model to GPU/CPU

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

#%%

# Training loop
num_epochs = 100  # You can increase this if necessary
batch_size = 1024  # Choose batch size based on available memory

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    permutation = torch.randperm(X_train.size()[0])  # Shuffle data for each epoch

    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update model parameters

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#%%
# Save trained model  <-- Added
torch.save(model, "IDS_DL-model.pth")
print("Model saved as IDS_DL-model.pth")

#%%

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = (y_pred > 0.5).float()

# Move tensors to CPU and convert to NumPy arrays
y_test_cpu = y_test.cpu().numpy()
y_pred_cpu = y_pred_cls.cpu().numpy()
y_prob_cpu = y_pred.cpu().numpy()

# Classification report
print("Classification Report:")
print(classification_report(y_test_cpu, y_pred_cpu, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test_cpu, y_pred_cpu)
print("Confusion Matrix:")
print(cm)

# --- Confusion Matrix Heatmap ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_heatmap.png", dpi=400)
plt.show()

# --- ROC Curve ---
fpr, tpr, thresholds = roc_curve(y_test_cpu, y_prob_cpu)
auc_score = roc_auc_score(y_test_cpu, y_prob_cpu)
print("AUC Score:", auc_score)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=400)
plt.show()

# --- MCC Score ---
mcc_score = matthews_corrcoef(y_test_cpu, y_pred_cpu)
print("MCC Score:", mcc_score)

#%%

# Save test dataset as network_log.txt with commas separating features
np.savetxt("network_log.txt", X_test.cpu().numpy(), fmt='%f', delimiter=',')
print("Testing data saved as network_log.txt")

