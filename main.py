import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for reproducibility
np.random.seed(42)

# number of time steps (more than 2000 as required)
n_steps = 2500
time = np.arange(n_steps)

# generate correlated multivariate time series
feature_1 = 0.03 * time + np.sin(0.02 * time) + np.random.normal(0, 0.4, n_steps)
feature_2 = 0.7 * feature_1 + np.random.normal(0, 0.3, n_steps)
feature_3 = np.cos(0.02 * time) + np.random.normal(0, 0.4, n_steps)

# create dataframe
data = pd.DataFrame({
    "Feature_1": feature_1,
    "Feature_2": feature_2,
    "Feature_3": feature_3
})

print(data.head())

# plot the dataset
data.plot(figsize=(10, 5))
plt.title("Synthetic Multivariate Time Series Dataset")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

from sklearn.preprocessing import MinMaxScaler

# scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# create sequences
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 30
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# train-test split (time based)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        return self.fc(out)
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # context vector
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return self.fc(context)
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# convert data to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

def train_model(model, train_loader, X_test, y_test, epochs=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            output = model(xb)
            loss = criterion(output, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    return rmse, mae
# convert numpy arrays to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# create DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# initialize models
input_size = X_train.shape[2]
hidden_size = 64

lstm_model = LSTMModel(input_size, hidden_size)
attention_model = AttentionLSTM(input_size, hidden_size)

# train models
print("\nTraining LSTM model...")
rmse_lstm, mae_lstm = train_model(lstm_model, train_loader, X_test_t, y_test)

print("\nTraining Attention LSTM model...")
rmse_attn, mae_attn = train_model(attention_model, train_loader, X_test_t, y_test)

# final output
print("\n--- FINAL RESULTS ---")
print(f"LSTM -> RMSE: {rmse_lstm:.4f}, MAE: {mae_lstm:.4f}")
print(f"Attention LSTM -> RMSE: {rmse_attn:.4f}, MAE: {mae_attn:.4f}")
