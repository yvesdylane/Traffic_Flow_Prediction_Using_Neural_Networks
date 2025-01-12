import os
# Set DRI_PRIME=1 for AMD GPU
os.environ['DRI_PRIME'] = '1'

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt

# Configure GPU settings
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Found the following GPU(s):")
        for gpu in gpus:
            print(f" - {gpu}")
            # Configure GPU memory growth
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\nGPU memory growth enabled")
    else:
        print("No GPU found. Running on CPU.")
except Exception as e:
    print(f"GPU configuration error: {e}")
    print("Falling back to CPU.")

# Custom MAPE metric to handle edge cases
def custom_mape(y_true, y_pred):
    epsilon = 1e-7  # Small constant to avoid division by zero
    diff = tf.abs((y_true - y_pred) / (y_true + epsilon))
    # Clip the values to avoid extreme percentages
    diff = tf.clip_by_value(diff, 0, 1)
    return tf.reduce_mean(diff) * 100

# Step 1: Load dataset with error handling
try:
    data = pd.read_csv('synthetic_traffic_data.csv')
    print(f"Loaded dataset with {len(data)} entries")
except FileNotFoundError:
    raise FileNotFoundError("Could not find 'synthetic_traffic_data.csv'. Please check the file path.")

# Step 2: Data Preprocessing
try:
    data['timestamp'] = pd.to_datetime(data['timestamp'])
except ValueError as e:
    print(f"Error converting timestamps: {e}")
    print("Please ensure timestamp format is consistent")
    raise

# 2.1 Extract time-based features
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# 2.2 Normalize weather data
weather_scaler = MinMaxScaler()
traffic_scaler = MinMaxScaler()

data[['temperature', 'humidity', 'rain']] = weather_scaler.fit_transform(
    data[['temperature', 'humidity', 'rain']].values
)

# 2.3 Create lag features
data.sort_values(by=['intersection_id', 'timestamp'], inplace=True)
data['traffic_flow_lag_1'] = data.groupby('intersection_id')['traffic_flow'].shift(1)
data['traffic_flow_lag_24'] = data.groupby('intersection_id')['traffic_flow'].shift(24)

# 2.4 Normalize traffic flow
data['traffic_flow'] = traffic_scaler.fit_transform(data[['traffic_flow']])

# Drop rows with NaN values
original_len = len(data)
data.dropna(inplace=True)
print(f"Dropped {original_len - len(data)} rows containing NaN values")

# Step 3: Dataset Splitting
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)

train_data = data.iloc[:train_size]
validation_data = data.iloc[train_size:train_size + val_size]
test_data = data.iloc[train_size + val_size:]

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(validation_data)}")
print(f"Test set size: {len(test_data)}")

features = ['intersection_id', 'hour', 'day_of_week', 'is_weekend', 'temperature',
            'humidity', 'rain', 'traffic_flow_lag_1', 'traffic_flow_lag_24']
target = 'traffic_flow'

# Verify features
for feature in features + [target]:
    if feature not in data.columns:
        raise ValueError(f"Missing required feature: {feature}")

X_train, y_train = train_data[features].values, train_data[target].values
X_val, y_val = validation_data[features].values, validation_data[target].values
X_test, y_test = test_data[features].values, test_data[target].values

# Feature scaling
feature_scaler = MinMaxScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_val_scaled = feature_scaler.transform(X_val)
X_test_scaled = feature_scaler.transform(X_test)


def create_sequences(X, y, timesteps):
    """
    Create sequences from numpy arrays
    """
    if len(X) <= timesteps:
        raise ValueError("Input length must be greater than timesteps")

    X_seq = []
    y_seq = []

    for i in range(len(X) - timesteps):
        X_seq.append(X[i:(i + timesteps)])
        y_seq.append(y[i + timesteps])

    return np.array(X_seq), np.array(y_seq)


timesteps = 5
try:
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, timesteps)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, timesteps)
    print(f"Training sequences shape: {X_train_seq.shape}")
    print(f"Validation sequences shape: {X_val_seq.shape}")
except ValueError as e:
    print(f"Error creating sequences: {e}")
    raise

# Define the model
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True,
         input_shape=(timesteps, X_train_seq.shape[2]),
         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.2),
    LSTM(32, activation='relu',
         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.2),
    Dense(16, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dense(1, activation='linear')
])

# Compile with custom MAPE
model.compile(optimizer='adam',
             loss='mse',
             metrics=['mae', custom_mape])

# Early stopping with more patience
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,  # Increased patience
    restore_best_weights=True,
    verbose=1,
    min_delta=0.0001  # Minimum change to qualify as an improvement
)

# Add learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=50,
    min_lr=0.0001,
    verbose=1
)

# Train with both callbacks
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate
val_metrics = model.evaluate(X_val_seq, y_val_seq, verbose=1)
print("\nValidation Metrics:")
print(f"Loss (MSE): {val_metrics[0]:.4f}")
print(f"MAE: {val_metrics[1]:.4f}")
print(f"MAPE: {val_metrics[2]:.4f}%")

# Calculate precision, recall, accuracy, and F1-score
val_predictions = model.predict(X_val_seq)
val_predictions_binary = (val_predictions > 0.5).astype(int)
y_val_seq_binary = (y_val_seq > 0.5).astype(int)

precision = precision_score(y_val_seq_binary, val_predictions_binary)
recall = recall_score(y_val_seq_binary, val_predictions_binary)
accuracy = accuracy_score(y_val_seq_binary, val_predictions_binary)
f1 = f1_score(y_val_seq_binary, val_predictions_binary)

print("\nMetrics for binary classification:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")

# Save model in the modern Keras format
try:
    model_path = 'traffic_prediction_model.keras'
    model.save(model_path)
    print(f"\nModel saved successfully as {model_path}")
except Exception as e:
    print(f"Error saving model: {e}")

# Print final training summary with min/max metrics
print("\nTraining Summary:")
print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
print(f"Best validation MAE: {min(history.history['val_mae']):.4f}")
print(f"Best validation MAPE: {min(history.history['val_custom_mape']):.4f}%")
print(f"\nFinal learning rate: {tf.keras.backend.get_value(model.optimizer.learning_rate):.6f}")

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.scatter(y_val_seq[:500], val_predictions[:500], alpha=0.5)
plt.plot([y_val_seq.min(), y_val_seq.max()], [y_val_seq.min(), y_val_seq.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Traffic Flow (First 500 samples)')
plt.grid(True)
plt.show()
