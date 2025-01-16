import os
# Set DRI_PRIME=1 for AMD GPU
os.environ['DRI_PRIME'] = '1'
from modelManager import save_model_with_metadata
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback

# Configure GPU settings
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Found the following GPU(s):")
        for gpu in gpus:
            print(f" - {gpu}")
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

data_train = data.iloc[:train_size]
data_test = data.iloc[train_size:]

print(f"Training set size: {len(data_train)}")
print(f"Test set size: {len(data_test)}")

features = ['intersection_id', 'hour', 'day_of_week', 'is_weekend', 'temperature',
            'humidity', 'rain', 'traffic_flow_lag_1', 'traffic_flow_lag_24']
target = 'traffic_flow'

# Verify features
for feature in features + [target]:
    if feature not in data.columns:
        raise ValueError(f"Missing required feature: {feature}")

X, y = data[features].values, data[target].values

# Feature scaling
feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(X)

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
    X_seq, y_seq = create_sequences(X_scaled, y, timesteps)
    print(f"Sequences shape: {X_seq.shape}")
except ValueError as e:
    print(f"Error creating sequences: {e}")
    raise

def build_model(timesteps, input_dim, units_1=64, units_2=32, dropout_rate=0.2):
    model = Sequential([
        LSTM(units_1, activation='relu', return_sequences=True,
             input_shape=(timesteps, input_dim),
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(dropout_rate),
        LSTM(units_2, activation='relu',
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(dropout_rate),
        Dense(16, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae', custom_mape])
    return model

# Step 6: Hyperparameter Tuning and Cross-Validation
kfold = KFold(n_splits=3, shuffle=True, random_state=42)

best_model = None
best_val_loss = float('inf')

units_1_options = [32, 64]
units_2_options = [16, 32]
dropout_options = [0.2, 0.3]

for units_1 in units_1_options:
    for units_2 in units_2_options:
        for dropout_rate in dropout_options:
            print(f"Testing configuration: units_1={units_1}, units_2={units_2}, dropout_rate={dropout_rate}")

            fold_val_losses = []

            for train_idx, val_idx in kfold.split(X_seq):
                X_train_fold, X_val_fold = X_seq[train_idx], X_seq[val_idx]
                y_train_fold, y_val_fold = y_seq[train_idx], y_seq[val_idx]

                model = build_model(timesteps, X_train_fold.shape[2], units_1, units_2, dropout_rate)

                history = model.fit(
                    X_train_fold, y_train_fold,
                    validation_data=(X_val_fold, y_val_fold),
                    epochs=20,
                    batch_size=32,
                    verbose=0,
                    callbacks=[TqdmCallback(verbose=1)]  # Add TQDM progress bar
                )

                val_loss = min(history.history['val_loss'])
                fold_val_losses.append(val_loss)

            avg_val_loss = np.mean(fold_val_losses)
            print(f"Average Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model
                print("New best model found!")

# Training summary
print("\nTraining Summary:")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print("Best Model Metrics:")
val_predictions = best_model.predict(X_seq[:500])

# Evaluate the model on test data
test_predictions = best_model.predict(X_seq[train_size:])
test_actual = y_seq[train_size:]

# Metrics
mse = mean_squared_error(test_actual, test_predictions)
mae = mean_absolute_error(test_actual, test_predictions)
precision = precision_score(test_actual.round(), test_predictions.round(), average='weighted')
recall = recall_score(test_actual.round(), test_predictions.round(), average='weighted')
f1 = f1_score(test_actual.round(), test_predictions.round(), average='weighted')
accuracy = accuracy_score(test_actual.round(), test_predictions.round())

# Final metrics
print("\nEvaluation Metrics on Test Data:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.plot(test_actual[:100], label="Actual", color='blue', marker='o')
plt.plot(test_predictions[:100], label="Predicted", color='red', linestyle='--', marker='x')
plt.title("Actual vs Predicted Traffic Flow")
plt.xlabel("Time Step")
plt.ylabel("Traffic Flow (Scaled)")
plt.legend()
plt.show()

# Final learning rate
final_lr = tf.keras.backend.get_value(best_model.optimizer.learning_rate)
print(f"Final Learning Rate: {final_lr:.6f}")

# Prepare metrics dictionary
metrics = {
    'mse': mse,
    'mae': mae,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'accuracy': accuracy
}

# Save the model
saved_model_path = save_model_with_metadata(best_model, metrics)