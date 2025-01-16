import os
from datetime import datetime
import json


def save_model_with_metadata(model, metrics, model_dir='saved_models'):
    """
    Save the model along with its training metrics and timestamp

    Parameters:
    model: tensorflow model
    metrics: dict containing model metrics
    model_dir: directory to save the model
    """
    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Generate timestamp for unique model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f'traffic_model_{timestamp}')

    # Save the model
    model.save(model_path)

    # Save the metrics and parameters
    metadata = {
        'timestamp': timestamp,
        'metrics': {
            'mse': float(metrics['mse']),
            'mae': float(metrics['mae']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'accuracy': float(metrics['accuracy'])
        },
        'model_parameters': {
            'timesteps': model.input_shape[1],
            'input_dim': model.input_shape[2]
        }
    }

    # Save metadata
    with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Model saved to: {model_path}")
    return model_path


def load_model_with_metadata(model_path):
    """
    Load a saved model and its metadata

    Parameters:
    model_path: path to the saved model directory

    Returns:
    model: loaded tensorflow model
    metadata: dict containing model metadata
    """
    # Load the model
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'custom_mape': custom_mape})

    # Load metadata
    metadata_path = os.path.join(model_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = None
        print("No metadata file found")

    return model, metadata