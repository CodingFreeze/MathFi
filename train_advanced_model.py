"""
Train a CNN model for advanced handwritten math symbol recognition.

This script:
1. Loads our enhanced dataset with advanced math symbols
2. Trains a CNN model for recognition of basic and advanced symbols
3. Evaluates the model performance
4. Saves the trained model for use in the app
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime
import time
from tqdm import tqdm
import pandas as pd
import glob
import cv2
from IPython.display import clear_output

from app.models.cnn_model import create_model

class TrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_times = []
        self.metrics_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print("\nStarting training...")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{self.total_epochs}")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # Update metrics history
        for metric in self.metrics_history:
            if metric in logs:
                self.metrics_history[metric].append(logs[metric])
        
        # Calculate and display statistics
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.total_epochs - (epoch + 1)
        estimated_time_remaining = avg_epoch_time * remaining_epochs
        
        # Display progress
        clear_output(wait=True)
        print(f"\nEpoch {epoch + 1}/{self.total_epochs}")
        print(f"Time taken: {epoch_time:.2f}s")
        print(f"Average epoch time: {avg_epoch_time:.2f}s")
        print(f"Estimated time remaining: {estimated_time_remaining/60:.1f} minutes")
        print("\nCurrent Metrics:")
        print(f"Training Loss: {logs['loss']:.4f}")
        print(f"Training Accuracy: {logs['accuracy']:.4f}")
        print(f"Validation Loss: {logs['val_loss']:.4f}")
        print(f"Validation Accuracy: {logs['val_accuracy']:.4f}")
        
        # Plot metrics
        self.plot_metrics()
        
    def plot_metrics(self):
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics_history['accuracy'], label='Training')
        plt.plot(self.metrics_history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics_history['loss'], label='Training')
        plt.plot(self.metrics_history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

def load_advanced_dataset(data_dir='data/math_symbols', test_split=0.2):
    """
    Load the enhanced dataset with advanced math symbols.
    
    Args:
        data_dir: Directory containing the dataset
        test_split: Proportion of data to use for testing
        
    Returns:
        (x_train, y_train, x_test, y_test), label_map
    """
    print(f"Loading advanced math symbols dataset from {data_dir}...")
    
    # Get all subdirectories (each corresponds to a symbol class)
    symbol_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d)) and d != 'equations']
    
    # Sort to ensure consistent label ordering
    symbol_dirs.sort()
    
    # Create a mapping from symbol to class index
    label_map = {i: symbol for i, symbol in enumerate(symbol_dirs)}
    symbol_to_idx = {symbol: i for i, symbol in label_map.items()}
    
    images = []
    labels = []
    
    for symbol in tqdm(symbol_dirs, desc="Loading symbols"):
        symbol_dir = os.path.join(data_dir, symbol)
        files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
        
        for file_path in files:
            # Read and preprocess image
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Ensure 28x28 size
            if img.shape[0] != 28 or img.shape[1] != 28:
                img = cv2.resize(img, (28, 28))
                
            # Normalize pixel values to [0, 1]
            img = img.astype('float32') / 255.0
            
            # Add to dataset
            images.append(img.reshape(28, 28, 1))  # Add channel dimension
            labels.append(symbol_to_idx[symbol])
    
    # Convert to numpy arrays
    x_data = np.array(images)
    y_data = np.array(labels)
    
    # Shuffle the data
    indices = np.arange(len(x_data))
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]
    
    # Split into train and test sets
    split_idx = int(len(x_data) * (1 - test_split))
    x_train, x_test = x_data[:split_idx], x_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]
    
    # Convert labels to one-hot
    num_classes = len(symbol_dirs)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    
    print(f"Dataset loaded: {len(x_train)} training samples, {len(x_test)} test samples")
    print(f"Number of classes: {num_classes}")
    
    return (x_train, y_train, x_test, y_test), label_map

def train_advanced_recognition_model(model_type='custom_cnn', epochs=100, batch_size=64, learning_rate=0.001):
    """
    Train a model for advanced handwritten math symbol recognition.
    
    Args:
        model_type: Type of model architecture to use ('custom_cnn', 'mobilenet', 'resnet')
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        
    Returns:
        Trained model and training history
    """
    print("=" * 50)
    print(f"Training advanced handwritten math symbol recognition model")
    print(f"Model type: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 50)
    
    # Load the dataset
    (x_train, y_train, x_test, y_test), label_map = load_advanced_dataset()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Create the model
    print("Creating the model...")
    num_classes = len(label_map)
    model = create_model(input_shape=(28, 28, 1), num_classes=num_classes, model_type=model_type)
    
    # Compile with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Ensure the model directory exists
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f'models/advanced_recognition_model_{model_type}_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)
    
    # Save the label map
    with open(os.path.join(model_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)
    
    # Create data augmentation for training
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # Create TensorBoard callback
    tensorboard_dir = os.path.join(model_dir, 'logs')
    os.makedirs(tensorboard_dir, exist_ok=True)
    tensorboard_callback = TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )
    
    # Create training monitor
    training_monitor = TrainingMonitor(epochs)
    
    # Create callbacks
    callbacks = [
        # Save the best model
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Save model checkpoints every 10 epochs
        ModelCheckpoint(
            os.path.join(model_dir, 'checkpoints', 'model_epoch_{epoch:03d}.h5'),
            save_freq=10 * (len(x_train) // batch_size),  # Every 10 epochs
            save_best_only=False,
            verbose=1
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience for complex dataset
            restore_best_weights=True,
            verbose=1
        ),
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard logging
        tensorboard_callback,
        # Training monitor
        training_monitor
    ]
    
    # Train the model
    print(f"Training the model for {epochs} epochs...")
    
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=callbacks,
        verbose=0  # Set to 0 since we're using our custom monitor
    )
    
    # Evaluate the model
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
    
    # Save training history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(model_dir, 'training_history.csv'), index=False)
    
    # Save the final model (we also saved the best model during training)
    print("Saving the model...")
    model.save(os.path.join(model_dir, 'final_model.keras'))
    
    # Also save to the default location for app use
    default_model_dir = 'models/advanced_recognition_model'
    os.makedirs(default_model_dir, exist_ok=True)
    model.save(os.path.join(default_model_dir, 'model.keras'))
    
    # Copy the label map to the default location
    with open(os.path.join(default_model_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)
    
    print(f"Models saved to {model_dir} and {default_model_dir}")
    print("Done!")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an advanced handwritten math symbol recognition model')
    parser.add_argument('--model-type', type=str, default='custom_cnn',
                        choices=['custom_cnn', 'mobilenet', 'resnet'],
                        help='Type of model architecture to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate')
    
    args = parser.parse_args()
    
    train_advanced_recognition_model(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    ) 