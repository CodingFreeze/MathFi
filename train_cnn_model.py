"""
Train a CNN model for handwritten math symbol recognition.

This script:
1. Downloads or loads math symbol datasets
2. Trains a CNN model for symbol recognition
3. Evaluates the model performance
4. Saves the trained model for use in the app
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime
import time
from tqdm import tqdm
import pandas as pd
from IPython.display import clear_output

from app.models.cnn_model import create_model
from app.utils.data_preprocessing import prepare_dataset, download_kaggle_dataset

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

def train_symbol_recognition_model(model_type='custom_cnn', epochs=50, batch_size=64):
    """
    Train a model for handwritten math symbol recognition.
    
    Args:
        model_type: Type of model architecture to use ('custom_cnn', 'mobilenet', 'resnet')
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model and training history
    """
    print("=" * 50)
    print(f"Training handwritten math symbol recognition model")
    print(f"Model type: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("=" * 50)
    
    # Check if we should try to download Kaggle dataset
    try_kaggle = input("Try to download Kaggle dataset? (y/n): ").lower().strip() == 'y'
    
    if try_kaggle:
        print("Attempting to download Kaggle dataset...")
        download_kaggle_dataset(
            'xainano/handwritten-mathematical-expressions',
            'data/math_symbols'
        )
    
    # Prepare the dataset
    print("Preparing dataset...")
    x_train, y_train, x_test, y_test = prepare_dataset()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Create the model
    print("Creating the model...")
    num_classes = 20  # 10 digits (0-9) + 10 math symbols
    model = create_model(num_classes=num_classes, model_type=model_type)
    
    # Print model summary
    model.summary()
    
    # Create label map for inference (will be saved with the model)
    label_map = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
        5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: '+', 11: '-', 12: '×', 13: '÷', 14: '=',
        15: 'x', 16: 'y', 17: '(', 18: ')', 19: '.'
    }
    
    # Ensure the model directory exists
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f'models/symbol_recognition_model_{model_type}_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    
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
            save_freq='epoch',
            save_best_only=False,
            verbose=1
        ),
        # Early stopping with increased patience
        EarlyStopping(
            monitor='val_loss',
            patience=10,  # Stop if no improvement for 10 epochs
            restore_best_weights=True,
            verbose=1
        ),
        # More aggressive learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,  # Reduce LR if no improvement for 5 epochs
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
    default_model_dir = 'models/symbol_recognition_model'
    os.makedirs(default_model_dir, exist_ok=True)
    model.save(os.path.join(default_model_dir, 'model.keras'))
    
    # Copy the label map to the default location
    with open(os.path.join(default_model_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)
    
    print(f"Models saved to {model_dir} and {default_model_dir}")
    print("Done!")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a handwritten math symbol recognition model')
    parser.add_argument('--model-type', type=str, default='custom_cnn',
                        choices=['custom_cnn', 'mobilenet', 'resnet'],
                        help='Type of model architecture to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
    train_symbol_recognition_model(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size
    ) 