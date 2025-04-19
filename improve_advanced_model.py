"""
Improve the model accuracy for advanced math symbol recognition
by using:
1. Improved CNN model architecture with residual connections
2. Enhanced data augmentation
3. Optimized learning rate and batch size
4. Class weighting to handle imbalanced data
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import json
from datetime import datetime
import time
import glob
import cv2
from tqdm import tqdm
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import our model creation function
from app.models.cnn_model import create_model

def load_and_balance_dataset(data_dir='data/math_symbols', test_split=0.2, max_samples=None):
    """
    Load the dataset with advanced math symbols and balance the classes.
    
    Args:
        data_dir: Directory containing the dataset
        test_split: Proportion of data to use for testing
        max_samples: Maximum number of samples per class (for balancing)
        
    Returns:
        (x_train, y_train, x_test, y_test), label_map, class_weights
    """
    print(f"Loading and balancing advanced math symbols dataset from {data_dir}...")
    
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
    
    # Count samples per class for statistics
    class_counts = {}
    
    for symbol in tqdm(symbol_dirs, desc="Loading symbols"):
        symbol_dir = os.path.join(data_dir, symbol)
        files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
        
        # Limit samples per class if specified
        if max_samples and len(files) > max_samples:
            files = files[:max_samples]
            
        class_counts[symbol] = len(files)
        
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
    
    # Calculate class weights for imbalanced dataset
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_data),
        y=y_data
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Print class distribution
    print("Class distribution:")
    for symbol, count in class_counts.items():
        print(f"  {symbol}: {count} samples")
    
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
    y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
    y_test_one_hot = to_categorical(y_test, num_classes=num_classes)
    
    print(f"Dataset loaded: {len(x_train)} training samples, {len(x_test)} test samples")
    print(f"Number of classes: {num_classes}")
    
    return (x_train, y_train_one_hot, x_test, y_test_one_hot), label_map, class_weight_dict

def create_advanced_datagen():
    """
    Create an enhanced ImageDataGenerator with more aggressive augmentation.
    
    Returns:
        An ImageDataGenerator for training
    """
    return ImageDataGenerator(
        rotation_range=15,           # Increased rotation range
        width_shift_range=0.15,      # Increased shift range
        height_shift_range=0.15,     # Increased shift range
        zoom_range=0.15,            # Increased zoom range
        shear_range=0.15,           # Increased shear range
        fill_mode='nearest',
        brightness_range=[0.8, 1.2], # Brightness variation
        # Horizontal flip disabled (could affect symbol meaning)
    )

def train_improved_model(epochs=100, batch_size=32, learning_rate=0.0005, model_type='improved_cnn'):
    """
    Train an improved model for advanced handwritten math symbol recognition.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        model_type: Type of model architecture to use
        
    Returns:
        Trained model
    """
    print("=" * 60)
    print(f"Training improved model for advanced math symbol recognition")
    print(f"Model type: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)
    
    # Check for GPU availability
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Using GPU: {physical_devices}")
    else:
        print("No GPU found, using CPU")
    
    # Load the dataset
    (x_train, y_train, x_test, y_test), label_map, class_weights = load_and_balance_dataset()
    
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
    model_dir = f'models/improved_math_symbol_model_{model_type}_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)
    
    # Save the label map
    with open(os.path.join(model_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)
    
    # Create enhanced data augmentation for training
    datagen = create_advanced_datagen()
    
    # Callbacks for training
    callbacks = [
        # Save the best model
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Save checkpoints every 20 epochs
        ModelCheckpoint(
            os.path.join(model_dir, 'checkpoints', 'model_epoch_{epoch:03d}.h5'),
            monitor='val_accuracy',
            save_best_only=False,
            save_freq=20 * len(x_train) // batch_size,
            verbose=1
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,    # More generous patience for learning rate reduction
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model with class weighting
    print("Training model...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate the model
    print("Evaluating the model...")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
    
    # Save the final model in both formats
    print("Saving the model...")
    model.save(os.path.join(model_dir, 'final_model.h5'))
    
    # Also save to the default location for the app
    model.save('models/advanced_recognition_model')
    
    # Plot and save training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    
    print(f"Models saved to {model_dir} and models/advanced_recognition_model")
    print("Done!")
    
    return model, history

if __name__ == "__main__":
    train_improved_model(
        epochs=100,
        batch_size=32,
        learning_rate=0.0005,
        model_type='improved_cnn'
    ) 