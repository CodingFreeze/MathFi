"""
Quick training script focused on just a few basic math symbols.
This will create a small, focused model that should work well for basic symbols.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import json
from datetime import datetime
import glob
from tensorflow.keras.utils import to_categorical
import cv2

# Ensure TensorFlow is using the GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices}")
else:
    print("No GPU found, using CPU")

def create_model():
    """Create a simple CNN model for recognizing basic math symbols."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 classes: +, -, =, x
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_custom_dataset():
    """Load our custom dataset of basic math symbols."""
    data_dir = 'data/math_symbols'
    symbols = ['+', '-', '=', 'x']
    symbol_to_label = {'+': 0, '-': 1, '=': 2, 'x': 3}
    
    x_data = []
    y_data = []
    
    for symbol in symbols:
        symbol_dir = os.path.join(data_dir, symbol)
        if not os.path.exists(symbol_dir):
            print(f"Warning: Directory {symbol_dir} not found")
            continue
            
        for img_path in glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png")):
            # Read image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
                
            # Resize to 28x28
            img = cv2.resize(img, (28, 28))
            
            # Normalize
            img = img.astype('float32') / 255.0
            
            # Add channel dimension
            img = img.reshape(28, 28, 1)
            
            # Add to dataset
            x_data.append(img)
            y_data.append(symbol_to_label[symbol])
    
    if not x_data:
        raise ValueError("No data loaded! Check your dataset directory.")
    
    # Convert to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # Convert labels to one-hot encoding
    y_data = to_categorical(y_data, num_classes=4)
    
    # Shuffle the data
    indices = np.arange(len(x_data))
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]
    
    # Split into train/test
    split = int(0.8 * len(x_data))
    x_train, x_test = x_data[:split], x_data[split:]
    y_train, y_test = y_data[:split], y_data[split:]
    
    print(f"Loaded {len(x_train)} training samples and {len(x_test)} test samples")
    return x_train, y_train, x_test, y_test

def train_model():
    """Train a model on our custom math symbols dataset."""
    print("Loading custom dataset...")
    x_train, y_train, x_test, y_test = load_custom_dataset()
    
    print("Creating model...")
    model = create_model()
    model.summary()
    
    # Create a label map
    label_map = {
        0: '+', 1: '-', 2: '=', 3: 'x'
    }
    
    # Create directories for saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f'models/basic_symbols_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the label map
    with open(os.path.join(model_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train the model
    print("Training model...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    print("Evaluating model...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
    
    # Save the final model
    model.save(os.path.join(model_dir, 'final_model.keras'))
    
    # Also save to the default location for app use
    default_model_dir = 'models/symbol_recognition_model'
    os.makedirs(default_model_dir, exist_ok=True)
    model.save(os.path.join(default_model_dir, 'model.keras'))
    
    # Update the default label map to include our basic symbols
    full_label_map = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
        5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: '+', 11: '-', 12: '×', 13: '÷', 14: '=',
        15: 'x', 16: 'y', 17: '(', 18: ')', 19: '.'
    }
    
    # Copy the label map to the default location
    with open(os.path.join(default_model_dir, 'label_map.json'), 'w') as f:
        json.dump(full_label_map, f)
    
    print(f"Models saved to {model_dir} and {default_model_dir}")
    print("Training complete!")

if __name__ == "__main__":
    train_model() 