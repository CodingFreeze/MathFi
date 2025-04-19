"""
Improved training script for handwritten math symbol recognition.
This script uses data augmentation and a more robust CNN architecture.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
from datetime import datetime
import glob
from tensorflow.keras.utils import to_categorical
import cv2
from tqdm import tqdm
import pandas as pd

# Ensure TensorFlow is using the GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices}")
else:
    print("No GPU found, using CPU")

def create_model(num_classes=20):
    """Create an improved CNN model for recognizing handwritten math symbols."""
    # Use a more sophisticated architecture
    model = tf.keras.Sequential([
        # First convolution block
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second convolution block
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Third convolution block
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Fully connected layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use a better optimizer with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_dataset():
    """Load the dataset of handwritten math symbols."""
    data_dir = 'data/math_symbols'
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found!")
        print("Please run download_crohme_dataset.py first to prepare the dataset.")
        return None
    
    # Define symbol classes
    symbol_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                      '+', '-', '×', '÷', '=', 'x', 'y', '(', ')', '.']
    
    # Create symbol to index mapping
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbol_classes)}
    
    # Initialize data arrays
    x_data = []
    y_data = []
    
    # Count available samples per class
    total_per_class = {}
    
    # Load each symbol class
    for symbol in symbol_classes:
        symbol_dir = os.path.join(data_dir, symbol)
        
        # Skip if directory doesn't exist
        if not os.path.exists(symbol_dir):
            print(f"Warning: No directory for symbol {symbol}")
            total_per_class[symbol] = 0
            continue
        
        # Get all image files for this symbol
        image_files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
        total_per_class[symbol] = len(image_files)
        
        # Load each image
        for img_path in image_files:
            # Read image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Normalize to [0, 1]
            img = img.astype('float32') / 255.0
            
            # Add channel dimension
            img = img.reshape(28, 28, 1)
            
            # Add to dataset
            x_data.append(img)
            y_data.append(symbol_to_idx[symbol])
    
    if not x_data:
        print("No data loaded! Check your dataset directory.")
        return None
    
    # Convert to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # Print statistics
    print("Dataset statistics:")
    print(f"Total samples: {len(x_data)}")
    for symbol, count in total_per_class.items():
        print(f"{symbol}: {count} samples")
    
    # Convert labels to one-hot encoding
    y_data = to_categorical(y_data, num_classes=len(symbol_classes))
    
    # Shuffle the data
    indices = np.arange(len(x_data))
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]
    
    # Split into train/test
    split = int(0.8 * len(x_data))
    x_train, x_test = x_data[:split], x_data[split:]
    y_train, y_test = y_data[:split], y_data[split:]
    
    print(f"Training set: {len(x_train)} samples")
    print(f"Test set: {len(x_test)} samples")
    
    return x_train, y_train, x_test, y_test, symbol_classes

def train_model():
    """Train the model on real handwritten math symbols."""
    print("Loading dataset...")
    dataset = load_dataset()
    if dataset is None:
        return
    
    x_train, y_train, x_test, y_test, symbol_classes = dataset
    
    print("Creating model...")
    model = create_model(num_classes=len(symbol_classes))
    model.summary()
    
    # Create a label map
    label_map = {idx: symbol for idx, symbol in enumerate(symbol_classes)}
    
    # Create directories for saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f'models/handwritten_math_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the label map
    with open(os.path.join(model_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)
    
    # Set up data augmentation to improve robustness
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # Callbacks for training
    callbacks = [
        # Save the best model
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Stop early if validation loss doesn't improve
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    print("Training model...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        validation_data=(x_test, y_test),
        epochs=100,  # We're using early stopping, so this is just a maximum
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
    
    # Copy the label map to the default location
    with open(os.path.join(default_model_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)
    
    # Save training history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(model_dir, 'training_history.csv'), index=False)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    
    print(f"Models saved to {model_dir} and {default_model_dir}")
    print("Training complete!")
    
    # Generate confusion matrix
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(15, 15))
    cm = tf.math.confusion_matrix(y_true_classes, y_pred_classes).numpy()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Add labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=symbol_classes,
           yticklabels=symbol_classes,
           title='Confusion Matrix',
           ylabel='True Symbol',
           xlabel='Predicted Symbol')
    
    # Rotate x tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    
    return model, history

if __name__ == "__main__":
    # First, check if we have the CROHME dataset ready
    if not os.path.exists('data/math_symbols'):
        print("Dataset not found. Running CROHME dataset preparation first...")
        import download_crohme_dataset
        download_crohme_dataset.process_crohme_dataset()
    
    train_model() 