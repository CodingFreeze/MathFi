"""
Radical approach to improve advanced math symbol recognition:
1. Create a specialized model that only focuses on advanced symbols
2. Heavily oversample advanced symbols to address extreme class imbalance
3. Apply stronger augmentation for advanced symbols
4. Use transfer learning from our base model
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
from datetime import datetime
import glob
import cv2
from tqdm import tqdm
from sklearn.utils import class_weight
from PIL import Image, ImageEnhance, ImageFilter

from app.models.cnn_model import create_model

# Define advanced symbols to focus on
ADVANCED_SYMBOLS = [
    "sin", "cos", "tan", "∫", "∂", "∑", "lim", "dx", "dy", 
    "π", "θ", "α", "β", "γ", "λ", "√", "∞", "^", ">", "<", "≥", "≤"
]

def load_advanced_symbols_only(data_dir='data/math_symbols', augmentation_factor=20):
    """
    Load ONLY the advanced symbols and heavily oversample them.
    
    Args:
        data_dir: Directory containing the dataset
        augmentation_factor: How many times to oversample each advanced symbol
        
    Returns:
        x_data, y_data, class_map
    """
    print(f"Loading advanced symbols dataset from {data_dir}...")
    
    # Only include directories for advanced symbols
    symbol_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d)) and d in ADVANCED_SYMBOLS]
    
    if not symbol_dirs:
        raise ValueError(f"No advanced symbol directories found in {data_dir}")
        
    # Sort to ensure consistent label ordering
    symbol_dirs.sort()
    
    # Create a mapping from symbol to class index
    class_map = {symbol: i for i, symbol in enumerate(symbol_dirs)}
    
    # Create a Keras ImageDataGenerator for augmentation instead of imgaug
    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=15,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )
    
    images = []
    labels = []
    
    # Track samples per class
    class_counts = {}
    
    # First load all original images
    for symbol in tqdm(symbol_dirs, desc="Loading original symbols"):
        symbol_dir = os.path.join(data_dir, symbol)
        files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
        
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
            images.append(img.reshape(28, 28, 1))
            labels.append(class_map[symbol])
    
    # Now heavily augment each symbol using Keras ImageDataGenerator
    for symbol in tqdm(symbol_dirs, desc="Augmenting symbols"):
        symbol_dir = os.path.join(data_dir, symbol)
        files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
        
        for file_path in files:
            # Read image
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Ensure 28x28 size
            if img.shape[0] != 28 or img.shape[1] != 28:
                img = cv2.resize(img, (28, 28))
            
            # Add batch dimension and channel dimension for ImageDataGenerator
            img_batch = np.expand_dims(np.expand_dims(img, axis=0), axis=-1)
            
            # Generate augmented samples
            aug_iter = datagen.flow(img_batch, batch_size=1)
            for _ in range(augmentation_factor):
                aug_img = next(aug_iter)[0]  # Get the next augmented image
                
                # Reshape and normalize
                aug_img = aug_img.reshape(28, 28, 1).astype('float32') / 255.0
                
                # Add to dataset
                images.append(aug_img)
                labels.append(class_map[symbol])
    
    # Convert to numpy arrays
    x_data = np.array(images)
    y_data = np.array(labels)
    
    # Print statistics
    print(f"Dataset created with {len(x_data)} samples")
    print("Class distribution:")
    for symbol in symbol_dirs:
        orig_count = class_counts[symbol]
        total_count = orig_count * (augmentation_factor + 1)
        print(f"  {symbol}: {orig_count} original + {total_count - orig_count} augmented = {total_count}")
    
    return x_data, y_data, class_map

def train_specialized_model(pretrained_model_path=None, epochs=100, batch_size=16, learning_rate=0.0002):
    """
    Train a specialized model focused only on advanced math symbols.
    
    Args:
        pretrained_model_path: Path to pretrained model for transfer learning
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        
    Returns:
        Trained model
    """
    print("=" * 60)
    print("Training specialized model for advanced symbols")
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
    
    # Load specialized dataset
    x_data, y_data, class_map = load_advanced_symbols_only(augmentation_factor=20)
    
    # Shuffle dataset
    indices = np.random.permutation(len(x_data))
    x_data = x_data[indices]
    y_data = y_data[indices]
    
    # Split into train and validation sets (80/20)
    split_idx = int(len(x_data) * 0.8)
    x_train, x_val = x_data[:split_idx], x_data[split_idx:]
    y_train, y_val = y_data[:split_idx], y_data[split_idx:]
    
    # Convert labels to one-hot
    num_classes = len(class_map)
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    
    # Create model
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Using transfer learning from {pretrained_model_path}")
        base_model = load_model(pretrained_model_path)
        
        # Create new model using the convolutional base
        # Find the layer before the classifier
        for i, layer in enumerate(base_model.layers):
            if isinstance(layer, Flatten) or isinstance(layer, GlobalAveragePooling2D):
                feature_layer_idx = i
                break
        
        # Create new model
        feature_extractor = Model(inputs=base_model.input, 
                                outputs=base_model.layers[feature_layer_idx].output)
        
        # Freeze the feature extractor
        feature_extractor.trainable = False
        
        # Create new model
        model = tf.keras.Sequential([
            feature_extractor,
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
    else:
        print("Creating new model from scratch")
        model = create_model(num_classes=num_classes, model_type='improved_cnn')
    
    # Compile with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Create class weights to handle any remaining imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_data),
        y=y_data
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Create model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f'models/specialized_advanced_symbols_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save class map
    inverse_class_map = {i: symbol for symbol, i in class_map.items()}
    with open(os.path.join(model_dir, 'class_map.json'), 'w') as f:
        json.dump(inverse_class_map, f)
    
    # Create callbacks
    callbacks = [
        # Save the best model
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    print("Training specialized model...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Save the final model
    model.save(os.path.join(model_dir, 'final_model.h5'))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    
    # Evaluate model
    val_loss, val_acc = model.evaluate(x_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    print(f"Model saved to {model_dir}")
    return model, history, model_dir

def generate_advanced_symbol_samples(output_dir='data/advanced_samples', samples_per_class=100):
    """
    Generate additional synthetic samples of advanced symbols for testing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a Keras ImageDataGenerator for augmentation instead of imgaug
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=20,
        zoom_range=0.2,
        brightness_range=[0.5, 1.5],
        fill_mode='nearest'
    )
    
    for symbol in ADVANCED_SYMBOLS:
        symbol_dir = os.path.join(output_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Check if we already have enough samples
        existing = glob.glob(os.path.join(symbol_dir, '*.png'))
        if len(existing) >= samples_per_class:
            print(f"Already have {len(existing)} samples for {symbol}")
            continue
            
        # Create the new samples
        print(f"Generating {samples_per_class} samples for {symbol}")
        
        # For now, we'll copy existing samples and augment them
        source_dir = os.path.join('data/math_symbols', symbol)
        if not os.path.exists(source_dir):
            print(f"Warning: No source directory found for {symbol}")
            continue
            
        source_files = glob.glob(os.path.join(source_dir, f"{symbol}_*.png"))
        if not source_files:
            print(f"Warning: No source files found for {symbol}")
            continue
        
        for i in range(samples_per_class):
            # Select a random source file
            source_file = np.random.choice(source_files)
            
            # Read image
            img = cv2.imread(source_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Resize to 28x28 if needed
            if img.shape[0] != 28 or img.shape[1] != 28:
                img = cv2.resize(img, (28, 28))
            
            # Add batch dimension and channel dimension for ImageDataGenerator
            img_batch = np.expand_dims(np.expand_dims(img, axis=0), axis=-1)
            
            # Generate augmented sample
            aug_img = next(datagen.flow(img_batch, batch_size=1))[0]
            
            # Save the image
            output_path = os.path.join(symbol_dir, f"{symbol}_synth_{i:03d}.png")
            cv2.imwrite(output_path, aug_img.reshape(28, 28))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a specialized model for advanced math symbols')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model for transfer learning')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0002,
                        help='Initial learning rate')
    parser.add_argument('--generate-samples', action='store_true',
                        help='Generate additional synthetic samples')
    
    args = parser.parse_args()
    
    if args.generate_samples:
        generate_advanced_symbol_samples()
    
    train_specialized_model(
        pretrained_model_path=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    ) 