"""
Create a hybrid two-stage recognition system:
1. First model classifies between basic and advanced symbols
2. Based on classification, use either:
   - Original model for basic symbols
   - Specialized model for advanced symbols
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Concatenate
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import json
import glob
import cv2
from tqdm import tqdm
import pickle

from app.models.cnn_model import create_model
from oversample_advanced_symbols import ADVANCED_SYMBOLS

class HybridRecognizer:
    """A two-stage hybrid model for math symbol recognition."""
    
    def __init__(self, basic_model_path, advanced_model_path, classifier_model_path=None):
        """
        Initialize the hybrid recognizer.
        
        Args:
            basic_model_path: Path to the model trained on basic symbols
            advanced_model_path: Path to the specialized model trained on advanced symbols
            classifier_model_path: Path to the model that decides if a symbol is basic or advanced
        """
        self.basic_model = load_model(basic_model_path)
        self.advanced_model = load_model(advanced_model_path)
        
        # Load label maps
        self.basic_label_map = self._load_label_map(basic_model_path)
        self.advanced_label_map = self._load_label_map(advanced_model_path)
        
        # Load classifier model if provided
        if classifier_model_path and os.path.exists(classifier_model_path):
            self.classifier_model = load_model(classifier_model_path)
        else:
            self.classifier_model = None
            
        # Create inverse label maps for lookup
        self.basic_inverse_map = {label: idx for idx, label in self.basic_label_map.items()}
        self.advanced_inverse_map = {label: idx for idx, label in self.advanced_label_map.items()}
        
        # Create unified label map
        self.unified_label_map = {}
        self.unified_label_map.update(self.basic_label_map)
        
        # Add advanced symbols that might not be in the basic label map
        advanced_labels = set(self.advanced_label_map.values())
        for idx, label in self.advanced_label_map.items():
            if label in ADVANCED_SYMBOLS:
                self.unified_label_map[f"adv_{idx}"] = label
    
    def _load_label_map(self, model_path):
        """Load the label map for a model."""
        # Check if model_path is a directory or a file
        if os.path.isdir(model_path):
            label_map_path = os.path.join(model_path, 'label_map.json')
            if not os.path.exists(label_map_path):
                label_map_path = os.path.join(model_path, 'class_map.json')
        else:
            # Try to find a label map in the same directory
            model_dir = os.path.dirname(model_path)
            label_map_path = os.path.join(model_dir, 'label_map.json')
            if not os.path.exists(label_map_path):
                label_map_path = os.path.join(model_dir, 'class_map.json')
        
        # Load the label map
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"No label map found for model at {model_path}")
    
    def train_classifier(self, data_dir='data/math_symbols', output_dir='models/classifier'):
        """
        Train a classifier model to determine if a symbol is basic or advanced.
        
        Args:
            data_dir: Directory containing all symbol data
            output_dir: Directory to save the trained classifier
        
        Returns:
            Trained classifier model
        """
        print("Training classifier model to distinguish between basic and advanced symbols...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all symbol samples
        images = []
        labels = []  # 0 for basic, 1 for advanced
        
        # Get all subdirectories
        symbol_dirs = [d for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d)) and d != 'equations']
        
        for symbol in tqdm(symbol_dirs, desc="Loading symbols"):
            # Determine if this is an advanced symbol
            is_advanced = symbol in ADVANCED_SYMBOLS
            
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
                images.append(img.reshape(28, 28, 1))
                labels.append(1 if is_advanced else 0)
        
        # Convert to numpy arrays
        x_data = np.array(images)
        y_data = np.array(labels)
        
        # Print class distribution
        print(f"Dataset loaded: {len(x_data)} samples")
        print(f"Basic symbols: {np.sum(y_data == 0)} samples")
        print(f"Advanced symbols: {np.sum(y_data == 1)} samples")
        
        # Shuffle the data
        indices = np.random.permutation(len(x_data))
        x_data = x_data[indices]
        y_data = y_data[indices]
        
        # Split into train and test sets
        split_idx = int(len(x_data) * 0.8)
        x_train, x_test = x_data[:split_idx], x_data[split_idx:]
        y_train, y_test = y_data[:split_idx], y_data[split_idx:]
        
        # Convert to one-hot encoded labels for binary classification
        y_train_one_hot = to_categorical(y_train, num_classes=2)
        y_test_one_hot = to_categorical(y_test, num_classes=2)
        
        # Create a simple CNN classifier
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2, activation='softmax')  # Binary classification: basic vs advanced
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Set up callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(output_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        print("Training classifier model...")
        history = model.fit(
            x_train, y_train_one_hot,
            validation_data=(x_test, y_test_one_hot),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the model
        model.save(os.path.join(output_dir, 'classifier_model.h5'))
        
        # Evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test_one_hot)
        print(f"Classifier accuracy: {accuracy:.4f}")
        
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
        plt.savefig(os.path.join(output_dir, 'classifier_training_history.png'))
        
        # Save the class labels
        class_labels = {0: 'basic', 1: 'advanced'}
        with open(os.path.join(output_dir, 'class_labels.json'), 'w') as f:
            json.dump(class_labels, f)
        
        # Set the classifier model
        self.classifier_model = model
        
        return model
    
    def predict(self, image):
        """
        Predict the symbol from an image using the hybrid approach.
        
        Args:
            image: A preprocessed 28x28x1 image (normalized to [0,1])
            
        Returns:
            Predicted symbol and confidence
        """
        if len(image.shape) == 2:  # If image is 28x28, add channel dimension
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3 and image.shape[0] == 28:  # If image is 28x28x1
            image = image.reshape(1, 28, 28, 1)
        
        # Ensure we have a batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # First, classify if it's a basic or advanced symbol
        if self.classifier_model is not None:
            classifier_pred = self.classifier_model.predict(image)[0]
            is_advanced = np.argmax(classifier_pred) == 1
            classifier_confidence = classifier_pred[np.argmax(classifier_pred)]
            
            print(f"Classifier prediction: {'Advanced' if is_advanced else 'Basic'} "
                  f"with confidence {classifier_confidence:.4f}")
        else:
            # Without a classifier, try both models and use the one with higher confidence
            is_advanced = None
        
        # Predict with appropriate model based on classifier
        if is_advanced is None:
            # Without classifier, try both models
            basic_pred = self.basic_model.predict(image)[0]
            advanced_pred = self.advanced_model.predict(image)[0]
            
            basic_label_idx = np.argmax(basic_pred)
            basic_confidence = basic_pred[basic_label_idx]
            basic_label = self.basic_label_map.get(str(basic_label_idx), 
                                                 self.basic_label_map.get(basic_label_idx, "Unknown"))
            
            advanced_label_idx = np.argmax(advanced_pred)
            advanced_confidence = advanced_pred[advanced_label_idx]
            advanced_label = self.advanced_label_map.get(str(advanced_label_idx), 
                                                       self.advanced_label_map.get(advanced_label_idx, "Unknown"))
            
            # Use the prediction with higher confidence
            if basic_confidence > advanced_confidence:
                predicted_label = basic_label
                confidence = basic_confidence
                model_used = "basic"
            else:
                predicted_label = advanced_label
                confidence = advanced_confidence
                model_used = "advanced"
                
            print(f"Used {model_used} model. Predicted {predicted_label} with confidence {confidence:.4f}")
            
        elif is_advanced:
            # Use advanced model
            advanced_pred = self.advanced_model.predict(image)[0]
            
            advanced_label_idx = np.argmax(advanced_pred)
            advanced_confidence = advanced_pred[advanced_label_idx]
            advanced_label = self.advanced_label_map.get(str(advanced_label_idx), 
                                                       self.advanced_label_map.get(advanced_label_idx, "Unknown"))
            
            predicted_label = advanced_label
            confidence = advanced_confidence
            
            print(f"Used advanced model. Predicted {predicted_label} with confidence {confidence:.4f}")
            
        else:
            # Use basic model
            basic_pred = self.basic_model.predict(image)[0]
            
            basic_label_idx = np.argmax(basic_pred)
            basic_confidence = basic_pred[basic_label_idx]
            basic_label = self.basic_label_map.get(str(basic_label_idx), 
                                                 self.basic_label_map.get(basic_label_idx, "Unknown"))
            
            predicted_label = basic_label
            confidence = basic_confidence
            
            print(f"Used basic model. Predicted {predicted_label} with confidence {confidence:.4f}")
        
        return predicted_label, confidence
    
    def save(self, path='models/hybrid_recognizer'):
        """Save the hybrid recognizer configuration."""
        os.makedirs(path, exist_ok=True)
        
        # Save the configuration
        config = {
            'basic_model_path': self.basic_model.name if hasattr(self.basic_model, 'name') else 'basic_model',
            'advanced_model_path': self.advanced_model.name if hasattr(self.advanced_model, 'name') else 'advanced_model',
            'classifier_model_path': self.classifier_model.name if self.classifier_model and hasattr(self.classifier_model, 'name') else None,
            'basic_label_map': self.basic_label_map,
            'advanced_label_map': self.advanced_label_map,
            'unified_label_map': self.unified_label_map
        }
        
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        # Save the models if they don't already exist in this path
        if not os.path.exists(os.path.join(path, 'basic_model.h5')):
            self.basic_model.save(os.path.join(path, 'basic_model.h5'))
            
        if not os.path.exists(os.path.join(path, 'advanced_model.h5')):
            self.advanced_model.save(os.path.join(path, 'advanced_model.h5'))
            
        if self.classifier_model and not os.path.exists(os.path.join(path, 'classifier_model.h5')):
            self.classifier_model.save(os.path.join(path, 'classifier_model.h5'))
        
        print(f"Hybrid recognizer saved to {path}")
    
    @classmethod
    def load(cls, path='models/hybrid_recognizer'):
        """Load a hybrid recognizer from a saved configuration."""
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Check if models are in the config path or absolute paths
        basic_model_path = config['basic_model_path']
        if not os.path.exists(basic_model_path):
            basic_model_path = os.path.join(path, 'basic_model.h5')
            
        advanced_model_path = config['advanced_model_path']
        if not os.path.exists(advanced_model_path):
            advanced_model_path = os.path.join(path, 'advanced_model.h5')
            
        classifier_model_path = config.get('classifier_model_path')
        if classifier_model_path and not os.path.exists(classifier_model_path):
            classifier_model_path = os.path.join(path, 'classifier_model.h5')
        
        # Create instance
        recognizer = cls(basic_model_path, advanced_model_path, classifier_model_path)
        
        # Override label maps from config
        recognizer.basic_label_map = config['basic_label_map']
        recognizer.advanced_label_map = config['advanced_label_map']
        recognizer.unified_label_map = config['unified_label_map']
        
        return recognizer

def create_hybrid_recognizer(basic_model_path, advanced_model_path, train_classifier=True):
    """
    Create a hybrid recognizer that combines the basic and advanced models.
    
    Args:
        basic_model_path: Path to the model trained on basic symbols
        advanced_model_path: Path to the specialized model trained on advanced symbols
        train_classifier: Whether to train a classifier model
        
    Returns:
        HybridRecognizer instance
    """
    print(f"Creating hybrid recognizer with:")
    print(f"  Basic model: {basic_model_path}")
    print(f"  Advanced model: {advanced_model_path}")
    
    # Create the hybrid recognizer
    recognizer = HybridRecognizer(basic_model_path, advanced_model_path)
    
    # Train classifier if requested
    if train_classifier:
        recognizer.train_classifier()
    
    # Save the recognizer
    output_dir = 'models/hybrid_recognizer'
    os.makedirs(output_dir, exist_ok=True)
    recognizer.save(output_dir)
    
    print(f"Hybrid recognizer created and saved to {output_dir}")
    return recognizer

def evaluate_hybrid_model(recognizer, data_dir='data/math_symbols', num_samples=100):
    """
    Evaluate the hybrid model on a test set.
    
    Args:
        recognizer: HybridRecognizer instance
        data_dir: Directory containing the test data
        num_samples: Number of random samples to test
        
    Returns:
        Overall accuracy and per-class accuracy
    """
    print(f"Evaluating hybrid model on {num_samples} random samples...")
    
    # Load test data
    test_images = []
    test_labels = []
    symbol_to_files = {}
    
    # Get all subdirectories
    symbol_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d)) and d != 'equations']
    
    # First collect all files per symbol
    for symbol in symbol_dirs:
        symbol_dir = os.path.join(data_dir, symbol)
        files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
        symbol_to_files[symbol] = files
    
    # Determine how many samples to take from each class for balanced evaluation
    samples_per_class = max(1, num_samples // len(symbol_dirs))
    
    # Collect test samples
    for symbol, files in symbol_to_files.items():
        # Take a random subset
        if len(files) > samples_per_class:
            selected_files = np.random.choice(files, samples_per_class, replace=False)
        else:
            selected_files = files
        
        for file_path in selected_files:
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
            test_images.append(img.reshape(28, 28, 1))
            test_labels.append(symbol)
    
    # Print test set info
    print(f"Test set: {len(test_images)} images from {len(symbol_dirs)} classes")
    
    # Run predictions
    correct = 0
    class_correct = {}
    class_total = {}
    
    for i, (image, true_label) in enumerate(zip(test_images, test_labels)):
        predicted_label, confidence = recognizer.predict(image)
        
        # Check if prediction is correct
        is_correct = predicted_label == true_label
        
        if is_correct:
            correct += 1
        
        # Update per-class accuracy
        class_total[true_label] = class_total.get(true_label, 0) + 1
        class_correct[true_label] = class_correct.get(true_label, 0) + (1 if is_correct else 0)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_images)} images. Current accuracy: {correct/(i+1):.4f}")
    
    # Calculate overall accuracy
    overall_accuracy = correct / len(test_images)
    
    # Calculate per-class accuracy
    per_class_accuracy = {}
    for label in class_total:
        per_class_accuracy[label] = class_correct.get(label, 0) / class_total[label]
    
    # Print results
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print("Per-class accuracy:")
    
    # Sort classes by accuracy
    sorted_classes = sorted(per_class_accuracy.items(), key=lambda x: x[1])
    
    for label, acc in sorted_classes:
        print(f"  {label}: {acc:.4f} ({class_correct.get(label, 0)}/{class_total[label]})")
    
    # Calculate accuracy for basic vs advanced symbols
    basic_correct = sum(class_correct.get(label, 0) for label in class_total if label not in ADVANCED_SYMBOLS)
    basic_total = sum(class_total.get(label, 0) for label in class_total if label not in ADVANCED_SYMBOLS)
    
    advanced_correct = sum(class_correct.get(label, 0) for label in class_total if label in ADVANCED_SYMBOLS)
    advanced_total = sum(class_total.get(label, 0) for label in class_total if label in ADVANCED_SYMBOLS)
    
    if basic_total > 0:
        print(f"Basic symbols accuracy: {basic_correct/basic_total:.4f} ({basic_correct}/{basic_total})")
    
    if advanced_total > 0:
        print(f"Advanced symbols accuracy: {advanced_correct/advanced_total:.4f} ({advanced_correct}/{advanced_total})")
    
    return overall_accuracy, per_class_accuracy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create and evaluate a hybrid math symbol recognizer')
    parser.add_argument('--basic-model', type=str, default='models/symbol_recognition_model',
                        help='Path to the basic symbol recognition model')
    parser.add_argument('--advanced-model', type=str, default='models/specialized_advanced_symbols',
                        help='Path to the advanced symbol recognition model')
    parser.add_argument('--train-classifier', action='store_true',
                        help='Train a classifier to distinguish between basic and advanced symbols')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the hybrid model on a test set')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of random samples to use for evaluation')
    
    args = parser.parse_args()
    
    # Create the hybrid recognizer
    recognizer = create_hybrid_recognizer(
        args.basic_model,
        args.advanced_model,
        args.train_classifier
    )
    
    # Evaluate if requested
    if args.evaluate:
        evaluate_hybrid_model(recognizer, num_samples=args.num_samples) 