import numpy as np
import tensorflow as tf
import os
from typing import List, Dict, Tuple
import cv2
import string
import json

# Path to the model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                          'models', 'symbol_recognition_model')
LABEL_MAP_PATH = os.path.join(MODEL_PATH, 'label_map.json')

# Define the list of classes our model can recognize
# This will be loaded from the JSON file if available
DEFAULT_CLASSES = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '+', 11: '-', 12: '×', 13: '÷', 14: '=',
    15: 'x', 16: 'y', 17: '(', 18: ')', 19: '.'
}

def load_label_map() -> Dict[str, str]:
    """
    Load the label map from the JSON file.
    
    Returns:
        Dictionary mapping class indices to symbols
    """
    if os.path.exists(LABEL_MAP_PATH):
        try:
            with open(LABEL_MAP_PATH, 'r') as f:
                # JSON keys are strings, so convert them back to integers
                str_label_map = json.load(f)
                label_map = {int(k): v for k, v in str_label_map.items()}
                return label_map
        except Exception as e:
            print(f"Error loading label map: {str(e)}")
    
    print("Using default label map")
    return DEFAULT_CLASSES

def load_model():
    """
    Load the trained model for symbol recognition.
    
    Returns:
        Loaded TensorFlow model or None if loading fails
    """
    # Check if model directory exists
    if os.path.exists(MODEL_PATH):
        # Try to load the best model from h5 file first
        h5_path = os.path.join(MODEL_PATH, 'best_model.h5')
        if os.path.exists(h5_path):
            try:
                print(f"Loading model from {h5_path}")
                model = tf.keras.models.load_model(h5_path)
                return model
            except Exception as e:
                print(f"Error loading h5 model: {str(e)}")
        
        # Try to load from SavedModel format
        try:
            print(f"Loading model from {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    # If we reach here, no model was successfully loaded
    print("No model found, using fallback recognition method")
    return None

def preprocess_for_model(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for model prediction.
    
    Args:
        image: Input image (can be any size)
        
    Returns:
        Preprocessed image ready for model input
    """
    # Ensure image is grayscale
    if len(image.shape) == 3 and image.shape[2] > 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [0, 1]
    normalized = resized.astype('float32') / 255.0
    
    # Add batch and channel dimensions
    processed = normalized.reshape(1, 28, 28, 1)
    
    return processed

def recognize_single_symbol(symbol_image: np.ndarray, model=None) -> Tuple[str, float]:
    """
    Recognize a single symbol from an image.
    
    Args:
        symbol_image: Image of a single symbol
        model: Pre-trained model (optional)
        
    Returns:
        Tuple of (recognized symbol, confidence)
    """
    # Load label map
    label_map = load_label_map()
    
    # If model is available, use it for prediction
    if model is not None:
        try:
            # Preprocess image for the model
            input_image = preprocess_for_model(symbol_image)
            
            # Get model predictions
            predictions = model.predict(input_image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Convert class index to symbol using label map
            if predicted_class in label_map:
                return label_map[predicted_class], confidence
            else:
                print(f"Warning: Predicted class {predicted_class} not in label map")
        except Exception as e:
            print(f"Error using model for prediction: {str(e)}")
            # Fall back to heuristic method if model prediction fails
    
    # Fallback: Use simple heuristic approach for demo purposes
    # We'll check the percentage of white pixels in the image
    white_percentage = np.sum(symbol_image > 0) / (symbol_image.shape[0] * symbol_image.shape[1])
    
    # Simple heuristics for a few symbols as placeholder
    confidence = 0.5  # Lower confidence for heuristic method
    if white_percentage < 0.1:
        return '.', confidence
    elif white_percentage < 0.2:
        return '-', confidence
    elif white_percentage < 0.25:
        return '1', confidence
    elif 0.25 <= white_percentage < 0.35:
        if np.mean(symbol_image[5:15, 5:15]) > 200:
            return '0', confidence
        return '+', confidence
    elif 0.35 <= white_percentage < 0.4:
        return '=', confidence
    elif 0.4 <= white_percentage < 0.45:
        return 'x', confidence
    elif 0.45 <= white_percentage < 0.5:
        return '3', confidence
    elif 0.5 <= white_percentage < 0.55:
        return '8', confidence
    else:
        return '2', confidence

def recognize_symbols(symbol_images: List[np.ndarray]) -> str:
    """
    Recognize all symbols and construct the equation.
    
    Args:
        symbol_images: List of segmented symbol images
        
    Returns:
        The recognized equation as a string
    """
    # Load the model once for all symbols
    model = load_model()
    recognized_symbols = []
    
    # Process each symbol image
    for symbol_image in symbol_images:
        symbol, confidence = recognize_single_symbol(symbol_image, model)
        recognized_symbols.append(symbol)
    
    # Join the symbols to form the equation
    equation = ''.join(recognized_symbols)
    
    # Apply basic syntax fixes
    equation = fix_equation_syntax(equation)
    
    # Replace '×' with '*' for compatibility with SymPy
    equation = equation.replace('×', '*')
    # Replace '÷' with '/' for compatibility with SymPy
    equation = equation.replace('÷', '/')
    
    return equation

def fix_equation_syntax(equation: str) -> str:
    """
    Fix common syntax issues in the recognized equation.
    
    Args:
        equation: The raw recognized equation
        
    Returns:
        Equation with syntax fixes applied
    """
    # List of operators
    operators = ['+', '-', '*', '×', '÷', '/']
    
    # Fix consecutive operators (keep only the first one)
    for i in range(len(equation) - 1, 0, -1):
        if equation[i] in operators and equation[i-1] in operators:
            equation = equation[:i] + equation[i+1:]
    
    # Remove operators at the beginning except minus (which could be a negative sign)
    if equation and equation[0] in operators and equation[0] != '-':
        equation = equation[1:]
    
    # Remove operators at the end
    if equation and equation[-1] in operators:
        equation = equation[:-1]
    
    # Fix implicit multiplication (e.g., 2x -> 2*x, )x -> x, x( -> x*()
    fixed_eq = ""
    for i in range(len(equation)):
        fixed_eq += equation[i]
        
        if i < len(equation) - 1:
            # Digit followed by variable or parenthesis
            if equation[i].isdigit() and (equation[i+1].isalpha() or equation[i+1] == '('):
                fixed_eq += '*'
            # Close parenthesis followed by variable or digit or open parenthesis
            elif equation[i] == ')' and (equation[i+1].isalpha() or equation[i+1].isdigit() or equation[i+1] == '('):
                fixed_eq += '*'
    
    return fixed_eq

def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
    """
    Get a color based on confidence level.
    
    Args:
        confidence: Confidence value (0-1)
        
    Returns:
        BGR color tuple
    """
    if confidence > 0.8:
        return (0, 255, 0)  # Green for high confidence
    elif confidence > 0.6:
        return (0, 255, 255)  # Yellow for medium confidence
    else:
        return (0, 0, 255)  # Red for low confidence

def visualize_recognition(image: np.ndarray, symbols: List[Tuple[str, Tuple[int, int, int, int], float]]) -> np.ndarray:
    """
    Visualize the recognized symbols on the original image.
    
    Args:
        image: Original image
        symbols: List of (symbol, bounding_box, confidence) tuples
        
    Returns:
        Visualization image
    """
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Draw bounding boxes and labels for each symbol
    for symbol, bbox, confidence in symbols:
        x, y, w, h = bbox
        
        # Get color based on confidence
        color = get_confidence_color(confidence)
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        # Draw label with confidence
        label = f"{symbol} ({confidence:.2f})"
        cv2.putText(vis_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis_image

def build_training_data():
    """
    Function to build a dataset for training the model.
    This would be used to create a custom dataset combining MNIST digits
    and math symbols from CROHME.
    
    In a full implementation, this would:
    1. Download and preprocess MNIST and CROHME datasets
    2. Combine them into a unified dataset
    3. Split into training and validation sets
    4. Save for later use in training
    """
    pass

def train_model():
    """
    Train the model on the prepared dataset.
    In a full implementation, this would:
    1. Load the prepared dataset
    2. Define a CNN architecture
    3. Train the model
    4. Evaluate and save the best model
    """
    pass 