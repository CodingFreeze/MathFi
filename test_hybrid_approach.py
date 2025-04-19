"""
Test the hybrid approach with a few sample images.

This script:
1. Sets up the hybrid recognizer
2. Tests it on a few sample images of both basic and advanced symbols
3. Compares the results with the original model approach
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
from tqdm import tqdm

# Set quieter TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import our hybrid recognizer
try:
    from create_hybrid_model import HybridRecognizer, evaluate_hybrid_model, ADVANCED_SYMBOLS
except ImportError:
    try:
        from oversample_advanced_symbols import ADVANCED_SYMBOLS
        from create_hybrid_model import HybridRecognizer, evaluate_hybrid_model
    except ImportError:
        print("Error: Could not import required modules. Make sure create_hybrid_model.py and oversample_advanced_symbols.py are available.")
        sys.exit(1)

def get_sample_images(data_dir='data/math_symbols', num_samples=5, seed=42):
    """Get random sample images of both basic and advanced symbols."""
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Define advanced symbols - use the imported list
    # ADVANCED_SYMBOLS is now imported from create_hybrid_model.py
    
    # Get all subdirectories
    symbol_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d)) and d != 'equations']
    
    # Separate advanced and basic symbols
    basic_symbols = [s for s in symbol_dirs if s not in ADVANCED_SYMBOLS]
    adv_symbols = [s for s in symbol_dirs if s in ADVANCED_SYMBOLS]
    
    # Sample symbols
    if basic_symbols:
        basic_sample = random.sample(basic_symbols, min(num_samples, len(basic_symbols)))
    else:
        basic_sample = []
        
    if adv_symbols:
        adv_sample = random.sample(adv_symbols, min(num_samples, len(adv_symbols)))
    else:
        adv_sample = []
    
    # Get sample images
    sample_images = []
    sample_labels = []
    
    # Get basic symbol images
    for symbol in basic_sample:
        symbol_dir = os.path.join(data_dir, symbol)
        files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
        
        if files:
            file_path = random.choice(files)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Ensure 28x28 size
                if img.shape[0] != 28 or img.shape[1] != 28:
                    img = cv2.resize(img, (28, 28))
                
                # Normalize
                img = img.astype('float32') / 255.0
                
                sample_images.append(img)
                sample_labels.append({"symbol": symbol, "type": "basic", "file": file_path})
    
    # Get advanced symbol images
    for symbol in adv_sample:
        symbol_dir = os.path.join(data_dir, symbol)
        files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
        
        if files:
            file_path = random.choice(files)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Ensure 28x28 size
                if img.shape[0] != 28 or img.shape[1] != 28:
                    img = cv2.resize(img, (28, 28))
                
                # Normalize
                img = img.astype('float32') / 255.0
                
                sample_images.append(img)
                sample_labels.append({"symbol": symbol, "type": "advanced", "file": file_path})
    
    return sample_images, sample_labels

def predict_with_original_model(image, model_path='models/symbol_recognition_model'):
    """Use the original model to predict a symbol."""
    # Load the model
    model = load_model(model_path)
    
    # Reshape the image if needed
    if len(image.shape) == 2:
        image = image.reshape(1, 28, 28, 1)
    elif len(image.shape) == 3 and image.shape[-1] == 1:
        image = image.reshape(1, 28, 28, 1)
    
    # Make prediction
    pred = model.predict(image)[0]
    label_idx = np.argmax(pred)
    confidence = pred[label_idx]
    
    # Get label map
    label_map_path = os.path.join(model_path, 'label_map.json')
    
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        
        label = label_map.get(str(label_idx), str(label_idx))
    else:
        label = str(label_idx)
    
    return label, confidence

def test_hybrid_approach():
    """Test the hybrid approach against the original model."""
    print("Testing the hybrid approach for math symbol recognition...")
    
    # Check if we have trained models
    basic_model_path = 'models/symbol_recognition_model'
    if not os.path.exists(basic_model_path):
        print(f"Error: Basic model not found at {basic_model_path}")
        return
    
    # Look for specialized advanced model
    adv_model_paths = glob.glob('models/specialized_advanced_symbols*')
    if not adv_model_paths:
        print("Warning: No specialized advanced model found. Trying alternative path...")
        adv_model_paths = ['models/advanced_recognition_model']
    
    adv_model_path = adv_model_paths[0]
    if not os.path.exists(adv_model_path):
        print(f"Error: Advanced model not found at {adv_model_path}")
        return
    
    print(f"Using basic model: {basic_model_path}")
    print(f"Using advanced model: {adv_model_path}")
    
    # Get test images
    sample_images, sample_labels = get_sample_images(num_samples=5)
    
    if not sample_images:
        print("Error: No sample images found")
        return
    
    print(f"Testing with {len(sample_images)} sample images")
    
    # Create hybrid recognizer
    try:
        recognizer = HybridRecognizer(basic_model_path, adv_model_path)
    except Exception as e:
        print(f"Error creating hybrid recognizer: {e}")
        return
    
    # Test each image
    results = []
    
    for i, (image, label_info) in enumerate(zip(sample_images, sample_labels)):
        print(f"\nTesting image {i+1}: {label_info['symbol']} ({label_info['type']})")
        
        # Predict with hybrid model
        hybrid_label, hybrid_conf = recognizer.predict(image)
        
        # Predict with original model
        orig_label, orig_conf = predict_with_original_model(image, basic_model_path)
        
        results.append({
            "true_label": label_info['symbol'],
            "type": label_info['type'],
            "hybrid_prediction": hybrid_label,
            "hybrid_confidence": float(hybrid_conf),
            "original_prediction": orig_label,
            "original_confidence": float(orig_conf)
        })
        
        print(f"True label: {label_info['symbol']} ({label_info['type']})")
        print(f"Hybrid prediction: {hybrid_label} (confidence: {hybrid_conf:.4f})")
        print(f"Original prediction: {orig_label} (confidence: {orig_conf:.4f})")
        
        # Display the results
        hybrid_correct = hybrid_label == label_info['symbol']
        orig_correct = orig_label == label_info['symbol']
        
        print(f"Hybrid model: {'✅ CORRECT' if hybrid_correct else '❌ WRONG'}")
        print(f"Original model: {'✅ CORRECT' if orig_correct else '❌ WRONG'}")
    
    # Calculate accuracy
    hybrid_correct = sum(1 for r in results if r['hybrid_prediction'] == r['true_label'])
    orig_correct = sum(1 for r in results if r['original_prediction'] == r['true_label'])
    
    print("\nResults summary:")
    print(f"Hybrid model accuracy: {hybrid_correct/len(results):.2f} ({hybrid_correct}/{len(results)})")
    print(f"Original model accuracy: {orig_correct/len(results):.2f} ({orig_correct}/{len(results)})")
    
    # Separate by type
    basic_results = [r for r in results if r['type'] == 'basic']
    adv_results = [r for r in results if r['type'] == 'advanced']
    
    if basic_results:
        basic_hybrid_correct = sum(1 for r in basic_results if r['hybrid_prediction'] == r['true_label'])
        basic_orig_correct = sum(1 for r in basic_results if r['original_prediction'] == r['true_label'])
        
        print(f"Basic symbols - Hybrid: {basic_hybrid_correct/len(basic_results):.2f}, Original: {basic_orig_correct/len(basic_results):.2f}")
    
    if adv_results:
        adv_hybrid_correct = sum(1 for r in adv_results if r['hybrid_prediction'] == r['true_label'])
        adv_orig_correct = sum(1 for r in adv_results if r['original_prediction'] == r['true_label'])
        
        print(f"Advanced symbols - Hybrid: {adv_hybrid_correct/len(adv_results):.2f}, Original: {adv_orig_correct/len(adv_results):.2f}")
    
    # Train a classifier if needed
    train_classifier = input("\nDo you want to train a classifier for the hybrid model? (y/n): ").lower() == 'y'
    
    if train_classifier:
        print("Training classifier...")
        recognizer.train_classifier()
        
        # Test again with classifier
        print("\nTesting with classifier...")
        hybrid_correct_with_classifier = 0
        
        for i, (image, label_info) in enumerate(zip(sample_images, sample_labels)):
            hybrid_label, hybrid_conf = recognizer.predict(image)
            hybrid_correct_with_classifier += (hybrid_label == label_info['symbol'])
            
            print(f"Image {i+1}: {label_info['symbol']} - Prediction: {hybrid_label} ({'✅' if hybrid_label == label_info['symbol'] else '❌'})")
        
        print(f"Hybrid model with classifier accuracy: {hybrid_correct_with_classifier/len(sample_images):.2f}")
    
    # Run full evaluation if requested
    run_full_eval = input("\nDo you want to run a full evaluation? (y/n): ").lower() == 'y'
    
    if run_full_eval:
        num_samples = int(input("How many samples to evaluate? (default: 100): ") or "100")
        evaluate_hybrid_model(recognizer, num_samples=num_samples)

if __name__ == "__main__":
    test_hybrid_approach() 