"""
A quick test script to check our hybrid approach with existing models
"""

import os
import sys
import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import random

# Set quieter TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Advanced symbols list
ADVANCED_SYMBOLS = [
    "sin", "cos", "tan", "∫", "∂", "∑", "lim", "dx", "dy", 
    "π", "θ", "α", "β", "γ", "λ", "√", "∞", "^", ">", "<", "≥", "≤"
]

def find_models():
    """Find available models in the models directory."""
    models = {}
    
    # Find basic model
    if os.path.exists('models/symbol_recognition_model'):
        models['basic'] = 'models/symbol_recognition_model'
    
    # Find advanced models
    advanced_models = glob.glob('models/advanced_recognition_model*')
    if advanced_models:
        models['advanced'] = advanced_models[0]
    
    # Find specialized models
    specialized_models = glob.glob('models/specialized_advanced_symbols*')
    if specialized_models:
        models['specialized'] = specialized_models[0]
        
    return models

def get_sample_images(data_dir='data/math_symbols', num_basic=3, num_advanced=3):
    """Get sample images for testing."""
    samples = []
    
    # Get all directories
    symbol_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d)) and d != 'equations']
    
    # Split into basic and advanced
    basic_symbols = [s for s in symbol_dirs if s not in ADVANCED_SYMBOLS]
    adv_symbols = [s for s in symbol_dirs if s in ADVANCED_SYMBOLS]
    
    # Sample basic symbols
    if basic_symbols and num_basic > 0:
        basic_sample = random.sample(basic_symbols, min(num_basic, len(basic_symbols)))
        for symbol in basic_sample:
            symbol_dir = os.path.join(data_dir, symbol)
            files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
            if files:
                file_path = random.choice(files)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    if img.shape[0] != 28 or img.shape[1] != 28:
                        img = cv2.resize(img, (28, 28))
                    img = img.astype('float32') / 255.0
                    samples.append({
                        'image': img,
                        'true_label': symbol,
                        'type': 'basic',
                        'path': file_path
                    })
    
    # Sample advanced symbols
    if adv_symbols and num_advanced > 0:
        adv_sample = random.sample(adv_symbols, min(num_advanced, len(adv_symbols)))
        for symbol in adv_sample:
            symbol_dir = os.path.join(data_dir, symbol)
            files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
            if files:
                file_path = random.choice(files)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    if img.shape[0] != 28 or img.shape[1] != 28:
                        img = cv2.resize(img, (28, 28))
                    img = img.astype('float32') / 255.0
                    samples.append({
                        'image': img,
                        'true_label': symbol,
                        'type': 'advanced',
                        'path': file_path
                    })
    
    return samples

def predict_with_model(image, model_path):
    """Make a prediction with a model."""
    try:
        # Try loading the model with TFSMLayer approach for Keras 3.0
        try:
            if os.path.isdir(model_path):
                # Load as SavedModel format
                tfsm_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
                
                # Prepare image
                if len(image.shape) == 2:
                    image_batch = image.reshape(1, 28, 28, 1)
                else:
                    image_batch = image.reshape(1, 28, 28, 1)
                
                # Make prediction using the layer
                prediction = tfsm_layer(image_batch)
                
                # Get the output tensor (depends on the model's output structure)
                if isinstance(prediction, dict):
                    # Find the output tensor
                    output_key = list(prediction.keys())[0]
                    pred = prediction[output_key].numpy()[0]
                else:
                    pred = prediction.numpy()[0]
            else:
                # Try direct loading for .h5 files
                model = load_model(model_path)
                
                # Prepare image
                if len(image.shape) == 2:
                    image_batch = image.reshape(1, 28, 28, 1)
                else:
                    image_batch = image.reshape(1, 28, 28, 1)
                
                # Make prediction
                pred = model.predict(image_batch, verbose=0)[0]
        except Exception as e:
            # Fallback to loading specific model file for .h5 format
            if os.path.isdir(model_path):
                # Try to find .h5 or .keras files
                h5_files = glob.glob(os.path.join(model_path, '*.h5'))
                keras_files = glob.glob(os.path.join(model_path, '*.keras'))
                model_files = h5_files + keras_files
                
                if model_files:
                    model_file = model_files[0]
                    model = load_model(model_file)
                    
                    # Prepare image
                    if len(image.shape) == 2:
                        image_batch = image.reshape(1, 28, 28, 1)
                    else:
                        image_batch = image.reshape(1, 28, 28, 1)
                    
                    # Make prediction
                    pred = model.predict(image_batch, verbose=0)[0]
                else:
                    raise FileNotFoundError(f"No .h5 or .keras file found in {model_path}")
            else:
                raise e
        
        # Get the class index and confidence
        pred_idx = np.argmax(pred)
        confidence = pred[pred_idx]
        
        # Load label map
        label_map_path = os.path.join(model_path, 'label_map.json')
        class_map_path = os.path.join(model_path, 'class_map.json')
        
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                label_map = json.load(f)
            label = label_map.get(str(pred_idx), str(pred_idx))
        elif os.path.exists(class_map_path):
            with open(class_map_path, 'r') as f:
                label_map = json.load(f)
            label = label_map.get(str(pred_idx), str(pred_idx))
        else:
            label = str(pred_idx)
        
        return label, confidence
    except Exception as e:
        print(f"Error predicting with model {model_path}: {e}")
        return "Error", 0.0

def hybrid_predict(image, basic_model_path, advanced_model_path):
    """Make a prediction using our hybrid approach."""
    # For simplicity, we'll use a heuristic:
    # - Use advanced model for advanced symbols
    # - Use basic model for basic symbols
    # In a real implementation, we'd use a classifier or confidence-based approach
    
    # Predict with both models
    basic_label, basic_conf = predict_with_model(image, basic_model_path)
    advanced_label, advanced_conf = predict_with_model(image, advanced_model_path)
    
    # Use confidence to determine which prediction to use
    if advanced_conf > basic_conf:
        return advanced_label, advanced_conf, "advanced"
    else:
        return basic_label, basic_conf, "basic"

def main():
    """Main test function."""
    print("Quick test of hybrid math symbol recognition")
    print("-" * 50)
    
    # Find available models
    models = find_models()
    print(f"Found models: {list(models.keys())}")
    
    if 'basic' not in models:
        print("Error: No basic model found!")
        return
    
    if 'advanced' not in models and 'specialized' not in models:
        print("Error: No advanced or specialized model found!")
        return
    
    # Get advanced model (prefer specialized if available)
    advanced_model_path = models.get('specialized', models.get('advanced'))
    print(f"Using basic model: {models['basic']}")
    print(f"Using advanced model: {advanced_model_path}")
    
    # Get sample images
    samples = get_sample_images(num_basic=2, num_advanced=3)
    if not samples:
        print("Error: No sample images found!")
        return
    
    print(f"Testing with {len(samples)} sample images\n")
    
    # Test each sample
    hybrid_correct = 0
    basic_correct = 0
    
    for i, sample in enumerate(samples):
        image = sample['image']
        true_label = sample['true_label']
        type_label = sample['type']
        
        print(f"Sample {i+1}: {true_label} ({type_label})")
        
        # Predict with basic model only
        basic_label, basic_conf = predict_with_model(image, models['basic'])
        
        # Predict with hybrid approach
        hybrid_label, hybrid_conf, model_used = hybrid_predict(
            image, models['basic'], advanced_model_path
        )
        
        # Check if predictions are correct
        basic_is_correct = basic_label == true_label
        hybrid_is_correct = hybrid_label == true_label
        
        # Update counters
        if basic_is_correct:
            basic_correct += 1
        if hybrid_is_correct:
            hybrid_correct += 1
        
        # Print results
        print(f"  True label: {true_label}")
        print(f"  Basic model: {basic_label} (confidence: {basic_conf:.4f}) - {'✓' if basic_is_correct else '✗'}")
        print(f"  Hybrid model: {hybrid_label} (confidence: {hybrid_conf:.4f}, using {model_used} model) - {'✓' if hybrid_is_correct else '✗'}")
        print()
    
    # Print summary
    print("-" * 50)
    print("Results summary:")
    print(f"  Basic model accuracy: {basic_correct/len(samples):.2f} ({basic_correct}/{len(samples)})")
    print(f"  Hybrid model accuracy: {hybrid_correct/len(samples):.2f} ({hybrid_correct}/{len(samples)})")
    
    # Calculate accuracy by type
    basic_samples = [s for s in samples if s['type'] == 'basic']
    adv_samples = [s for s in samples if s['type'] == 'advanced']
    
    if basic_samples:
        basic_on_basic = sum(1 for s in basic_samples if predict_with_model(s['image'], models['basic'])[0] == s['true_label'])
        hybrid_on_basic = sum(1 for s in basic_samples if hybrid_predict(s['image'], models['basic'], advanced_model_path)[0] == s['true_label'])
        print(f"  Basic symbols - Basic model: {basic_on_basic/len(basic_samples):.2f}, Hybrid model: {hybrid_on_basic/len(basic_samples):.2f}")
    
    if adv_samples:
        basic_on_adv = sum(1 for s in adv_samples if predict_with_model(s['image'], models['basic'])[0] == s['true_label'])
        hybrid_on_adv = sum(1 for s in adv_samples if hybrid_predict(s['image'], models['basic'], advanced_model_path)[0] == s['true_label'])
        print(f"  Advanced symbols - Basic model: {basic_on_adv/len(adv_samples):.2f}, Hybrid model: {hybrid_on_adv/len(adv_samples):.2f}")

if __name__ == "__main__":
    main() 