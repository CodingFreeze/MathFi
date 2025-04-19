"""
Evaluate model performance per class to identify which advanced math symbols
are most challenging for the model to recognize.

This will help target future improvements to specific problematic symbols.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import json
import glob
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import argparse

def load_test_data(data_dir='data/math_symbols', test_split=0.2):
    """
    Load test data with ground truth labels.
    
    Args:
        data_dir: Directory containing the dataset
        test_split: Proportion of data to use for testing
        
    Returns:
        (x_test, y_test_true), label_map
    """
    print(f"Loading test data from {data_dir}...")
    
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
    label_counts = {}
    
    for symbol in tqdm(symbol_dirs, desc="Loading symbols"):
        symbol_dir = os.path.join(data_dir, symbol)
        files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
        
        label_counts[symbol] = len(files)
        
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
    x_test = x_data[split_idx:]
    y_test = y_data[split_idx:]
    
    print(f"Test dataset loaded: {len(x_test)} samples")
    print(f"Number of classes: {len(symbol_dirs)}")
    
    for symbol, count in label_counts.items():
        test_count = int(count * test_split)
        print(f"  Class '{symbol}': {test_count} test samples")
    
    return (x_test, y_test), label_map

def evaluate_model_per_class(model_path, output_dir=None):
    """
    Evaluate the model's performance per class.
    
    Args:
        model_path: Path to the trained model
        output_dir: Directory to save evaluation results, defaults to model directory
    """
    print(f"Evaluating model: {model_path}")
    
    # Set output directory
    if output_dir is None:
        if os.path.isdir(model_path):
            output_dir = model_path
        else:
            output_dir = os.path.dirname(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Try to load label map
    label_map_path = os.path.join(output_dir, 'label_map.json')
    if not os.path.exists(label_map_path) and os.path.isdir(model_path):
        label_map_path = os.path.join(model_path, 'label_map.json')
    
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        print(f"Loaded label map with {len(label_map)} classes")
    else:
        print("No label map found. Evaluation will use numeric indices.")
        label_map = None
    
    # Load test data
    (x_test, y_test_true), dataset_label_map = load_test_data()
    
    # Use dataset label map if no model label map found
    if label_map is None:
        label_map = dataset_label_map
    
    # Make predictions
    print("Making predictions...")
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate overall accuracy
    accuracy = np.mean(y_pred == y_test_true)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Generate classification report
    class_names = [label_map[str(i)] if str(i) in label_map else label_map.get(i, f"Class {i}") for i in range(len(label_map))]
    
    # Generate and print classification report
    report = classification_report(y_test_true, y_pred, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    report_df = pd.DataFrame(report).transpose()
    print(report_df.round(4))
    
    # Save the report to CSV
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test_true, y_pred)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot per-class accuracy
    plt.figure(figsize=(14, 8))
    class_metrics = {}
    advanced_symbols = ["sin", "cos", "tan", "∫", "∂", "∑", "lim", "dx", "dy", 
                        "π", "θ", "α", "β", "γ", "λ", "√", "∞", "^", ">", "<", "≥", "≤"]
    
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            precision = report[str(i)]['precision']
            recall = report[str(i)]['recall']
            f1 = report[str(i)]['f1-score']
            class_metrics[class_name] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    # Sort by F1 score for better visualization
    sorted_classes = sorted(class_metrics.keys(), key=lambda x: class_metrics[x]['f1'])
    
    # Prepare data for the plot
    x_pos = np.arange(len(sorted_classes))
    f1_scores = [class_metrics[c]['f1'] for c in sorted_classes]
    precisions = [class_metrics[c]['precision'] for c in sorted_classes]
    recalls = [class_metrics[c]['recall'] for c in sorted_classes]
    
    # Create bar colors (highlight advanced symbols)
    colors = ['#1f77b4' if c not in advanced_symbols else '#ff7f0e' for c in sorted_classes]
    
    # Plot F1 scores
    plt.figure(figsize=(14, 8))
    bars = plt.bar(x_pos, f1_scores, align='center', alpha=0.7, color=colors)
    plt.xticks(x_pos, sorted_classes, rotation=90)
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Symbol Class')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a legend
    basic_patch = plt.Rectangle((0, 0), 1, 1, fc='#1f77b4', alpha=0.7)
    advanced_patch = plt.Rectangle((0, 0), 1, 1, fc='#ff7f0e', alpha=0.7)
    plt.legend([basic_patch, advanced_patch], ['Basic Symbols', 'Advanced Symbols'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_score_by_class.png'))
    
    # Plot precision and recall
    plt.figure(figsize=(14, 8))
    width = 0.35
    plt.bar(x_pos - width/2, precisions, width, label='Precision', alpha=0.7, color='#2ca02c')
    plt.bar(x_pos + width/2, recalls, width, label='Recall', alpha=0.7, color='#d62728')
    plt.xticks(x_pos, sorted_classes, rotation=90)
    plt.ylabel('Score')
    plt.title('Precision and Recall by Symbol Class')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_by_class.png'))
    
    # Generate report of problematic symbols (lowest F1 scores)
    problematic_threshold = 0.6  # symbols with F1 score below this are considered problematic
    problematic_symbols = [c for c in sorted_classes if class_metrics[c]['f1'] < problematic_threshold]
    
    print("\nProblematic Symbols (F1 < 0.6):")
    for symbol in problematic_symbols:
        metrics = class_metrics[symbol]
        print(f"  {symbol}: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    # Save list of problematic symbols to file
    with open(os.path.join(output_dir, 'problematic_symbols.txt'), 'w') as f:
        f.write("Problematic Symbols (F1 < 0.6):\n")
        for symbol in problematic_symbols:
            metrics = class_metrics[symbol]
            f.write(f"{symbol}: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}\n")
    
    print(f"\nEvaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model performance per class')
    parser.add_argument('--model-path', type=str, default='models/advanced_recognition_model',
                        help='Path to the trained model')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_model_per_class(args.model_path, args.output_dir) 