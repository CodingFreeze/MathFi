import numpy as np
import os
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import urllib.request
import zipfile
import shutil
import tarfile
import gzip
from typing import Tuple, Dict, List
import pandas as pd
import requests
import tqdm
import glob
import random
from PIL import Image, ImageDraw, ImageFont

def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the MNIST dataset for digits.
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape to add channel dimension (required for CNN)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')
    
    # Normalize pixel values to [0, 1]
    x_train /= 255.0
    x_test /= 255.0
    
    # We'll only keep digits 0-9 from MNIST
    # Filter out only the required digits
    digit_indices_train = np.where(y_train < 10)[0]
    digit_indices_test = np.where(y_test < 10)[0]
    
    x_train_digits = x_train[digit_indices_train]
    y_train_digits = y_train[digit_indices_train]
    x_test_digits = x_test[digit_indices_test]
    y_test_digits = y_test[digit_indices_test]
    
    return x_train_digits, y_train_digits, x_test_digits, y_test_digits

def download_file(url: str, target_path: str):
    """
    Download a file from a URL to a target path.
    
    Args:
        url: URL to download from
        target_path: Path to save the file
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(target_path):
        print(f"File already exists: {target_path}")
        return
    
    print(f"Downloading {url} to {target_path}")
    
    # Using requests with a progress bar for better UX
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(target_path, 'wb') as f, tqdm.tqdm(
            desc=os.path.basename(target_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def download_kaggle_dataset(kaggle_dataset: str, data_dir: str):
    """
    Download a dataset from Kaggle.
    Note: Requires Kaggle API credentials to be set up.
    
    Args:
        kaggle_dataset: Name of the Kaggle dataset (e.g., 'xainano/handwritten-mathematical-expressions')
        data_dir: Directory to save the dataset
    """
    try:
        import kaggle
        
        os.makedirs(data_dir, exist_ok=True)
        kaggle.api.dataset_download_files(
            kaggle_dataset, 
            path=data_dir, 
            unzip=True
        )
        print(f"Downloaded and extracted Kaggle dataset to {data_dir}")
    except ImportError:
        print("Kaggle API not found. Please install it with 'pip install kaggle'.")
    except Exception as e:
        print(f"Error downloading Kaggle dataset: {str(e)}")
        print("Make sure your Kaggle API credentials are set up in ~/.kaggle/kaggle.json")

def download_crohme_dataset(data_dir: str = 'data/crohme') -> str:
    """
    Download the CROHME dataset for math symbols (simplified version).
    
    Args:
        data_dir: Directory to save the dataset
        
    Returns:
        Path to the extracted dataset
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # URL for the CROHME 2019 dataset
    # This is a placeholder - the actual dataset needs proper credentials
    url = "https://tc11.cvc.uab.es/datasets/ICDAR2019-CROHME-TDF_1"
    
    print("Note: CROHME dataset requires registration.")
    print(f"Please visit {url} to download the dataset manually and place it in {data_dir}")
    
    # Alternative: Use Kaggle's handwritten math symbols dataset
    kaggle_dataset = "xainano/handwritten-mathematical-expressions"
    print(f"Alternatively, you can use the Kaggle dataset: {kaggle_dataset}")
    
    # Check if we have data already
    if len(os.listdir(data_dir)) > 0:
        print(f"Found existing data in {data_dir}")
        return data_dir
    
    # If no real data, create synthetic data as fallback
    print("No dataset found. Creating synthetic data as a fallback.")
    create_synthetic_math_symbols(os.path.join(data_dir, 'synthetic'))
    
    return data_dir

def load_handwritten_math_symbols(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load handwritten math symbols from a dataset directory.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        Tuple of (images, labels)
    """
    # Collect all image files
    image_files = []
    labels = []
    label_map = {}
    
    # Check if we're using Kaggle dataset format
    if os.path.exists(os.path.join(data_dir, 'train')):
        # Process Kaggle dataset structure
        for symbol_dir in glob.glob(os.path.join(data_dir, 'train', '*')):
            symbol = os.path.basename(symbol_dir)
            
            # Map the symbol to a label index if not already done
            if symbol not in label_map:
                label_map[symbol] = len(label_map) + 10  # Start after digits (0-9)
            
            # Collect images for this symbol
            for img_file in glob.glob(os.path.join(symbol_dir, '*.png')):
                image_files.append(img_file)
                labels.append(label_map[symbol])
    else:
        # Generic dataset structure: assume each subdirectory is a symbol class
        for symbol_dir in glob.glob(os.path.join(data_dir, '*')):
            if os.path.isdir(symbol_dir):
                symbol = os.path.basename(symbol_dir)
                
                # Map the symbol to a label index if not already done
                if symbol not in label_map:
                    label_map[symbol] = len(label_map) + 10  # Start after digits (0-9)
                
                # Collect images for this symbol
                for img_file in glob.glob(os.path.join(symbol_dir, '*.png')):
                    image_files.append(img_file)
                    labels.append(label_map[symbol])
    
    # If no real data found, return empty arrays
    if not image_files:
        return np.array([]), np.array([])
    
    # Load and preprocess images
    images = []
    for img_file in image_files:
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Resize to 28x28
        img = cv2.resize(img, (28, 28))
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        # Add channel dimension
        img = img.reshape(28, 28, 1)
        
        images.append(img)
    
    # Save the label map for reference
    with open(os.path.join(data_dir, 'label_map.json'), 'w') as f:
        import json
        json.dump({str(v): k for k, v in label_map.items()}, f)
    
    return np.array(images), np.array(labels)

def create_synthetic_math_symbols(data_dir: str, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic math symbols for training.
    
    Args:
        data_dir: Directory to save the synthetic symbols
        num_samples: Number of samples to generate per symbol
        
    Returns:
        Tuple of (x_data, y_data)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Define the math symbols we want to generate
    symbols = ['+', '-', '×', '÷', '=', 'x', 'y', '(', ')', '.']
    
    # Initialize arrays to store the data
    x_data = np.zeros((num_samples * len(symbols), 28, 28, 1), dtype=np.float32)
    y_data = np.zeros(num_samples * len(symbols), dtype=np.int32)
    
    # Try to use PIL with fonts for better-looking symbols
    try:
        # Check for some common fonts that should have math symbols
        font_paths = [
            '/System/Library/Fonts/Supplemental/Symbol.ttf',  # macOS
            '/usr/share/fonts/truetype/freefont/FreeMono.ttf',  # Linux
            'C:\\Windows\\Fonts\\Arial.ttf',  # Windows
            '/Users/abdullahrana/Documents/Coding Side Projects/MathFi/data/fonts/DejaVuSans.ttf',  # Custom path
        ]
        
        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break
        
        if font_path:
            font_sizes = [18, 20, 22, 24]
            positions = [(10, 6), (9, 7), (8, 8), (7, 9), (6, 10)]
            
            for i, symbol in enumerate(symbols):
                for j in range(num_samples):
                    # Create a blank image
                    img = Image.new('L', (28, 28), color=0)
                    draw = ImageDraw.Draw(img)
                    
                    # Randomly select font size and position
                    font_size = random.choice(font_sizes)
                    position = random.choice(positions)
                    
                    # Try to use the font
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        draw.text(position, symbol, fill=255, font=font)
                    except Exception:
                        # If font doesn't work, use default
                        draw.text(position, symbol, fill=255)
                    
                    # Convert to numpy array
                    img_array = np.array(img).reshape(28, 28, 1) / 255.0
                    
                    # Add some noise and random rotation
                    noise = np.random.normal(0, 0.01, img_array.shape)
                    img_array += noise
                    img_array = np.clip(img_array, 0, 1)
                    
                    # Store the image and label
                    index = i * num_samples + j
                    x_data[index] = img_array
                    y_data[index] = i + 10  # Labels for math symbols start from 10
                    
                    # Save a few examples for inspection
                    if j < 5:
                        os.makedirs(os.path.join(data_dir, symbol), exist_ok=True)
                        img.save(os.path.join(data_dir, symbol, f"{j}.png"))
            
            return x_data, y_data
    except Exception as e:
        print(f"Error using PIL for synthetic data: {str(e)}")
    
    # Fallback to OpenCV if PIL approach fails
    for i, symbol in enumerate(symbols):
        # Create a directory for each symbol
        symbol_dir = os.path.join(data_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        for j in range(num_samples):
            # Create a blank image
            img = np.zeros((28, 28, 1), dtype=np.float32)
            
            # Draw the symbol
            if symbol == '+':
                # Draw a plus sign
                img[10:18, 5:23, 0] = 1.0  # Horizontal line
                img[5:23, 10:18, 0] = 1.0  # Vertical line
            elif symbol == '-':
                # Draw a minus sign
                img[12:16, 5:23, 0] = 1.0  # Horizontal line
            elif symbol == '×':
                # Draw a multiplication sign (×)
                for k in range(20):
                    img[4+k, 4+k, 0] = 1.0  # Diagonal line from top-left to bottom-right
                    img[4+k, 24-k, 0] = 1.0  # Diagonal line from top-right to bottom-left
            elif symbol == '÷':
                # Draw a division sign (÷)
                img[12:16, 5:23, 0] = 1.0  # Horizontal line
                cv2.circle(img, (14, 7), 3, 1.0, -1)  # Top dot
                cv2.circle(img, (14, 21), 3, 1.0, -1)  # Bottom dot
            elif symbol == '=':
                # Draw an equals sign
                img[8:12, 5:23, 0] = 1.0  # Top horizontal line
                img[16:20, 5:23, 0] = 1.0  # Bottom horizontal line
            elif symbol == 'x':
                # Draw an x variable
                for k in range(16):
                    img[6+k, 6+k, 0] = 1.0  # Diagonal line from top-left to bottom-right
                    img[6+k, 22-k, 0] = 1.0  # Diagonal line from top-right to bottom-left
            elif symbol == 'y':
                # Draw a y variable
                for k in range(8):
                    img[6+k, 10+k, 0] = 1.0  # Top-left to middle
                    img[6+k, 18-k, 0] = 1.0  # Top-right to middle
                for k in range(8):
                    img[14+k, 14, 0] = 1.0  # Middle to bottom
            elif symbol == '(':
                # Draw a left parenthesis
                for k in range(16):
                    angle = (k / 15.0) * np.pi
                    x = int(18 - 8 * np.cos(angle))
                    y = int(14 + 10 * np.sin(angle))
                    if 0 <= x < 28 and 0 <= y < 28:
                        img[y, x, 0] = 1.0
            elif symbol == ')':
                # Draw a right parenthesis
                for k in range(16):
                    angle = (k / 15.0) * np.pi
                    x = int(10 + 8 * np.cos(angle))
                    y = int(14 + 10 * np.sin(angle))
                    if 0 <= x < 28 and 0 <= y < 28:
                        img[y, x, 0] = 1.0
            elif symbol == '.':
                # Draw a dot
                cv2.circle(img, (14, 21), 2, 1.0, -1)
            
            # Add some noise for variation
            noise = np.random.normal(0, 0.01, img.shape)
            img += noise
            img = np.clip(img, 0, 1)
            
            # Add slight random rotation for variation
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((14, 14), angle, 1)
            img = cv2.warpAffine(img, M, (28, 28))[:,:,np.newaxis]
            
            # Store the image and label
            index = i * num_samples + j
            x_data[index] = img
            y_data[index] = i + 10  # Labels for math symbols start from 10
            
            # Save a few examples for inspection
            if j < 5:
                cv2.imwrite(os.path.join(symbol_dir, f"{j}.png"), img.reshape(28, 28) * 255)
    
    return x_data, y_data

def combine_datasets(x_digits: np.ndarray, y_digits: np.ndarray,
                     x_symbols: np.ndarray, y_symbols: np.ndarray,
                     test_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine the digit and symbol datasets and split into train/test sets.
    
    Args:
        x_digits: Digit images
        y_digits: Digit labels
        x_symbols: Symbol images
        y_symbols: Symbol labels
        test_split: Proportion of data to use for testing
        
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    # Check if we have actual symbol data
    if len(x_symbols) == 0:
        print("No symbol data found. Creating synthetic data.")
        x_symbols, y_symbols = create_synthetic_math_symbols('data/synthetic_symbols')
    
    # Combine the datasets
    x_combined = np.vstack([x_digits, x_symbols])
    y_combined = np.concatenate([y_digits, y_symbols])
    
    # Shuffle the data
    indices = np.arange(len(x_combined))
    np.random.shuffle(indices)
    x_combined = x_combined[indices]
    y_combined = y_combined[indices]
    
    # Split into train and test sets
    split_idx = int(len(x_combined) * (1 - test_split))
    x_train, x_test = x_combined[:split_idx], x_combined[split_idx:]
    y_train, y_test = y_combined[:split_idx], y_combined[split_idx:]
    
    # Convert to one-hot encoding
    y_train = to_categorical(y_train, num_classes=20)
    y_test = to_categorical(y_test, num_classes=20)
    
    return x_train, y_train, x_test, y_test

def prepare_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare the complete dataset for training the model.
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    # Load MNIST for digits
    x_digits, y_digits, x_test_digits, y_test_digits = load_mnist()
    
    # Try to load real handwritten math symbols
    data_dir = 'data/math_symbols'
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if we have a Kaggle dataset
    x_symbols_real, y_symbols_real = load_handwritten_math_symbols(data_dir)
    
    if len(x_symbols_real) > 0:
        print(f"Loaded {len(x_symbols_real)} real math symbols from {data_dir}")
        x_symbols, y_symbols = x_symbols_real, y_symbols_real
    else:
        # Create synthetic math symbols as fallback
        print("No real math symbol data found. Creating synthetic data.")
        x_symbols, y_symbols = create_synthetic_math_symbols('data/synthetic_symbols')
    
    # Split digits into train and test (MNIST already comes split)
    x_train_digits = x_digits
    y_train_digits = y_digits
    
    # Split symbols into train and test
    split_idx = int(len(x_symbols) * 0.8)
    x_train_symbols = x_symbols[:split_idx]
    y_train_symbols = y_symbols[:split_idx]
    x_test_symbols = x_symbols[split_idx:]
    y_test_symbols = y_symbols[split_idx:]
    
    # Combine train sets
    x_train = np.vstack([x_train_digits, x_train_symbols])
    y_train = np.concatenate([y_train_digits, y_train_symbols])
    
    # Combine test sets
    x_test = np.vstack([x_test_digits, x_test_symbols])
    y_test = np.concatenate([y_test_digits, y_test_symbols])
    
    # Shuffle the training data
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    # Shuffle the test data
    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    y_test = y_test[indices]
    
    # Convert to one-hot encoding
    y_train = to_categorical(y_train, num_classes=20)
    y_test = to_categorical(y_test, num_classes=20)
    
    print(f"Final dataset shapes: x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"                      x_test: {x_test.shape}, y_test: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test 