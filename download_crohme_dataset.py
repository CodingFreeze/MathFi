"""
Download and process the CROHME dataset for handwritten math symbols recognition.
This dataset contains real handwritten mathematical expressions and symbols.
"""

import os
import requests
import zipfile
import shutil
import numpy as np
import cv2
from tqdm import tqdm
import glob
import random
import tarfile
import matplotlib.pyplot as plt
from PIL import Image

def download_file(url, target_path):
    """Download a file from URL to target path with progress bar."""
    print(f"Downloading {url} to {target_path}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(target_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            pbar.update(len(data))
    
    return target_path

def download_and_extract_crohme():
    """Download and extract the CROHME dataset."""
    data_dir = 'data/crohme'
    os.makedirs(data_dir, exist_ok=True)
    
    # Alternative dataset source since original CROHME might require login
    # This is a preprocessed version available on GitHub
    url = "https://github.com/ThomasLech/CROHME_extractor/archive/refs/heads/master.zip"
    target_path = os.path.join(data_dir, "crohme_github.zip")
    
    # Download the file if it doesn't exist
    if not os.path.exists(target_path):
        download_file(url, target_path)
    
    # Extract the dataset if not already extracted
    extract_dir = os.path.join(data_dir, "extracted")
    if not os.path.exists(extract_dir):
        print(f"Extracting {target_path} to {extract_dir}")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(target_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    return extract_dir

def process_crohme_dataset():
    """Process the extracted CROHME dataset for symbol classification."""
    extract_dir = download_and_extract_crohme()
    crohme_dir = os.path.join(extract_dir, "CROHME_extractor-master")
    
    if not os.path.exists(crohme_dir):
        print(f"CROHME extractor directory not found: {crohme_dir}")
        return None
    
    # Create directories for our processed dataset
    output_dir = 'data/math_symbols'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the symbols we want to extract
    # Map CROHME labels to our simplified labels
    symbol_map = {
        '+': '+',
        '-': '-',
        '=': '=',
        'x': 'x',
        'y': 'y',
        '0': '0',
        '1': '1',
        '2': '2',
        '3': '3',
        '4': '4',
        '5': '5',
        '6': '6',
        '7': '7',
        '8': '8',
        '9': '9',
        '(': '(',
        ')': ')',
        'times': '×',  # multiplication
        'div': '÷',    # division
        '.': '.',      # decimal point
    }
    
    # Check for the extracted data structure
    data_path = os.path.join(crohme_dir, "extracted", "symbols")
    if not os.path.exists(data_path):
        # Try alternative paths
        data_path = os.path.join(crohme_dir, "data", "symbols")
        if not os.path.exists(data_path):
            print(f"Symbol data not found. Checked paths: {os.path.join(crohme_dir, 'extracted', 'symbols')} and {os.path.join(crohme_dir, 'data', 'symbols')}")
            return None
    
    print(f"Found symbol data in {data_path}")
    
    # Count total available symbols
    total_symbols = 0
    for symbol_dir in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, symbol_dir)):
            symbol_files = glob.glob(os.path.join(data_path, symbol_dir, "*.png"))
            total_symbols += len(symbol_files)
    
    print(f"Found {total_symbols} symbol images in total")
    
    # Create output directories for each symbol
    for symbol in symbol_map.values():
        os.makedirs(os.path.join(output_dir, symbol), exist_ok=True)
    
    # Track how many symbols we've processed
    processed_counts = {symbol: 0 for symbol in symbol_map.values()}
    max_per_symbol = 500  # Limit to prevent dataset imbalance
    
    # Process each symbol directory
    for symbol_dir in tqdm(os.listdir(data_path), desc="Processing symbols"):
        if not os.path.isdir(os.path.join(data_path, symbol_dir)):
            continue
            
        # Map the directory name to our symbol names
        mapped_symbol = None
        for src, dst in symbol_map.items():
            if symbol_dir.lower() == src.lower():
                mapped_symbol = dst
                break
        
        if mapped_symbol is None:
            continue  # Skip symbols we don't want
            
        # Skip if we already have enough of this symbol
        if processed_counts[mapped_symbol] >= max_per_symbol:
            continue
            
        # Process symbol images
        symbol_files = glob.glob(os.path.join(data_path, symbol_dir, "*.png"))
        random.shuffle(symbol_files)  # Shuffle to get variety
        
        for img_path in symbol_files:
            if processed_counts[mapped_symbol] >= max_per_symbol:
                break
                
            try:
                # Read image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Invert if necessary (ensure white symbols on black background)
                mean_val = np.mean(img)
                if mean_val > 127:
                    img = 255 - img
                
                # Center the symbol using center of mass
                M = cv2.moments(img)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Calculate translation to center
                    tx = img.shape[1] // 2 - cX
                    ty = img.shape[0] // 2 - cY
                    
                    # Create translation matrix
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                
                # Add padding to make it square
                h, w = img.shape
                size = max(h, w) + 20  # Add padding
                square_img = np.zeros((size, size), dtype=np.uint8)
                offset_h = (size - h) // 2
                offset_w = (size - w) // 2
                square_img[offset_h:offset_h+h, offset_w:offset_w+w] = img
                
                # Resize to standard size
                img = cv2.resize(square_img, (28, 28))
                
                # Enhance contrast
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                
                # Save processed symbol
                output_path = os.path.join(output_dir, mapped_symbol, f"{mapped_symbol}_{processed_counts[mapped_symbol]}.png")
                cv2.imwrite(output_path, img)
                processed_counts[mapped_symbol] += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    # Report statistics
    print("\nProcessed symbols:")
    for symbol, count in processed_counts.items():
        print(f"{symbol}: {count} images")
    
    total_processed = sum(processed_counts.values())
    print(f"Total processed symbols: {total_processed}")
    
    # Generate synthetic data for symbols with too few samples
    min_samples = 100
    for symbol, count in processed_counts.items():
        if count < min_samples:
            print(f"Generating additional synthetic samples for {symbol} (current: {count})")
            generate_synthetic_samples(symbol, min_samples - count, output_dir)
    
    return output_dir

def generate_synthetic_samples(symbol, count, output_dir):
    """Generate synthetic samples for symbols with too few examples."""
    symbol_dir = os.path.join(output_dir, symbol)
    
    # Get existing samples as templates
    existing_samples = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
    if not existing_samples:
        print(f"No existing samples for {symbol}, cannot generate synthetic data")
        return
    
    # Create synthetic variations
    for i in range(count):
        # Pick a random template
        template_path = random.choice(existing_samples)
        img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply random transformations
        # 1. Rotation
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((14, 14), angle, 1)
        img = cv2.warpAffine(img, M, (28, 28))
        
        # 2. Slight scaling
        scale = random.uniform(0.9, 1.1)
        h, w = img.shape
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h > 0 and new_w > 0:
            img = cv2.resize(img, (new_w, new_h))
            canvas = np.zeros((28, 28), dtype=np.uint8)
            offset_h = (28 - new_h) // 2
            offset_w = (28 - new_w) // 2
            if offset_h >= 0 and offset_w >= 0 and offset_h + new_h <= 28 and offset_w + new_w <= 28:
                canvas[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = img
                img = canvas
            else:
                img = cv2.resize(img, (28, 28))
        
        # 3. Add slight noise
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # 4. Random erosion/dilation for thickness variation
        if random.choice([True, False]):
            kernel_size = random.choice([1, 2])
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            if random.choice([True, False]):
                img = cv2.erode(img, kernel, iterations=1)
            else:
                img = cv2.dilate(img, kernel, iterations=1)
        
        # Save the synthetic sample
        next_idx = len(existing_samples) + i
        output_path = os.path.join(symbol_dir, f"{symbol}_{next_idx}.png")
        cv2.imwrite(output_path, img)

def create_preview_grid():
    """Create a preview grid of processed symbols."""
    output_dir = 'data/math_symbols'
    
    symbols = os.listdir(output_dir)
    symbols = [s for s in symbols if os.path.isdir(os.path.join(output_dir, s))]
    
    # Set up the grid
    grid_size = min(5, len(symbols))
    samples_per_symbol = 5
    
    fig, axes = plt.subplots(len(symbols), samples_per_symbol, figsize=(12, 2*len(symbols)))
    
    for i, symbol in enumerate(sorted(symbols)):
        symbol_dir = os.path.join(output_dir, symbol)
        files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
        
        for j in range(samples_per_symbol):
            if j < len(files):
                img = cv2.imread(files[j], cv2.IMREAD_GRAYSCALE)
                if len(symbols) > 1:
                    axes[i, j].imshow(img, cmap='gray')
                    axes[i, j].set_title(f"{symbol}" if j == 0 else "")
                    axes[i, j].axis('off')
                else:
                    axes[j].imshow(img, cmap='gray')
                    axes[j].set_title(f"{symbol}" if j == 0 else "")
                    axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig('data/symbol_preview.png')
    print("Created preview grid at data/symbol_preview.png")

if __name__ == "__main__":
    print("Processing CROHME dataset for handwritten math symbols...")
    output_dir = process_crohme_dataset()
    
    if output_dir:
        create_preview_grid()
        print(f"Dataset prepared in {output_dir}")
        print("Ready for training!")
    else:
        print("Failed to process dataset.") 