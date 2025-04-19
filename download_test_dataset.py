import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import numpy as np
import cv2
import random

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

def prepare_test_dataset():
    # Create directory
    data_dir = 'data/math_symbols'
    os.makedirs(data_dir, exist_ok=True)
    
    # Basic symbols we want to create
    symbols = ['+', '-', '=', 'x']
    
    # Create directories for each symbol
    for symbol in symbols:
        os.makedirs(os.path.join(data_dir, symbol), exist_ok=True)
    
    # Generate sample test data (simple, high-contrast)
    for symbol in symbols:
        for i in range(20):
            img = np.zeros((200, 200), dtype=np.uint8)
            
            # Draw the symbol
            if symbol == '+':
                # Draw a plus sign
                cv2.line(img, (50, 100), (150, 100), 255, 10)
                cv2.line(img, (100, 50), (100, 150), 255, 10)
            elif symbol == '-':
                # Draw a minus sign
                cv2.line(img, (50, 100), (150, 100), 255, 10)
            elif symbol == '=':
                # Draw an equals sign
                cv2.line(img, (50, 80), (150, 80), 255, 8)
                cv2.line(img, (50, 120), (150, 120), 255, 8)
            elif symbol == 'x':
                # Draw an x
                cv2.line(img, (50, 50), (150, 150), 255, 8)
                cv2.line(img, (150, 50), (50, 150), 255, 8)
            
            # Add slight random rotation
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((100, 100), angle, 1)
            img = cv2.warpAffine(img, M, (200, 200))
            
            # Save the image
            filename = os.path.join(data_dir, symbol, f"{symbol}_{i}.png")
            cv2.imwrite(filename, img)
            print(f"Created {filename}")
    
    print(f"Created test dataset with {len(symbols)} symbols, 20 samples each")
    return data_dir

if __name__ == "__main__":
    prepare_test_dataset()
    print("Dataset preparation complete. Ready for training.") 