"""
Download and prepare dataset of handwritten math symbols.
This script will create a dataset by:
1. Generating synthetic symbols with various transformations
2. Downloading available math symbol datasets
"""

import os
import numpy as np
import cv2
import random
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import zipfile
import requests
import shutil

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

def generate_synthetic_symbols(output_dir, samples_per_symbol=500):
    """Generate synthetic math symbols using drawing primitives and fonts."""
    print("Generating synthetic math symbols dataset...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the symbols we want to generate
    symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
               '+', '-', '×', '÷', '=', 'x', 'y', '(', ')', '.']
    
    # Create directories for each symbol
    for symbol in symbols:
        os.makedirs(os.path.join(output_dir, symbol), exist_ok=True)
    
    # Check for available fonts
    font_paths = [
        '/System/Library/Fonts/Supplemental/Symbol.ttf',  # macOS
        '/System/Library/Fonts/SFNS.ttf',  # macOS
        '/usr/share/fonts/truetype/freefont/FreeMono.ttf',  # Linux
        'C:\\Windows\\Fonts\\Arial.ttf',  # Windows
        '/Users/abdullahrana/Documents/Coding Side Projects/MathFi/data/fonts/DejaVuSans.ttf',  # Custom path
    ]
    
    available_fonts = []
    for path in font_paths:
        if os.path.exists(path):
            available_fonts.append(path)
    
    if not available_fonts:
        # Create a fallback font directory
        font_dir = os.path.join('data', 'fonts')
        os.makedirs(font_dir, exist_ok=True)
        
        # Try downloading a font
        try:
            font_url = "https://github.com/google/fonts/raw/main/ofl/roboto/Roboto-Regular.ttf"
            font_path = os.path.join(font_dir, "Roboto-Regular.ttf")
            if not os.path.exists(font_path):
                download_file(font_url, font_path)
            available_fonts.append(font_path)
        except Exception as e:
            print(f"Error downloading font: {str(e)}")
    
    # Function to generate symbol using PIL
    def generate_with_pil(symbol, idx, font_path=None):
        # Random variations for size and position
        font_size = random.randint(18, 24)
        image_size = 28
        
        # Create a blank image
        img = Image.new('L', (image_size, image_size), color=0)
        draw = ImageDraw.Draw(img)
        
        # Position the symbol in the center with slight randomness
        x_offset = random.randint(-2, 2)
        y_offset = random.randint(-2, 2)
        
        # Default position in the center
        position = (image_size // 2 - font_size // 3 + x_offset, 
                   image_size // 2 - font_size // 3 + y_offset)
        
        try:
            if font_path:
                # Use the provided font
                font = ImageFont.truetype(font_path, font_size)
                draw.text(position, symbol, fill=255, font=font)
            else:
                # Fallback to default
                draw.text(position, symbol, fill=255)
        except Exception:
            # If font fails, handle special symbols manually
            if symbol == '+':
                draw.line([(10, 14), (18, 14)], fill=255, width=2)  # Horizontal
                draw.line([(14, 10), (14, 18)], fill=255, width=2)  # Vertical
            elif symbol == '-':
                draw.line([(10, 14), (18, 14)], fill=255, width=2)  # Horizontal
            elif symbol == '×':
                draw.line([(10, 10), (18, 18)], fill=255, width=2)  # Diagonal 1
                draw.line([(18, 10), (10, 18)], fill=255, width=2)  # Diagonal 2
            elif symbol == '÷':
                draw.line([(10, 14), (18, 14)], fill=255, width=2)  # Line
                draw.ellipse([(13, 8), (15, 10)], fill=255)  # Upper dot
                draw.ellipse([(13, 18), (15, 20)], fill=255)  # Lower dot
            elif symbol == '=':
                draw.line([(10, 12), (18, 12)], fill=255, width=2)  # Upper line
                draw.line([(10, 16), (18, 16)], fill=255, width=2)  # Lower line
            else:
                # For other symbols, try default text
                draw.text(position, symbol, fill=255)
        
        # Apply random transformations for variety
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
        
        # Add random noise
        img_array = np.array(img)
        noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Random thickness variation
        kernel_size = random.choice([1, 2])
        if random.choice([True, False, False]):  # 1/3 chance
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            if random.choice([True, False]):
                img_array = cv2.erode(img_array, kernel, iterations=1)
            else:
                img_array = cv2.dilate(img_array, kernel, iterations=1)
        
        # Save the image
        output_path = os.path.join(output_dir, symbol, f"{symbol}_{idx}.png")
        cv2.imwrite(output_path, img_array)
        
        return output_path
    
    # Function to generate symbol using OpenCV
    def generate_with_cv2(symbol, idx):
        # Create a blank image
        img = np.zeros((28, 28), dtype=np.uint8)
        
        # Draw the symbol based on type
        if symbol == '+':
            # Draw a plus sign
            cv2.line(img, (7, 14), (21, 14), 255, 2)  # Horizontal line
            cv2.line(img, (14, 7), (14, 21), 255, 2)  # Vertical line
        elif symbol == '-':
            # Draw a minus sign
            cv2.line(img, (7, 14), (21, 14), 255, 2)  # Horizontal line
        elif symbol == '×':
            # Draw a multiplication sign (×)
            cv2.line(img, (8, 8), (20, 20), 255, 2)  # Diagonal line from top-left to bottom-right
            cv2.line(img, (8, 20), (20, 8), 255, 2)  # Diagonal line from top-right to bottom-left
        elif symbol == '÷':
            # Draw a division sign (÷)
            cv2.line(img, (7, 14), (21, 14), 255, 2)  # Horizontal line
            cv2.circle(img, (14, 7), 2, 255, -1)  # Top dot
            cv2.circle(img, (14, 21), 2, 255, -1)  # Bottom dot
        elif symbol == '=':
            # Draw an equals sign
            cv2.line(img, (7, 10), (21, 10), 255, 2)  # Top horizontal line
            cv2.line(img, (7, 18), (21, 18), 255, 2)  # Bottom horizontal line
        elif symbol == 'x':
            # Draw an x variable (italic)
            cv2.line(img, (10, 8), (18, 20), 255, 2)  # Main diagonal
            cv2.line(img, (8, 14), (16, 8), 255, 2)  # Second part
        elif symbol == 'y':
            # Draw a y variable
            cv2.line(img, (10, 8), (14, 14), 255, 2)  # Top-left to middle
            cv2.line(img, (18, 8), (14, 14), 255, 2)  # Top-right to middle
            cv2.line(img, (14, 14), (14, 20), 255, 2)  # Middle to bottom
        elif symbol == '(':
            # Draw a left parenthesis
            cv2.ellipse(img, (18, 14), (8, 10), 0, -90, 90, 255, 2)
        elif symbol == ')':
            # Draw a right parenthesis
            cv2.ellipse(img, (10, 14), (8, 10), 0, 90, 270, 255, 2)
        elif symbol == '.':
            # Draw a dot
            cv2.circle(img, (14, 20), 2, 255, -1)
        elif symbol.isdigit():
            # For digits, use putText
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(symbol, font, 0.7, 2)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            text_y = (img.shape[0] + text_size[1]) // 2
            cv2.putText(img, symbol, (text_x, text_y), font, 0.7, 255, 2)
        
        # Add random transformations
        # 1. Random rotation
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((14, 14), angle, 1)
        img = cv2.warpAffine(img, M, (28, 28))
        
        # 2. Add noise
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # 3. Random thickness variation
        if random.choice([True, False, False]):  # 1/3 chance
            kernel_size = random.choice([1, 2])
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            if random.choice([True, False]):
                img = cv2.erode(img, kernel, iterations=1)
            else:
                img = cv2.dilate(img, kernel, iterations=1)
        
        # Save the image
        output_path = os.path.join(output_dir, symbol, f"{symbol}_{idx}.png")
        cv2.imwrite(output_path, img)
        
        return output_path
    
    # Generate samples for each symbol
    total_generated = 0
    for symbol in tqdm(symbols, desc="Generating symbols"):
        for i in range(samples_per_symbol):
            # Alternate between PIL and OpenCV generation methods
            if available_fonts and i % 2 == 0:
                font_path = random.choice(available_fonts)
                generate_with_pil(symbol, i, font_path)
            else:
                generate_with_cv2(symbol, i)
            total_generated += 1
    
    print(f"Generated {total_generated} synthetic symbol images in {output_dir}")
    return output_dir

def create_preview_grid(output_dir):
    """Create a preview grid of the generated symbols."""
    symbols = os.listdir(output_dir)
    symbols = [s for s in symbols if os.path.isdir(os.path.join(output_dir, s))]
    
    # Set up the grid
    fig, axes = plt.subplots(len(symbols), 5, figsize=(12, 2*len(symbols)))
    
    for i, symbol in enumerate(sorted(symbols)):
        symbol_dir = os.path.join(output_dir, symbol)
        files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
        
        if not files:
            continue
        
        # Select 5 random samples
        if len(files) > 5:
            files = random.sample(files, 5)
        
        for j, file_path in enumerate(files):
            if j < 5:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if len(symbols) > 1:
                    axes[i, j].imshow(img, cmap='gray')
                    axes[i, j].set_title(f"{symbol}" if j == 0 else "")
                    axes[i, j].axis('off')
                else:
                    axes[j].imshow(img, cmap='gray')
                    axes[j].set_title(f"{symbol}" if j == 0 else "")
                    axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '..', 'symbol_preview.png'))
    print(f"Created preview grid at {os.path.join(output_dir, '..', 'symbol_preview.png')}")

def main():
    """Main function to prepare the dataset."""
    print("Preparing handwritten math symbols dataset...")
    
    # Create output directory
    output_dir = 'data/math_symbols'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic data
    generate_synthetic_symbols(output_dir)
    
    # Create preview grid
    create_preview_grid(output_dir)
    
    print("Dataset preparation complete!")
    print(f"Dataset saved in {output_dir}")
    print("Ready for training!")

if __name__ == "__main__":
    main() 