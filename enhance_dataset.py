"""
Enhance our dataset of handwritten math symbols by incorporating:
1. MNIST dataset for digits
2. More handwriting variations
3. Realistic handwritten math symbols

This script will supplement our synthetic dataset with more real-world examples.
"""

import os
import numpy as np
import cv2
import random
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.datasets import mnist
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps, ImageTransform
import requests
import shutil
import zipfile
import io

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

def incorporate_mnist_digits(output_dir, samples_per_digit=200):
    """
    Incorporate MNIST digits into our dataset.
    MNIST provides high-quality handwritten digits.
    """
    print("Incorporating MNIST digits into the dataset...")
    
    # Create directories for each digit if they don't exist
    for i in range(10):
        digit_dir = os.path.join(output_dir, str(i))
        os.makedirs(digit_dir, exist_ok=True)
    
    # Load MNIST dataset
    (x_train, y_train), _ = mnist.load_data()
    
    # Count existing samples for each digit
    existing_samples = {}
    for i in range(10):
        digit_dir = os.path.join(output_dir, str(i))
        existing_samples[i] = len(glob.glob(os.path.join(digit_dir, f"{i}_*.png")))
    
    # Process MNIST digits
    for digit in range(10):
        # Find all indices for this digit
        indices = np.where(y_train == digit)[0]
        
        # Shuffle to get random samples
        np.random.shuffle(indices)
        
        # Determine how many samples to add
        needed_samples = max(0, samples_per_digit - existing_samples[digit])
        indices = indices[:needed_samples]
        
        # Save the selected samples
        for i, idx in enumerate(indices):
            img = x_train[idx]
            
            # Add random noise for variety
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # Apply random transformations
            if random.choice([True, False, False]):  # 1/3 chance
                angle = random.uniform(-10, 10)
                rows, cols = img.shape
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                img = cv2.warpAffine(img, M, (cols, rows))
            
            # Save the image
            sample_idx = existing_samples[digit] + i
            output_path = os.path.join(output_dir, str(digit), f"{digit}_{sample_idx}.png")
            cv2.imwrite(output_path, img)
    
    # Report how many images we've added
    new_samples = {}
    for i in range(10):
        digit_dir = os.path.join(output_dir, str(i))
        new_samples[i] = len(glob.glob(os.path.join(digit_dir, f"{i}_*.png")))
        added = new_samples[i] - existing_samples[i]
        print(f"Added {added} MNIST samples for digit {i}")
    
    return sum(new_samples.values()) - sum(existing_samples.values())

def create_variations_from_existing(output_dir, variations_per_sample=3):
    """
    Create variations from existing symbols in the dataset.
    This increases diversity by applying different transformations.
    """
    print("Creating variations from existing samples...")
    
    # Get all symbol directories
    symbol_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    total_added = 0
    
    for symbol in symbol_dirs:
        symbol_dir = os.path.join(output_dir, symbol)
        existing_files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
        
        # Get the next index to use
        if existing_files:
            try:
                next_idx = max([int(os.path.basename(f).split('_')[1].split('.')[0]) 
                            for f in existing_files]) + 1
            except:
                next_idx = len(existing_files)
        else:
            next_idx = 0
        
        # Only use a subset of files to create variations
        if len(existing_files) > 20:
            existing_files = random.sample(existing_files, 20)
        
        for file_path in existing_files:
            # Read the image
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
                
            # Create variations
            for i in range(variations_per_sample):
                # Apply a series of transformations
                
                # Convert to PIL for easier transformations
                pil_img = Image.fromarray(image)
                
                # Random rotation
                angle = random.uniform(-15, 15)
                pil_img = pil_img.rotate(angle, resample=Image.BICUBIC, expand=False)
                
                # Random perspective distortion (simpler than quad transform)
                if random.choice([True, False]):
                    width, height = pil_img.size
                    scale = 0.1
                    
                    # Random perspective distortion
                    x1 = random.uniform(0, width * scale)
                    y1 = random.uniform(0, height * scale)
                    x2 = width - random.uniform(0, width * scale)
                    y2 = random.uniform(0, height * scale)
                    x3 = width - random.uniform(0, width * scale)
                    y3 = height - random.uniform(0, height * scale)
                    x4 = random.uniform(0, width * scale)
                    y4 = height - random.uniform(0, height * scale)
                    
                    # Define perspective transform coefficients
                    from_coords = [(0, 0), (width, 0), (width, height), (0, height)]
                    to_coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                    
                    try:
                        # Apply perspective transform
                        coeffs = find_coeffs(to_coords, from_coords)
                        pil_img = pil_img.transform(
                            (width, height),
                            Image.PERSPECTIVE,
                            coeffs,
                            Image.BICUBIC
                        )
                    except (AttributeError, ImportError, NameError):
                        # Fallback for older PIL versions or if transform is not available
                        # Just use an affine transform instead (less distortion but still adds variation)
                        angle = random.uniform(-5, 5)
                        scale = random.uniform(0.9, 1.1)
                        shift_x = random.uniform(-2, 2)
                        shift_y = random.uniform(-2, 2)
                        
                        # Create an affine transform matrix
                        transform_matrix = [
                            scale * np.cos(np.radians(angle)), -scale * np.sin(np.radians(angle)), shift_x,
                            scale * np.sin(np.radians(angle)), scale * np.cos(np.radians(angle)), shift_y
                        ]
                        
                        pil_img = pil_img.transform((width, height), 
                                                 Image.AFFINE, 
                                                 transform_matrix,
                                                 Image.BICUBIC)
                
                # Random brightness/contrast
                if random.choice([True, False]):
                    enhancer = ImageEnhance.Brightness(pil_img)
                    factor = random.uniform(0.8, 1.2)
                    pil_img = enhancer.enhance(factor)
                    
                    enhancer = ImageEnhance.Contrast(pil_img)
                    factor = random.uniform(0.8, 1.2)
                    pil_img = enhancer.enhance(factor)
                
                # Occasional blur or sharpen
                if random.choice([True, False, False]):
                    if random.choice([True, False]):
                        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
                    else:
                        pil_img = pil_img.filter(ImageFilter.SHARPEN)
                
                # Occasional noise or speckles
                if random.choice([True, False, False]):
                    img_array = np.array(pil_img)
                    noise_type = random.choice(['gaussian', 'salt_pepper'])
                    
                    if noise_type == 'gaussian':
                        noise = np.random.normal(0, random.uniform(3, 8), img_array.shape).astype(np.uint8)
                        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                    else:  # salt and pepper
                        prob = random.uniform(0.01, 0.03)
                        rnd = np.random.rand(*img_array.shape)
                        img_array[rnd < prob/2] = 0
                        img_array[rnd > 1 - prob/2] = 255
                    
                    pil_img = Image.fromarray(img_array)
                
                # Occasional thickness change
                if random.choice([True, False]):
                    img_array = np.array(pil_img)
                    if random.choice([True, False]):
                        # Thin
                        kernel = np.ones((2, 2), np.uint8)
                        img_array = cv2.erode(img_array, kernel, iterations=1)
                    else:
                        # Thick
                        kernel = np.ones((2, 2), np.uint8)
                        img_array = cv2.dilate(img_array, kernel, iterations=1)
                    
                    pil_img = Image.fromarray(img_array)
                
                # Convert back to numpy and save
                img_array = np.array(pil_img)
                output_path = os.path.join(symbol_dir, f"{symbol}_{next_idx + i}.png")
                cv2.imwrite(output_path, img_array)
                total_added += 1
            
            next_idx += variations_per_sample
    
    print(f"Added {total_added} variations from existing samples")
    return total_added

def download_kaggle_handwritten_data():
    """
    Attempt to download handwritten math symbol datasets from Kaggle or other sources.
    This is a placeholder; in practice, you would need a Kaggle account and API key.
    """
    # Try to download a handwritten math dataset from HASYv2
    try:
        # HASYv2 dataset: https://zenodo.org/record/259444
        hasy_url = "https://zenodo.org/record/259444/files/HASYv2.tar.bz2"
        os.makedirs("data/external", exist_ok=True)
        target_path = os.path.join("data/external", "HASYv2.tar.bz2")
        
        if not os.path.exists(target_path):
            print("Downloading HASYv2 dataset (may take some time)...")
            response = requests.get(hasy_url, stream=True)
            
            # Only download if the request was successful
            if response.status_code == 200:
                with open(target_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded HASYv2 dataset to {target_path}")
                
                # Extract the dataset
                import tarfile
                with tarfile.open(target_path, "r:bz2") as tar:
                    extract_dir = os.path.join("data/external", "HASYv2")
                    os.makedirs(extract_dir, exist_ok=True)
                    print(f"Extracting HASYv2 dataset to {extract_dir}")
                    tar.extractall(path=extract_dir)
                
                return extract_dir
            else:
                print(f"Failed to download HASYv2 dataset: {response.status_code}")
        else:
            print(f"HASYv2 dataset already downloaded at {target_path}")
            extract_dir = os.path.join("data/external", "HASYv2")
            return extract_dir
    
    except Exception as e:
        print(f"Error downloading external dataset: {str(e)}")
    
    return None

def process_hasy_dataset(hasy_dir, output_dir, symbols_map, max_per_symbol=200):
    """
    Process the HASYv2 dataset to extract relevant math symbols.
    
    Args:
        hasy_dir: Directory containing the HASYv2 dataset
        output_dir: Directory to save extracted symbols
        symbols_map: Mapping from HASYv2 symbol names to our symbol names
        max_per_symbol: Maximum number of samples per symbol
    """
    if not hasy_dir or not os.path.exists(hasy_dir):
        print("HASYv2 dataset directory not found")
        return 0
    
    # Check for the symbols.csv file which contains labels
    symbols_csv = os.path.join(hasy_dir, "symbols.csv")
    if not os.path.exists(symbols_csv):
        print(f"Symbols file not found: {symbols_csv}")
        return 0
    
    print("Processing HASYv2 dataset...")
    
    # Read the symbols.csv file to get file paths and labels
    import csv
    symbol_files = []
    
    with open(symbols_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 2:
                file_path = os.path.join(hasy_dir, row[0])
                symbol_name = row[1]
                
                # Check if this symbol is in our mapping
                for hasy_pattern, our_symbol in symbols_map.items():
                    if hasy_pattern in symbol_name:
                        symbol_files.append((file_path, our_symbol))
                        break
    
    # Shuffle the files to get different variations if we need to limit
    random.shuffle(symbol_files)
    
    # Count how many we've processed for each symbol
    processed_counts = {}
    for _, symbol in symbol_files:
        if symbol not in processed_counts:
            processed_counts[symbol] = 0
    
    total_added = 0
    
    # Process each file
    for file_path, symbol in tqdm(symbol_files, desc="Processing HASYv2"):
        # Skip if we've reached the limit for this symbol
        if processed_counts[symbol] >= max_per_symbol:
            continue
        
        try:
            # Read the image
            if os.path.exists(file_path):
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # Ensure it's binary (white on black)
                _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
                
                # Resize to 28x28
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                
                # Create the output directory if it doesn't exist
                symbol_dir = os.path.join(output_dir, symbol)
                os.makedirs(symbol_dir, exist_ok=True)
                
                # Get next index for this symbol
                existing_files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
                if existing_files:
                    try:
                        next_idx = max([int(os.path.basename(f).split('_')[1].split('.')[0]) 
                                    for f in existing_files]) + 1
                    except:
                        next_idx = len(existing_files)
                else:
                    next_idx = 0
                
                # Save the image
                output_path = os.path.join(symbol_dir, f"{symbol}_{next_idx}.png")
                cv2.imwrite(output_path, img)
                
                processed_counts[symbol] += 1
                total_added += 1
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    print(f"Added {total_added} symbols from HASYv2 dataset")
    return total_added

def simulate_written_equations(output_dir, num_equations=10):
    """
    Create images that simulate complete handwritten equations.
    This helps with training on more realistic scenarios where symbols are connected.
    """
    print("Creating simulated handwritten equations...")
    
    # Create directory for equations
    equations_dir = os.path.join(output_dir, "equations")
    os.makedirs(equations_dir, exist_ok=True)
    
    # Define some simple equation templates
    equation_templates = [
        "a+b=c",
        "a-b=c",
        "a×b=c",
        "a÷b=c",
        "a+b-c=d",
        "a×b+c=d",
        "a×(b+c)=d",
        "(a+b)÷c=d",
        "a+b+c+d=e",
        "a^2+b^2=c^2"
    ]
    
    # Symbols we can use to substitute variables
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    operators = ['+', '-', '×', '÷', '=', '(', ')']
    
    # Get all available symbol images
    symbol_images = {}
    for symbol in digits + operators:
        symbol_dir = os.path.join(output_dir, symbol)
        if os.path.exists(symbol_dir):
            files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
            if files:
                symbol_images[symbol] = files
    
    # Create equations
    for i in range(num_equations):
        # Choose a template
        template = random.choice(equation_templates)
        
        # Create a blank canvas (larger for equations)
        equation_width = len(template) * 28
        canvas = np.zeros((56, equation_width), dtype=np.uint8)
        
        # Keep track of current x position
        x_pos = 5  # Start with a small margin
        
        # For each character in the template
        for char in template:
            if char in symbol_images:
                # Choose a random image for this symbol
                img_path = random.choice(symbol_images[char])
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Apply random transformations
                    # 1. Scale
                    scale = random.uniform(0.9, 1.3)
                    new_size = int(28 * scale)
                    img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_AREA)
                    
                    # 2. Rotation
                    angle = random.uniform(-10, 10)
                    M = cv2.getRotationMatrix2D((new_size/2, new_size/2), angle, 1)
                    img = cv2.warpAffine(img, M, (new_size, new_size))
                    
                    # 3. Random vertical position
                    y_offset = random.randint(5, 28 - 5)
                    
                    # Place the symbol on the canvas
                    # Ensure we don't go out of bounds
                    y_max = min(56, y_offset + img.shape[0])
                    x_max = min(equation_width, x_pos + img.shape[1])
                    img_height = y_max - y_offset
                    img_width = x_max - x_pos
                    
                    if img_height > 0 and img_width > 0:
                        try:
                            canvas[y_offset:y_max, x_pos:x_max] = img[:img_height, :img_width]
                        except:
                            pass  # Skip if there's a shape mismatch
                    
                    # Move x position
                    x_pos += img.shape[1] + random.randint(0, 5)  # Add a small random gap
            else:
                # For variables or unsupported symbols, use a placeholder
                x_pos += 28  # Skip space
        
        # Save the equation image
        output_path = os.path.join(equations_dir, f"equation_{i}.png")
        cv2.imwrite(output_path, canvas)
    
    print(f"Created {num_equations} simulated equations")
    return num_equations

def enhance_existing_dataset(output_dir):
    """
    Enhance our existing dataset with more varied samples.
    """
    print(f"Enhancing dataset in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Track how many new samples we add
    total_added = 0
    
    # 1. Add MNIST digits
    total_added += incorporate_mnist_digits(output_dir)
    
    # 2. Create variations from existing samples
    total_added += create_variations_from_existing(output_dir)
    
    # 3. Try to download and process external dataset (HASYv2)
    hasy_dir = download_kaggle_handwritten_data()
    if hasy_dir:
        # Map HASYv2 symbol names to our symbol names
        symbols_map = {
            "plus": "+",
            "minus": "-",
            "div": "÷",
            "times": "×",
            "eq": "=",
            "lparen": "(",
            "rparen": ")",
            "0": "0",
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4",
            "5": "5",
            "6": "6",
            "7": "7",
            "8": "8",
            "9": "9",
            "x_": "x",
            "y_": "y",
            "dot": "."
        }
        
        total_added += process_hasy_dataset(hasy_dir, output_dir, symbols_map)
    
    # 4. Create simulated equation images
    total_added += simulate_written_equations(output_dir)
    
    print(f"Enhanced dataset with {total_added} new samples")
    return total_added

def create_preview_grid(output_dir):
    """Create a preview grid of the enhanced dataset."""
    symbols = [s for s in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, s))]
    
    # Special handling for equations folder
    if "equations" in symbols:
        symbols.remove("equations")
        has_equations = True
    else:
        has_equations = False
    
    # Sort the symbols in a logical order
    sorted_symbols = []
    # First digits
    for i in range(10):
        if str(i) in symbols:
            sorted_symbols.append(str(i))
    # Then operators
    for op in ['+', '-', '×', '÷', '=', '(', ')', '.']:
        if op in symbols:
            sorted_symbols.append(op)
    # Then variables
    for var in ['x', 'y']:
        if var in symbols:
            sorted_symbols.append(var)
    # Any remaining symbols
    for s in symbols:
        if s not in sorted_symbols:
            sorted_symbols.append(s)
    
    if not sorted_symbols:
        print("No symbol directories found!")
        return
    
    # Set up the grid for symbols
    fig, axes = plt.subplots(len(sorted_symbols), 5, figsize=(12, 2*len(sorted_symbols)))
    
    for i, symbol in enumerate(sorted_symbols):
        symbol_dir = os.path.join(output_dir, symbol)
        files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
        
        if not files:
            continue
        
        # Select random samples
        if len(files) > 5:
            files = random.sample(files, 5)
        
        for j, file_path in enumerate(files):
            if j < 5:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    if len(sorted_symbols) > 1:
                        axes[i, j].imshow(img, cmap='gray')
                        axes[i, j].set_title(f"{symbol}" if j == 0 else "")
                        axes[i, j].axis('off')
                    else:
                        axes[j].imshow(img, cmap='gray')
                        axes[j].set_title(f"{symbol}" if j == 0 else "")
                        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '..', 'enhanced_symbols_preview.png'))
    print(f"Created preview grid at {os.path.join(output_dir, '..', 'enhanced_symbols_preview.png')}")
    
    # Create a separate preview for equations if any
    if has_equations:
        equations_dir = os.path.join(output_dir, "equations")
        files = glob.glob(os.path.join(equations_dir, "equation_*.png"))
        
        if files:
            # Limit to 5 files
            if len(files) > 5:
                files = random.sample(files, 5)
            
            # Create a new figure
            fig, axes = plt.subplots(1, len(files), figsize=(15, 3))
            
            for j, file_path in enumerate(files):
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    if len(files) > 1:
                        axes[j].imshow(img, cmap='gray')
                        axes[j].set_title(f"Equation {j+1}")
                        axes[j].axis('off')
                    else:
                        axes.imshow(img, cmap='gray')
                        axes.set_title(f"Equation {j+1}")
                        axes.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '..', 'equations_preview.png'))
            print(f"Created equations preview at {os.path.join(output_dir, '..', 'equations_preview.png')}")

def find_coeffs(pa, pb):
    """
    Find coefficients for perspective transformation.
    pa: points after transformation
    pb: points before transformation
    """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float64)
    B = np.array(pb).reshape(8)
    
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

if __name__ == "__main__":
    output_dir = "data/math_symbols"
    enhance_existing_dataset(output_dir)
    create_preview_grid(output_dir)
    print("Dataset enhancement complete! Ready for training.") 