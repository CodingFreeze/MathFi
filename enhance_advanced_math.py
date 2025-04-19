"""
Enhance our dataset with advanced mathematical symbols needed for calculus and higher-level mathematics.
This script adds symbols such as:
- Trigonometric functions (sin, cos, tan)
- Calculus symbols (∫, ∑, ∂, lim)
- Greek letters (π, θ, α, β, etc.)
- Other advanced math notation (√, ∞, etc.)
"""

import os
import numpy as np
import cv2
import random
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import shutil
import sys

# Import functions from the original enhance_dataset.py
from enhance_dataset import (
    find_coeffs,
    create_variations_from_existing,
    create_preview_grid
)

# Define advanced math symbols to create
ADVANCED_SYMBOLS = {
    # Trigonometric functions
    "sin": "sin",
    "cos": "cos", 
    "tan": "tan",
    
    # Calculus symbols
    "∫": "integral",
    "∂": "partial",
    "∑": "sum",
    "lim": "limit",
    "dx": "dx",
    "dy": "dy",
    
    # Greek letters
    "π": "pi",
    "θ": "theta",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "λ": "lambda",
    
    # Advanced operators
    "√": "sqrt",
    "∞": "infinity",
    "^": "power",
    ">": "greater",
    "<": "less",
    "≥": "greater_equal",
    "≤": "less_equal"
}

def create_advanced_symbol(symbol, output_dir, num_samples=50):
    """
    Create images for an advanced math symbol.
    
    Args:
        symbol: The symbol to create
        output_dir: The output directory
        num_samples: Number of samples to create
    
    Returns:
        Number of samples created
    """
    print(f"Creating {num_samples} samples for symbol: {symbol}")
    
    # Create directory for the symbol
    symbol_dir = os.path.join(output_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    
    # Find a suitable font that contains math symbols
    font_paths = [
        '/System/Library/Fonts/Supplemental/Symbol.ttf',  # macOS
        '/Library/Fonts/STIXGeneral.otf',  # macOS math fonts
        '/System/Library/Fonts/Supplemental/STIXGeneral.otf',  # Another macOS location
        '/usr/share/fonts/truetype/stix-word/STIXMath-Regular.otf',  # Linux
        '/usr/share/fonts/truetype/freefont/FreeSerif.ttf',  # Linux
        'C:\\Windows\\Fonts\\seguisym.ttf',  # Windows
        'C:\\Windows\\Fonts\\STIX-Regular.otf',  # Windows
    ]
    
    # Also check for fonts in the project
    project_fonts = glob.glob("data/fonts/*.ttf") + glob.glob("data/fonts/*.otf")
    font_paths.extend(project_fonts)
    
    # Create a directory for fonts if it doesn't exist
    os.makedirs("data/fonts", exist_ok=True)
    
    # Find a usable font
    font_path = None
    font_size = 20
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                # Test if this font can render our symbol
                font = ImageFont.truetype(path, font_size)
                # For multi-character symbols like "sin", we can always render them
                if len(symbol) > 1 or font.getbbox(symbol) is not None:
                    font_path = path
                    break
            except Exception:
                continue
    
    # If no suitable font found, download or use a fallback
    if font_path is None:
        print(f"No suitable font found for {symbol}, using default")
        # Use default font or PIL's built-in default
        font = None
    else:
        print(f"Using font: {font_path}")
        font = ImageFont.truetype(font_path, font_size)
    
    # Generate samples
    samples_created = 0
    
    for i in range(num_samples):
        # Create a blank image
        img = Image.new('L', (28, 28), color=0)
        draw = ImageDraw.Draw(img)
        
        # For multi-character symbols, use a smaller size
        if len(symbol) > 1:
            actual_font_size = int(font_size * (1.0 / len(symbol) * 1.5))
            if actual_font_size < 10:
                actual_font_size = 10
            if font_path:
                font = ImageFont.truetype(font_path, actual_font_size)
        
        # Position variations
        position_x = random.randint(1, 8)
        position_y = random.randint(1, 8)
        
        # Draw the text
        if font:
            draw.text((position_x, position_y), symbol, fill=255, font=font)
        else:
            draw.text((position_x, position_y), symbol, fill=255)
        
        # Apply transformations for variety
        # 1. Random rotation
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
        
        # 2. Random perspective distortion (only 25% of the time)
        if random.random() < 0.25:
            width, height = img.size
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
                img = img.transform(
                    (width, height),
                    Image.PERSPECTIVE,
                    coeffs,
                    Image.BICUBIC
                )
            except Exception as e:
                print(f"Perspective transform failed: {e}")
        
        # 3. Random brightness/contrast adjustment
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)
            
            enhancer = ImageEnhance.Contrast(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)
        
        # 4. Occasional blur
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        
        # 5. Add some noise
        img_array = np.array(img)
        noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Save the image
        output_path = os.path.join(symbol_dir, f"{symbol}_{i}.png")
        img.save(output_path)
        samples_created += 1
    
    return samples_created

def create_calculus_equations(output_dir, num_equations=10):
    """
    Create simulated calculus equations.
    
    Args:
        output_dir: The output directory
        num_equations: Number of equations to create
        
    Returns:
        Number of equations created
    """
    print(f"Creating {num_equations} calculus equations...")
    
    # Create directory for equations
    equations_dir = os.path.join(output_dir, "equations")
    os.makedirs(equations_dir, exist_ok=True)
    
    # Define calculus equation templates
    equation_templates = [
        "∫x^2dx=x^3/3",                   # Integral
        "∂f/∂x=2x",                       # Partial derivative
        "limx→∞1/x=0",                    # Limit
        "∑i=1^n i=n(n+1)/2",              # Summation
        "sin^2θ+cos^2θ=1",                # Trig identity
        "f'(x)=lim∆x→0(f(x+∆x)-f(x))/∆x", # Derivative definition
        "∫_0^π sin(x)dx=2",               # Definite integral
        "e^(iπ)+1=0",                     # Euler's identity
        "y=sinx+cosx",                    # Basic trig function
        "dy/dx=f'(x)",                    # Derivative notation
    ]
    
    # Get all available symbol images
    available_symbols = set()
    for symbol_dir in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, symbol_dir)) and symbol_dir != "equations":
            available_symbols.add(symbol_dir)
    
    # Track created equations
    created_equations = 0
    
    # Create a larger blank canvas for equations
    for i in range(num_equations):
        # Choose a template
        template = random.choice(equation_templates)
        
        # Create a blank canvas for the equation
        canvas_width = 350  # Width to accommodate longer equations
        canvas_height = 60  # Height for calculus notation
        canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        
        # Render the equation using a mono-spaced approach
        x_pos = 5  # Starting position
        y_pos = 20  # Vertical position
        
        # Process the template and place symbols
        # This is a simplified approach - a real implementation would need 
        # to handle mathematical layout properly
        for char in template:
            # Skip unknown symbols
            if char not in available_symbols and char.lower() not in available_symbols:
                x_pos += 12  # Skip with smaller space
                continue
                
            # Get the actual symbol to use
            symbol = char if char in available_symbols else char.lower()
            
            # Find images for this symbol
            symbol_dir = os.path.join(output_dir, symbol)
            files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.png"))
            
            if files:
                # Choose a random image
                img_path = random.choice(files)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize while maintaining aspect ratio
                    # For subscripts and superscripts, make them smaller
                    if char in "^_":
                        new_height = 18
                    else:
                        new_height = 28
                    
                    aspect_ratio = img.shape[1] / img.shape[0]
                    new_width = int(new_height * aspect_ratio)
                    
                    img = cv2.resize(img, (new_width, new_height))
                    
                    # Adjust vertical position for subscripts and superscripts
                    if char == "^":  # Superscript position
                        y_offset = y_pos - 15
                    elif char == "_":  # Subscript position
                        y_offset = y_pos + 5
                    else:
                        y_offset = y_pos - new_height // 2
                    
                    # Check boundaries
                    y_start = max(0, y_offset)
                    y_end = min(canvas_height, y_offset + new_height)
                    x_start = x_pos
                    x_end = min(canvas_width, x_pos + new_width)
                    
                    # Calculate image slices
                    img_y_start = y_start - y_offset
                    img_y_end = img_y_start + (y_end - y_start)
                    img_x_end = x_end - x_pos
                    
                    # Place symbol on canvas
                    if y_end > y_start and x_end > x_start and img_y_end <= img.shape[0] and img_x_end <= img.shape[1]:
                        try:
                            canvas[y_start:y_end, x_start:x_end] = img[img_y_start:img_y_end, 0:img_x_end]
                        except:
                            pass  # Skip if there's a shape mismatch
                    
                    # Move x position based on symbol width
                    x_pos += new_width + 2
            else:
                # If image not found, just advance position
                x_pos += 15
        
        # Save the equation
        output_path = os.path.join(equations_dir, f"calculus_eq_{i}.png")
        cv2.imwrite(output_path, canvas)
        created_equations += 1
    
    print(f"Created {created_equations} calculus equations")
    return created_equations

def enhance_dataset_with_advanced_math(data_dir="data/math_symbols"):
    """
    Enhance our dataset with advanced math symbols.
    
    Args:
        data_dir: The directory containing the math symbols dataset
    
    Returns:
        Number of new samples added
    """
    print(f"Enhancing dataset with advanced math symbols in {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)
    
    # Track how many new samples we add
    total_added = 0
    
    # Add each advanced symbol
    for symbol, name in ADVANCED_SYMBOLS.items():
        try:
            samples = create_advanced_symbol(symbol, data_dir)
            total_added += samples
        except Exception as e:
            print(f"Error creating symbol {symbol}: {e}")
    
    # Create variations from the newly added symbols
    print("Creating variations from advanced symbols...")
    total_added += create_variations_from_existing(data_dir)
    
    # Create calculus equations
    total_added += create_calculus_equations(data_dir)
    
    print(f"Enhanced dataset with {total_added} advanced math samples and equations")
    return total_added

if __name__ == "__main__":
    data_dir = "data/math_symbols"
    
    # Enhance the dataset with advanced math symbols
    enhance_dataset_with_advanced_math(data_dir)
    
    # Create a preview grid of all symbols including the new ones
    create_preview_grid(data_dir)
    
    print("Advanced math dataset enhancement complete! Ready for training.") 