import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import math

def preprocess_image(image: np.ndarray, use_adaptive_threshold: bool = True) -> np.ndarray:
    """
    Preprocess the input image for better character recognition.
    
    Args:
        image: Input image in BGR format
        use_adaptive_threshold: Whether to use adaptive thresholding
        
    Returns:
        Preprocessed binary image
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if use_adaptive_threshold:
        # Use adaptive thresholding for better results with varying lighting
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
    else:
        # Apply Otsu's thresholding as a fallback
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary

def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Deskew an image containing text to make it horizontal.
    
    Args:
        image: Input image (binary)
        
    Returns:
        Deskewed image
    """
    # Calculate the skew angle
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust the angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def segment_symbols(binary_image: np.ndarray, min_area: int = 100) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Segment individual symbols from the preprocessed binary image.
    
    Args:
        binary_image: Preprocessed binary image
        min_area: Minimum contour area to consider as a symbol
        
    Returns:
        Tuple of (symbol_images, bounding_boxes) where symbol_images is a 
        list of segmented symbol images and bounding_boxes is a list of 
        (x, y, w, h) tuples
    """
    # Try to deskew the image first
    deskewed = deskew_image(binary_image)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(deskewed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours (noise)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # If we don't have enough contours, try using the original image
    if len(filtered_contours) < 2 and len(contours) > 2:
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Sort contours from left to right, considering that symbols can be at different heights
    def get_leftmost_x(contour):
        x, _, _, _ = cv2.boundingRect(contour)
        return x
    
    # Sort by x-coordinate
    filtered_contours.sort(key=get_leftmost_x)
    
    # Extract each symbol
    symbol_images = []
    bounding_boxes = []
    
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add some padding
        padding = 5
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(binary_image.shape[1], x + w + padding)
        y_max = min(binary_image.shape[0], y + h + padding)
        
        # Extract the symbol from the original binary image for consistency
        symbol = binary_image[y_min:y_max, x_min:x_max]
        
        # Skip if the extracted region is too small
        if symbol.size == 0 or symbol.shape[0] < 5 or symbol.shape[1] < 5:
            continue
            
        # Resize to a standard size (28x28) for the model
        symbol = cv2.resize(symbol, (28, 28), interpolation=cv2.INTER_AREA)
        
        symbol_images.append(symbol)
        bounding_boxes.append((x, y, w, h))
    
    return symbol_images, bounding_boxes

def merge_touching_symbols(bounding_boxes: List[Tuple[int, int, int, int]], 
                          image_shape: Tuple[int, int], 
                          threshold: float = 0.1) -> List[Tuple[int, int, int, int]]:
    """
    Merge bounding boxes that are likely part of the same symbol.
    
    Args:
        bounding_boxes: List of bounding box tuples (x, y, w, h)
        image_shape: Shape of the image (height, width)
        threshold: Overlap threshold for merging
        
    Returns:
        List of merged bounding boxes
    """
    if not bounding_boxes:
        return []
    
    # Convert to numpy array for easier manipulation
    boxes = np.array(bounding_boxes)
    
    # Calculate the area of each box
    areas = boxes[:, 2] * boxes[:, 3]
    
    # Flag to track if we merged any boxes
    merged_any = True
    
    while merged_any:
        merged_any = False
        i = 0
        
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                # Calculate overlap
                x1, y1, w1, h1 = boxes[i]
                x2, y2, w2, h2 = boxes[j]
                
                # Check if boxes overlap horizontally
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                
                # Check if boxes are close vertically
                vertical_distance = abs((y1 + h1/2) - (y2 + h2/2))
                
                # Merge if horizontal overlap is significant and vertical distance is reasonable
                if (overlap_x > threshold * min(w1, w2) and 
                    vertical_distance < max(h1, h2) * 1.5):
                    
                    # Create a new merged box
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1 + w1, x2 + w2) - new_x
                    new_h = max(y1 + h1, y2 + h2) - new_y
                    
                    # Replace the first box with the merged box
                    boxes[i] = np.array([new_x, new_y, new_w, new_h])
                    
                    # Remove the second box
                    boxes = np.delete(boxes, j, axis=0)
                    areas = np.delete(areas, j)
                    
                    merged_any = True
                else:
                    j += 1
            i += 1
    
    return [tuple(box) for box in boxes]

def create_visualization(original_image: np.ndarray, binary_image: np.ndarray, 
                         bounding_boxes: List[Tuple[int, int, int, int]], 
                         recognized_symbols: Optional[List[str]] = None) -> np.ndarray:
    """
    Create a visualization of the segmented symbols.
    
    Args:
        original_image: Original input image
        binary_image: Preprocessed binary image
        bounding_boxes: List of bounding box coordinates (x, y, w, h)
        recognized_symbols: Optional list of recognized symbols
        
    Returns:
        Visualization image with bounding boxes and labels
    """
    # Convert binary image to BGR for colored annotations
    if len(original_image.shape) == 2:
        visualization = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        visualization = original_image.copy()
    
    # Draw bounding boxes and labels
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # Draw rectangle
        cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label if symbols are provided
        if recognized_symbols and i < len(recognized_symbols):
            label = recognized_symbols[i]
            cv2.putText(visualization, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return visualization

def clean_up_after_segmentation(symbol_images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply additional processing to segmented symbols to improve recognition.
    
    Args:
        symbol_images: List of segmented symbol images
        
    Returns:
        List of processed symbol images
    """
    processed_symbols = []
    
    for symbol in symbol_images:
        # Make sure the image is binary
        if len(symbol.shape) == 3:
            symbol = cv2.cvtColor(symbol, cv2.COLOR_BGR2GRAY)
        
        # Ensure good contrast
        if np.max(symbol) - np.min(symbol) < 50:
            _, symbol = cv2.threshold(symbol, 127, 255, cv2.THRESH_BINARY)
        
        # Center the symbol in the image
        if np.sum(symbol) > 0:  # Make sure there's actually a symbol
            # Find the center of mass
            M = cv2.moments(symbol)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Calculate shift to center
                shift_x = symbol.shape[1]//2 - cX
                shift_y = symbol.shape[0]//2 - cY
                
                # Create transformation matrix
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                
                # Apply shift
                centered = cv2.warpAffine(symbol, M, (symbol.shape[1], symbol.shape[0]))
                symbol = centered
        
        processed_symbols.append(symbol)
    
    return processed_symbols

def group_symbols_into_equation(symbols: List[Tuple[str, np.ndarray]]) -> List[List[Tuple[str, np.ndarray]]]:
    """
    Group symbols into equations or expressions.
    This is a simplified version that assumes all symbols are part of a single line equation.
    
    Args:
        symbols: List of (symbol, bounding_box) tuples
        
    Returns:
        List of grouped equations
    """
    # For now, we assume everything is a single equation
    return [symbols] 