import os
import sys
import pytest
import numpy as np
import cv2
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.image_processing import preprocess_image, segment_symbols
from app.utils.recognition import fix_equation_syntax
from app.utils.equation_solver import parse_and_solve_equation

def test_preprocess_image():
    # Create a simple test image
    img = np.zeros((100, 200), dtype=np.uint8)
    cv2.putText(img, "2+2=4", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    # Preprocess with default params
    processed = preprocess_image(img)
    
    # Check that the result is binary
    assert set(np.unique(processed)).issubset({0, 255})
    
    # Check that preprocessing with adaptive thresholding also works
    processed_adaptive = preprocess_image(img, use_adaptive_threshold=True)
    assert set(np.unique(processed_adaptive)).issubset({0, 255})

def test_segment_symbols():
    # Create a test image with digits
    img = np.zeros((100, 200), dtype=np.uint8)
    cv2.putText(img, "1 2 3", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    # Preprocess
    binary = preprocess_image(img)
    
    # Segment with default params
    symbols, boxes = segment_symbols(binary)
    
    # We should detect 3 symbols
    assert len(symbols) == 3
    assert len(boxes) == 3
    
    # Each symbol should be 28x28
    for symbol in symbols:
        assert symbol.shape == (28, 28)

def test_fix_equation_syntax():
    # Test fixing consecutive operators
    assert fix_equation_syntax("1++2") == "1+2"
    
    # Test removing operators at the beginning
    assert fix_equation_syntax("+1+2") == "1+2"
    
    # Test removing operators at the end
    assert fix_equation_syntax("1+2+") == "1+2"
    
    # Test implicit multiplication
    assert fix_equation_syntax("2x") == "2*x"
    assert fix_equation_syntax("2(3+4)") == "2*(3+4)"
    assert fix_equation_syntax(")5") == ")*5"

def test_parse_and_solve_equation():
    # Test simple equation
    solution, steps = parse_and_solve_equation("x+5=10")
    assert solution == "x = 5"
    assert len(steps) > 0
    
    # Test expression
    solution, steps = parse_and_solve_equation("2+3*4")
    assert solution == "14"
    assert len(steps) > 0

if __name__ == "__main__":
    # Create the tests directory if it doesn't exist
    Path("tests").mkdir(exist_ok=True)
    
    # Run the tests
    pytest.main(["-xvs", __file__]) 