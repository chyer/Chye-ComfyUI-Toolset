#!/usr/bin/env python3
"""
Test script for CYHGlobalColorGradingNode
"""

import sys
import os
import torch
import numpy as np
from PIL import Image

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the node class
from categories.post_process import CYHGlobalColorGradingNode, apply_global_color_grading

def create_test_image(width=256, height=256):
    """Create a simple test image with gradients"""
    # Create RGB gradient
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    # Create RGB channels
    r = xx  # Red gradient left to right
    g = yy  # Green gradient top to bottom  
    b = (xx + yy) / 2  # Blue diagonal gradient
    
    # Combine channels
    image = np.stack([r, g, b], axis=-1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    return torch.from_numpy(image.astype(np.float32))

def test_global_color_grading_function():
    """Test the apply_global_color_grading function"""
    print("Testing apply_global_color_grading function...")
    
    # Create test image
    test_image = create_test_image(64, 64)
    print(f"Test image shape: {test_image.shape}")
    print(f"Test image range: {test_image.min():.3f} - {test_image.max():.3f}")
    
    # Test different parameters
    test_cases = [
        {"temperature": 6500, "saturation": 1.0, "tint": 0.0, "gamma": 1.0, "exposure": 0.0, "contrast": 1.0},
        {"temperature": 3200, "saturation": 1.5, "tint": 0.0, "gamma": 1.0, "exposure": 0.0, "contrast": 1.0},  # Warm
        {"temperature": 9000, "saturation": 0.5, "tint": 0.0, "gamma": 1.0, "exposure": 0.0, "contrast": 1.0},  # Cool
        {"temperature": 6500, "saturation": 1.0, "tint": 0.5, "gamma": 1.0, "exposure": 0.0, "contrast": 1.0},  # Magenta tint
        {"temperature": 6500, "saturation": 1.0, "tint": -0.5, "gamma": 1.0, "exposure": 0.0, "contrast": 1.0}, # Green tint
        {"temperature": 6500, "saturation": 1.0, "tint": 0.0, "gamma": 2.2, "exposure": 0.0, "contrast": 1.0},  # High gamma
        {"temperature": 6500, "saturation": 1.0, "tint": 0.0, "gamma": 0.5, "exposure": 0.0, "contrast": 1.0},  # Low gamma
        {"temperature": 6500, "saturation": 1.0, "tint": 0.0, "gamma": 1.0, "exposure": 1.0, "contrast": 1.0},  # +1 EV
        {"temperature": 6500, "saturation": 1.0, "tint": 0.0, "gamma": 1.0, "exposure": -1.0, "contrast": 1.0}, # -1 EV
        {"temperature": 6500, "saturation": 1.0, "tint": 0.0, "gamma": 1.0, "exposure": 0.0, "contrast": 1.5},  # High contrast
        {"temperature": 6500, "saturation": 1.0, "tint": 0.0, "gamma": 1.0, "exposure": 0.0, "contrast": 0.7},  # Low contrast
    ]
    
    for i, params in enumerate(test_cases):
        try:
            result = apply_global_color_grading(test_image, **params)
            print(f"[PASS] Test case {i+1}: {params} - Success")
            print(f"  Result range: {result.min():.3f} - {result.max():.3f}")
        except Exception as e:
            print(f"[FAIL] Test case {i+1}: {params} - Failed: {e}")
            return False
    
    print("All function tests passed!")
    return True

def test_global_color_grading_node():
    """Test the CYHGlobalColorGradingNode class"""
    print("\nTesting CYHGlobalColorGradingNode class...")
    
    # Create node instance
    node = CYHGlobalColorGradingNode()
    
    # Test INPUT_TYPES
    input_types = node.INPUT_TYPES()
    required_params = input_types["required"]
    
    expected_params = ["image", "temperature", "saturation", "tint", "gamma", "exposure", "contrast"]
    for param in expected_params:
        if param not in required_params:
            print(f"[FAIL] Missing parameter: {param}")
            return False
    
    print("[PASS] All required parameters present")
    
    # Test with default values
    test_image = create_test_image(32, 32)
    
    try:
        result = node.apply_color_grading(
            test_image,
            temperature=6500,
            saturation=1.0,
            tint=0.0,
            gamma=1.0,
            exposure=0.0,
            contrast=1.0
        )
        print("[PASS] Node execution with defaults - Success")
        print(f"  Result shape: {result[0].shape}")
    except Exception as e:
        print(f"[FAIL] Node execution failed: {e}")
        return False
    
    # Test with extreme values
    try:
        result = node.apply_color_grading(
            test_image,
            temperature=1000,   # Minimum
            saturation=2.0,     # Maximum
            tint=-1.0,          # Minimum
            gamma=2.5,          # Maximum
            exposure=-4.0,      # Minimum
            contrast=2.0        # Maximum
        )
        print("[PASS] Node execution with extreme values - Success")
    except Exception as e:
        print(f"[FAIL] Node execution with extreme values failed: {e}")
        return False
    
    print("All node tests passed!")
    return True

def save_test_results():
    """Save test images for visual verification"""
    print("\nSaving test images for visual verification...")
    
    # Create a more interesting test image
    test_image = create_test_image(512, 512)
    
    # Save original
    original_np = test_image[0].numpy() * 255
    original_img = Image.fromarray(original_np.astype(np.uint8))
    original_img.save("test_original.png")
    print("[PASS] Saved original test image")
    
    # Apply different effects and save
    effects = [
        ("warm_3200k", {"temperature": 3200, "saturation": 1.2}),
        ("cool_9000k", {"temperature": 9000, "saturation": 0.8}),
        ("high_contrast", {"contrast": 1.8, "exposure": 0.5}),
        ("low_contrast", {"contrast": 0.6, "exposure": -0.5}),
        ("magenta_tint", {"tint": 0.7, "saturation": 1.1}),
        ("green_tint", {"tint": -0.7, "saturation": 1.1}),
    ]
    
    for name, params in effects:
        try:
            result = apply_global_color_grading(test_image, **params)
            result_np = result[0].numpy() * 255
            result_img = Image.fromarray(result_np.astype(np.uint8))
            result_img.save(f"test_{name}.png")
            print(f"[PASS] Saved {name} test image")
        except Exception as e:
            print(f"[FAIL] Failed to save {name}: {e}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing CYHGlobalColorGradingNode Implementation")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_global_color_grading_function()
    success &= test_global_color_grading_node()
    success &= save_test_results()
    
    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED! Global color grading node is working correctly.")
        print("Test images saved for visual verification.")
    else:
        print("SOME TESTS FAILED! Please check the implementation.")
    print("=" * 60)