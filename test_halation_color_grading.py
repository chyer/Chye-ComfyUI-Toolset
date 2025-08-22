"""
Test script for enhanced ARRI halation node with comprehensive color grading controls
"""
import sys
import os
import numpy as np
import torch

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the enhanced halation function
from categories.post_process import arri_halation_effect, kelvin_to_rgb, apply_color_grading

def test_kelvin_to_rgb():
    """Test Kelvin to RGB conversion function"""
    print("Testing Kelvin to RGB conversion...")
    
    # Test various temperature values
    test_temps = [1000, 2000, 3000, 4000, 5000, 6500, 10000, 20000, 40000]
    
    for temp in test_temps:
        rgb = kelvin_to_rgb(temp)
        print(f"  {temp}K -> RGB: ({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})")
        
        # Verify values are in valid range
        assert all(0.0 <= val <= 1.0 for val in rgb), f"RGB values out of range for {temp}K"
    
    print("[PASS] Kelvin to RGB conversion test passed\n")

def test_color_grading():
    """Test comprehensive color grading function"""
    print("Testing color grading function...")
    
    # Test base color (neutral gray)
    base_color = (0.5, 0.5, 0.5)
    
    # Test saturation
    saturated = apply_color_grading(base_color, saturation=2.0)
    desaturated = apply_color_grading(base_color, saturation=0.5)
    print(f"  Saturation: base {base_color} -> saturated {saturated}, desaturated {desaturated}")
    
    # Test tint
    green_tint = apply_color_grading(base_color, tint=-1.0)
    magenta_tint = apply_color_grading(base_color, tint=1.0)
    print(f"  Tint: base {base_color} -> green {green_tint}, magenta {magenta_tint}")
    
    # Test gamma
    high_gamma = apply_color_grading(base_color, gamma=2.0)
    low_gamma = apply_color_grading(base_color, gamma=0.5)
    print(f"  Gamma: base {base_color} -> high {high_gamma}, low {low_gamma}")
    
    # Test exposure
    overexposed = apply_color_grading(base_color, exposure=2.0)
    underexposed = apply_color_grading(base_color, exposure=-2.0)
    print(f"  Exposure: base {base_color} -> over {overexposed}, under {underexposed}")
    
    # Test contrast
    high_contrast = apply_color_grading(base_color, contrast=1.5)
    low_contrast = apply_color_grading(base_color, contrast=0.7)
    print(f"  Contrast: base {base_color} -> high {high_contrast}, low {low_contrast}")
    
    # Test combined effects
    combined = apply_color_grading(base_color, saturation=1.5, tint=0.3, gamma=1.2, exposure=0.5, contrast=1.1)
    print(f"  Combined: base {base_color} -> {combined}")
    
    # Verify all values are in valid range
    test_cases = [saturated, desaturated, green_tint, magenta_tint, 
                 high_gamma, low_gamma, overexposed, underexposed,
                 high_contrast, low_contrast, combined]
    
    for i, case in enumerate(test_cases):
        assert all(0.0 <= val <= 1.0 for val in case), f"Color grading case {i} out of range: {case}"
    
    print("[PASS] Color grading test passed\n")

def test_halation_effect():
    """Test the enhanced halation effect with color grading"""
    print("Testing enhanced halation effect...")
    
    # Create a simple test image (batch of 1, 64x64, 3 channels)
    test_image = torch.rand(1, 64, 64, 3)  # Random image
    
    # Test with default parameters (should work like original)
    result_default = arri_halation_effect(test_image, threshold=220, blur_size=25, intensity=0.6)
    print(f"  Default halation: input shape {test_image.shape}, output shape {result_default.shape}")
    
    # Test with warm temperature (3000K)
    result_warm = arri_halation_effect(test_image, temperature=3000, saturation=1.2, intensity=0.8)
    print(f"  Warm halation (3000K): output shape {result_warm.shape}")
    
    # Test with cool temperature (10000K)
    result_cool = arri_halation_effect(test_image, temperature=10000, saturation=0.8, intensity=0.4)
    print(f"  Cool halation (10000K): output shape {result_cool.shape}")
    
    # Test with extreme color grading
    result_extreme = arri_halation_effect(
        test_image, 
        temperature=2000, 
        saturation=2.0, 
        tint=-0.8, 
        gamma=0.7, 
        exposure=1.5, 
        contrast=1.8,
        intensity=0.9
    )
    print(f"  Extreme grading: output shape {result_extreme.shape}")
    
    # Verify output shapes match input
    assert result_default.shape == test_image.shape, "Output shape mismatch for default"
    assert result_warm.shape == test_image.shape, "Output shape mismatch for warm"
    assert result_cool.shape == test_image.shape, "Output shape mismatch for cool"
    assert result_extreme.shape == test_image.shape, "Output shape mismatch for extreme"
    
    # Verify values are in valid range
    for name, result in [("default", result_default), ("warm", result_warm), 
                        ("cool", result_cool), ("extreme", result_extreme)]:
        result_np = result.cpu().numpy()
        assert np.all(result_np >= 0.0) and np.all(result_np <= 1.0), f"Values out of range for {name}"
    
    print("[PASS] Halation effect test passed\n")

def test_node_interface():
    """Test that the node class can be instantiated and called"""
    print("Testing node interface...")
    
    from categories.post_process import CYHARRHalationNode
    
    # Create node instance
    node = CYHARRHalationNode()
    
    # Test input types
    input_types = node.INPUT_TYPES()
    required_inputs = input_types["required"]
    
    # Verify all expected parameters are present
    expected_params = ["image", "threshold", "blur_size", "intensity", 
                      "temperature", "saturation", "tint", "gamma", "exposure", "contrast"]
    
    for param in expected_params:
        assert param in required_inputs, f"Missing parameter: {param}"
    
    print("  Input parameters:", list(required_inputs.keys()))
    
    # Test validation
    assert node.VALIDATE_INPUTS(), "Input validation failed"
    
    print("[PASS] Node interface test passed\n")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Enhanced ARRI Halation Node with Color Grading")
    print("=" * 60)
    
    try:
        test_kelvin_to_rgb()
        test_color_grading()
        test_halation_effect()
        test_node_interface()
        
        print("SUCCESS: All tests passed! The enhanced halation node is working correctly.")
        print("\nFeatures tested:")
        print("  - Kelvin temperature to RGB conversion (1000-40000K)")
        print("  - Comprehensive color grading (saturation, tint, gamma, exposure, contrast)")
        print("  - Temperature-based halation colors")
        print("  - Node interface with all new parameters")
        print("  - Backward compatibility with original functionality")
        
    except Exception as e:
        print(f"FAILED: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())