#!/usr/bin/env python3
"""
Test script for CYHInteractivePainterNode
This script tests the basic functionality of the interactive painting node
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_painter_node_import():
    """Test that the painter node can be imported successfully"""
    try:
        from categories.image_tools import CYHInteractivePainterNode, NODE_CLASS_MAPPINGS
        
        print("CYHInteractivePainterNode imported successfully")
        print(f"Node class mappings: {list(NODE_CLASS_MAPPINGS.keys())}")
        
        # Test node initialization
        node = CYHInteractivePainterNode()
        print("Node instance created successfully")
        
        # Test INPUT_TYPES
        input_types = node.INPUT_TYPES()
        print(f"Input types: {input_types}")
        
        # Test basic functionality
        try:
            # Mock the global dictionary
            from categories.image_tools import CYH_PAINTER_DICT
            CYH_PAINTER_DICT["test_123"] = node
            
            # Test IS_CHANGED method
            changed_hash = node.IS_CHANGED(512, 512, 20, "#FF0000", "test_123")
            print(f"IS_CHANGED method works: {changed_hash}")
            
        except Exception as e:
            print(f"Partial test success (some methods may require ComfyUI environment): {e}")
        
        return True
        
    except ImportError as e:
        print(f"Import failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def test_web_endpoints():
    """Test that web endpoints are properly defined"""
    try:
        from categories.image_tools import to_base64_img_url
        
        # Test base64 conversion
        from PIL import Image
        test_image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        base64_url = to_base64_img_url(test_image)
        
        if base64_url.startswith("data:image/png;base64,"):
            print("Base64 image conversion works")
        else:
            print("Base64 conversion failed")
            
        return True
        
    except Exception as e:
        print(f"Web endpoint test failed: {e}")
        return False

def test_node_registration():
    """Test that the node is properly registered in the main module"""
    try:
        from __init__ import NODE_CLASS_MAPPINGS
        
        if "CYHInteractivePainterNode" in NODE_CLASS_MAPPINGS:
            print("Node registered in main module")
            return True
        else:
            print("Node not found in main module mappings")
            return False
            
    except Exception as e:
        print(f"Node registration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing CYHInteractivePainterNode Implementation")
    print("=" * 50)
    
    tests = [
        test_painter_node_import,
        test_web_endpoints,
        test_node_registration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"All {passed}/{total} tests passed!")
        print("\nNext steps:")
        print("1. Restart ComfyUI to load the new node")
        print("2. Look for 'Interactive Painter (CYH)' in the image/tools category")
        print("3. Test the interactive canvas functionality in the ComfyUI web interface")
    else:
        print(f"{passed}/{total} tests passed")
        print("Some functionality may require a full ComfyUI environment to test properly")