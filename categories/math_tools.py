"""
Math tools for Chye ComfyUI Toolset
"""

import sys
import os

# Add parent directory to Python path for ComfyUI compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from shared.constants import (
        MATH_CATEGORY,
        DEFAULT_AB_VALUE, MIN_AB_VALUE, MAX_AB_VALUE,
        DEFAULT_DIVISOR
    )
    from shared.helpers import round_to_multiple
except ImportError:
    # Fallback import for ComfyUI environments
    import importlib.util
    
    # Import constants
    constants_path = os.path.join(parent_dir, "shared", "constants.py")
    spec = importlib.util.spec_from_file_location("constants", constants_path)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    
    MATH_CATEGORY = constants.MATH_CATEGORY
    DEFAULT_AB_VALUE = constants.DEFAULT_AB_VALUE
    MIN_AB_VALUE = constants.MIN_AB_VALUE
    MAX_AB_VALUE = constants.MAX_AB_VALUE
    DEFAULT_DIVISOR = constants.DEFAULT_DIVISOR
    
    # Import helpers
    helpers_path = os.path.join(parent_dir, "shared", "helpers.py")
    spec = importlib.util.spec_from_file_location("helpers", helpers_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    
    round_to_multiple = helpers.round_to_multiple


class CYHABSwitchNode:
    """
    A ComfyUI node that takes Width and Height inputs, ensures they are 
    divisible by 32 (ComfyUI standard), and provides a switch to swap the output order.
    
    The UI step size is fixed at 32 to match ComfyUI's resolution requirements.
    
    Perfect for:
    - Resolution validation and adjustment (32-pixel alignment)
    - ComfyUI-compatible dimension processing
    - Dimension swapping for orientation changes (portrait/landscape)
    - VAE encoder/decoder compatible resolutions
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": DEFAULT_AB_VALUE,
                    "min": MIN_AB_VALUE,
                    "max": MAX_AB_VALUE,
                    "step": DEFAULT_DIVISOR  # Fixed 32-step for ComfyUI resolution standard
                }),
                "height": ("INT", {
                    "default": DEFAULT_AB_VALUE,
                    "min": MIN_AB_VALUE,
                    "max": MAX_AB_VALUE,
                    "step": DEFAULT_DIVISOR  # Fixed 32-step for ComfyUI resolution standard
                }),
                "swap_order": ("BOOLEAN", {
                    "default": False
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calculate"
    CATEGORY = MATH_CATEGORY
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, width, height, **kwargs):
        if width <= 0 or height <= 0:
            return "Width and height must be positive integers"
        
        return True

    def calculate(self, width, height, swap_order):
        """
        Process the input values and return them in the specified order.
        Values are automatically rounded to multiples of 32 (ComfyUI standard).
        
        Args:
            width (int): Width value
            height (int): Height value
            swap_order (bool): Whether to swap the output order
            
        Returns:
            tuple: (width, height) - processed values in specified order
        """
        
        # Fixed divisor of 32 for ComfyUI resolution standard
        divisor = DEFAULT_DIVISOR  # 32
        
        # Round both values to the nearest multiple of 32
        processed_width = round_to_multiple(width, divisor)
        processed_height = round_to_multiple(height, divisor)
        
        # Apply order swapping if requested
        if swap_order:
            return (processed_height, processed_width)
        else:
            return (processed_width, processed_height)


# Node registration for this category
NODE_CLASS_MAPPINGS = {
    "CYHABSwitchNode": CYHABSwitchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CYHABSwitchNode": "🔢 CYH Math | A B Switch Res 32 Step",
}
