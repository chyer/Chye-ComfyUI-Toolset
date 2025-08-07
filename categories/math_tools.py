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
        DEFAULT_MULTIPLIER_VALUE, MIN_MULTIPLIER_VALUE, MAX_MULTIPLIER_VALUE, MULTIPLIER_VALUE_STEP,
        DEFAULT_INCREMENT, MIN_INCREMENT, MAX_INCREMENT, INCREMENT_STEP
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
    DEFAULT_MULTIPLIER_VALUE = constants.DEFAULT_MULTIPLIER_VALUE
    MIN_MULTIPLIER_VALUE = constants.MIN_MULTIPLIER_VALUE
    MAX_MULTIPLIER_VALUE = constants.MAX_MULTIPLIER_VALUE
    MULTIPLIER_VALUE_STEP = constants.MULTIPLIER_VALUE_STEP
    DEFAULT_INCREMENT = constants.DEFAULT_INCREMENT
    MIN_INCREMENT = constants.MIN_INCREMENT
    MAX_INCREMENT = constants.MAX_INCREMENT
    INCREMENT_STEP = constants.INCREMENT_STEP
    
    # Import helpers
    helpers_path = os.path.join(parent_dir, "shared", "helpers.py")
    spec = importlib.util.spec_from_file_location("helpers", helpers_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    
    round_to_multiple = helpers.round_to_multiple


class CYHResolutionMultiplierNode:
    """
    A math node that outputs resolution values based on a multiplier.
    The value field will increment/decrement in steps of the multiplier.
    Perfect for use with resolution width/height inputs in ComfyUI.
    For example, if the multiplier is 32, the value will snap to multiples of 32 (32, 64, 96, etc.).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiplier": ("INT", {"default": DEFAULT_MULTIPLIER_VALUE, "min": MIN_MULTIPLIER_VALUE, "max": MAX_MULTIPLIER_VALUE, "step": MULTIPLIER_VALUE_STEP}),
                "width": ("INT", {"default": DEFAULT_MULTIPLIER_VALUE, "min": MIN_MULTIPLIER_VALUE, "max": MAX_MULTIPLIER_VALUE * 20, "step": MULTIPLIER_VALUE_STEP}),
                "height": ("INT", {"default": DEFAULT_MULTIPLIER_VALUE, "min": MIN_MULTIPLIER_VALUE, "max": MAX_MULTIPLIER_VALUE * 20, "step": MULTIPLIER_VALUE_STEP}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    FUNCTION = "calculate"
    CATEGORY = MATH_CATEGORY
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def calculate(self, multiplier, width, height):
        # Round both width and height to the nearest multiple of the multiplier
        result_width = round_to_multiple(width, multiplier)
        result_height = round_to_multiple(height, multiplier)
        return (result_width, result_height)


# Node registration for this category
NODE_CLASS_MAPPINGS = {
    "CYHResolutionMultiplierNode": CYHResolutionMultiplierNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CYHResolutionMultiplierNode": "🔢 CYH Math | Resolution Multiplier",
}