"""
Latent generation tools for Chye ComfyUI Toolset
"""

import torch
import sys
import os

# Add parent directory to Python path for ComfyUI compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from shared.constants import (
        MODEL_RESOLUTIONS, ASPECT_RATIOS, LATENT_CATEGORY,
        DEFAULT_MULTIPLIER, MIN_MULTIPLIER, MAX_MULTIPLIER, MULTIPLIER_STEP,
        DEFAULT_BATCH_SIZE, MIN_BATCH_SIZE, MAX_BATCH_SIZE,
        DEFAULT_ORIENTATION, ORIENTATIONS
    )
    from shared.helpers import parse_aspect_ratio, calculate_final_dimensions
except ImportError:
    # Fallback import for ComfyUI environments
    import importlib.util
    
    # Import constants
    constants_path = os.path.join(parent_dir, "shared", "constants.py")
    spec = importlib.util.spec_from_file_location("constants", constants_path)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    
    MODEL_RESOLUTIONS = constants.MODEL_RESOLUTIONS
    ASPECT_RATIOS = constants.ASPECT_RATIOS
    LATENT_CATEGORY = constants.LATENT_CATEGORY
    DEFAULT_MULTIPLIER = constants.DEFAULT_MULTIPLIER
    MIN_MULTIPLIER = constants.MIN_MULTIPLIER
    MAX_MULTIPLIER = constants.MAX_MULTIPLIER
    MULTIPLIER_STEP = constants.MULTIPLIER_STEP
    DEFAULT_BATCH_SIZE = constants.DEFAULT_BATCH_SIZE
    MIN_BATCH_SIZE = constants.MIN_BATCH_SIZE
    MAX_BATCH_SIZE = constants.MAX_BATCH_SIZE
    DEFAULT_ORIENTATION = constants.DEFAULT_ORIENTATION
    ORIENTATIONS = constants.ORIENTATIONS
    
    # Import helpers
    helpers_path = os.path.join(parent_dir, "shared", "helpers.py")
    spec = importlib.util.spec_from_file_location("helpers", helpers_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    
    parse_aspect_ratio = helpers.parse_aspect_ratio
    calculate_final_dimensions = helpers.calculate_final_dimensions


class CYHLatentFluxAspectRatio:
    """
    Generates empty latent images with Flux-specific aspect ratios and resolutions
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (ASPECT_RATIOS["FLUX"], {"default": "16:9 (Widescreen) - 1344×768"}),
                "orientation": (ORIENTATIONS, {"default": DEFAULT_ORIENTATION}),
                "multiplier": ("FLOAT", {"default": DEFAULT_MULTIPLIER, "min": MIN_MULTIPLIER, "max": MAX_MULTIPLIER, "step": MULTIPLIER_STEP}),
                "batch_size": ("INT", {"default": DEFAULT_BATCH_SIZE, "min": MIN_BATCH_SIZE, "max": MAX_BATCH_SIZE}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = LATENT_CATEGORY
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def generate(self, aspect_ratio, orientation, multiplier, batch_size):
        # Extract actual aspect ratio from dropdown text
        actual_aspect_ratio = parse_aspect_ratio(aspect_ratio)
        
        # Get base resolution for Flux
        width, height = MODEL_RESOLUTIONS["FLUX"][actual_aspect_ratio]
        
        # Calculate final dimensions with orientation and multiplier
        final_width, final_height = calculate_final_dimensions(width, height, orientation, multiplier)
        
        # Generate empty latent
        latent = torch.zeros([batch_size, 4, final_height // 8, final_width // 8])
        
        return ({"samples": latent},)


class CYHLatentQwenAspectRatio:
    """
    Generates empty latent images with Qwen Image-specific aspect ratios and resolutions
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (ASPECT_RATIOS["QWEN"], {"default": "16:9 (Widescreen) - 1664×928"}),
                "orientation": (ORIENTATIONS, {"default": DEFAULT_ORIENTATION}),
                "multiplier": ("FLOAT", {"default": DEFAULT_MULTIPLIER, "min": MIN_MULTIPLIER, "max": MAX_MULTIPLIER, "step": MULTIPLIER_STEP}),
                "batch_size": ("INT", {"default": DEFAULT_BATCH_SIZE, "min": MIN_BATCH_SIZE, "max": MAX_BATCH_SIZE}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = LATENT_CATEGORY
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def generate(self, aspect_ratio, orientation, multiplier, batch_size):
        # Extract actual aspect ratio from dropdown text
        actual_aspect_ratio = parse_aspect_ratio(aspect_ratio)
        
        # Get base resolution for Qwen
        width, height = MODEL_RESOLUTIONS["QWEN"][actual_aspect_ratio]
        
        # Calculate final dimensions with orientation and multiplier
        final_width, final_height = calculate_final_dimensions(width, height, orientation, multiplier)
        
        # Generate empty latent
        latent = torch.zeros([batch_size, 4, final_height // 8, final_width // 8])
        
        return ({"samples": latent},)


class CYHLatentSDXLAspectRatio:
    """
    Generates empty latent images with SDXL-specific aspect ratios and resolutions
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (ASPECT_RATIOS["SDXL"], {"default": "16:9 (Widescreen) - 1024×576"}),
                "orientation": (ORIENTATIONS, {"default": DEFAULT_ORIENTATION}),
                "multiplier": ("FLOAT", {"default": DEFAULT_MULTIPLIER, "min": MIN_MULTIPLIER, "max": MAX_MULTIPLIER, "step": MULTIPLIER_STEP}),
                "batch_size": ("INT", {"default": DEFAULT_BATCH_SIZE, "min": MIN_BATCH_SIZE, "max": MAX_BATCH_SIZE}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = LATENT_CATEGORY
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def generate(self, aspect_ratio, orientation, multiplier, batch_size):
        # Extract actual aspect ratio from dropdown text
        actual_aspect_ratio = parse_aspect_ratio(aspect_ratio)
        
        # Get base resolution for SDXL
        width, height = MODEL_RESOLUTIONS["SDXL"][actual_aspect_ratio]
        
        # Calculate final dimensions with orientation and multiplier
        final_width, final_height = calculate_final_dimensions(width, height, orientation, multiplier)
        
        # Generate empty latent
        latent = torch.zeros([batch_size, 4, final_height // 8, final_width // 8])
        
        return ({"samples": latent},)


# Node registration for this category
NODE_CLASS_MAPPINGS = {
    "CYHLatentFluxAspectRatio": CYHLatentFluxAspectRatio,
    "CYHLatentQwenAspectRatio": CYHLatentQwenAspectRatio,
    "CYHLatentSDXLAspectRatio": CYHLatentSDXLAspectRatio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CYHLatentFluxAspectRatio": "🔹 CYH Latent | Flux Aspect Ratio",
    "CYHLatentQwenAspectRatio": "🔹 CYH Latent | Qwen Aspect Ratio", 
    "CYHLatentSDXLAspectRatio": "🔹 CYH Latent | SDXL Aspect Ratio"
}