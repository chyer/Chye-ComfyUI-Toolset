import torch
import math
from comfy.sd import VAE
from nodes import common_ksampler

class ASPLatentGenerator:
    """
    Generates empty latent images with model-specific aspect ratios and resolutions
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Create model-specific aspect ratio options
        flux_options = [
            "1:1 (Square) - 1024×1024",
            "4:3 (Standard) - 1280×960",
            "3:2 (Photo) - 1152×768",
            "16:9 (Widescreen) - 1344×768",
            "21:9 (Ultrawide) - 1792×768"
        ]
        
        qwen_options = [
            "1:1 (Square) - 1328×1328",
            "4:3 (Standard) - 1472×1140",
            "3:2 (Photo) - 1536×1024",
            "16:9 (Widescreen) - 1664×928",
            "21:9 (Ultrawide) - 1984×864"
        ]
        
        sdxl_options = [
            "1:1 (Square) - 1024×1024",
            "4:3 (Standard) - 1024×768",
            "3:2 (Photo) - 1152×768",
            "16:9 (Widescreen) - 1024×576",
            "21:9 (Ultrawide) - 1344×576"
        ]
        
        return {
            "required": {
                "model_type": (["Flux", "Qwen Image", "SDXL"], {"default": "Flux"}),
                "flux_aspect_ratio": (flux_options, {"default": "16:9 (Widescreen) - 1344×768"}),
                "qwen_aspect_ratio": (qwen_options, {"default": "16:9 (Widescreen) - 1664×928"}),
                "sdxl_aspect_ratio": (sdxl_options, {"default": "16:9 (Widescreen) - 1024×576"}),
                "orientation": (["Portrait", "Landscape"], {"default": "Portrait"}),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "latent"

    MODEL_RESOLUTIONS = {
        "Flux": {
            "1:1": (1024, 1024),
            "4:3": (1280, 960),
            "3:2": (1152, 768),
            "16:9": (1344, 768),
            "21:9": (1792, 768),
        },
        "Qwen Image": {
            "1:1": (1328, 1328),
            "4:3": (1472, 1140),
            "3:2": (1536, 1024),
            "16:9": (1664, 928),
            "21:9": (1984, 864),
        },
        "SDXL": {
            "1:1": (1024, 1024),
            "4:3": (1024, 768),
            "3:2": (1152, 768),
            "16:9": (1024, 576),
            "21:9": (1344, 576),
        }
    }

    def round_to_multiple(self, value, multiple=32):
        """Round to nearest multiple of 32"""
        return multiple * round(value / multiple)

    def generate(self, model_type, flux_aspect_ratio, qwen_aspect_ratio, sdxl_aspect_ratio, orientation, multiplier, batch_size):
        # Select the appropriate aspect ratio based on model type
        if model_type == "Flux":
            selected_aspect_text = flux_aspect_ratio
        elif model_type == "Qwen Image":
            selected_aspect_text = qwen_aspect_ratio
        else:  # SDXL
            selected_aspect_text = sdxl_aspect_ratio
        
        # Extract actual aspect ratio from dropdown text (e.g., "1:1 (Square) - 1024×1024" -> "1:1")
        actual_aspect_ratio = selected_aspect_text.split(" ")[0]
        
        # Get base resolution
        width, height = self.MODEL_RESOLUTIONS[model_type][actual_aspect_ratio]
        
        # Apply orientation
        if orientation == "Portrait" and width > height:
            width, height = height, width
        elif orientation == "Landscape" and height > width:
            width, height = height, width
        
        # Apply multiplier and round to multiples of 32
        width = self.round_to_multiple(width * multiplier)
        height = self.round_to_multiple(height * multiplier)
        
        # Generate empty latent
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        
        return ({"samples": latent},)

NODE_CLASS_MAPPINGS = {
    "ASPLatentGenerator": ASPLatentGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASPLatentGenerator": "🔹 CYH Aspect Ratio"
}