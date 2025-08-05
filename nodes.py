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
        # Create comprehensive aspect ratio options with all model resolutions
        aspect_options = [
            "1:1 (Square) | Flux:1024×1024 | Qwen:1328×1328 | SDXL:1024×1024",
            "4:3 (Standard) | Flux:1280×960 | Qwen:1472×1140 | SDXL:1024×768",
            "3:2 (Photo) | Flux:1152×768 | Qwen:1536×1024 | SDXL:1152×768",
            "16:9 (Widescreen) | Flux:1344×768 | Qwen:1664×928 | SDXL:1024×576",
            "21:9 (Ultrawide) | Flux:1792×768 | Qwen:1984×864 | SDXL:1344×576"
        ]
        
        return {
            "required": {
                "model_type": (["Flux", "Qwen Image", "SDXL"], {"default": "Flux"}),
                "aspect_ratio": (aspect_options, {"default": "16:9 (Widescreen) | Flux:1344×768 | Qwen:1664×928 | SDXL:1024×576"}),
                "orientation": (["Portrait", "Landscape"], {"default": "Portrait"}),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, model_type, aspect_ratio, **kwargs):
        return True

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

    def get_resolution_text(self, model_type, aspect_ratio):
        """Get resolution text for display purposes"""
        if model_type not in self.MODEL_RESOLUTIONS or aspect_ratio not in self.MODEL_RESOLUTIONS[model_type]:
            return ""
        
        width, height = self.MODEL_RESOLUTIONS[model_type][aspect_ratio]
        aspect_names = {
            "1:1": "Square",
            "4:3": "Standard",
            "3:2": "Photo",
            "16:9": "Widescreen",
            "21:9": "Ultrawide"
        }
        
        name = aspect_names.get(aspect_ratio, "")
        return f"{aspect_ratio} ({name}) → {width}×{height}"

    def generate(self, model_type, aspect_ratio, orientation, multiplier, batch_size):
        # Extract actual aspect ratio from dropdown text (e.g., "1:1 (Square) | Flux:1024×1024..." -> "1:1")
        actual_aspect_ratio = aspect_ratio.split(" ")[0]
        
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