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
        return {
            "required": {
                "model_type": (["Flux", "Qwen Image", "SDXL"], {"default": "Flux"}),
                "aspect_ratio": (["1:1", "4:3", "3:2", "16:9", "21:9"], {"default": "16:9"}),
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

    def generate(self, model_type, aspect_ratio, orientation, multiplier, batch_size):
        # Get base resolution
        width, height = self.MODEL_RESOLUTIONS[model_type][aspect_ratio]
        
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