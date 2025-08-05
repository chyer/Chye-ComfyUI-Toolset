import torch
import math
from comfy.sd import VAE
from nodes import common_ksampler

class CYHFluxASPLatentGenerator:
    """
    Generates empty latent images with Flux-specific aspect ratios and resolutions
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ([
                    "1:1 (Square) - 1024×1024",
                    "4:3 (Standard) - 1280×960",
                    "3:2 (Photo) - 1152×768", 
                    "16:9 (Widescreen) - 1344×768",
                    "21:9 (Ultrawide) - 1792×768"
                ], {"default": "16:9 (Widescreen) - 1344×768"}),
                "orientation": (["Portrait", "Landscape"], {"default": "Portrait"}),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "latent"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    MODEL_RESOLUTIONS = {
        "1:1": (1024, 1024),
        "4:3": (1280, 960),
        "3:2": (1152, 768),
        "16:9": (1344, 768),
        "21:9": (1792, 768),
    }

    def round_to_multiple(self, value, multiple=32):
        """Round to nearest multiple of 32"""
        return multiple * round(value / multiple)

    def generate(self, aspect_ratio, orientation, multiplier, batch_size):
        # Extract actual aspect ratio from dropdown text
        actual_aspect_ratio = aspect_ratio.split(" ")[0]
        
        # Get base resolution
        width, height = self.MODEL_RESOLUTIONS[actual_aspect_ratio]
        
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

class CYHQwenASPLatentGenerator:
    """
    Generates empty latent images with Qwen Image-specific aspect ratios and resolutions
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ([
                    "1:1 (Square) - 1328×1328",
                    "4:3 (Standard) - 1472×1140",
                    "3:2 (Photo) - 1536×1024",
                    "16:9 (Widescreen) - 1664×928",
                    "21:9 (Ultrawide) - 1984×864"
                ], {"default": "16:9 (Widescreen) - 1664×928"}),
                "orientation": (["Portrait", "Landscape"], {"default": "Portrait"}),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "latent"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    MODEL_RESOLUTIONS = {
        "1:1": (1328, 1328),
        "4:3": (1472, 1140),
        "3:2": (1536, 1024),
        "16:9": (1664, 928),
        "21:9": (1984, 864),
    }

    def round_to_multiple(self, value, multiple=32):
        """Round to nearest multiple of 32"""
        return multiple * round(value / multiple)

    def generate(self, aspect_ratio, orientation, multiplier, batch_size):
        # Extract actual aspect ratio from dropdown text
        actual_aspect_ratio = aspect_ratio.split(" ")[0]
        
        # Get base resolution
        width, height = self.MODEL_RESOLUTIONS[actual_aspect_ratio]
        
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

class CYHSDXLASPLatentGenerator:
    """
    Generates empty latent images with SDXL-specific aspect ratios and resolutions
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ([
                    "1:1 (Square) - 1024×1024",
                    "4:3 (Standard) - 1024×768",
                    "3:2 (Photo) - 1152×768",
                    "16:9 (Widescreen) - 1024×576",
                    "21:9 (Ultrawide) - 1344×576"
                ], {"default": "16:9 (Widescreen) - 1024×576"}),
                "orientation": (["Portrait", "Landscape"], {"default": "Portrait"}),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "latent"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    MODEL_RESOLUTIONS = {
        "1:1": (1024, 1024),
        "4:3": (1024, 768),
        "3:2": (1152, 768),
        "16:9": (1024, 576),
        "21:9": (1344, 576),
    }

    def round_to_multiple(self, value, multiple=32):
        """Round to nearest multiple of 32"""
        return multiple * round(value / multiple)

    def generate(self, aspect_ratio, orientation, multiplier, batch_size):
        # Extract actual aspect ratio from dropdown text
        actual_aspect_ratio = aspect_ratio.split(" ")[0]
        
        # Get base resolution
        width, height = self.MODEL_RESOLUTIONS[actual_aspect_ratio]
        
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

WEB_DIRECTORY = "./web"
__version__ = "1.0.0"

NODE_CLASS_MAPPINGS = {
    "CYHFluxASPLatentGenerator": CYHFluxASPLatentGenerator,
    "CYHQwenASPLatentGenerator": CYHQwenASPLatentGenerator,
    "CYHSDXLASPLatentGenerator": CYHSDXLASPLatentGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CYHFluxASPLatentGenerator": "🔹 CYH Flux Aspect Ratio",
    "CYHQwenASPLatentGenerator": "🔹 CYH Qwen Aspect Ratio", 
    "CYHSDXLASPLatentGenerator": "🔹 CYH SDXL Aspect Ratio"
}
