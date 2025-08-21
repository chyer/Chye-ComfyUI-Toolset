"""
Post process tools for Chye ComfyUI Toolset
"""

import sys
import os
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

# Add parent directory to Python path for ComfyUI compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from shared.constants import POST_PROCESS_CATEGORY
except ImportError:
    # Fallback import for ComfyUI environments
    import importlib.util
    
    # Import constants
    constants_path = os.path.join(parent_dir, "shared", "constants.py")
    spec = importlib.util.spec_from_file_location("constants", constants_path)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    
    # Define POST_PROCESS_CATEGORY if not in constants
    if hasattr(constants, 'POST_PROCESS_CATEGORY'):
        POST_PROCESS_CATEGORY = constants.POST_PROCESS_CATEGORY
    else:
        POST_PROCESS_CATEGORY = "post_process"


def generate_correlated_noise(shape, grain_size):
    """Generate spatially correlated noise to simulate realistic film grain"""
    # Generate random noise
    noise = np.random.normal(0, 1, shape)
    
    # Apply Gaussian filter to create spatial correlation
    # The grain_size parameter controls the correlation radius
    sigma = max(0.5, grain_size / 10.0)  # Scale grain_size to appropriate sigma
    correlated_noise = gaussian_filter(noise, sigma=sigma)
    
    # Normalize to maintain consistent noise power
    correlated_noise = correlated_noise / np.std(correlated_noise)
    
    return correlated_noise


def apply_realistic_grain(image_tensor, strength, iso, grain_size, colored=False):
    """
    Apply realistic film grain to an image tensor
    
    Args:
        image_tensor: Input image tensor (B, H, W, C)
        strength: Overall grain strength (0.0 to 1.0)
        iso: ISO value to simulate (100 to 6400)
        grain_size: Size of grain particles (1.0 to 10.0)
        colored: Whether to apply colored grain (True/False)
    
    Returns:
        Tensor with applied film grain
    """
    # Convert to numpy
    image_np = image_tensor.cpu().numpy()
    
    # Generate grain mask based on luminance (shadows get more apparent grain)
    luminance = 0.299 * image_np[0,:,:,0] + 0.587 * image_np[0,:,:,1] + 0.114 * image_np[0,:,:,2]
    shadow_boost = 1.0 + (1.0 - luminance) * 0.5  # Boost grain in shadows
    
    # Generate spatially correlated noise
    grain_pattern = generate_correlated_noise(image_np.shape[1:3], grain_size)
    
    # Apply ISO-based scaling
    iso_factor = np.log10(iso / 100)
    final_grain = grain_pattern * iso_factor * strength * shadow_boost
    
    # Apply to image
    if colored:
        # Generate separate grain patterns for each color channel
        grain_r = generate_correlated_noise(image_np.shape[1:3], grain_size)
        grain_g = generate_correlated_noise(image_np.shape[1:3], grain_size)
        grain_b = generate_correlated_noise(image_np.shape[1:3], grain_size)
        
        # Apply ISO-based scaling for each channel
        grain_r = grain_r * iso_factor * strength * shadow_boost
        grain_g = grain_g * iso_factor * strength * shadow_boost
        grain_b = grain_b * iso_factor * strength * shadow_boost
        
        # Apply colored grain to each channel
        result = image_np.copy()
        result[0,:,:,0] = image_np[0,:,:,0] + grain_r
        result[0,:,:,1] = image_np[0,:,:,1] + grain_g
        result[0,:,:,2] = image_np[0,:,:,2] + grain_b
    else:
        # Apply monochrome grain to all channels
        result = image_np.copy()
        for c in range(image_np.shape[3]):
            result[0,:,:,c] = image_np[0,:,:,c] + final_grain
    
    # Clip values to valid range
    result = np.clip(result, 0.0, 1.0)
    
    # Convert back to tensor
    return torch.from_numpy(result).to(image_tensor.device)


class CYHFilmGrainNode:
    """
    A post-processing node that applies realistic film grain to images.
    Simulates the look of photographic film with customizable grain characteristics.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "iso": ("INT", {"default": 400, "min": 100, "max": 6400, "step": 100}),
                "grain_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "colored": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_grain"
    CATEGORY = POST_PROCESS_CATEGORY
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def apply_grain(self, image, strength, iso, grain_size, colored):
        # Apply realistic film grain to the image
        result = apply_realistic_grain(image, strength, iso, grain_size, colored)
        return (result,)


def arri_halation_effect(image_tensor, threshold=220, blur_size=25, intensity=0.6):
    """
    Apply ARRI-style halation effect to simulate film bloom around highlights
    
    Args:
        image_tensor: Input image tensor (B, H, W, C)
        threshold: Highlight threshold (0-255)
        blur_size: Gaussian blur kernel size (odd number)
        intensity: Halation effect strength (0.0 to 1.0)
    
    Returns:
        Tensor with applied halation effect
    """
    import cv2
    import numpy as np
    
    # Convert to numpy
    image_np = image_tensor.cpu().numpy()
    
    # Convert from (B, H, W, C) to (H, W, C) for single image processing
    if len(image_np.shape) == 4:
        image_np = image_np[0]  # Take first image in batch
    
    # Convert from float [0,1] to uint8 [0,255]
    image_uint8 = (image_np * 255).astype(np.uint8)
    
    # Convert BGR to RGB (OpenCV uses BGR, ComfyUI uses RGB)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    
    # Convert to float32 for precision
    img_float = image_bgr.astype(np.float32) / 255.0

    # Step 1: Extract Highlights (luminance)
    lum = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, highlights = cv2.threshold(lum, threshold, 255, cv2.THRESH_BINARY)
    highlights = highlights.astype(np.float32) / 255.0

    # Step 2: Create halation color mask (red-orange tint)
    # Start with zero image, add shifted values in R and G channels
    halation_mask = np.zeros_like(img_float)
    halation_mask[:, :, 2] = highlights * 1.0     # enhance red channel (BGR order: B=0, G=1, R=2)
    halation_mask[:, :, 1] = highlights * 0.5     # add some green for orange tint

    # Step 3: Blur the mask to create glow
    # Ensure blur_size is odd
    blur_size = max(1, blur_size)
    if blur_size % 2 == 0:
        blur_size += 1
        
    glow = cv2.GaussianBlur(halation_mask, (blur_size, blur_size), 0)

    # Step 4: Additive blend glow onto original
    result = np.clip(img_float + glow * intensity, 0, 1)

    # Convert back to RGB and uint8
    result_uint8 = (result * 255).astype(np.uint8)
    result_rgb = cv2.cvtColor(result_uint8, cv2.COLOR_BGR2RGB)
    
    # Convert back to float [0,1] and restore batch dimension
    result_float = result_rgb.astype(np.float32) / 255.0
    result_float = np.expand_dims(result_float, axis=0)  # Add batch dimension
    
    # Convert back to tensor
    return torch.from_numpy(result_float).to(image_tensor.device)


class CYHARRHalationNode:
    """
    A post-processing node that applies ARRI-style halation effect to images.
    Simulates the film bloom around highlights with customizable parameters.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("INT", {"default": 220, "min": 0, "max": 255, "step": 1}),
                "blur_size": ("INT", {"default": 25, "min": 1, "max": 101, "step": 2}),
                "intensity": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_halation"
    CATEGORY = POST_PROCESS_CATEGORY
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def apply_halation(self, image, threshold, blur_size, intensity):
        # Apply ARRI halation effect to the image
        result = arri_halation_effect(image, threshold, blur_size, intensity)
        return (result,)


import cv2
import functools
import time
from collections import OrderedDict

# Cache for distortion maps to avoid recomputation
_DISTORTION_CACHE = OrderedDict()
_CACHE_MAX_SIZE = 10  # Keep last 10 distortion maps in cache

def _get_cache_key(shape, k1, k2, k3, center_x, center_y):
    """Generate a unique cache key for distortion parameters"""
    return f"{shape[0]}x{shape[1]}_{k1:.6f}_{k2:.6f}_{k3:.6f}_{center_x:.3f}_{center_y:.3f}"

def _clean_cache():
    """Clean cache if it exceeds maximum size"""
    while len(_DISTORTION_CACHE) > _CACHE_MAX_SIZE:
        _DISTORTION_CACHE.popitem(last=False)

def apply_barrel_distortion(image_tensor, k1, k2, k3, center_x=0.5, center_y=0.5, interpolation=cv2.INTER_LINEAR):
    """
    Apply barrel distortion to an image tensor using OpenCV remap
    
    Args:
        image_tensor: Input image tensor (B, H, W, C)
        k1, k2, k3: Barrel distortion coefficients
        center_x, center_y: Normalized center coordinates (0.0-1.0)
        interpolation: OpenCV interpolation method
        
    Returns:
        Tensor with applied barrel distortion
    """
    import cv2
    import numpy as np
    
    # Convert to numpy
    image_np = image_tensor.cpu().numpy()
    
    # Convert from (B, H, W, C) to (H, W, C) for single image processing
    if len(image_np.shape) == 4:
        image_np = image_np[0]  # Take first image in batch
    
    height, width = image_np.shape[:2]
    
    # Calculate actual center coordinates
    cx = int(center_x * width)
    cy = int(center_y * height)
    
    # Generate cache key
    cache_key = _get_cache_key((height, width), k1, k2, k3, center_x, center_y)
    
    # Check cache first
    if cache_key in _DISTORTION_CACHE:
        map_x, map_y = _DISTORTION_CACHE[cache_key]
        # Move to end to mark as recently used
        _DISTORTION_CACHE.move_to_end(cache_key)
    else:
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Normalize coordinates to [-1, 1] range relative to center
        x_norm = (x - cx) / (width / 2)
        y_norm = (y - cy) / (height / 2)
        
        # Calculate radius from center
        r = np.sqrt(x_norm**2 + y_norm**2)
        
        # Apply barrel distortion formula: r' = r * (1 + k1*r^2 + k2*r^4 + k3*r^6)
        r_distorted = r * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
        
        # Avoid division by zero
        r_distorted = np.where(r == 0, 0, r_distorted)
        scale = np.where(r == 0, 0, r_distorted / r)
        
        # Calculate distorted coordinates
        x_distorted = scale * x_norm * (width / 2) + cx
        y_distorted = scale * y_norm * (height / 2) + cy
        
        # Create mapping arrays for remap
        map_x = x_distorted.astype(np.float32)
        map_y = y_distorted.astype(np.float32)
        
        # Store in cache
        _DISTORTION_CACHE[cache_key] = (map_x, map_y)
        _clean_cache()
    
    # Apply distortion using remap
    # Convert from RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    distorted_bgr = cv2.remap(image_bgr, map_x, map_y, interpolation)
    
    # Convert back to RGB
    distorted_rgb = cv2.cvtColor(distorted_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert back to float [0,1] and restore batch dimension
    result_float = distorted_rgb.astype(np.float32) / 255.0
    result_float = np.expand_dims(result_float, axis=0)  # Add batch dimension
    
    # Convert back to tensor
    return torch.from_numpy(result_float).to(image_tensor.device)

def apply_chromatic_aberration(image_tensor, preset="none", intensity=1.0,
                              k1_r=0.0, k2_r=0.0, k3_r=0.0,
                              k1_g=0.0, k2_g=0.0, k3_g=0.0,
                              k1_b=0.0, k2_b=0.0, k3_b=0.0,
                              center_x=0.5, center_y=0.5, quality="fast"):
    """
    Apply chromatic aberration effect using barrel distortion per RGB channel
    
    Args:
        image_tensor: Input image tensor
        preset: Preset configuration ("none", "vintage", "modern", "extreme")
        intensity: Master intensity multiplier (0.0-2.0)
        k1_r, k2_r, k3_r: Red channel distortion coefficients
        k1_g, k2_g, k3_g: Green channel distortion coefficients
        k1_b, k2_b, k3_b: Blue channel distortion coefficients
        center_x, center_y: Normalized center coordinates (0.0-1.0)
        quality: Interpolation quality ("fast" or "high")
        
    Returns:
        Tensor with applied chromatic aberration
    """
    # Apply preset configurations
    if preset != "custom":
        if preset == "none":
            k1_r = k2_r = k3_r = 0.0
            k1_g = k2_g = k3_g = 0.0
            k1_b = k2_b = k3_b = 0.0
        elif preset == "vintage":
            k1_r, k2_r, k3_r = 0.15, 0.0, 0.0
            k1_g, k2_g, k3_g = 0.0, 0.0, 0.0
            k1_b, k2_b, k3_b = -0.15, 0.0, 0.0
        elif preset == "modern":
            k1_r, k2_r, k3_r = 0.08, -0.02, 0.0
            k1_g, k2_g, k3_g = 0.0, 0.0, 0.0
            k1_b, k2_b, k3_b = -0.08, 0.02, 0.0
        elif preset == "extreme":
            k1_r, k2_r, k3_r = 0.25, -0.1, 0.02
            k1_g, k2_g, k3_g = 0.0, 0.0, 0.0
            k1_b, k2_b, k3_b = -0.25, 0.1, -0.02
    
    # Apply intensity multiplier
    k1_r *= intensity; k2_r *= intensity; k3_r *= intensity
    k1_g *= intensity; k2_g *= intensity; k3_g *= intensity
    k1_b *= intensity; k2_b *= intensity; k3_b *= intensity
    
    # Choose interpolation method
    interpolation = cv2.INTER_LINEAR if quality == "fast" else cv2.INTER_CUBIC
    
    # Convert to numpy and split channels
    image_np = image_tensor.cpu().numpy()
    if len(image_np.shape) == 4:
        image_np = image_np[0]  # Take first image in batch
    
    # Process each channel separately
    channels = []
    for i, (k1, k2, k3) in enumerate([(k1_r, k2_r, k3_r),
                                     (k1_g, k2_g, k3_g),
                                     (k1_b, k2_b, k3_b)]):
        # Create single channel image
        channel_img = np.zeros_like(image_np)
        channel_img[:, :, i] = image_np[:, :, i]
        
        # Apply distortion to this channel
        channel_tensor = torch.from_numpy(channel_img).to(image_tensor.device)
        distorted_channel = apply_barrel_distortion(
            channel_tensor, k1, k2, k3, center_x, center_y, interpolation
        )
        
        # Extract the distorted channel
        distorted_np = distorted_channel.cpu().numpy()[0]
        channels.append(distorted_np[:, :, i:i+1])
    
    # Combine channels
    result_np = np.concatenate(channels, axis=2)
    result_np = np.expand_dims(result_np, axis=0)  # Add batch dimension
    
    # Convert back to tensor
    return torch.from_numpy(result_np).to(image_tensor.device)


class CYHChromaticAberrationNode:
    """
    A post-processing node that applies realistic chromatic aberration to images.
    Simulates lens color fringing with barrel distortion per RGB channel.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["none", "vintage", "modern", "extreme", "custom"], {"default": "vintage"}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "quality": (["fast", "high"], {"default": "fast"}),
                "k1_r": ("FLOAT", {"default": 0.15, "min": -1.0, "max": 1.0, "step": 0.001}),
                "k2_r": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.001}),
                "k3_r": ("FLOAT", {"default": 0.0, "min": -0.2, "max": 0.2, "step": 0.001}),
                "k1_g": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "k2_g": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.001}),
                "k3_g": ("FLOAT", {"default": 0.0, "min": -0.2, "max": 0.2, "step": 0.001}),
                "k1_b": ("FLOAT", {"default": -0.15, "min": -1.0, "max": 1.0, "step": 0.001}),
                "k2_b": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.001}),
                "k3_b": ("FLOAT", {"default": 0.0, "min": -0.2, "max": 0.2, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_chromatic_aberration"
    CATEGORY = POST_PROCESS_CATEGORY
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def apply_chromatic_aberration(self, image, preset="vintage", intensity=1.0,
                                  center_x=0.5, center_y=0.5, quality="fast",
                                  k1_r=0.15, k2_r=0.0, k3_r=0.0,
                                  k1_g=0.0, k2_g=0.0, k3_g=0.0,
                                  k1_b=-0.15, k2_b=0.0, k3_b=0.0):
        # Apply chromatic aberration effect
        result = apply_chromatic_aberration(
            image, preset, intensity,
            k1_r, k2_r, k3_r,
            k1_g, k2_g, k3_g,
            k1_b, k2_b, k3_b,
            center_x, center_y, quality
        )
        return (result,)


# Node registration for this category
NODE_CLASS_MAPPINGS = {
    "CYHFilmGrainNode": CYHFilmGrainNode,
    "CYHARRHalationNode": CYHARRHalationNode,
    "CYHChromaticAberrationNode": CYHChromaticAberrationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CYHFilmGrainNode": "🎬 CYH Post Process | Film Grain",
    "CYHARRHalationNode": "🎬 CYH Post Process | ARRI Halation",
    "CYHChromaticAberrationNode": "🌈 CYH Post Process | Chromatic Aberration",
}