"""
Chye ComfyUI Toolset - A comprehensive collection of ComfyUI custom nodes
"""

import sys
import os

# Add current directory to Python path for ComfyUI compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from categories.latent_tools import (
        NODE_CLASS_MAPPINGS as LATENT_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as LATENT_DISPLAY_MAPPINGS
    )
except ImportError:
    # Fallback for ComfyUI environments
    import importlib.util
    spec = importlib.util.spec_from_file_location("latent_tools", os.path.join(current_dir, "categories", "latent_tools.py"))
    latent_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(latent_tools)
    
    LATENT_CLASS_MAPPINGS = latent_tools.NODE_CLASS_MAPPINGS
    LATENT_DISPLAY_MAPPINGS = latent_tools.NODE_DISPLAY_NAME_MAPPINGS

# Combine all category mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add latent tools
NODE_CLASS_MAPPINGS.update(LATENT_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LATENT_DISPLAY_MAPPINGS)

# Future categories will be added here:
# from categories.image_tools import NODE_CLASS_MAPPINGS as IMAGE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as IMAGE_DISPLAY_MAPPINGS
# NODE_CLASS_MAPPINGS.update(IMAGE_CLASS_MAPPINGS)
# NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_DISPLAY_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']