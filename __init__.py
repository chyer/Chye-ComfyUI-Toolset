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
    from categories.math_tools import (
        NODE_CLASS_MAPPINGS as MATH_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as MATH_DISPLAY_MAPPINGS
    )
    from categories.file_tools import (
        NODE_CLASS_MAPPINGS as FILE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as FILE_DISPLAY_MAPPINGS
    )
    from categories.post_process import (
        NODE_CLASS_MAPPINGS as POST_PROCESS_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as POST_PROCESS_DISPLAY_MAPPINGS
    )
    from categories.prompt_tools import (
        NODE_CLASS_MAPPINGS as PROMPT_TOOLS_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as PROMPT_TOOLS_DISPLAY_MAPPINGS
    )
    from categories.image_tools import (
        NODE_CLASS_MAPPINGS as IMAGE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as IMAGE_DISPLAY_MAPPINGS
    )
except ImportError:
    # Fallback for ComfyUI environments
    import importlib.util
    
    # Import latent tools
    spec = importlib.util.spec_from_file_location("latent_tools", os.path.join(current_dir, "categories", "latent_tools.py"))
    latent_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(latent_tools)
    
    LATENT_CLASS_MAPPINGS = latent_tools.NODE_CLASS_MAPPINGS
    LATENT_DISPLAY_MAPPINGS = latent_tools.NODE_DISPLAY_NAME_MAPPINGS
    
    # Import math tools
    spec = importlib.util.spec_from_file_location("math_tools", os.path.join(current_dir, "categories", "math_tools.py"))
    math_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_tools)
    
    MATH_CLASS_MAPPINGS = math_tools.NODE_CLASS_MAPPINGS
    MATH_DISPLAY_MAPPINGS = math_tools.NODE_DISPLAY_NAME_MAPPINGS
    
    # Import file tools
    spec = importlib.util.spec_from_file_location("file_tools", os.path.join(current_dir, "categories", "file_tools.py"))
    file_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(file_tools)
    
    FILE_CLASS_MAPPINGS = file_tools.NODE_CLASS_MAPPINGS
    FILE_DISPLAY_MAPPINGS = file_tools.NODE_DISPLAY_NAME_MAPPINGS
    
    # Import post process tools
    spec = importlib.util.spec_from_file_location("post_process", os.path.join(current_dir, "categories", "post_process.py"))
    post_process = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(post_process)
    
    POST_PROCESS_CLASS_MAPPINGS = post_process.NODE_CLASS_MAPPINGS
    POST_PROCESS_DISPLAY_MAPPINGS = post_process.NODE_DISPLAY_NAME_MAPPINGS
    
    # Import prompt tools
    spec = importlib.util.spec_from_file_location("prompt_tools", os.path.join(current_dir, "categories", "prompt_tools.py"))
    prompt_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_tools)
    
    PROMPT_TOOLS_CLASS_MAPPINGS = prompt_tools.NODE_CLASS_MAPPINGS
    PROMPT_TOOLS_DISPLAY_MAPPINGS = prompt_tools.NODE_DISPLAY_NAME_MAPPINGS
    
    # Import image tools
    spec = importlib.util.spec_from_file_location("image_tools", os.path.join(current_dir, "categories", "image_tools.py"))
    image_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(image_tools)
    
    IMAGE_CLASS_MAPPINGS = image_tools.NODE_CLASS_MAPPINGS
    IMAGE_DISPLAY_MAPPINGS = image_tools.NODE_DISPLAY_NAME_MAPPINGS

# Combine all category mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add latent tools
NODE_CLASS_MAPPINGS.update(LATENT_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LATENT_DISPLAY_MAPPINGS)

# Add math tools
NODE_CLASS_MAPPINGS.update(MATH_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MATH_DISPLAY_MAPPINGS)

# Add file tools
NODE_CLASS_MAPPINGS.update(FILE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(FILE_DISPLAY_MAPPINGS)

# Add post process tools
NODE_CLASS_MAPPINGS.update(POST_PROCESS_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(POST_PROCESS_DISPLAY_MAPPINGS)

# Add prompt tools
NODE_CLASS_MAPPINGS.update(PROMPT_TOOLS_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(PROMPT_TOOLS_DISPLAY_MAPPINGS)

# Add image tools
NODE_CLASS_MAPPINGS.update(IMAGE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_DISPLAY_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']