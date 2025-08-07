"""
Helper utility functions for Chye ComfyUI Toolset
"""

def round_to_multiple(value: float, multiple: int = 32) -> int:
    """Round value to nearest multiple (default 32 for ComfyUI)"""
    return multiple * round(value / multiple)

def parse_aspect_ratio(aspect_ratio_text: str) -> str:
    """Extract actual aspect ratio from dropdown text (e.g., '16:9 (Widescreen) - 1344×768' -> '16:9')"""
    return aspect_ratio_text.split(" ")[0]

def apply_orientation(width: int, height: int, orientation: str) -> tuple[int, int]:
    """Apply orientation to dimensions, swapping if needed"""
    if orientation == "Portrait" and width > height:
        return height, width
    elif orientation == "Landscape" and height > width:
        return height, width
    return width, height

def get_model_resolution(model: str, aspect_ratio: str, model_resolutions: dict) -> tuple[int, int]:
    """Get base resolution for a specific model and aspect ratio"""
    return model_resolutions.get(model, {}).get(aspect_ratio, (1024, 1024))

def calculate_final_dimensions(width: int, height: int, orientation: str, multiplier: float) -> tuple[int, int]:
    """Calculate final dimensions with orientation and multiplier applied"""
    # Apply orientation
    final_width, final_height = apply_orientation(width, height, orientation)
    
    # Apply multiplier and round to multiples of 32
    final_width = round_to_multiple(final_width * multiplier)
    final_height = round_to_multiple(final_height * multiplier)
    
    return final_width, final_height