"""
Constants and configuration data for Chye ComfyUI Toolset
"""

# Model-specific aspect ratio resolutions
MODEL_RESOLUTIONS = {
    "FLUX": {
        "1:1": (1024, 1024),
        "4:3": (1280, 960),
        "3:2": (1152, 768),
        "16:9": (1344, 768),
        "21:9": (1792, 768),
    },
    "QWEN": {
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

# Aspect ratio display options for each model
ASPECT_RATIOS = {
    "FLUX": [
        "1:1 (Square) - 1024×1024",
        "4:3 (Standard) - 1280×960",
        "3:2 (Photo) - 1152×768", 
        "16:9 (Widescreen) - 1344×768",
        "21:9 (Ultrawide) - 1792×768"
    ],
    "QWEN": [
        "1:1 (Square) - 1328×1328",
        "4:3 (Standard) - 1472×1140",
        "3:2 (Photo) - 1536×1024",
        "16:9 (Widescreen) - 1664×928",
        "21:9 (Ultrawide) - 1984×864"
    ],
    "SDXL": [
        "1:1 (Square) - 1024×1024",
        "4:3 (Standard) - 1024×768",
        "3:2 (Photo) - 1152×768",
        "16:9 (Widescreen) - 1024×576",
        "21:9 (Ultrawide) - 1344×576"
    ]
}

# Common settings
DEFAULT_MULTIPLIER = 1.0
MIN_MULTIPLIER = 0.1
MAX_MULTIPLIER = 10.0
MULTIPLIER_STEP = 0.1

DEFAULT_BATCH_SIZE = 1
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 64

DEFAULT_ORIENTATION = "Portrait"
ORIENTATIONS = ["Portrait", "Landscape"]

# Node categories
LATENT_CATEGORY = "latent"