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
MATH_CATEGORY = "math"
FILE_CATEGORY = "file"
POST_PROCESS_CATEGORY = "post_process"
PROMPT_TOOLS_CATEGORY = "prompt_tools"

# Math node constants
DEFAULT_MULTIPLIER_VALUE = 32
MIN_MULTIPLIER_VALUE = 1
MAX_MULTIPLIER_VALUE = 1024
MULTIPLIER_VALUE_STEP = 1

DEFAULT_INCREMENT = 1
MIN_INCREMENT = 1
MAX_INCREMENT = 100
INCREMENT_STEP = 1

# Phone screen resolutions
PHONE_RESOLUTIONS = {
    "16:9": (1080, 1920),
    "19.5:9": (1080, 2340),
    "20:9": (1080, 2400),
    "18:9": (1080, 2160)
}

# Phone aspect ratio display options
PHONE_ASPECT_RATIOS = [
    "16:9 (Standard) - 1080×1920",
    "19.5:9 (Modern Tall) - 1080×2340",
    "20:9 (Ultra Tall) - 1080×2400",
    "18:9 (Tall) - 1080×2160"
]

# Default orientation for phone screens
DEFAULT_PHONE_ORIENTATION = "Portrait"

# Video resolutions
VIDEO_RESOLUTIONS = {
    "16:9": (1920, 1080),
    "21:9": (1920, 810),
    "4:3": (1024, 768),
    "9:16": (1080, 1920)
}

# Video aspect ratio display options
VIDEO_ASPECT_RATIOS = [
    "16:9 (Standard) - 1920×1080",
    "21:9 (Cinema) - 1920×810",
    "4:3 (Traditional) - 1024×768",
    "9:16 (Portrait) - 1080×1920"
]

# Default orientations for video formats
DEFAULT_VIDEO_ORIENTATIONS = {
    "16:9": "Landscape",
    "21:9": "Landscape",
    "4:3": "Landscape",
    "9:16": "Portrait"
}

# Social media resolutions
SOCIAL_RESOLUTIONS = {
    "Instagram Square (1:1)": (1080, 1080),
    "Instagram Portrait (4:5)": (1080, 1350),
    "Instagram Landscape (1.91:1)": (1080, 608),
    "Instagram Stories (9:16)": (1080, 1920),
    "TikTok Standard (9:16)": (1080, 1920),
    "TikTok Wide (16:9)": (1920, 1080),
    "YouTube Standard (16:9)": (1920, 1080),
    "YouTube Shorts (9:16)": (1080, 1920),
    "Facebook Feed (1.91:1)": (1200, 630),
    "Facebook Stories (9:16)": (1080, 1920),
    "Twitter Tweet (16:9)": (1200, 675),
    "Twitter Header (3:1)": (1500, 500)
}

# Social media aspect ratio display options
SOCIAL_ASPECT_RATIOS = [
    "Instagram Square (1:1) - 1080×1080",
    "Instagram Portrait (4:5) - 1080×1350",
    "Instagram Landscape (1.91:1) - 1080×608",
    "Instagram Stories (9:16) - 1080×1920",
    "TikTok Standard (9:16) - 1080×1920",
    "TikTok Wide (16:9) - 1920×1080",
    "YouTube Standard (16:9) - 1920×1080",
    "YouTube Shorts (9:16) - 1080×1920",
    "Facebook Feed (1.91:1) - 1200×630",
    "Facebook Stories (9:16) - 1080×1920",
    "Twitter Tweet (16:9) - 1200×675",
    "Twitter Header (3:1) - 1500×500"
]

# Default orientations for social media formats
DEFAULT_SOCIAL_ORIENTATIONS = {
    "Instagram Square (1:1)": "Landscape",
    "Instagram Portrait (4:5)": "Portrait",
    "Instagram Landscape (1.91:1)": "Landscape",
    "Instagram Stories (9:16)": "Portrait",
    "TikTok Standard (9:16)": "Portrait",
    "TikTok Wide (16:9)": "Landscape",
    "YouTube Standard (16:9)": "Landscape",
    "YouTube Shorts (9:16)": "Portrait",
    "Facebook Feed (1.91:1)": "Landscape",
    "Facebook Stories (9:16)": "Portrait",
    "Twitter Tweet (16:9)": "Landscape",
    "Twitter Header (3:1)": "Landscape"
}