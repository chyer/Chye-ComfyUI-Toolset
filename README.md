# Chye ComfyUI Toolset

A comprehensive collection of ComfyUI custom nodes organized by category for enhanced workflow productivity.

## 🎯 Features

### 🔹 Latent Tools
- **Model-Specific Aspect Ratios**: Optimized presets for Flux, Qwen Image, and SDXL models
- **Phone Screen Sizes**: Common mobile device resolutions including modern tall screens
- **Video Formats**: Standard video resolutions from 480p to 4K including cinema formats
- **Social Media Formats**: Optimized sizes for Instagram, TikTok, YouTube, Facebook, and Twitter
- **Smart Orientation Control**: Automatic Portrait/Landscape switching
- **Precision Scaling**: Multiplier with automatic rounding to multiples of 32
- **Batch Generation**: Support for multiple latents at once

### 🔸 Post Process Tools
- **Realistic Film Grain**: Apply photographic film grain simulation with ISO control
- **ARRI Halation Effect**: Simulate film bloom around highlights with red-orange tint
- **Chromatic Aberration**: Lens color fringing simulation with barrel distortion
- **Spatially Correlated Noise**: Advanced noise generation using scipy for realistic grain patterns
- **Color Channel Control**: Option for monochrome or colored grain effects

### 🔧 Utility Tools *(Coming Soon)*
- Workflow helpers
- Data conversion utilities
- Debug and analysis tools

## 📦 Installation

### Method 1: Git Clone (Recommended)
```bash
cd custom_nodes
git clone https://github.com/chyer/Chye-ComfyUI-Toolset.git
cd Chye-ComfyUI-Toolset
echo "Chye-ComfyUI-Toolset" > .git/.cnr-id

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Manual Copy
1. Copy the `Chye-ComfyUI-Toolset` folder to your ComfyUI `custom_nodes` directory
2. **Important**: Create a file named `.cnr-id` in the `.git` folder with content `Chye-ComfyUI-Toolset` to avoid workflow errors
   ```bash
   echo "Chye-ComfyUI-Toolset" > .git/.cnr-id
   ```
3. Install dependencies:
   ```bash
   cd Chye-ComfyUI-Toolset
   pip install -r requirements.txt
   ```
4. Restart ComfyUI

## 🚀 Usage

### Latent Tools
Find the nodes under the **latent** category:

#### Model-Specific Nodes
- **🔹 CYH Latent | Flux Aspect Ratio** - Flux-optimized latent generation
- **🔹 CYH Latent | Qwen Aspect Ratio** - Qwen Image-optimized latent generation
- **🔹 CYH Latent | SDXL Aspect Ratio** - SDXL-optimized latent generation

#### Phone Screen Nodes
- **🔹 CYH Latent | Phone Aspect Ratio** - Common mobile device resolutions

#### Video Format Nodes
- **🔹 CYH Latent | Video Aspect Ratio** - Standard video resolutions and formats

#### Social Media Nodes
- **🔹 CYH Latent | Social Media Aspect Ratio** - Optimized sizes for social platforms

#### Configuration Options:
- **Aspect Ratio**: Choose from category-specific presets
- **Orientation**: Portrait or Landscape (with smart defaults based on media type)
- **Multiplier**: Scale resolution (0.1-10.0)
- **Batch Size**: Number of latents to generate (1-64)

#### Example Workflow:
```
[🔹 CYH Latent | Flux Aspect Ratio] → [KSampler] → [VAE Decode]
```

#### Multiplier Example:
```
Model: SDXL
Aspect: 16:9
Orientation: Landscape
Multiplier: 1.5
→ Output: 1536×864 latent
```

## 🎬 Post Process Tools

Find the nodes under the **post_process** category:

- **🎬 CYH Post Process | Film Grain** - Apply realistic film grain simulation
  - **Strength**: Overall grain intensity (0.0 to 1.0)
  - **ISO**: Simulated film sensitivity (100-6400)
  - **Grain Size**: Particle size control (1.0-10.0)
  - **Colored**: Toggle between monochrome or colored grain

- **🎬 CYH Post Process | ARRI Halation** - Apply ARRI-style film bloom effect
  - **Threshold**: Highlight detection threshold (0-255)
  - **Blur Size**: Glow effect size (1-101, odd numbers)
  - **Intensity**: Halation strength (0.0 to 1.0)

- **🌈 CYH Post Process | Chromatic Aberration** - Apply lens color fringing effect
  - **Preset**: Quick configurations (None, Vintage, Modern, Extreme, Custom)
  - **Intensity**: Master effect strength multiplier (0.0-2.0)
  - **Center**: Adjustable distortion center point (0.0-1.0)
  - **Quality**: Fast (bilinear) or High (bicubic) interpolation
  - **Advanced Controls**: Individual RGB channel distortion coefficients (k1, k2, k3)

#### Preset Details:
- **None**: No chromatic aberration
- **Vintage**: Classic lens look with red/blue separation
- **Modern**: Subtle modern lens character
- **Extreme**: Strong artistic effect with complex distortion

#### Example Workflows:
```
[KSampler] → [VAE Decode] → [🎬 CYH Post Process | Film Grain]
```
```
[KSampler] → [VAE Decode] → [🎬 CYH Post Process | ARRI Halation]
```
```
[KSampler] → [VAE Decode] → [🌈 CYH Post Process | Chromatic Aberration]
```

## 📱 Phone Screen Resolutions

| Aspect Ratio | Resolution | Description |
|-------------|------------|-------------|
| 16:9 | 1080×1920 | Standard phone screens |
| 19.5:9 | 1080×2340 | Modern tall phones |
| 20:9 | 1080×2400 | Ultra tall phones |
| 18:9 | 1080×2160 | Tall phones |

**Default Orientation**: Portrait (optimized for mobile viewing)

## 🎬 Video Format Resolutions

| Aspect Ratio | Resolution | Description |
|-------------|------------|-------------|
| 16:9 | 1920×1080 | Standard video (Full HD) |
| 21:9 | 1920×810 | Cinema format |
| 4:3 | 1024×768 | Traditional video |
| 9:16 | 1080×1920 | Portrait video |

**Default Orientation**: Landscape (except 9:16 which defaults to Portrait)

## 📱 Social Media Resolutions

| Platform | Format | Resolution | Aspect Ratio |
|----------|--------|------------|-------------|
| Instagram | Square | 1080×1080 | 1:1 |
| Instagram | Portrait | 1080×1350 | 4:5 |
| Instagram | Landscape | 1080×608 | 1.91:1 |
| Instagram | Stories/Reels | 1080×1920 | 9:16 |
| TikTok | Standard | 1080×1920 | 9:16 |
| TikTok | Wide | 1920×1080 | 16:9 |
| YouTube | Standard | 1920×1080 | 16:9 |
| YouTube | Shorts | 1080×1920 | 9:16 |
| Facebook | Feed | 1200×630 | 1.91:1 |
| Facebook | Stories | 1080×1920 | 9:16 |
| Twitter | Tweet | 1200×675 | 16:9 |
| Twitter | Header | 1500×500 | 3:1 |

**Default Orientation**: Platform-specific (Portrait for Instagram Stories/TikTok, Landscape for YouTube Standard, etc.)

## 📐 Resolution Reference

| Model      | 1:1      | 4:3      | 3:2      | 16:9     | 21:9     |
|------------|----------|----------|----------|----------|----------|
| **Flux**   | 1024×1024| 1280×960 | 1152×768 | 1344×768 | 1792×768 |
| **Qwen**   | 1328×1328| 1472×1140| 1536×1024| 1664×928 | 1984×864 |
| **SDXL**   | 1024×1024| 1024×768 | 1152×768 | 1024×576 | 1344×576 |

## 🏗️ Project Structure

```
Chye-ComfyUI-Toolset/
├── categories/
│   ├── latent_tools.py      # Aspect ratio latent generators
│   ├── post_process.py      # Image post-processing tools (film grain, etc.)
│   ├── image_tools.py       # (Future) Image processing tools
│   ├── utility_tools.py     # (Future) Workflow utilities
│   └── __init__.py
├── shared/
│   ├── constants.py         # Common configuration data
│   ├── validators.py        # Input validation utilities
│   ├── helpers.py          # Shared utility functions
│   └── __init__.py
├── __init__.py             # Main entry point
├── pyproject.toml          # Project metadata
└── README.md              # This file
```

## 📝 Notes

- All dimensions are automatically rounded to multiples of 32 for ComfyUI compatibility
- Portrait orientation swaps width/height when the base resolution is landscape-oriented
- Multiplier scales base resolution before rounding
- Modular design allows for easy expansion with new tool categories
- Phone screen nodes default to Portrait orientation for optimal mobile viewing
- Video format nodes have smart defaults based on the aspect ratio (Landscape for most, Portrait for 9:16)
- Social media nodes have platform-specific default orientations for optimal results

## 🔄 Version History

- **v2.2.0**: Added Chromatic Aberration node with realistic lens simulation
- **v2.1.0**: Added Phone, Video, and Social Media aspect ratio nodes
- **v2.0.0**: Complete restructure into modular toolset with shared utilities
- **v1.x.x**: Individual aspect ratio nodes

## 🤝 Contributing

This toolset is designed for extensibility. New categories and tools can be easily added following the established patterns in the `categories/` and `shared/` directories.

---

**Happy Creating! 🎨**