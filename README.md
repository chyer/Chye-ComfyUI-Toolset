# Chye ComfyUI Toolset

A comprehensive collection of ComfyUI custom nodes organized by category for enhanced workflow productivity.

## 🎯 Features

### 🔹 Latent Tools
- **Model-Specific Aspect Ratios**: Optimized presets for Flux, Qwen Image, and SDXL models
- **Smart Orientation Control**: Automatic Portrait/Landscape switching
- **Precision Scaling**: Multiplier with automatic rounding to multiples of 32
- **Batch Generation**: Support for multiple latents at once

### 🔸 Image Tools *(Coming Soon)*
- Image processing utilities
- Format conversion tools
- Enhancement filters

### 🔧 Utility Tools *(Coming Soon)*
- Workflow helpers
- Data conversion utilities
- Debug and analysis tools

## 📦 Installation

1. Copy the `Chye-ComfyUI-Toolset` folder to your ComfyUI `custom_nodes` directory
2. **Important**: Create a file named `.cnr-id` in the `.git` folder with content `Chye-ComfyUI-Toolset` to avoid workflow errors
   ```bash
   echo "Chye-ComfyUI-Toolset" > .git/.cnr-id
   ```
3. Restart ComfyUI

## 🚀 Usage

### Latent Tools
Find the nodes under the **latent** category:
- **🔹 CYH Latent | Flux Aspect Ratio** - Flux-optimized latent generation
- **🔹 CYH Latent | Qwen Aspect Ratio** - Qwen Image-optimized latent generation  
- **🔹 CYH Latent | SDXL Aspect Ratio** - SDXL-optimized latent generation

#### Configuration Options:
- **Aspect Ratio**: Choose from model-specific presets (1:1, 4:3, 3:2, 16:9, 21:9)
- **Orientation**: Portrait (default) or Landscape
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

## 🔄 Version History

- **v2.0.0**: Complete restructure into modular toolset with shared utilities
- **v1.x.x**: Individual aspect ratio nodes

## 🤝 Contributing

This toolset is designed for extensibility. New categories and tools can be easily added following the established patterns in the `categories/` and `shared/` directories.

---

**Happy Creating! 🎨**