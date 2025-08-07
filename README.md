# ComfyUI CYH Aspect Ratio

A ComfyUI custom node that generates empty latent images with model-specific aspect ratios and resolutions.

## Features

- **Model Selection**: Choose between Flux, Qwen Image, and SDXL models
- **Aspect Ratios**: 1:1, 4:3, 3:2, 16:9, 21:9 presets
- **Orientation Control**: Portrait or Landscape modes
- **Resolution Scaling**: Multiplier with automatic rounding to multiples of 32
- **Batch Support**: Generate multiple latents at once

## Installation

1. Copy the `Chye_ASPLatent` folder to your ComfyUI `custom_nodes` directory
2. **Important**: Create a file named `.cnr-id` in the `.git` folder with content `Chye-ASPlatent` to avoid workflow errors
   ```
   echo "Chye-ASPlatent" > .git/.cnr-id
   ```
3. Restart ComfyUI

## Usage

1. Add the **🔹 CYH Aspect Ratio** node to your workflow
2. Configure settings:
   - **Model Type**: Select your target model
   - **Aspect Ratio**: Choose desired ratio
   - **Orientation**: Portrait (default) or Landscape
   - **Multiplier**: Scale resolution (0.1-10.0)
   - **Batch Size**: Number of latents to generate

3. Connect the LATENT output to your sampler

## Example Workflows

### Basic Generation
```
[CYH Aspect Ratio] → [KSampler] → [VAE Decode]
```

### Multiplier Example
```
Model: SDXL
Aspect: 16:9 
Orientation: Landscape
Multiplier: 1.5
→ Output: 1536×864 latent
```

## Resolution Reference

| Model      | 1:1      | 4:3      | 3:2      | 16:9     | 21:9     |
|------------|----------|----------|----------|----------|----------|
| **Flux**   | 1024×1024| 1280×960 | 1152×768 | 1344×768 | 1792×768 |
| **Qwen**   | 1328×1328| 1472×1140| 1536×1024| 1664×928 | 1984×864 |
| **SDXL**   | 1024×1024| 1024×768 | 1152×768 | 1024×576 | 1344×576 |

## Notes

- All dimensions are automatically rounded to multiples of 32
- Portrait orientation swaps width/height when appropriate
- Multiplier scales base resolution before rounding