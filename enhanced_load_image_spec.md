# Enhanced Load Image Node Specification

## Overview
This document specifies the design for an enhanced Load Image node for ComfyUI that extends the standard LoadImage node with additional features including filename output and advanced resizing capabilities.

## Features

### Core Features
1. **Standard Outputs**:
   - IMAGE (standard image tensor)
   - MASK (standard mask from alpha channel)

2. **Enhanced Outputs**:
   - FILENAME (string containing the loaded image filename)

3. **Resizing Capabilities**:
   - Resize to desired dimensions
   - Padding with solid color option
   - Cropping option for resize down
   - Position control for padding/cropping (center, left, right, top, bottom)

## Node Design Specification

### Input Parameters
1. **image** - Standard image selection from input directory
2. **resize_mode** - Enum: ["disabled", "resize", "pad", "crop"]
3. **width** - INT: Desired width (when resize_mode != "disabled")
4. **height** - INT: Desired height (when resize_mode != "disabled")
5. **pad_color** - INT: Color for padding (RGB hex value)
6. **upscale_method** - Enum: ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
7. **pad_position** - Enum: ["center", "left", "right", "top", "bottom"] (for pad mode)
8. **crop_position** - Enum: ["center", "left", "right", "top", "bottom"] (for crop mode)

### Output Parameters
1. **IMAGE** - Standard image tensor
2. **MASK** - Standard mask tensor
3. **FILENAME** - String containing the image filename

## Implementation Approach

### 1. Extend Standard LoadImage
- Inherit from the standard LoadImage class
- Override the load_image method to add filename extraction
- Add resizing functionality using comfy.utils.common_upscale

### 2. Filename Extraction
- Extract filename from the image path using os.path.basename
- Return as additional string output

### 3. Resizing Features
- **Resize mode**: Standard scaling to target dimensions
- **Pad mode**: Scale to fit one dimension, pad the other with solid color
  * Position options: center, left, right, top, bottom
- **Crop mode**: Scale to fill both dimensions, crop excess
  * Position options: center, left, right, top, bottom

### 4. Integration
- Add to image category
- Follow existing node naming conventions
- Maintain compatibility with existing workflows

## Technical Implementation Details

### Class Structure
```python
class EnhancedLoadImage(LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        # Extend standard inputs with resizing options
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filename")
    FUNCTION = "load_image_enhanced"
```

### Resizing Logic
- Use comfy.utils.common_upscale for image scaling
- Implement padding with torch operations and position control
- Implement cropping with tensor slicing and position control

### Filename Extraction
- Extract from image_path using os.path.basename
- Handle annotated filepaths from folder_paths

### Position Control Implementation
- **Padding**: Calculate padding amounts for each side based on position
- **Cropping**: Calculate crop coordinates based on position

## Future Considerations
1. Add support for batch processing
2. Add aspect ratio preservation options
3. Add configurable interpolation methods for padding
4. Add support for different color spaces

## Dependencies
- comfy.utils.common_upscale
- torch for tensor operations
- os.path for filename extraction
- Standard ComfyUI LoadImage functionality