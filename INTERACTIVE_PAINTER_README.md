# Interactive Painter Node (CYH)

A simplified interactive painting node for ComfyUI that provides a hard round brush for freehand drawing with solid colors.

## Features

- **Hard Round Brush**: Simple, precise brush for freehand drawing
- **Real-time Canvas**: Interactive drawing surface with mouse/touch support
- **Color Selection**: Hex color input with visual color picker
- **Brush Size Control**: Adjustable brush size from 1-100 pixels
- **Clear Canvas**: One-click canvas clearing
- **Alpha Channel Support**: Transparent background with proper mask output

## Installation

1. Copy the entire `Chye-ComfyUI-Toolset` folder to your ComfyUI `custom_nodes` directory
2. Restart ComfyUI
3. The node will be available under `image/tools` → `Interactive Painter (CYH)`

## Usage

### Basic Setup

1. Add the `Interactive Painter (CYH)` node to your workflow
2. Configure the canvas dimensions (width/height)
3. Set your desired brush size and color
4. Click the node to open the interactive canvas
5. Draw with your mouse or touch device
6. The painted image will be output for use in your workflow

### Node Inputs

- **width**: Canvas width in pixels (default: 512, range: 64-2048)
- **height**: Canvas height in pixels (default: 512, range: 64-2048)  
- **brush_size**: Brush diameter in pixels (default: 20, range: 1-100)
- **brush_color**: Hex color code (default: #FF0000)

### Node Outputs

- **image**: The painted image as a tensor (RGBA format)
- **mask**: Alpha channel mask for transparency

## Technical Details

### Implementation

Based on patterns from AlekPet's PainterNode, but simplified to focus on:
- Hard round brush only (no complex brush system)
- Basic color selection (no gradient/pattern support)
- Real-time canvas updates via WebSocket
- Base64 image data transmission

### File Structure

- `categories/image_tools.py` - Main Python implementation
- `js/cyh_painter.js` - Frontend JavaScript for canvas interaction
- WebSocket endpoints for real-time communication

### Web Endpoints

- `POST /cyh_painter/canvas_update` - Handle canvas data updates
- `GET /cyh_painter/get_image/{unique_id}` - Retrieve canvas image

## Examples

### Basic Painting Workflow

1. Create a new canvas with desired dimensions
2. Draw your design using the interactive brush
3. Connect the output image to any image processing node
4. Use the mask for selective processing

### Integration with ControlNet

The painted images work well with ControlNet for guided image generation:
- Draw rough sketches or masks
- Use as conditioning images
- Combine with other image processing nodes

## Limitations

- Currently supports only hard round brushes
- No brush opacity control
- No layer system
- No undo/redo functionality
- Basic color selection (hex codes only)

## Future Enhancements

Potential improvements could include:
- Multiple brush types (soft, textured, etc.)
- Brush opacity controls
- Layer support
- Undo/redo functionality
- Color palette system
- Image import/export

## Troubleshooting

### Canvas Not Appearing
- Ensure ComfyUI is restarted after installation
- Check browser console for JavaScript errors

### Drawing Not Working
- Verify mouse/touch events are supported by your browser
- Check that WebSocket connections are allowed

### Performance Issues
- Reduce canvas dimensions for better performance
- Use smaller brush sizes for complex drawings

## Support

For issues or feature requests, please check the project repository or create an issue with detailed information about your environment and the problem you're experiencing.