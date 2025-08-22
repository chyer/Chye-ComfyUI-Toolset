# Interactive Painting Node Implementation Plan

## Based on AlekPet PainterNode Analysis

### Key Implementation Patterns Identified:
1. **WebSocket Communication**: Custom HTTP endpoints via `PromptServer.instance.routes`
2. **Global State Management**: `PAINTER_DICT` tracks node instances by unique_id
3. **Real-time Canvas Updates**: `wait_canvas_change()` with timeout mechanism
4. **Base64 Image Handling**: `toBase64ImgUrl()` converts PIL images to data URLs
5. **Simplified Approach**: Focus only on hard round brush functionality

### Implementation Strategy:
1. **Simplified Version**: Remove complex settings persistence (JSON files)
2. **Hard Round Brush Only**: Single brush type with configurable size
3. **Basic Color Selection**: Simple color picker or hex input
4. **Real-time Preview**: Canvas updates during painting
5. **Minimal Dependencies**: Only essential functionality

### Core Components:

#### 1. WebSocket Endpoints (Simplified):
- `/cyh_painter/canvas_update` - Handle canvas changes
- `/cyh_painter/get_image` - Retrieve current canvas state

#### 2. Global State Management:
```python
CYH_PAINTER_DICT = {}  # Track painting node instances
```

#### 3. Hard Round Brush Implementation:
- Simple circle drawing with configurable size
- Mouse coordinate tracking
- RGBA color support with opacity

#### 4. Node Interface:
- Brush size slider (1-100px)
- Color picker (hex input or basic color selection)
- Canvas display area
- Clear canvas button

### Implementation Steps:

1. **Basic Infrastructure**: WebSocket endpoints and global state
2. **Canvas System**: Base64 image handling and real-time updates
3. **Brush Implementation**: Hard round brush with mouse tracking
4. **UI Controls**: Brush size, color selection, clear functionality
5. **Integration**: Connect to existing image processing pipeline
6. **Testing**: Verify functionality and latent space compatibility

### File Structure:
- `categories/image_tools.py`: Main painting node implementation
- JavaScript: Canvas drawing logic (embedded in Python)
- WebSocket handlers for real-time communication

### Dependencies:
- PIL/Pillow for image processing
- Base64 encoding for canvas data
- Torch for tensor conversion (if needed for latent space)
- Standard ComfyUI server components

### Testing Strategy:
1. Basic brush functionality testing
2. Color application verification
3. Canvas clearing and reset
4. Integration with image processing nodes
5. Latent space compatibility testing