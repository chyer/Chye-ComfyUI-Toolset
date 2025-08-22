import base64
import time
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import torch
from server import PromptServer
from aiohttp import web

# Global dictionary to track painting node instances
CYH_PAINTER_DICT = {}

def to_base64_img_url(img):
    """Convert PIL Image to base64 data URL"""
    bytes_io = BytesIO()
    img.save(bytes_io, format="PNG")
    img_bytes = bytes_io.getvalue()
    img_base64 = base64.b64encode(img_bytes)
    return f"data:image/png;base64,{img_base64.decode('utf-8')}"

@PromptServer.instance.routes.post("/cyh_painter/canvas_update")
async def canvas_update(request):
    """Handle canvas updates from the frontend"""
    try:
        json_data = await request.json()
        unique_id = json_data.get("unique_id")
        canvas_data = json_data.get("canvas_data")
        
        if unique_id and unique_id in CYH_PAINTER_DICT:
            # Store the canvas data for the node instance
            CYH_PAINTER_DICT[unique_id].canvas_data = canvas_data
            CYH_PAINTER_DICT[unique_id].canvas_updated = True
            
            return web.json_response({"status": "success"}, status=200)
        
        return web.json_response({"status": "error", "message": "Node not found"}, status=404)
    
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)

@PromptServer.instance.routes.get("/cyh_painter/get_image/{unique_id}")
async def get_image(request):
    """Get the current canvas image for a node"""
    unique_id = request.match_info.get("unique_id")
    
    if unique_id and unique_id in CYH_PAINTER_DICT:
        node = CYH_PAINTER_DICT[unique_id]
        if hasattr(node, 'current_image') and node.current_image:
            img_url = to_base64_img_url(node.current_image)
            return web.json_response({"image": img_url}, status=200)
    
    return web.json_response({"image": None}, status=404)

def wait_canvas_update(unique_id, timeout=30):
    """Wait for canvas update with timeout"""
    for _ in range(timeout):
        if (unique_id in CYH_PAINTER_DICT and 
            hasattr(CYH_PAINTER_DICT[unique_id], 'canvas_updated') and 
            CYH_PAINTER_DICT[unique_id].canvas_updated):
            
            CYH_PAINTER_DICT[unique_id].canvas_updated = False
            return True
        time.sleep(0.1)
    return False

class CYHInteractivePainterNode:
    """Interactive painting node with hard round brush"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "brush_size": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "brush_color": ("STRING", {"default": "#FF0000"}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "paint"
    CATEGORY = "image/tools"
    DESCRIPTION = "Interactive painting node with hard round brush for freehand drawing"
    
    def __init__(self):
        self.canvas_data = None
        self.canvas_updated = False
        self.current_image = None
    
    def paint(self, width, height, brush_size, brush_color, unique_id, image=None):
        # Register this node instance
        CYH_PAINTER_DICT[unique_id] = self
        
        # Handle image input - convert to PIL Image if provided
        if image is not None:
            # Convert tensor to PIL Image
            image_np = image[0].cpu().numpy() * 255.0
            image_np = image_np.astype(np.uint8)
            
            if image_np.shape[2] == 3:  # RGB
                self.current_image = Image.fromarray(image_np, 'RGB').convert("RGBA")
            elif image_np.shape[2] == 4:  # RGBA
                self.current_image = Image.fromarray(image_np, 'RGBA')
            else:
                self.current_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        elif self.current_image is None:
            # Create initial blank canvas if no image provided
            self.current_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        
        # Convert hex color to RGB tuple
        try:
            if brush_color.startswith('#'):
                brush_color = brush_color.lstrip('#')
                if len(brush_color) == 6:
                    r = int(brush_color[0:2], 16)
                    g = int(brush_color[2:4], 16)
                    b = int(brush_color[4:6], 16)
                    brush_rgb = (r, g, b, 255)  # Full opacity
                else:
                    brush_rgb = (255, 0, 0, 255)  # Default red
            else:
                brush_rgb = (255, 0, 0, 255)  # Default red
        except:
            brush_rgb = (255, 0, 0, 255)  # Default red on error
        
        # Send current image to frontend for painting
        img_url = to_base64_img_url(self.current_image)
        
        PromptServer.instance.send_sync(
            "cyh_painter_start",
            {
                "unique_id": unique_id,
                "image": img_url,
                "brush_size": brush_size,
                "brush_color": brush_color,
                "width": width,
                "height": height
            }
        )
        
        # Wait for canvas updates
        if wait_canvas_update(unique_id):
            print(f"Painting node {unique_id}: Canvas updated successfully")
            
            # Process the canvas data if received
            if self.canvas_data:
                try:
                    # Convert base64 data URL to PIL Image
                    if self.canvas_data.startswith('data:image/png;base64,'):
                        base64_data = self.canvas_data.split(',', 1)[1]
                        image_data = base64.b64decode(base64_data)
                        self.current_image = Image.open(BytesIO(image_data)).convert("RGBA")
                except Exception as e:
                    print(f"Error processing canvas data: {e}")
        else:
            print(f"Painting node {unique_id}: Timeout waiting for canvas update")
        
        # Convert final image to tensor format
        image_array = np.array(self.current_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array)[None,]
        
        # Create mask from alpha channel
        if image_array.shape[2] == 4:  # RGBA
            mask = image_array[:, :, 3]  # Alpha channel
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, height, width), dtype=torch.float32)
        
        return (image_tensor, mask_tensor)
    
    @classmethod
    def IS_CHANGED(cls, width, height, brush_size, brush_color, unique_id, image=None):
        # Force re-execution on parameter changes
        image_hash = hash(str(image.shape)) if image is not None else "no_image"
        return f"{width}_{height}_{brush_size}_{brush_color}_{image_hash}"

# JavaScript for the frontend canvas interaction
NODE_CLASS_MAPPINGS = {
    "CYHInteractivePainterNode": CYHInteractivePainterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CYHInteractivePainterNode": "Interactive Painter (CYH)"
}

# Frontend JavaScript will be injected by ComfyUI
# The JavaScript handles the canvas drawing and mouse interaction