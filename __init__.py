"""
Chye ComfyUI Toolset - A comprehensive collection of ComfyUI custom nodes
"""

import sys
import os

# Add current directory to Python path for ComfyUI compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from categories.latent_tools import (
        NODE_CLASS_MAPPINGS as LATENT_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as LATENT_DISPLAY_MAPPINGS
    )
    from categories.math_tools import (
        NODE_CLASS_MAPPINGS as MATH_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as MATH_DISPLAY_MAPPINGS
    )
    from categories.file_tools import (
        NODE_CLASS_MAPPINGS as FILE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as FILE_DISPLAY_MAPPINGS
    )
    from categories.post_process import (
        NODE_CLASS_MAPPINGS as POST_PROCESS_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as POST_PROCESS_DISPLAY_MAPPINGS
    )
    from categories.prompt_tools import (
        NODE_CLASS_MAPPINGS as PROMPT_TOOLS_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as PROMPT_TOOLS_DISPLAY_MAPPINGS
    )
    from categories.video_tools import (
        NODE_CLASS_MAPPINGS as VIDEO_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as VIDEO_DISPLAY_MAPPINGS
    )
except ImportError:
    # Fallback for ComfyUI environments
    import importlib.util
    
    # Import latent tools
    spec = importlib.util.spec_from_file_location("latent_tools", os.path.join(current_dir, "categories", "latent_tools.py"))
    latent_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(latent_tools)
    
    LATENT_CLASS_MAPPINGS = latent_tools.NODE_CLASS_MAPPINGS
    LATENT_DISPLAY_MAPPINGS = latent_tools.NODE_DISPLAY_NAME_MAPPINGS
    
    # Import math tools
    spec = importlib.util.spec_from_file_location("math_tools", os.path.join(current_dir, "categories", "math_tools.py"))
    math_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_tools)
    
    MATH_CLASS_MAPPINGS = math_tools.NODE_CLASS_MAPPINGS
    MATH_DISPLAY_MAPPINGS = math_tools.NODE_DISPLAY_NAME_MAPPINGS
    
    # Import file tools
    spec = importlib.util.spec_from_file_location("file_tools", os.path.join(current_dir, "categories", "file_tools.py"))
    file_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(file_tools)
    
    FILE_CLASS_MAPPINGS = file_tools.NODE_CLASS_MAPPINGS
    FILE_DISPLAY_MAPPINGS = file_tools.NODE_DISPLAY_NAME_MAPPINGS
    
    # Import post process tools
    spec = importlib.util.spec_from_file_location("post_process", os.path.join(current_dir, "categories", "post_process.py"))
    post_process = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(post_process)
    
    POST_PROCESS_CLASS_MAPPINGS = post_process.NODE_CLASS_MAPPINGS
    POST_PROCESS_DISPLAY_MAPPINGS = post_process.NODE_DISPLAY_NAME_MAPPINGS
    
    # Import prompt tools
    spec = importlib.util.spec_from_file_location("prompt_tools", os.path.join(current_dir, "categories", "prompt_tools.py"))
    prompt_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_tools)
    
    PROMPT_TOOLS_CLASS_MAPPINGS = prompt_tools.NODE_CLASS_MAPPINGS
    PROMPT_TOOLS_DISPLAY_MAPPINGS = prompt_tools.NODE_DISPLAY_NAME_MAPPINGS
    
    # Import video tools
    spec = importlib.util.spec_from_file_location("video_tools", os.path.join(current_dir, "categories", "video_tools.py"))
    video_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(video_tools)
    
    VIDEO_CLASS_MAPPINGS = video_tools.NODE_CLASS_MAPPINGS
    VIDEO_DISPLAY_MAPPINGS = video_tools.NODE_DISPLAY_NAME_MAPPINGS
    

# Combine all category mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add latent tools
NODE_CLASS_MAPPINGS.update(LATENT_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LATENT_DISPLAY_MAPPINGS)

# Add math tools
NODE_CLASS_MAPPINGS.update(MATH_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MATH_DISPLAY_MAPPINGS)

# Add file tools
NODE_CLASS_MAPPINGS.update(FILE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(FILE_DISPLAY_MAPPINGS)

# Add post process tools
NODE_CLASS_MAPPINGS.update(POST_PROCESS_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(POST_PROCESS_DISPLAY_MAPPINGS)

# Add prompt tools
NODE_CLASS_MAPPINGS.update(PROMPT_TOOLS_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(PROMPT_TOOLS_DISPLAY_MAPPINGS)

# Add video tools
NODE_CLASS_MAPPINGS.update(VIDEO_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(VIDEO_DISPLAY_MAPPINGS)


# API endpoints for web extensions
import json
from aiohttp import web

# Store the server instance for API registration
_server_instance = None

def get_server():
    """Get the ComfyUI server instance"""
    global _server_instance
    return _server_instance

def set_server(server):
    """Set the ComfyUI server instance for API registration"""
    global _server_instance
    _server_instance = server

async def text_file_loader_load_handler(request):
    """Handle file loading requests from the JavaScript extension"""
    try:
        data = await request.json()
        file_path = data.get('file_path', '')
        
        if not file_path:
            return web.json_response({"error": "No file path provided"}, status=400)
        
        # Use the existing CYHTextFileLoaderNode logic to load the file
        from categories.file_tools import CYHTextFileLoaderNode
        node = CYHTextFileLoaderNode()
        
        # Simulate the node execution to get the content
        result = node.load_text_file(file_path=file_path, trigger="Load File")
        
        if "result" in result and len(result["result"]) >= 2:
            content = result["result"][0]  # The loaded text content
            status = result["result"][1]   # The status message
            
            return web.json_response({
                "content": content,
                "status": status,
                "success": True
            })
        else:
            return web.json_response({
                "error": "Failed to load file content",
                "success": False
            }, status=500)
            
    except Exception as e:
        return web.json_response({
            "error": f"Internal server error: {str(e)}",
            "success": False
        }, status=500)

# Function to register API endpoints
def register_api_endpoints(server):
    """Register API endpoints with the ComfyUI server"""
    set_server(server)
    
    # Register the text file loader endpoint
    server.routes.post("/chye/text_file_loader/load")(text_file_loader_load_handler)
    print("[Chye Toolset] Registered API endpoint: /chye/text_file_loader/load")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'register_api_endpoints']