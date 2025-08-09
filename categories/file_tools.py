"""
File tools for Chye ComfyUI Toolset
"""

import sys
import os
import re

# Add parent directory to Python path for ComfyUI compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from shared.constants import FILE_CATEGORY
    from shared.helpers import sanitize_filename
except ImportError:
    # Fallback import for ComfyUI environments
    import importlib.util
    
    # Import constants
    constants_path = os.path.join(parent_dir, "shared", "constants.py")
    spec = importlib.util.spec_from_file_location("constants", constants_path)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    
    # Define FILE_CATEGORY if not in constants
    if hasattr(constants, 'FILE_CATEGORY'):
        FILE_CATEGORY = constants.FILE_CATEGORY
    else:
        FILE_CATEGORY = "file"
    
    # Import helpers
    helpers_path = os.path.join(parent_dir, "shared", "helpers.py")
    spec = importlib.util.spec_from_file_location("helpers", helpers_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    
    sanitize_filename = helpers.sanitize_filename if hasattr(helpers, 'sanitize_filename') else None

# Define sanitize_filename if not available from helpers
if sanitize_filename is None:
    def sanitize_filename(filename):
        """Remove invalid characters from filename"""
        return re.sub(r'[<>:"/\\|?*]', '', filename).strip()


class CYHFolderFilenameBuilderNode:
    """
    A node that constructs file paths by combining project name, subfolder, and filename.
    Allows customization of path delimiters and optional subfolder usage.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"default": "MyProject"}),
                "filename": ("STRING", {"default": "image01"}),
                "use_subfolders": ("BOOLEAN", {"default": True}),
                "delimiter": (["/", "_", "-", "\\"], {"default": "/"}),
            },
            "optional": {
                "subfolder": ("STRING", {"default": "images"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    FUNCTION = "build_path"
    CATEGORY = FILE_CATEGORY
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def build_path(self, project_name, filename, use_subfolders, delimiter, subfolder=""):
        # Sanitize inputs to remove invalid characters
        project_name = sanitize_filename(project_name)
        filename = sanitize_filename(filename)
        subfolder = sanitize_filename(subfolder)
        
        # Handle empty inputs with defaults
        if not project_name:
            project_name = "UntitledProject"
        if not filename:
            filename = "untitled"
        
        # Build the path components
        path_components = [project_name]
        
        if use_subfolders and subfolder:
            path_components.append(subfolder)
        
        path_components.append(filename)
        
        # Join components with the selected delimiter
        full_path = delimiter.join(path_components)
        
        return (full_path, filename)


# Node registration for this category
NODE_CLASS_MAPPINGS = {
    "CYHFolderFilenameBuilderNode": CYHFolderFilenameBuilderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CYHFolderFilenameBuilderNode": "📁 CYH File | Folder Filename Builder",
}