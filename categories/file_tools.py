"""
File tools for Chye ComfyUI Toolset
"""

import sys
import os
import re
try:
    import folder_paths
except ImportError:
    # folder_paths is only available in ComfyUI environment
    # Create a mock for testing purposes
    class MockFolderPaths:
        @staticmethod
        def get_input_directory():
            return "./input"
        
        @staticmethod
        def get_output_directory():
            return "./output"
        
        @staticmethod
        def get_temp_directory():
            return "./temp"
    
    folder_paths = MockFolderPaths()

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


class CYHTextFileLoaderNode:
    """
    A node that loads text content from a file with a load button.
    Supports .txt files and provides editable output with UI updates.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Select .txt file using file browser button →"
                }),
                "trigger": (["Load File", "Reload"], {"default": "Load File"}),
            },
            "optional": {
                "editable_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Loaded text will appear here (editable)"
                }),
                "encoding": (["utf-8", "utf-16", "ascii", "latin-1"], {"default": "utf-8"}),
                "auto_load": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("editable_text", "status")
    FUNCTION = "load_text_file"
    CATEGORY = FILE_CATEGORY
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, file_path="", trigger="Load File", auto_load=False, **kwargs):
        # Create unique hash based on file path and modification time
        import hashlib
        import time
        import random
        
        if auto_load and file_path and os.path.exists(file_path):
            try:
                # Include file modification time for auto-reloading
                mod_time = os.path.getmtime(file_path)
                # Add random component to ensure uniqueness
                hash_input = f"{file_path}_{mod_time}_{time.time()}_{random.random()}"
                return hashlib.md5(hash_input.encode()).hexdigest()
            except:
                pass
        
        # For manual trigger, always return unique hash to force execution
        # Include random component to ensure each execution is unique
        hash_input = f"{file_path}_{trigger}_{time.time()}_{random.random()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    # --- Helper functions ---
    def find_node_by_id(self, unique_id, workflow_info):
        if not workflow_info or "nodes" not in workflow_info:
            print(f"[{self.__class__.__name__}] Helper Error: Invalid workflow_info.")
            return None
        target_id = str(unique_id[0]) if isinstance(unique_id, list) else str(unique_id)
        for node_data in workflow_info["nodes"]:
            if str(node_data.get("id")) == target_id:
                return node_data
        print(f"[{self.__class__.__name__}] Helper Error: Node ID {target_id} not found in workflow.")
        return None

    def find_widget_index(self, node_data, widget_name):
        # Combine required and optional keys to get all widget names
        req_keys = list(self.INPUT_TYPES().get("required", {}).keys())
        opt_keys = list(self.INPUT_TYPES().get("optional", {}).keys())
        all_keys = req_keys + opt_keys
        try:
            idx = all_keys.index(widget_name)
            return idx
        except ValueError:
            print(f"[{self.__class__.__name__}] Helper Error: Widget '{widget_name}' not found in INPUT_TYPES keys: {all_keys}")
            return None

    def load_text_file(self, file_path="", trigger="Load File", editable_text="", encoding="utf-8", auto_load=False, unique_id=None, extra_pnginfo=None):
        # Handle empty file path
        if not file_path or not file_path.strip():
            if editable_text:
                return {"ui": {"text": [editable_text]}, "result": (editable_text, "[WARN] No file path provided - keeping current text")}
            return {"ui": {"text": [""]}, "result": ("", "[WARN] No file path provided")}
        
        # Clean the file path
        file_path = file_path.strip()
        
        # Check if file exists - first try direct path, then check in ComfyUI directories
        resolved_path = file_path
        if not os.path.exists(resolved_path):
            # Try to find the file in ComfyUI input directories
            try:
                input_dir = folder_paths.get_input_directory()
                full_path = os.path.join(input_dir, resolved_path)
                if os.path.exists(full_path):
                    resolved_path = full_path
                    print(f"[CYH Text File Loader] Found file in input directory: {resolved_path}")
                else:
                    # Try with just the filename in input directory
                    filename = os.path.basename(resolved_path)
                    full_path = os.path.join(input_dir, filename)
                    if os.path.exists(full_path):
                        resolved_path = full_path
                        print(f"[CYH Text File Loader] Found file by filename in input directory: {resolved_path}")
            except Exception as e:
                print(f"[CYH Text File Loader] Error resolving file path: {e}")
        
        # Check if resolved file exists
        if not os.path.exists(resolved_path):
            if editable_text:
                return {"ui": {"text": [editable_text]}, "result": (editable_text, f"[ERROR] File not found: {file_path} - keeping current text")}
            return {"ui": {"text": [""]}, "result": ("", f"[ERROR] File not found: {file_path}")}
        
        # Check if it's a text file
        if not resolved_path.lower().endswith('.txt'):
            if editable_text:
                return {"ui": {"text": [editable_text]}, "result": (editable_text, f"[ERROR] Not a text file: {file_path} - keeping current text")}
            return {"ui": {"text": [""]}, "result": ("", f"[ERROR] Not a text file: {file_path}")}
        
        try:
            # Read the file content
            with open(resolved_path, 'r', encoding=encoding) as file:
                content = file.read()
            
            # Update UI widget if workflow info is available
            if unique_id and extra_pnginfo:
                current_workflow_info = extra_pnginfo[0] if isinstance(extra_pnginfo, list) and extra_pnginfo else extra_pnginfo
                if current_workflow_info and isinstance(current_workflow_info, dict) and "workflow" in current_workflow_info:
                    node_data = self.find_node_by_id(unique_id, current_workflow_info["workflow"])
                    if node_data:
                        widget_index = self.find_widget_index(node_data, "editable_text")
                        if widget_index is not None:
                            if "widgets_values" not in node_data or not isinstance(node_data["widgets_values"], list):
                                req_widgets = len(self.INPUT_TYPES().get("required", {}))
                                opt_widgets = len(self.INPUT_TYPES().get("optional", {}))
                                num_widgets = req_widgets + opt_widgets
                                node_data["widgets_values"] = ["" for _ in range(num_widgets)]
                            while len(node_data["widgets_values"]) <= widget_index:
                                node_data["widgets_values"].append("")
                            node_data["widgets_values"][widget_index] = content
            
            return {"ui": {"text": [content]}, "result": (content, f"[SUCCESS] Successfully loaded: {os.path.basename(resolved_path)} ({len(content)} characters)")}
            
        except UnicodeDecodeError as e:
            # Try with different encoding if utf-8 fails
            if encoding != "utf-8":
                if editable_text:
                    return {"ui": {"text": [editable_text]}, "result": (editable_text, f"[ERROR] Encoding error with {encoding}: {str(e)} - keeping current text")}
                return {"ui": {"text": [""]}, "result": ("", f"[ERROR] Encoding error with {encoding}: {str(e)}")}
            
            # Try with latin-1 as fallback
            try:
                with open(resolved_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                
                # Update UI widget if workflow info is available
                if unique_id and extra_pnginfo:
                    current_workflow_info = extra_pnginfo[0] if isinstance(extra_pnginfo, list) and extra_pnginfo else extra_pnginfo
                    if current_workflow_info and isinstance(current_workflow_info, dict) and "workflow" in current_workflow_info:
                        node_data = self.find_node_by_id(unique_id, current_workflow_info["workflow"])
                        if node_data:
                            widget_index = self.find_widget_index(node_data, "editable_text")
                            if widget_index is not None:
                                if "widgets_values" not in node_data or not isinstance(node_data["widgets_values"], list):
                                    req_widgets = len(self.INPUT_TYPES().get("required", {}))
                                    opt_widgets = len(self.INPUT_TYPES().get("optional", {}))
                                    num_widgets = req_widgets + opt_widgets
                                    node_data["widgets_values"] = ["" for _ in range(num_widgets)]
                                while len(node_data["widgets_values"]) <= widget_index:
                                    node_data["widgets_values"].append("")
                                node_data["widgets_values"][widget_index] = content
                
                return {"ui": {"text": [content]}, "result": (content, f"[SUCCESS] Loaded with latin-1: {os.path.basename(resolved_path)} ({len(content)} characters)")}
            except Exception as fallback_error:
                if editable_text:
                    return {"ui": {"text": [editable_text]}, "result": (editable_text, f"[ERROR] Encoding error: {str(fallback_error)} - keeping current text")}
                return {"ui": {"text": [""]}, "result": ("", f"[ERROR] Encoding error: {str(fallback_error)}")}
                
        except Exception as e:
            if editable_text:
                return {"ui": {"text": [editable_text]}, "result": (editable_text, f"[ERROR] Error reading file: {str(e)} - keeping current text")}
            return {"ui": {"text": [""]}, "result": ("", f"[ERROR] Error reading file: {str(e)}")}


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
    RETURN_NAMES = ("full_path", "filename")
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

class CYHTextFileEditorNode:
    """
    A node that provides an editable text widget with action modes.
    Works with text input from other sources (like Text File Loader).
    Provides action modes for controlling output behavior.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        widget_default_text = (
            "Editable text widget\n"
            "Action modes:\n"
            "- use_input: Outputs connected text, updates widget\n"
            "- use_edit_mute_input: Outputs widget text, mutes input\n"
        )
        return {
            "required": {
                "action": (["use_input", "use_edit_mute_input"], {"default": "use_input"})
            },
            "optional": {
                "editable_text_widget": ("STRING", {
                    "multiline": True,
                    "default": widget_default_text
                }),
                "input_text": ("STRING", {"default": ""})
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_text",)
    FUNCTION = "process_text"
    CATEGORY = FILE_CATEGORY
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, action="use_input", **kwargs):
        # Always return unique hash to force execution
        import hashlib
        import time
        import random
        
        hash_input = f"{action}_{time.time()}_{random.random()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    # --- Helper functions ---
    def find_node_by_id(self, unique_id, workflow_info):
        if not workflow_info or "nodes" not in workflow_info:
            print(f"[{self.__class__.__name__}] Helper Error: Invalid workflow_info.");
            return None
        target_id = str(unique_id[0]) if isinstance(unique_id, list) else str(unique_id)
        for node_data in workflow_info["nodes"]:
            if str(node_data.get("id")) == target_id:
                return node_data
        print(f"[{self.__class__.__name__}] Helper Error: Node ID {target_id} not found in workflow.");
        return None

    def find_widget_index(self, node_data, widget_name):
        # Combine required and optional keys to get all widget names
        req_keys = list(self.INPUT_TYPES().get("required", {}).keys())
        opt_keys = list(self.INPUT_TYPES().get("optional", {}).keys())
        all_keys = req_keys + opt_keys
        try:
            idx = all_keys.index(widget_name)
            return idx
        except ValueError:
            print(f"[{self.__class__.__name__}] Helper Error: Widget '{widget_name}' not found in INPUT_TYPES keys: {all_keys}")
            return None

    # --- Main Processing Function ---
    def process_text(self, action="use_input",
                      editable_text_widget="", input_text="",
                      unique_id=None, extra_pnginfo=None):
        output_text = ""
        text_for_widget_update = None
        class_name_log = self.__class__.__name__

        print(f"[{class_name_log}] Action: '{action}', Node ID: {unique_id}")

        # Use default if input is None
        effective_input_text = input_text if input_text is not None else self.INPUT_TYPES()['optional']['input_text'][1].get('default', '')

        # Determine which text to use based on action mode
        if action == "use_input":
            output_text = effective_input_text
            text_for_widget_update = output_text
            print(f"[{class_name_log}] Chose 'use_input'. Outputting input text ('{output_text[:60]}...'). Attempting UI widget update.")
        elif action == "use_edit_mute_input":
            output_text = editable_text_widget
            print(f"[{class_name_log}] Chose 'use_edit_mute_input'. Outputting widget text ('{output_text[:60]}...').")
        else:
            print(f"[{class_name_log}] Warning: Unknown action '{action}'. Defaulting to outputting widget text.")
            output_text = editable_text_widget

        # --- Attempt to update the UI widget ---
        node_data_updated = False
        if text_for_widget_update is not None and unique_id and extra_pnginfo:
            print(f"[{class_name_log}] Attempting UI widget update for node {unique_id[0]}...")
            current_workflow_info = extra_pnginfo[0] if isinstance(extra_pnginfo, list) and extra_pnginfo else extra_pnginfo
            if current_workflow_info and isinstance(current_workflow_info, dict) and "workflow" in current_workflow_info:
                node_data = self.find_node_by_id(unique_id, current_workflow_info["workflow"])
                if node_data:
                    widget_index = self.find_widget_index(node_data, "editable_text_widget")
                    if widget_index is not None:
                        if "widgets_values" not in node_data or not isinstance(node_data["widgets_values"], list):
                            req_widgets = len(self.INPUT_TYPES().get("required", {}))
                            opt_widgets = len(self.INPUT_TYPES().get("optional", {}))
                            num_widgets = req_widgets + opt_widgets
                            node_data["widgets_values"] = ["" for _ in range(num_widgets)]
                            print(f"[{class_name_log}] Initialized/Reset widgets_values.")
                        while len(node_data["widgets_values"]) <= widget_index:
                            node_data["widgets_values"].append("")
                            print(f"[{class_name_log}] Padded widgets_values.")
                        current_widget_val = node_data["widgets_values"][widget_index]
                        if current_widget_val != text_for_widget_update:
                            node_data["widgets_values"][widget_index] = text_for_widget_update
                            print(f"[{class_name_log}] ---> Set widgets_values[{widget_index}].")
                            node_data_updated = True
                        else:
                            print(f"[{class_name_log}] Widget value already matches target.")
            elif text_for_widget_update is not None:
                print(f"[{class_name_log}] Cannot attempt UI update - missing unique_id or extra_pnginfo.")

        text_to_show_in_ui = text_for_widget_update if text_for_widget_update is not None else editable_text_widget
        print(f"[{class_name_log}] Final Output Text Type: {type(output_text)}, Value: '{str(output_text)[:60]}...'")

        return {"ui": {"text": [str(text_to_show_in_ui)]}, "result": (str(output_text),)}

# Node registration for this category
NODE_CLASS_MAPPINGS = {
    "CYHFolderFilenameBuilderNode": CYHFolderFilenameBuilderNode,
    "CYHTextFileLoaderNode": CYHTextFileLoaderNode,
    "CYHTextFileEditorNode": CYHTextFileEditorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CYHFolderFilenameBuilderNode": "📁 CYH File | Folder Filename Builder",
    "CYHTextFileLoaderNode": "📄 CYH File | Text File Loader",
    "CYHTextFileEditorNode": "📄 CYH File | Text File Editor",
}