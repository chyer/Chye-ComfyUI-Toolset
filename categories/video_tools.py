"""
Video tools for Chye ComfyUI Toolset
"""

import sys
import os
import shutil
import logging
import json
import subprocess
import tempfile
from pathlib import Path
from datetime import timedelta

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
    from shared.constants import VIDEO_CATEGORY
    from shared.helpers import sanitize_filename
except ImportError:
    # Fallback import for ComfyUI environments
    import importlib.util
    
    # Import constants
    constants_path = os.path.join(parent_dir, "shared", "constants.py")
    spec = importlib.util.spec_from_file_location("constants", constants_path)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    
    # Define VIDEO_CATEGORY if not in constants
    if hasattr(constants, 'VIDEO_CATEGORY'):
        VIDEO_CATEGORY = constants.VIDEO_CATEGORY
    else:
        VIDEO_CATEGORY = "video"
    
    # Import helpers
    helpers_path = os.path.join(parent_dir, "shared", "helpers.py")
    spec = importlib.util.spec_from_file_location("helpers", helpers_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    
    sanitize_filename = helpers.sanitize_filename if hasattr(helpers, 'sanitize_filename') else None

# Define sanitize_filename if not available from helpers
if sanitize_filename is None:
    import re
    def sanitize_filename(filename):
        """Remove invalid characters from filename"""
        return re.sub(r'[<>:"/\\|?*]', '', filename).strip()

logger = logging.getLogger(__name__)

class CYHPreviewVideo:
    """
    Preview Video
    
    A node that displays a video from a given file path without re-encoding or
    saving a new version. It's designed to preview videos that already exist on
    the hard drive, such as those produced by video generation or loading nodes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "placeholder": "/path/to/video.mp4",
                        "tooltip": "Absolute or relative path to the video file to preview.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_path", "video_info")
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = VIDEO_CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def get_video_info(self, video_path: str):
        """Extract video metadata information using ffprobe if available and return as formatted text"""
        if not video_path or not os.path.exists(video_path):
            filename = os.path.basename(video_path) if video_path else "unknown"
            return f"Filename: {filename}\nExists: No\nError: File does not exist"
        
        try:
            lines = []
            filename = os.path.basename(video_path)
            file_size = os.path.getsize(video_path)
            file_size_mb = round(file_size / (1024 * 1024), 2)
            modified_time = os.path.getmtime(video_path)
            
            # Format basic file information
            lines.append(f"Filename: {filename}")
            lines.append(f"File Size: {file_size_mb} MB ({file_size:,} bytes)")
            lines.append(f"Exists: Yes")
            
            # Convert Unix timestamp to readable date
            try:
                dt = datetime.fromtimestamp(modified_time)
                formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
                lines.append(f"Last Modified: {formatted_date} (Unix timestamp: {modified_time})")
            except:
                lines.append(f"Last Modified: Unix timestamp {modified_time}")
            
            # Try to use ffprobe for detailed video information if available
            try:
                # Check if ffprobe is available
                result = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    probe_data = json.loads(result.stdout)
                    
                    # Add separator before video-specific info
                    lines.append("")
                    
                    # Extract format information
                    if 'format' in probe_data:
                        format_info = probe_data['format']
                        format_name = format_info.get('format_name', 'unknown')
                        if format_name and format_name != 'unknown':
                            lines.append(f"Format Names: {format_name.replace(',', ', ')}")
                        
                        duration = float(format_info.get('duration', 0))
                        if duration > 0:
                            duration_formatted = str(timedelta(seconds=duration))
                            lines.append(f"Duration: {duration} seconds ({duration_formatted})")
                        
                        bit_rate = int(format_info.get('bit_rate', 0)) if format_info.get('bit_rate') else 0
                        if bit_rate > 0:
                            bit_rate_mbps = round(bit_rate / 1000000, 2)
                            lines.append(f"Bit Rate: {bit_rate_mbps} Mbps ({bit_rate:,} bps)")
                    
                    # Extract video stream information
                    video_streams = [s for s in probe_data.get('streams', []) if s.get('codec_type') == 'video']
                    if video_streams:
                        video_stream = video_streams[0]
                        lines.append("")
                        
                        video_codec = video_stream.get('codec_name', 'unknown')
                        if video_codec != 'unknown':
                            lines.append(f"Video Codec: {video_codec}")
                        
                        width = video_stream.get('width', 0)
                        height = video_stream.get('height', 0)
                        if width and height:
                            lines.append(f"Resolution: {width} x {height} pixels")
                        
                        frame_rate = video_stream.get('r_frame_rate', 'unknown')
                        if frame_rate != 'unknown':
                            lines.append(f"Frame Rate: {frame_rate} fps")
                        
                        pix_fmt = video_stream.get('pix_fmt', 'unknown')
                        if pix_fmt != 'unknown':
                            lines.append(f"Pixel Format: {pix_fmt}")
                    
                    # Extract audio stream information (keep as optional)
                    audio_streams = [s for s in probe_data.get('streams', []) if s.get('codec_type') == 'audio']
                    if audio_streams:
                        audio_stream = audio_streams[0]
                        audio_codec = audio_stream.get('codec_name', 'unknown')
                        if audio_codec != 'unknown':
                            lines.append("")
                            lines.append(f"Audio Codec: {audio_codec}")
                        
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
                # ffprobe not available or failed, add note
                lines.append("")
                lines.append("Note: Detailed video information requires ffprobe (FFmpeg)")
            except Exception as e:
                logger.warning("Failed to extract detailed video info: %s", e)
                lines.append("")
                lines.append(f"Note: Error extracting video info: {str(e)}")
            
            return "\n".join(lines)
            
        except Exception as e:
            # Fallback for basic file operations that might fail
            filename = os.path.basename(video_path) if video_path else "unknown"
            return f"Filename: {filename}\nExists: Yes\nError: Failed to get file info: {str(e)}"

    def preview(self, video_path: str):
        if not video_path or not os.path.exists(video_path):
            logger.warning("PreviewVideo: Video path is empty or file does not exist: %s", video_path)
            video_info_text = self.get_video_info(video_path)
            return {"ui": {"images": []}, "result": (None, video_path, video_info_text)}

        video_path = os.path.abspath(video_path)
        
        # Check if the video is in a web-accessible directory
        for dir_type, dir_path in [
            ("output", folder_paths.get_output_directory()),
            ("input", folder_paths.get_input_directory()),
            ("temp", folder_paths.get_temp_directory()),
        ]:
            try:
                abs_dir_path = os.path.abspath(dir_path)
                if os.path.commonpath([video_path, abs_dir_path]) == abs_dir_path:
                    relative_path = os.path.relpath(video_path, abs_dir_path)
                    subfolder, filename = os.path.split(relative_path)
                    video_info_text = self.get_video_info(video_path)
                    return {
                        "ui": {
                            "images": [
                                {
                                    "filename": filename,
                                    "subfolder": subfolder,
                                    "type": dir_type,
                                }
                            ],
                            "animated": (True,),
                        },
                        "result": (None, video_path, video_info_text),
                    }
            except Exception as e:
                logger.error("Error checking path %s against %s: %s", video_path, dir_path, e)

        # If not, copy to temp directory to make it accessible
        try:
            temp_dir = folder_paths.get_temp_directory()
            filename = os.path.basename(video_path)
            dest_path = os.path.join(temp_dir, filename)
            
            # To avoid re-copying, check if it already exists
            if not os.path.exists(dest_path) or os.path.getmtime(video_path) != os.path.getmtime(dest_path):
                shutil.copy2(video_path, dest_path)
                logger.info("Copied video to temp for preview: %s", dest_path)

            video_info_text = self.get_video_info(video_path)
            return {
                "ui": {
                    "images": [
                        {"filename": filename, "subfolder": "", "type": "temp"}
                    ],
                    "animated": (True,),
                },
                "result": (None, video_path, video_info_text),
            }
        except Exception as e:
            logger.error("Failed to copy video to temp directory for preview: %s", e, exc_info=True)
            video_info_text = self.get_video_info(video_path)
            return {"ui": {"images": []}, "result": (None, video_path, video_info_text)}

# Node registration for this category
NODE_CLASS_MAPPINGS = {
    "CYHPreviewVideo": CYHPreviewVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CYHPreviewVideo": "🎬 CYH Video | Preview Video",
}