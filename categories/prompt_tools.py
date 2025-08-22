"""
Prompt enhancement tools for Chye ComfyUI Toolset
Uses LLM APIs to refine and enhance image generation prompts
"""

import sys
import os
import json
import time
import logging
import requests
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add parent directory to Python path for ComfyUI compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from shared.constants import PROMPT_TOOLS_CATEGORY
except ImportError:
    # Fallback import for ComfyUI environments
    import importlib.util
    
    # Import constants
    constants_path = os.path.join(parent_dir, "shared", "constants.py")
    spec = importlib.util.spec_from_file_location("constants", constants_path)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    
    # Define PROMPT_TOOLS_CATEGORY if not in constants
    if hasattr(constants, 'PROMPT_TOOLS_CATEGORY'):
        PROMPT_TOOLS_CATEGORY = constants.PROMPT_TOOLS_CATEGORY
    else:
        PROMPT_TOOLS_CATEGORY = "prompt_tools"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PREPROMPTS_DIR = os.path.join(parent_dir, "preprompts")
API_KEYS_DIR = os.path.join(PREPROMPTS_DIR, "api_keys")
TEMPLATES_DIR = os.path.join(PREPROMPTS_DIR, "providers")

# Provider-specific configurations
PROVIDER_CONFIGS = {
    "deepseek": {
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
        "max_tokens": 2000,
        "temperature": 0.7,
    },
    "openai": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4-turbo-preview",
        "max_tokens": 2000,
        "temperature": 0.7,
    },
    "anthropic": {
        "api_url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 2000,
        "temperature": 0.7,
    }
}

# Example template content
PROFESSIONAL_PHOTO_TEMPLATE = """You are a professional photography prompt engineer. Enhance image generation prompts with:

Technical Excellence:
- Professional camera settings (aperture, shutter speed, ISO)
- Lighting conditions and quality (golden hour, studio lighting, natural light)
- Composition techniques (rule of thirds, leading lines, framing)

Artistic Quality:
- Professional photography terminology
- Realistic details and textures
- Mood and atmosphere descriptors

Enhance this prompt while maintaining its core meaning:
{prompt}"""

CINEMATIC_TEMPLATE = """You are a cinematic prompt engineer. Enhance image generation prompts with:

Cinematic Elements:
- Film camera terminology (anamorphic lenses, film stock, cinematic lighting)
- Director and cinematographer styles (Nolan, Villeneuve, Deakins)
- Movie genres and visual styles (sci-fi, noir, fantasy, drama)

Visual Storytelling:
- Camera angles and movements (dolly shot, Dutch angle, tracking shot)
- Color grading and cinematic color palettes
- Atmospheric effects (volumetric lighting, lens flares, film grain)

Enhance this prompt with cinematic quality:
{prompt}"""

HIGH_DETAIL_TEMPLATE = """You are a detail-oriented prompt engineer. Enhance image generation prompts with:

Ultra Detail:
- Hyper-realistic textures and materials
- Intricate patterns and fine details
- Microscopic and macroscopic elements

Technical Precision:
- Scientific and technical accuracy
- Engineering and architectural details
- Natural phenomena and physical properties

Add exceptional detail to this prompt:
{prompt}"""

class PromptToolsSetup:
    """
    Setup node for prompt enhancement tools.
    Creates folder structure and example templates.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_recreate": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "setup"
    CATEGORY = PROMPT_TOOLS_CATEGORY
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def setup(self, force_recreate=False):
        """Create folder structure and example templates"""
        try:
            # Create base directories
            base_dirs = [
                "api_keys",
                "providers/deepseek",
                "providers/openai", 
                "providers/anthropic",
                "styles",
                "quality",
                "models",
                "custom"
            ]
            
            for dir_path in base_dirs:
                full_path = os.path.join(PREPROMPTS_DIR, dir_path)
                os.makedirs(full_path, exist_ok=True)
                logger.info(f"Created directory: {full_path}")
            
            # Create example template files
            example_templates = {
                "providers/deepseek/professional_photo.txt": PROFESSIONAL_PHOTO_TEMPLATE,
                "providers/openai/cinematic.txt": CINEMATIC_TEMPLATE,
                "providers/anthropic/high_detail.txt": HIGH_DETAIL_TEMPLATE,
                "styles/cinematic.txt": CINEMATIC_TEMPLATE,
                "quality/high_detail.txt": HIGH_DETAIL_TEMPLATE
            }
            
            for rel_path, content in example_templates.items():
                full_path = os.path.join(PREPROMPTS_DIR, rel_path)
                if force_recreate or not os.path.exists(full_path):
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.info(f"Created template: {full_path}")
            
            # Create example key files with instructions
            key_files = {
                "api_keys/deepseek_api_key.txt": "# Replace with your DeepSeek API key\n# Format: sk-your-actual-api-key-here\n# Get from: https://platform.deepseek.com/api_keys",
                "api_keys/openai_api_key.txt": "# Replace with your OpenAI API key\n# Format: sk-your-actual-api-key-here\n# Get from: https://platform.openai.com/api-keys",
                "api_keys/anthropic_api_key.txt": "# Replace with your Anthropic API key\n# Format: sk-your-actual-api-key-here\n# Get from: https://console.anthropic.com/settings/keys"
            }
            
            for rel_path, content in key_files.items():
                full_path = os.path.join(PREPROMPTS_DIR, rel_path)
                if force_recreate or not os.path.exists(full_path):
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.info(f"Created key file: {full_path}")
            
            # Create .gitignore for API keys
            gitignore_path = os.path.join(PREPROMPTS_DIR, ".gitignore")
            gitignore_content = "# API keys should never be committed\napi_keys/*\n!api_keys/.gitkeep\n"
            
            if force_recreate or not os.path.exists(gitignore_path):
                with open(gitignore_path, "w", encoding="utf-8") as f:
                    f.write(gitignore_content)
                logger.info(f"Created .gitignore: {gitignore_path}")
            
            return ("✅ Setup completed successfully! Check the preprompts/ folder for templates and API key files.",)
            
        except Exception as e:
            error_msg = f"❌ Setup failed: {str(e)}"
            logger.error(error_msg)
            return (error_msg,)

class PromptEnhancer:
    """
    Main prompt enhancement node that uses LLM APIs to refine prompts.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "template_path": ("STRING", {"default": "providers/deepseek/professional_photo.txt"}),
                "provider": (["deepseek", "openai", "anthropic", "custom"], {"default": "deepseek"}),
            },
            "optional": {
                "api_key_override": ("STRING", {"default": "", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1000, "min": 100, "max": 4000, "step": 100}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance"
    CATEGORY = PROMPT_TOOLS_CATEGORY
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def load_template(self, template_path: str) -> str:
        """Load prompt template from file"""
        try:
            full_path = os.path.join(PREPROMPTS_DIR, template_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Template file not found: {template_path}")
            
            with open(full_path, "r", encoding="utf-8") as f:
                template = f.read().strip()
            
            if "{prompt}" not in template:
                raise ValueError("Template must contain {prompt} placeholder")
            
            return template
            
        except Exception as e:
            logger.error(f"Template loading error: {e}")
            raise

    def get_api_key(self, provider: str, api_key_override: str = "") -> str:
        """Get API key from file or override"""
        if api_key_override.strip():
            return api_key_override.strip()
        
        key_file = os.path.join(API_KEYS_DIR, f"{provider}_api_key.txt")
        if not os.path.exists(key_file):
            raise FileNotFoundError(f"API key file not found: {key_file}")
        
        with open(key_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        # Extract API key (skip comment lines)
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                return line
        
        raise ValueError(f"No valid API key found in {key_file}")

    def call_deepseek(self, prompt: str, template: str, api_key: str, temperature: float, max_tokens: int) -> str:
        """Call DeepSeek API for prompt enhancement"""
        full_prompt = template.format(prompt=prompt)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": PROVIDER_CONFIGS["deepseek"]["model"],
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = requests.post(
            PROVIDER_CONFIGS["deepseek"]["api_url"],
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def call_openai(self, prompt: str, template: str, api_key: str, temperature: float, max_tokens: int) -> str:
        """Call OpenAI API for prompt enhancement"""
        full_prompt = template.format(prompt=prompt)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": PROVIDER_CONFIGS["openai"]["model"],
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = requests.post(
            PROVIDER_CONFIGS["openai"]["api_url"],
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def call_anthropic(self, prompt: str, template: str, api_key: str, temperature: float, max_tokens: int) -> str:
        """Call Anthropic API for prompt enhancement"""
        full_prompt = template.format(prompt=prompt)
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": PROVIDER_CONFIGS["anthropic"]["model"],
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            PROVIDER_CONFIGS["anthropic"]["api_url"],
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Anthropic API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["content"][0]["text"].strip()

    def enhance(self, prompt: str, template_path: str, provider: str, 
                api_key_override: str = "", temperature: float = 0.7, max_tokens: int = 1000):
        """Main enhancement function"""
        try:
            # Check if setup is needed
            if not os.path.exists(PREPROMPTS_DIR):
                return ("❌ Please run PromptTools Setup node first to create folder structure",)
            
            # Load template
            template = self.load_template(template_path)
            
            # Get API key
            api_key = self.get_api_key(provider, api_key_override)
            
            # Call appropriate provider
            if provider == "deepseek":
                enhanced = self.call_deepseek(prompt, template, api_key, temperature, max_tokens)
            elif provider == "openai":
                enhanced = self.call_openai(prompt, template, api_key, temperature, max_tokens)
            elif provider == "anthropic":
                enhanced = self.call_anthropic(prompt, template, api_key, temperature, max_tokens)
            else:
                # Custom provider - use template directly
                enhanced = template.format(prompt=prompt)
            
            return (enhanced,)
            
        except FileNotFoundError as e:
            error_msg = f"❌ File error: {str(e)}. Please run PromptTools Setup."
            logger.error(error_msg)
            return (error_msg,)
        except ValueError as e:
            error_msg = f"❌ Configuration error: {str(e)}"
            logger.error(error_msg)
            return (error_msg,)
        except Exception as e:
            error_msg = f"❌ Enhancement failed: {str(e)}"
            logger.error(error_msg)
            return (error_msg,)

class PromptEnhancerEditable:
    """
    Alternative prompt enhancement node with editable preprompt template.
    Allows direct editing of the preprompt text instead of loading from files.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "preprompt_template": ("STRING", {
                    "multiline": True,
                    "default": PROFESSIONAL_PHOTO_TEMPLATE,
                    "placeholder": "Enter your preprompt template here. Must include {prompt} placeholder."
                }),
                "provider": (["deepseek", "openai", "anthropic", "custom"], {"default": "deepseek"}),
            },
            "optional": {
                "api_key_override": ("STRING", {"default": "", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1000, "min": 100, "max": 4000, "step": 100}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance"
    CATEGORY = PROMPT_TOOLS_CATEGORY
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def get_api_key(self, provider: str, api_key_override: str = "") -> str:
        """Get API key from file or override"""
        if api_key_override.strip():
            return api_key_override.strip()
        
        key_file = os.path.join(API_KEYS_DIR, f"{provider}_api_key.txt")
        if not os.path.exists(key_file):
            raise FileNotFoundError(f"API key file not found: {key_file}")
        
        with open(key_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        # Extract API key (skip comment lines)
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                return line
        
        raise ValueError(f"No valid API key found in {key_file}")

    def call_deepseek(self, prompt: str, template: str, api_key: str, temperature: float, max_tokens: int) -> str:
        """Call DeepSeek API for prompt enhancement"""
        full_prompt = template.format(prompt=prompt)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": PROVIDER_CONFIGS["deepseek"]["model"],
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = requests.post(
            PROVIDER_CONFIGS["deepseek"]["api_url"],
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def call_openai(self, prompt: str, template: str, api_key: str, temperature: float, max_tokens: int) -> str:
        """Call OpenAI API for prompt enhancement"""
        full_prompt = template.format(prompt=prompt)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": PROVIDER_CONFIGS["openai"]["model"],
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = requests.post(
            PROVIDER_CONFIGS["openai"]["api_url"],
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def call_anthropic(self, prompt: str, template: str, api_key: str, temperature: float, max_tokens: int) -> str:
        """Call Anthropic API for prompt enhancement"""
        full_prompt = template.format(prompt=prompt)
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": PROVIDER_CONFIGS["anthropic"]["model"],
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            PROVIDER_CONFIGS["anthropic"]["api_url"],
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Anthropic API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["content"][0]["text"].strip()

    def enhance(self, prompt: str, preprompt_template: str, provider: str,
                api_key_override: str = "", temperature: float = 0.7, max_tokens: int = 1000):
        """Main enhancement function with editable template"""
        try:
            # Check if setup is needed for API keys
            if not os.path.exists(PREPROMPTS_DIR) and provider != "custom":
                return ("❌ Please run PromptTools Setup node first to create API key files",)
            
            # Validate template contains {prompt} placeholder
            if "{prompt}" not in preprompt_template:
                return ("❌ Template must contain {prompt} placeholder. Please add it to your template.",)
            
            # Get API key (for non-custom providers)
            api_key = ""
            if provider != "custom":
                api_key = self.get_api_key(provider, api_key_override)
            
            # Call appropriate provider
            if provider == "deepseek":
                enhanced = self.call_deepseek(prompt, preprompt_template, api_key, temperature, max_tokens)
            elif provider == "openai":
                enhanced = self.call_openai(prompt, preprompt_template, api_key, temperature, max_tokens)
            elif provider == "anthropic":
                enhanced = self.call_anthropic(prompt, preprompt_template, api_key, temperature, max_tokens)
            else:
                # Custom provider - use template directly
                enhanced = preprompt_template.format(prompt=prompt)
            
            return (enhanced,)
            
        except FileNotFoundError as e:
            error_msg = f"❌ File error: {str(e)}. Please run PromptTools Setup."
            logger.error(error_msg)
            return (error_msg,)
        except ValueError as e:
            error_msg = f"❌ Configuration error: {str(e)}"
            logger.error(error_msg)
            return (error_msg,)
        except Exception as e:
            error_msg = f"❌ Enhancement failed: {str(e)}"
            logger.error(error_msg)
            return (error_msg,)

# Node registration for this category
NODE_CLASS_MAPPINGS = {
    "PromptToolsSetup": PromptToolsSetup,
    "PromptEnhancer": PromptEnhancer,
    "PromptEnhancerEditable": PromptEnhancerEditable,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptToolsSetup": "🔤 CYH Prompt Tools | Setup",
    "PromptEnhancer": "🔤 CYH Prompt Tools | Enhancer",
    "PromptEnhancerEditable": "🔤 CYH Prompt Tools | Enhancer (Editable)",
}