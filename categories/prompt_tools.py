"""
Prompt enhancement tools for Chye ComfyUI Toolset
Uses LLM APIs to refine and enhance image generation prompts

Folder Structure:
- Presets/ (main directory)
  - Preprompts/ (template files - .txt)
  - api_keys/ (API key files - .txt)
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

# Configure logging with cross-platform compatibility
logger = logging.getLogger(__name__)
try:
    # Try to use colored logging for better readability (works on most systems)
    import coloredlogs
    coloredlogs.install(level=logging.DEBUG, logger=logger,
                       fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
except ImportError:
    # Fallback to standard logging if coloredlogs is not available
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Constants with cross-platform path handling - Updated for Presets structure
PRESETS_DIR = Path(parent_dir) / "Presets"
TEMPLATES_DIR = PRESETS_DIR / "Preprompts"  # Main templates directory
API_KEYS_DIR = PRESETS_DIR / "api_keys"  # API keys directory
OLD_PREPROMPTS_DIR = Path(parent_dir) / "preprompts"  # For migration

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

def scan_template_files():
    """Scan the templates directory and return all .txt files with friendly names"""
    try:
        if not TEMPLATES_DIR.exists():
            logger.warning(f"Templates directory not found: {TEMPLATES_DIR}")
            return []
        
        template_files = []
        for file_path in TEMPLATES_DIR.glob("*.txt"):
            # Create friendly name from filename
            friendly_name = file_path.stem.replace('_', ' ').title()
            template_files.append((friendly_name, file_path.name))
        
        # Sort alphabetically by friendly name
        template_files.sort(key=lambda x: x[0])
        
        logger.debug(f"Found {len(template_files)} template files: {[f[0] for f in template_files]}")
        return template_files
        
    except Exception as e:
        logger.error(f"Error scanning template files: {e}")
        return []

def migrate_old_templates():
    """Migrate templates from old preprompts structure to new Presets structure"""
    try:
        if not OLD_PREPROMPTS_DIR.exists():
            logger.debug("No old preprompts directory found for migration")
            return False
        
        # Create new directories if they don't exist
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        API_KEYS_DIR.mkdir(parents=True, exist_ok=True)
        
        migrated_count = 0
        
        # Find all .txt files in old structure
        for old_file in OLD_PREPROMPTS_DIR.rglob("*.txt"):
            if "api_keys" in str(old_file):
                # Skip API key files - they'll be handled separately
                continue
                
            new_file = TEMPLATES_DIR / old_file.name
            
            # Handle filename conflicts
            counter = 1
            while new_file.exists():
                stem = old_file.stem
                suffix = old_file.suffix
                new_file = TEMPLATES_DIR / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Copy file to new location
            import shutil
            shutil.copy2(old_file, new_file)
            migrated_count += 1
            logger.info(f"Migrated template: {old_file} -> {new_file}")
        
        # Migrate API keys if they exist
        old_api_keys_dir = OLD_PREPROMPTS_DIR / "api_keys"
        if old_api_keys_dir.exists():
            for old_key_file in old_api_keys_dir.glob("*.txt"):
                new_key_file = API_KEYS_DIR / old_key_file.name
                if not new_key_file.exists():
                    shutil.copy2(old_key_file, new_key_file)
                    logger.info(f"Migrated API key: {old_key_file} -> {new_key_file}")
        
        logger.info(f"Migration completed. Moved {migrated_count} template files.")
        return migrated_count > 0
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False

class PromptToolsSetup:
    """
    Setup node for prompt enhancement tools.
    Creates new Presets folder structure and example templates.
    
    Creates:
    - Presets/Preprompts/ (for template files)
    - Presets/api_keys/ (for API key files)
    - Example template files
    - Example API key files with instructions
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
            # Create new Presets directory structure
            base_dirs = [
                "Preprompts",  # Main templates directory
                "api_keys", # API keys directory
            ]
            
            for dir_path in base_dirs:
                full_path = PRESETS_DIR / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {full_path}")
            
            # Create example template files in flat structure
            example_templates = {
                "professional_photo.txt": PROFESSIONAL_PHOTO_TEMPLATE,
                "cinematic.txt": CINEMATIC_TEMPLATE,
                "high_detail.txt": HIGH_DETAIL_TEMPLATE,
            }
            
            for filename, content in example_templates.items():
                full_path = TEMPLATES_DIR / filename
                if force_recreate or not full_path.exists():
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.info(f"Created template: {full_path}")
            
            # Create example key files with instructions
            key_files = {
                "deepseek_api_key.txt": "# Replace with your DeepSeek API key\n# Format: sk-your-actual-api-key-here\n# Get from: https://platform.deepseek.com/api_keys",
                "openai_api_key.txt": "# Replace with your OpenAI API key\n# Format: sk-your-actual-api-key-here\n# Get from: https://platform.openai.com/api-keys",
                "anthropic_api_key.txt": "# Replace with your Anthropic API key\n# Format: sk-your-actual-api-key-here\n# Get from: https://console.anthropic.com/settings/keys"
            }
            
            for filename, content in key_files.items():
                full_path = API_KEYS_DIR / filename
                if force_recreate or not full_path.exists():
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.info(f"Created key file: {full_path}")
            
            # Create .gitignore for API keys
            gitignore_path = PRESETS_DIR / ".gitignore"
            gitignore_content = "# API keys should never be committed\napi_keys/*\n!api_keys/.gitkeep\n"
            
            if force_recreate or not gitignore_path.exists():
                with open(gitignore_path, "w", encoding="utf-8") as f:
                    f.write(gitignore_content)
                logger.info(f"Created .gitignore: {gitignore_path}")
            
            # Try to migrate existing templates from old structure
            migrated = migrate_old_templates()
            migration_msg = " (migrated existing templates)" if migrated else ""
            
            return (f"✅ Setup completed successfully! Check the Presets/ folder for templates and API key files.{migration_msg}",)
            
        except Exception as e:
            error_msg = f"❌ Setup failed: {str(e)}"
            logger.error(error_msg)
            return (error_msg,)

class PromptEnhancer:
    """
    Main prompt enhancement node that uses LLM APIs to refine prompts.
    
    Features:
    - Dropdown menu for template selection (scans Presets/Presets/*.txt files)
    - Dynamic template scanning on menu open
    - Backward compatibility with old preprompts structure
    - Support for multiple AI providers (DeepSeek, OpenAI, Anthropic)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get template files dynamically for dropdown
        template_files = scan_template_files()
        template_options = ["custom"]  # Always include custom option
        
        if template_files:
            # Add all found templates to dropdown
            template_options.extend([f[0] for f in template_files])
        else:
            # Fallback to default templates if none found
            template_options.extend([
                "Professional Photo",
                "Cinematic",
                "High Detail"
            ])
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "template": (template_options, {"default": "Professional Photo"}),
                "provider": (["deepseek", "openai", "anthropic", "custom"], {"default": "deepseek"}),
            },
            "optional": {
                "custom_template_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to custom template file (relative to Presets/Preprompts/)"
                }),
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

    def load_template_from_file(self, template_name: str) -> str:
        """Load prompt template from file using friendly name"""
        try:
            # Find the actual filename from the friendly name
            template_files = scan_template_files()
            actual_filename = None
            
            for friendly_name, filename in template_files:
                if friendly_name == template_name:
                    actual_filename = filename
                    break
            
            if not actual_filename:
                logger.error(f"Template not found: {template_name}")
                raise FileNotFoundError(f"Template not found: {template_name}")
            
            full_path = TEMPLATES_DIR / actual_filename
            logger.debug(f"Loading template from: {full_path}")
            
            if not full_path.exists():
                logger.error(f"Template file not found: {full_path}")
                raise FileNotFoundError(f"Template file not found: {full_path}")
            
            with open(full_path, "r", encoding="utf-8") as f:
                template = f.read().strip()
            
            logger.debug(f"Template loaded. Length: {len(template)} characters")
            
            if "{prompt}" not in template:
                logger.error("Template missing {prompt} placeholder")
                raise ValueError("Template must contain {prompt} placeholder")
            
            logger.debug("Template validation successful")
            return template
            
        except Exception as e:
            logger.error(f"Template loading error: {e}")
            raise

    def load_custom_template(self, custom_template_path: str) -> str:
        """Load custom template from specified path"""
        try:
            full_path = TEMPLATES_DIR / custom_template_path
            logger.debug(f"Loading custom template from: {full_path}")
            
            if not full_path.exists():
                logger.error(f"Custom template file not found: {full_path}")
                raise FileNotFoundError(f"Custom template file not found: {custom_template_path}")
            
            with open(full_path, "r", encoding="utf-8") as f:
                template = f.read().strip()
            
            logger.debug(f"Custom template loaded. Length: {len(template)} characters")
            
            if "{prompt}" not in template:
                logger.error("Custom template missing {prompt} placeholder")
                raise ValueError("Custom template must contain {prompt} placeholder")
            
            logger.debug("Custom template validation successful")
            return template
            
        except Exception as e:
            logger.error(f"Custom template loading error: {e}")
            raise

    def get_api_key(self, provider: str, api_key_override: str = "") -> str:
        """Get API key from file or override"""
        logger.debug(f"Getting API key for provider: {provider}")
        logger.debug(f"API key override provided: {'Yes' if api_key_override.strip() else 'No'}")
        
        if api_key_override.strip():
            logger.debug("Using API key override")
            return api_key_override.strip()
        
        key_file = API_KEYS_DIR / f"{provider}_api_key.txt"
        logger.debug(f"Looking for API key file at: {key_file}")
        
        if not key_file.exists():
            logger.error(f"API key file not found: {key_file}")
            raise FileNotFoundError(f"API key file not found: {key_file}")
        
        with open(key_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        logger.debug(f"API key file content (first 50 chars): {content[:50]}...")
        
        # Extract API key (skip comment lines)
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                logger.debug(f"Found API key: {line[:10]}...{line[-4:]}")
                return line
        
        logger.error(f"No valid API key found in {key_file}")
        raise ValueError(f"No valid API key found in {key_file}")

    def call_deepseek(self, prompt: str, template: str, api_key: str, temperature: float, max_tokens: int) -> str:
        """Call DeepSeek API for prompt enhancement"""
        logger.debug("Calling DeepSeek API for prompt enhancement")
        logger.debug(f"Using model: {PROVIDER_CONFIGS['deepseek']['model']}")
        logger.debug(f"Temperature: {temperature}, Max tokens: {max_tokens}")
        
        full_prompt = template.format(prompt=prompt)
        logger.debug(f"Full prompt length: {len(full_prompt)} characters")
        
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
        
        logger.debug(f"Sending request to: {PROVIDER_CONFIGS['deepseek']['api_url']}")
        
        try:
            response = requests.post(
                PROVIDER_CONFIGS["deepseek"]["api_url"],
                headers=headers,
                json=payload,
                timeout=30
            )
            
            logger.debug(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
            
            result = response.json()
            enhanced_prompt = result["choices"][0]["message"]["content"].strip()
            logger.debug(f"Successfully enhanced prompt. Length: {len(enhanced_prompt)} characters")
            return enhanced_prompt
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception when calling DeepSeek API: {str(e)}")
            raise

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

    def enhance(self, prompt: str, template: str, provider: str,
                custom_template_path: str = "", api_key_override: str = "",
                temperature: float = 0.7, max_tokens: int = 1000):
        """Main enhancement function"""
        logger.info(f"Starting prompt enhancement with provider: {provider}")
        logger.debug(f"Input prompt length: {len(prompt)} characters")
        logger.debug(f"Selected template: {template}")
        
        try:
            # Check if setup is needed
            if not PRESETS_DIR.exists():
                # Try to migrate from old structure first
                if migrate_old_templates():
                    logger.info("Successfully migrated from old preprompts structure")
                else:
                    logger.error(f"Presets directory not found: {PRESETS_DIR}")
                    return ("❌ Please run PromptTools Setup node first to create folder structure",)
            
            # Load appropriate template
            if template == "custom":
                if not custom_template_path:
                    error_msg = "❌ Custom template selected but no custom template path provided"
                    logger.error(error_msg)
                    return (error_msg,)
                logger.debug(f"Loading custom template from path: {custom_template_path}")
                template_content = self.load_custom_template(custom_template_path)
            else:
                logger.debug(f"Loading built-in template: {template}")
                template_content = self.load_template_from_file(template)
            
            logger.debug(f"Template loaded successfully. Length: {len(template_content)} characters")
            
            # Get API key
            api_key = self.get_api_key(provider, api_key_override)
            
            # Call appropriate provider
            if provider == "deepseek":
                logger.info("Using DeepSeek provider for enhancement")
                enhanced = self.call_deepseek(prompt, template_content, api_key, temperature, max_tokens)
            elif provider == "openai":
                logger.info("Using OpenAI provider for enhancement")
                enhanced = self.call_openai(prompt, template_content, api_key, temperature, max_tokens)
            elif provider == "anthropic":
                logger.info("Using Anthropic provider for enhancement")
                enhanced = self.call_anthropic(prompt, template_content, api_key, temperature, max_tokens)
            else:
                # Custom provider - use template directly
                logger.info("Using custom provider (template only)")
                enhanced = template_content.format(prompt=prompt)
            
            logger.info(f"Successfully enhanced prompt. Result length: {len(enhanced)} characters")
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
        logger.debug(f"Getting API key for provider: {provider}")
        logger.debug(f"API key override provided: {'Yes' if api_key_override.strip() else 'No'}")
        
        if api_key_override.strip():
            logger.debug("Using API key override")
            return api_key_override.strip()
        
        key_file = API_KEYS_DIR / f"{provider}_api_key.txt"
        logger.debug(f"Looking for API key file at: {key_file}")
        
        if not key_file.exists():
            logger.error(f"API key file not found: {key_file}")
            raise FileNotFoundError(f"API key file not found: {key_file}")
        
        with open(key_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        logger.debug(f"API key file content (first 50 chars): {content[:50]}...")
        
        # Extract API key (skip comment lines)
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                logger.debug(f"Found API key: {line[:10]}...{line[-4:]}")
                return line
        
        logger.error(f"No valid API key found in {key_file}")
        raise ValueError(f"No valid API key found in {key_file}")

    def call_deepseek(self, prompt: str, template: str, api_key: str, temperature: float, max_tokens: int) -> str:
        """Call DeepSeek API for prompt enhancement"""
        logger.debug("Calling DeepSeek API for prompt enhancement")
        logger.debug(f"Using model: {PROVIDER_CONFIGS['deepseek']['model']}")
        logger.debug(f"Temperature: {temperature}, Max tokens: {max_tokens}")
        
        full_prompt = template.format(prompt=prompt)
        logger.debug(f"Full prompt length: {len(full_prompt)} characters")
        
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
        
        logger.debug(f"Sending request to: {PROVIDER_CONFIGS['deepseek']['api_url']}")
        
        try:
            response = requests.post(
                PROVIDER_CONFIGS["deepseek"]["api_url"],
                headers=headers,
                json=payload,
                timeout=30
            )
            
            logger.debug(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
            
            result = response.json()
            enhanced_prompt = result["choices"][0]["message"]["content"].strip()
            logger.debug(f"Successfully enhanced prompt. Length: {len(enhanced_prompt)} characters")
            return enhanced_prompt
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception when calling DeepSeek API: {str(e)}")
            raise

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
        logger.info(f"Starting prompt enhancement with provider: {provider}")
        logger.debug(f"Input prompt length: {len(prompt)} characters")
        logger.debug(f"Preprompt template length: {len(preprompt_template)} characters")
        
        try:
            # Check if setup is needed for API keys
            if not PRESETS_DIR.exists() and provider != "custom":
                logger.error(f"Presets directory not found: {PRESETS_DIR}")
                return ("❌ Please run PromptTools Setup node first to create API key files",)
            
            # Validate template contains {prompt} placeholder
            if "{prompt}" not in preprompt_template:
                logger.error("Template missing {prompt} placeholder")
                return ("❌ Template must contain {prompt} placeholder. Please add it to your template.",)
            
            # Get API key (for non-custom providers)
            api_key = ""
            if provider != "custom":
                api_key = self.get_api_key(provider, api_key_override)
            
            # Call appropriate provider
            if provider == "deepseek":
                logger.info("Using DeepSeek provider for enhancement")
                enhanced = self.call_deepseek(prompt, preprompt_template, api_key, temperature, max_tokens)
            elif provider == "openai":
                logger.info("Using OpenAI provider for enhancement")
                enhanced = self.call_openai(prompt, preprompt_template, api_key, temperature, max_tokens)
            elif provider == "anthropic":
                logger.info("Using Anthropic provider for enhancement")
                enhanced = self.call_anthropic(prompt, preprompt_template, api_key, temperature, max_tokens)
            else:
                # Custom provider - use template directly
                logger.info("Using custom provider (template only)")
                enhanced = preprompt_template.format(prompt=prompt)
            
            logger.info(f"Successfully enhanced prompt. Result length: {len(enhanced)} characters")
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