"""
Input validation utilities for Chye ComfyUI Toolset
"""

from .constants import (
    MIN_MULTIPLIER, MAX_MULTIPLIER,
    MIN_BATCH_SIZE, MAX_BATCH_SIZE,
    ORIENTATIONS
)

def validate_aspect_ratio(aspect_ratio: str, valid_ratios: list) -> bool:
    """Validate aspect ratio selection"""
    return aspect_ratio in valid_ratios

def validate_multiplier(multiplier: float) -> bool:
    """Validate multiplier is within acceptable range"""
    return MIN_MULTIPLIER <= multiplier <= MAX_MULTIPLIER

def validate_batch_size(batch_size: int) -> bool:
    """Validate batch size is within acceptable range"""
    return MIN_BATCH_SIZE <= batch_size <= MAX_BATCH_SIZE

def validate_orientation(orientation: str) -> bool:
    """Validate orientation selection"""
    return orientation in ORIENTATIONS

def validate_inputs(**kwargs) -> bool:
    """Comprehensive input validation - can be extended for specific node needs"""
    return True  # Base implementation - override in specific nodes if needed