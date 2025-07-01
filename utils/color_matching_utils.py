# utils/color_matching_utils.py
"""
Color Matching Utilities
Converts hex colors to color names for psychology analysis
"""

import colorsys
import math
from typing import Tuple, List, Dict, Optional

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to RGB tuple
    
    Args:
        hex_color: Hex color string (e.g., "#9c774b", "9c774b")
        
    Returns:
        RGB tuple (r, g, b) with values 0-255
    """
    try:
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        
        # Ensure 6 characters
        if len(hex_color) != 6:
            raise ValueError(f"Invalid hex color format: {hex_color}")
        
        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        return (r, g, b)
        
    except ValueError as e:
        print(f"Error converting hex to RGB: {e}")
        return (128, 128, 128)  # Default gray

def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    Convert RGB to HSV
    
    Args:
        r, g, b: RGB values 0-255
        
    Returns:
        HSV tuple (h, s, v) where:
        - h: hue 0-360 degrees
        - s: saturation 0-1
        - v: value 0-1
    """
    try:
        # Normalize RGB to 0-1
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        # Convert to HSV using colorsys
        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        
        # Convert hue to degrees
        h_degrees = h * 360.0
        
        return (h_degrees, s, v)
        
    except Exception as e:
        print(f"Error converting RGB to HSV: {e}")
        return (0.0, 0.0, 0.5)  # Default neutral

def hex_to_hsv(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert hex color directly to HSV
    
    Args:
        hex_color: Hex color string
        
    Returns:
        HSV tuple (h, s, v)
    """
    r, g, b = hex_to_rgb(hex_color)
    return rgb_to_hsv(r, g, b)

def normalize_hue(hue: float) -> float:
    """
    Normalize hue to 0-360 range
    
    Args:
        hue: Hue value
        
    Returns:
        Normalized hue 0-360
    """
    while hue < 0:
        hue += 360
    while hue >= 360:
        hue -= 360
    return hue

def hue_distance(hue1: float, hue2: float) -> float:
    """
    Calculate shortest distance between two hues (considering circular nature)
    
    Args:
        hue1, hue2: Hue values in degrees
        
    Returns:
        Shortest distance between hues
    """
    hue1 = normalize_hue(hue1)
    hue2 = normalize_hue(hue2)
    
    diff = abs(hue1 - hue2)
    return min(diff, 360 - diff)

def is_in_hue_range(hue: float, hue_ranges: List) -> bool:
    """
    Check if hue is within specified ranges
    
    Args:
        hue: Hue value to check
        hue_ranges: List of hue values or ranges
        
    Returns:
        True if hue is in any of the ranges
    """
    try:
        hue = normalize_hue(hue)
        
        # Handle different formats in hue_ranges
        for i in range(0, len(hue_ranges), 2):
            if i + 1 < len(hue_ranges):
                start_hue = hue_ranges[i]
                end_hue = hue_ranges[i + 1]
                
                # Handle wrapping around 360 (e.g., red: 350-360 and 0-10)
                if start_hue > end_hue:
                    if hue >= start_hue or hue <= end_hue:
                        return True
                else:
                    if start_hue <= hue <= end_hue:
                        return True
        
        return False
        
    except Exception as e:
        print(f"Error checking hue range: {e}")
        return False

def is_in_range(value: float, range_list: List[float]) -> bool:
    """
    Check if value is within range [min, max]
    
    Args:
        value: Value to check
        range_list: [min, max] range
        
    Returns:
        True if value is in range
    """
    try:
        if len(range_list) >= 2:
            return range_list[0] <= value <= range_list[1]
        return True
    except Exception:
        return True

def calculate_color_distance(hsv1: Tuple[float, float, float], hsv2: Tuple[float, float, float]) -> float:
    """
    Calculate perceptual distance between two HSV colors
    
    Args:
        hsv1, hsv2: HSV tuples
        
    Returns:
        Distance score (lower = more similar)
    """
    try:
        h1, s1, v1 = hsv1
        h2, s2, v2 = hsv2
        
        # Hue distance (weighted by saturation)
        hue_dist = hue_distance(h1, h2) / 180.0  # Normalize to 0-2
        hue_weight = (s1 + s2) / 2.0  # Average saturation
        weighted_hue_dist = hue_dist * hue_weight
        
        # Saturation and value distances
        sat_dist = abs(s1 - s2)
        val_dist = abs(v1 - v2)
        
        # Combined distance
        total_distance = (weighted_hue_dist * 0.6) + (sat_dist * 0.2) + (val_dist * 0.2)
        
        return total_distance
        
    except Exception as e:
        print(f"Error calculating color distance: {e}")
        return 1.0  # Maximum distance

def get_color_brightness_category(hsv: Tuple[float, float, float]) -> str:
    """
    Categorize color brightness
    
    Args:
        hsv: HSV tuple
        
    Returns:
        Brightness category ("very_dark", "dark", "medium", "light", "very_light")
    """
    h, s, v = hsv
    
    if v <= 0.2:
        return "very_dark"
    elif v <= 0.4:
        return "dark"
    elif v <= 0.7:
        return "medium"
    elif v <= 0.9:
        return "light"
    else:
        return "very_light"

def get_color_saturation_category(hsv: Tuple[float, float, float]) -> str:
    """
    Categorize color saturation
    
    Args:
        hsv: HSV tuple
        
    Returns:
        Saturation category ("gray", "muted", "moderate", "vibrant", "intense")
    """
    h, s, v = hsv
    
    if s <= 0.1:
        return "gray"
    elif s <= 0.3:
        return "muted"
    elif s <= 0.6:
        return "moderate"
    elif s <= 0.8:
        return "vibrant"
    else:
        return "intense"

# Export all functions
__all__ = [
    "hex_to_rgb",
    "rgb_to_hsv", 
    "hex_to_hsv",
    "normalize_hue",
    "hue_distance",
    "is_in_hue_range",
    "is_in_range",
    "calculate_color_distance",
    "get_color_brightness_category",
    "get_color_saturation_category"
]