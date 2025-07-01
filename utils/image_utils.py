import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import numpy as np
from typing import Tuple, Optional

def data_uri_to_pil_image(data_uri: str) -> Image.Image:
    """Converts a data URI to a PIL Image object."""
    try:
        if not data_uri.startswith('data:image'):
            raise ValueError("Data URI does not seem to represent an image.")
        
        header, encoded = data_uri.split(",", 1)
        # Further validation of header if necessary, e.g., check for base64
        if ';base64' not in header:
            raise ValueError("Data URI does not appear to be base64 encoded.")

        image_data = base64.b64decode(encoded)
        image_stream = BytesIO(image_data)
        image = Image.open(image_stream)
        return image
    except UnidentifiedImageError:
        raise ValueError("Cannot identify image file from data URI. The data may be corrupt or not a supported image format.")
    except base64.binascii.Error: # Error from b64decode
        raise ValueError("Invalid base64 encoding in data URI.")
    except ValueError as e: # Catch our own ValueErrors or others
        raise ValueError(f"Invalid data URI: {e}")
    except Exception as e: # Catch-all for other unexpected errors
        # It's good to log this error for debugging.
        print(f"Unexpected error converting data URI to PIL Image: {type(e).__name__} - {e}")
        raise ValueError(f"Could not process data URI: {type(e).__name__}")

def pil_image_to_data_uri(image: Image.Image, fmt: str = "PNG") -> str:
    """Converts a PIL Image object to a data URI."""
    # Ensure format is supported by PIL for saving
    supported_formats = Image.SAVE.keys()
    if fmt.upper() not in supported_formats and fmt.lower() not in supported_formats:
        # Fallback or raise error
        print(f"Warning: Format '{fmt}' might not be supported for saving. Defaulting to PNG.")
        fmt = "PNG" # Default to PNG if format is problematic

    buffered = BytesIO()
    
    # Handle potential issues with saving specific modes to specific formats
    # For example, saving a 'P' mode image (palette) as JPEG might require conversion.
    img_to_save = image
    if fmt.upper() == 'JPEG' and image.mode not in ('RGB', 'L', 'CMYK'):
        img_to_save = image.convert('RGB')
    elif image.mode == 'RGBA' and fmt.upper() == 'JPEG': # JPEG doesn't support alpha
        img_to_save = image.convert('RGB')

    img_to_save.save(buffered, format=fmt.upper())
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    mime_type = f"image/{fmt.lower()}"
    if fmt.upper() == "SVG": # Special case for SVG if ever supported by PIL save
        mime_type = "image/svg+xml"
        
    return f"data:{mime_type};base64,{img_str}"

def rgb_to_hex(rgb_tuple: Tuple[float, float, float]) -> str:
    """Converts an RGB tuple (components 0-255) to a hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2]))

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Converts a hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color format: {hex_color}")
    try:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        raise ValueError(f"Invalid hex color format: {hex_color}")

def rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to HSV color space."""
    r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # Hue calculation
    if diff == 0:
        h = 0
    elif max_val == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    # Saturation calculation
    s = 0 if max_val == 0 else diff / max_val
    
    # Value calculation
    v = max_val
    
    return h, s, v

def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to LAB color space (simplified conversion)."""
    try:
        # Normalize RGB values
        r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
        
        # Convert to XYZ (simplified sRGB to XYZ conversion)
        # Apply gamma correction
        def gamma_correct(c):
            if c > 0.04045:
                return pow((c + 0.055) / 1.055, 2.4)
            else:
                return c / 12.92
        
        r = gamma_correct(r)
        g = gamma_correct(g)
        b = gamma_correct(b)
        
        # Convert to XYZ using sRGB matrix
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        
        # Normalize by D65 illuminant
        x = x / 0.95047
        y = y / 1.00000
        z = z / 1.08883
        
        # Convert XYZ to LAB
        def xyz_to_lab_component(t):
            if t > 0.008856:
                return pow(t, 1/3)
            else:
                return (7.787 * t) + (16/116)
        
        fx = xyz_to_lab_component(x)
        fy = xyz_to_lab_component(y)
        fz = xyz_to_lab_component(z)
        
        L = (116 * fy) - 16
        a = 500 * (fx - fy)
        b_lab = 200 * (fy - fz)
        
        return L, a, b_lab
        
    except Exception as e:
        print(f"Error in RGB to LAB conversion: {e}")
        # Fallback to RGB values scaled to LAB-like range
        return rgb[0] * 100/255, (rgb[1] - 128) * 2, (rgb[2] - 128) * 2

def calculate_color_temperature(rgb: Tuple[int, int, int]) -> float:
    """Calculate the color temperature of an RGB color (warm vs cool)."""
    r, g, b = rgb
    
    # Simple heuristic: more red/yellow = warmer, more blue = cooler
    # Returns a value between 0 (cool) and 1 (warm)
    warm_component = (r + g * 0.5) / 255.0
    cool_component = b / 255.0
    
    # Normalize to 0-1 range where 0.5 is neutral
    temperature = warm_component / (warm_component + cool_component + 0.001)  # Add small value to avoid division by zero
    return temperature

def calculate_color_contrast(colors: list) -> float:
    """Calculate the overall contrast/variance in a list of RGB colors."""
    if len(colors) < 2:
        return 0.0
    
    try:
        # Convert to numpy array for easier calculation
        colors_array = np.array(colors)
        
        # Calculate standard deviation across all color channels
        contrast = np.std(colors_array) / 255.0  # Normalize to 0-1
        
        return float(contrast)
    except Exception as e:
        print(f"Error calculating color contrast: {e}")
        return 0.0

def preprocess_image_for_analysis(image: Image.Image, max_size: Tuple[int, int] = (150, 150)) -> np.ndarray:
    """Enhanced preprocessing with better error handling and options."""
    if image.mode != 'RGB':
        image = image.convert("RGB")
    
    img_copy = image.copy()
    img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)  # Use high-quality resampling
        
    np_image = np.array(img_copy)
    
    if np_image.ndim == 2:  # Grayscale
        np_image = np.stack((np_image,)*3, axis=-1)
    elif np_image.shape[2] == 4:  # RGBA
        np_image = np_image[:, :, :3]

    if np_image.shape[2] != 3:
        raise ValueError("Image could not be converted to a 3-channel RGB format.")
    
    return np_image.reshape(-1, 3)  # Return flat list of pixels

def validate_image_quality(image: Image.Image) -> dict:
    """Validate and analyze image quality for color extraction."""
    metrics = {
        "valid": True,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
        "warnings": []
    }
    
    # Check minimum size
    if image.width < 10 or image.height < 10:
        metrics["warnings"].append("Image is very small, color extraction may be unreliable")
    
    # Check if image is mostly uniform
    try:
        pixels = preprocess_image_for_analysis(image, (50, 50))
        variance = np.var(pixels)
        if variance < 100:  # Very low variance indicates uniform color
            metrics["warnings"].append("Image appears to have very uniform colors")
        metrics["color_variance"] = float(variance)
    except Exception as e:
        metrics["warnings"].append(f"Could not analyze color variance: {e}")
        metrics["color_variance"] = 0.0
    
    # Check for transparency
    if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
        metrics["warnings"].append("Image has transparency, which will be ignored in color extraction")
    
    return metrics

def data_uri_to_pil_image(data_uri: str) -> Image.Image:
    """Converts a data URI to a PIL Image object."""
    try:
        if not data_uri.startswith('data:image'):
            raise ValueError("Data URI does not seem to represent an image.")
        
        header, encoded = data_uri.split(",", 1)
        # Further validation of header if necessary, e.g., check for base64
        if ';base64' not in header:
            raise ValueError("Data URI does not appear to be base64 encoded.")

        image_data = base64.b64decode(encoded)
        image_stream = BytesIO(image_data)
        image = Image.open(image_stream)
        return image
    except UnidentifiedImageError:
        raise ValueError("Cannot identify image file from data URI. The data may be corrupt or not a supported image format.")
    except base64.binascii.Error: # Error from b64decode
        raise ValueError("Invalid base64 encoding in data URI.")
    except ValueError as e: # Catch our own ValueErrors or others
        raise ValueError(f"Invalid data URI: {e}")
    except Exception as e: # Catch-all for other unexpected errors
        # It's good to log this error for debugging.
        print(f"Unexpected error converting data URI to PIL Image: {type(e).__name__} - {e}")
        raise ValueError(f"Could not process data URI: {type(e).__name__}")

def pil_image_to_data_uri(image: Image.Image, fmt: str = "PNG") -> str:
    """Converts a PIL Image object to a data URI."""
    # Ensure format is supported by PIL for saving
    supported_formats = Image.SAVE.keys()
    if fmt.upper() not in supported_formats and fmt.lower() not in supported_formats:
        # Fallback or raise error
        print(f"Warning: Format '{fmt}' might not be supported for saving. Defaulting to PNG.")
        fmt = "PNG" # Default to PNG if format is problematic

    buffered = BytesIO()
    
    # Handle potential issues with saving specific modes to specific formats
    # For example, saving a 'P' mode image (palette) as JPEG might require conversion.
    img_to_save = image
    if fmt.upper() == 'JPEG' and image.mode not in ('RGB', 'L', 'CMYK'):
        img_to_save = image.convert('RGB')
    elif image.mode == 'RGBA' and fmt.upper() == 'JPEG': # JPEG doesn't support alpha
        img_to_save = image.convert('RGB')

    img_to_save.save(buffered, format=fmt.upper())
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    mime_type = f"image/{fmt.lower()}"
    if fmt.upper() == "SVG": # Special case for SVG if ever supported by PIL save
        mime_type = "image/svg+xml"
        
    return f"data:{mime_type};base64,{img_str}"

def rgb_to_hex(rgb_tuple: Tuple[float, float, float]) -> str:
    """Converts an RGB tuple (components 0-255) to a hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2]))

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Converts a hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color format: {hex_color}")
    try:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        raise ValueError(f"Invalid hex color format: {hex_color}")

def rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to HSV color space."""
    r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # Hue calculation
    if diff == 0:
        h = 0
    elif max_val == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    # Saturation calculation
    s = 0 if max_val == 0 else diff / max_val
    
    # Value calculation
    v = max_val
    
    return h, s, v

def hsv_to_rgb(hsv: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Convert HSV to RGB color space."""
    try:
        h, s, v = hsv
        
        # Normalize hue to 0-1 range for colorsys
        h_normalized = h / 360.0 if h > 1 else h
        
        # Use colorsys for conversion
        r, g, b = colorsys.hsv_to_rgb(h_normalized, s, v)
        
        # Convert back to 0-255 range
        return (int(r * 255), int(g * 255), int(b * 255))
        
    except Exception as e:
        print(f"Error in HSV to RGB conversion: {e}")
        # Fallback to gray
        gray_val = int(hsv[2] * 255) if len(hsv) > 2 else 128
        return (gray_val, gray_val, gray_val)

def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to LAB color space (simplified conversion)."""
    try:
        # Normalize RGB values
        r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
        
        # Convert to XYZ (simplified sRGB to XYZ conversion)
        # Apply gamma correction
        def gamma_correct(c):
            if c > 0.04045:
                return pow((c + 0.055) / 1.055, 2.4)
            else:
                return c / 12.92
        
        r = gamma_correct(r)
        g = gamma_correct(g)
        b = gamma_correct(b)
        
        # Convert to XYZ using sRGB matrix
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        
        # Normalize by D65 illuminant
        x = x / 0.95047
        y = y / 1.00000
        z = z / 1.08883
        
        # Convert XYZ to LAB
        def xyz_to_lab_component(t):
            if t > 0.008856:
                return pow(t, 1/3)
            else:
                return (7.787 * t) + (16/116)
        
        fx = xyz_to_lab_component(x)
        fy = xyz_to_lab_component(y)
        fz = xyz_to_lab_component(z)
        
        L = (116 * fy) - 16
        a = 500 * (fx - fy)
        b_lab = 200 * (fy - fz)
        
        return L, a, b_lab
        
    except Exception as e:
        print(f"Error in RGB to LAB conversion: {e}")
        # Fallback to RGB values scaled to LAB-like range
        return rgb[0] * 100/255, (rgb[1] - 128) * 2, (rgb[2] - 128) * 2

def lab_to_rgb(lab: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Convert LAB to RGB color space."""
    try:
        L, a, b = lab
        
        # Convert LAB to XYZ
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        def lab_to_xyz_component(t):
            if t > 0.206893:
                return t ** 3
            else:
                return (t - 16/116) / 7.787
        
        x = lab_to_xyz_component(fx) * 0.95047
        y = lab_to_xyz_component(fy) * 1.00000
        z = lab_to_xyz_component(fz) * 1.08883
        
        # Convert XYZ to RGB
        r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
        g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
        b = x *  0.0556434 + y * -0.2040259 + z *  1.0572252
        
        # Apply inverse gamma correction
        def inverse_gamma_correct(c):
            if c > 0.0031308:
                return 1.055 * pow(c, 1/2.4) - 0.055
            else:
                return 12.92 * c
        
        r = inverse_gamma_correct(r)
        g = inverse_gamma_correct(g)
        b = inverse_gamma_correct(b)
        
        # Clamp to 0-1 range and convert to 0-255
        r = max(0, min(1, r)) * 255
        g = max(0, min(1, g)) * 255
        b = max(0, min(1, b)) * 255
        
        return (int(r), int(g), int(b))
        
    except Exception as e:
        print(f"Error in LAB to RGB conversion: {e}")
        # Fallback: use L component as grayscale
        gray_val = int(max(0, min(100, lab[0])) * 255 / 100) if len(lab) > 0 else 128
        return (gray_val, gray_val, gray_val)

def calculate_color_temperature(rgb: Tuple[int, int, int]) -> float:
    """Calculate the color temperature of an RGB color (warm vs cool)."""
    r, g, b = rgb
    
    # Simple heuristic: more red/yellow = warmer, more blue = cooler
    # Returns a value between 0 (cool) and 1 (warm)
    warm_component = (r + g * 0.5) / 255.0
    cool_component = b / 255.0
    
    # Normalize to 0-1 range where 0.5 is neutral
    temperature = warm_component / (warm_component + cool_component + 0.001)  # Add small value to avoid division by zero
    return temperature

def calculate_color_contrast(colors: list) -> float:
    """Calculate the overall contrast/variance in a list of RGB colors."""
    if len(colors) < 2:
        return 0.0
    
    try:
        # Convert to numpy array for easier calculation
        colors_array = np.array(colors)
        
        # Calculate standard deviation across all color channels
        contrast = np.std(colors_array) / 255.0  # Normalize to 0-1
        
        return float(contrast)
    except Exception as e:
        print(f"Error calculating color contrast: {e}")
        return 0.0

def preprocess_image_for_analysis(image: Image.Image, max_size: Tuple[int, int] = (150, 150)) -> np.ndarray:
    """Enhanced preprocessing with better error handling and options."""
    if image.mode != 'RGB':
        image = image.convert("RGB")
    
    img_copy = image.copy()
    img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)  # Use high-quality resampling
        
    np_image = np.array(img_copy)
    
    if np_image.ndim == 2:  # Grayscale
        np_image = np.stack((np_image,)*3, axis=-1)
    elif np_image.shape[2] == 4:  # RGBA
        np_image = np_image[:, :, :3]

    if np_image.shape[2] != 3:
        raise ValueError("Image could not be converted to a 3-channel RGB format.")
    
    return np_image.reshape(-1, 3)  # Return flat list of pixels

def validate_image_quality(image: Image.Image) -> dict:
    """Validate and analyze image quality for color extraction."""
    metrics = {
        "valid": True,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
        "warnings": []
    }
    
    # Check minimum size
    if image.width < 10 or image.height < 10:
        metrics["warnings"].append("Image is very small, color extraction may be unreliable")
    
    # Check if image is mostly uniform
    try:
        pixels = preprocess_image_for_analysis(image, (50, 50))
        variance = np.var(pixels)
        if variance < 100:  # Very low variance indicates uniform color
            metrics["warnings"].append("Image appears to have very uniform colors")
        metrics["color_variance"] = float(variance)
    except Exception as e:
        metrics["warnings"].append(f"Could not analyze color variance: {e}")
        metrics["color_variance"] = 0.0
    
    # Check for transparency
    if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
        metrics["warnings"].append("Image has transparency, which will be ignored in color extraction")
    
    return metrics