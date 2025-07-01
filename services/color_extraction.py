# services/color_extraction.py
"""
Enhanced Color Extraction Service v3.0
NOW WITH COLOR PSYCHOLOGY INTEGRATION
Extracts dominant colors with multiple strategies and optional psychology analysis
"""

import numpy as np
from PIL import Image, ImageStat
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter
import colorsys

# Import utilities with fallback
try:
    from utils.image_utils import (
    hex_to_rgb, rgb_to_hex, rgb_to_hsv, hsv_to_rgb, rgb_to_lab, lab_to_rgb,  # 🆕 ADD lab_to_rgb
    preprocess_image_for_analysis, validate_image_quality
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.image_utils import (
        hex_to_rgb, rgb_to_hex, rgb_to_hsv, hsv_to_rgb, rgb_to_lab,
        preprocess_image_for_analysis, validate_image_quality
    )

# Import sklearn with fallback
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn not available. Some advanced features may be limited.")
    SKLEARN_AVAILABLE = False

def extract_dominant_colors_from_image(
    image,
    num_colors: int = 5,
    strategy: str = "enhanced_kmeans",
    strategy_option: Optional[str] = None,
    color_space: str = "rgb",
    # NEW: Psychology enhancement parameters (all optional for backward compatibility)
    include_psychology: bool = False,
    cultural_context: str = "universal",
    psychology_confidence_threshold: float = 0.5,
    return_detailed: bool = False
):
    """
    Enhanced color extraction with optional psychology analysis
    
    Args:
        image: PIL Image object
        num_colors: Number of colors to extract
        strategy: Color extraction strategy ("enhanced_kmeans", "frequency_based", "dominant_hues", "color_harmony", "adaptive")
        strategy_option: Strategy-specific option
        color_space: Color space for extraction ("rgb", "hsv", "lab")
        include_psychology: Whether to include psychology analysis (NEW)
        cultural_context: Cultural context for psychology ("universal", "english", "chinese") (NEW)
        psychology_confidence_threshold: Minimum confidence for color classification (NEW)
        return_detailed: Whether to return detailed dict or just colors (NEW)
        
    Returns:
        If include_psychology=False and return_detailed=False: List[str] (backward compatible)
        If include_psychology=True or return_detailed=True: Dict with colors and psychology
    """
    try:
        # Step 1: Extract base colors using existing functionality
        pixels = preprocess_image_for_analysis(image)
        
        if len(pixels) == 0:
            if include_psychology or return_detailed:
                return {
                    "colors": [],
                    "error": "No pixels found in image",
                    "psychology": {"enabled": include_psychology, "error": "No pixels"} if include_psychology else None
                }
            else:
                return []
        
        # Use existing color extraction logic
        if strategy == "enhanced_kmeans":
            colors_hex, weights, metrics = _extract_enhanced_kmeans_colors(pixels, num_colors, color_space)
        elif strategy == "frequency_based":
            colors_hex, weights, metrics = _extract_palette_by_frequency(pixels, num_colors)
        elif strategy == "dominant_hues":
            colors_hex, weights, metrics = _extract_dominant_hues(pixels, num_colors)
        elif strategy == "color_harmony":
            colors_hex, weights, metrics = _extract_color_harmony_palette(pixels, num_colors)
        elif strategy == "adaptive":
            # Auto-select best strategy based on image characteristics
            strategy = _select_adaptive_strategy(pixels, image)
            return extract_dominant_colors_from_image(
                image, num_colors, strategy, strategy_option, color_space,
                include_psychology, cultural_context, psychology_confidence_threshold, return_detailed
            )
        else:
            print(f"Using enhanced k-means for unknown strategy: {strategy}")
            colors_hex, weights, metrics = _extract_enhanced_kmeans_colors(pixels, num_colors, color_space)
        
        # Step 2: Return basic colors if no psychology requested (backward compatibility)
        if not include_psychology and not return_detailed:
            return colors_hex
        
        # Step 3: Build detailed response
        response = {
            "colors": colors_hex,
            "extraction_metrics": {
                "strategy": strategy,
                "color_space": color_space,
                "num_colors_extracted": len(colors_hex),
                "weights": weights,
                "strategy_metrics": metrics
            }
        }
        
        # Step 4: Add psychology analysis if requested
        if include_psychology:
            psychology_data = _analyze_colors_psychology(
                colors=colors_hex,
                cultural_context=cultural_context,
                confidence_threshold=psychology_confidence_threshold
            )
            response["psychology"] = psychology_data
        
        return response
        
    except Exception as e:
        print(f"❌ Error in color extraction: {e}")
        
        # Return appropriate fallback based on expected format
        if include_psychology or return_detailed:
            return {
                "colors": [],
                "error": str(e),
                "psychology": {"enabled": include_psychology, "error": str(e)} if include_psychology else None
            }
        else:
            return []  # Backward compatible fallback

def _analyze_colors_psychology(
    colors: List[str],
    cultural_context: str = "universal",
    confidence_threshold: float = 0.5
) -> Dict:
    """
    Analyze psychology for extracted colors
    
    Args:
        colors: List of hex color strings
        cultural_context: Cultural context for analysis
        confidence_threshold: Minimum confidence for classification
        
    Returns:
        Dict with psychology analysis
    """
    try:
        # Import psychology service here to avoid circular imports
        from services.color_psychology_service import ColorPsychologyService
        
        # Initialize psychology service
        psychology_service = ColorPsychologyService()
        
        # Analyze each color
        color_analyses = []
        all_psychology_themes = []
        successful_classifications = 0
        
        for hex_color in colors:
            analysis = psychology_service.get_psychology_for_hex_color(
                hex_color=hex_color,
                cultural_context=cultural_context,
                confidence_threshold=confidence_threshold
            )
            
            color_analyses.append(analysis)
            
            if analysis.get("status") == "success":
                successful_classifications += 1
                psychology_themes = analysis.get("psychology", [])
                all_psychology_themes.extend(psychology_themes)
        
        # Create palette analysis
        palette_analysis = _create_palette_psychology_analysis(
            all_psychology_themes=all_psychology_themes,
            total_colors=len(colors),
            successful_classifications=successful_classifications,
            cultural_context=cultural_context
        )
        
        # Analyze cultural variations
        cultural_comparison = _analyze_cultural_variations(
            colors=colors,
            psychology_service=psychology_service,
            base_context=cultural_context
        )
        
        return {
            "enabled": True,
            "culturalContext": cultural_context,
            "confidenceThreshold": confidence_threshold,
            "colorAnalyses": color_analyses,
            "paletteAnalysis": palette_analysis,
            "culturalComparison": cultural_comparison,
            "processingMetrics": {
                "totalColors": len(colors),
                "successfulClassifications": successful_classifications,
                "classificationRate": round(successful_classifications / max(len(colors), 1), 2),
                "availableCulturalContexts": psychology_service.get_cultural_contexts()
            }
        }
        
    except ImportError:
        print("⚠️ Color psychology service not available")
        return {
            "enabled": False,
            "error": "Psychology service not available",
            "culturalContext": cultural_context
        }
    except Exception as e:
        print(f"❌ Error in psychology analysis: {e}")
        return {
            "enabled": True,
            "error": str(e),
            "culturalContext": cultural_context
        }

def _create_palette_psychology_analysis(
    all_psychology_themes: List[str],
    total_colors: int,
    successful_classifications: int,
    cultural_context: str
) -> Dict:
    """Create aggregate psychology analysis for the color palette"""
    try:
        if not all_psychology_themes:
            return {
                "dominantPsychologyThemes": [],
                "overallMood": "unclassified",
                "psychologyClassificationRate": 0.0,
                "note": "No colors could be classified for psychology analysis"
            }
        
        # Count theme frequencies
        theme_counts = {}
        for theme in all_psychology_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        # Get dominant themes
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        dominant_themes = [theme for theme, count in sorted_themes[:5]]
        
        # Determine overall mood and color family
        overall_mood = _determine_overall_mood(dominant_themes, cultural_context)
        color_family = _determine_color_family(dominant_themes)
        
        return {
            "dominantPsychologyThemes": dominant_themes,
            "commonThemesCounts": dict(sorted_themes),
            "overallMood": overall_mood,
            "colorFamily": color_family,
            "psychologyClassificationRate": round(successful_classifications / max(total_colors, 1), 2),
            "successfulClassifications": successful_classifications,
            "totalColors": total_colors,
            "themeDiversity": len(theme_counts),
            "mostFrequentTheme": sorted_themes[0][0] if sorted_themes else None
        }
        
    except Exception as e:
        print(f"❌ Error creating palette analysis: {e}")
        return {"error": str(e)}

def _determine_overall_mood(themes: List[str], cultural_context: str) -> str:
    """Determine overall mood based on dominant psychology themes"""
    try:
        if not themes:
            return "neutral"
        
        # Mood mappings
        mood_mappings = {
            "energetic": ["Energy", "Enthusiasm", "Attention", "Cheerfulness", "Optimism"],
            "calming": ["Calmness", "Passivity", "Openness", "Freshness", "Peace"],
            "grounded": ["Earth", "Stability", "Natural", "Reliability", "Security"],
            "elegant": ["Royalty", "Luxury", "Nobility", "Sophistication"],
            "warm": ["Warmth", "Movement", "Spontaneity", "Comfort"],
            "peaceful": ["Nature", "Health", "Growth", "Harmony", "Balance"]
        }
        
        # Score each mood
        mood_scores = {}
        for mood, mood_themes in mood_mappings.items():
            score = sum(1 for theme in themes if theme in mood_themes)
            if score > 0:
                mood_scores[mood] = score
        
        if mood_scores:
            best_mood = max(mood_scores.items(), key=lambda x: x[1])
            return f"{best_mood[0]} ({best_mood[1]} matching themes)"
        
        return f"dominated by {themes[0].lower()} themes"
        
    except Exception:
        return "unclassified"

def _determine_color_family(themes: List[str]) -> str:
    """Determine color family based on psychology themes"""
    try:
        if any(theme in ["Earth", "Natural", "Stability"] for theme in themes):
            return "earth tones"
        elif any(theme in ["Sky", "Water", "Calmness"] for theme in themes):
            return "cool tones"  
        elif any(theme in ["Energy", "Warmth", "Sunshine"] for theme in themes):
            return "warm tones"
        elif any(theme in ["Royalty", "Luxury", "Mystery"] for theme in themes):
            return "sophisticated tones"
        elif any(theme in ["Nature", "Growth", "Health"] for theme in themes):
            return "natural tones"
        else:
            return "mixed tones"
    except Exception:
        return "unclassified"

def _analyze_cultural_variations(colors: List[str], psychology_service, base_context: str) -> Dict:
    """Analyze cultural variations in color psychology"""
    try:
        available_contexts = psychology_service.get_cultural_contexts()
        
        if len(available_contexts) <= 1:
            return {
                "hasVariation": False,
                "note": "Only one cultural context available"
            }
        
        cultural_differences = []
        
        # Sample first few colors to avoid heavy processing
        for color in colors[:3]:
            color_name, _ = psychology_service.classify_hex_color(color)
            if color_name:
                comparison = psychology_service.compare_cultural_psychology(color_name)
                if comparison.get("culturalVariation", False):
                    cultural_differences.append({
                        "color": color,
                        "colorName": color_name,
                        "differences": comparison.get("contextSpecific", {})
                    })
        
        return {
            "hasVariation": len(cultural_differences) > 0,
            "baseCulturalContext": base_context,
            "availableContexts": available_contexts,
            "culturalDifferences": cultural_differences,
            "note": f"Found cultural variations in {len(cultural_differences)} colors" if cultural_differences else "No significant cultural variations found"
        }
        
    except Exception as e:
        print(f"❌ Error analyzing cultural variations: {e}")
        return {"hasVariation": False, "error": str(e)}

def _select_adaptive_strategy(pixels, image) -> str:
    """Select the best color extraction strategy based on image characteristics"""
    try:
        # Analyze image characteristics
        pixel_count = len(pixels)
        unique_colors = len(set(tuple(p) for p in pixels))
        color_diversity = unique_colors / max(pixel_count, 1)
        
        # Calculate color variance
        color_variance = np.var(pixels, axis=0).mean()
        
        # Select strategy based on characteristics
        if color_diversity > 0.7 and color_variance > 1000:
            # High diversity, high variance -> use frequency-based
            return "frequency_based"
        elif color_diversity < 0.3:
            # Low diversity -> use enhanced k-means
            return "enhanced_kmeans"
        elif color_variance < 500:
            # Low variance -> use dominant hues
            return "dominant_hues"
        else:
            # Default to enhanced k-means
            return "enhanced_kmeans"
            
    except Exception:
        return "enhanced_kmeans"

# ============= EXISTING COLOR EXTRACTION STRATEGIES =============

def _extract_enhanced_kmeans_colors(
    pixels: np.ndarray, 
    num_colors: int,
    color_space: str = "rgb"
) -> Tuple[List[str], List[float], Dict]:
    """Enhanced K-means clustering with multiple color spaces and quality metrics."""
    if not SKLEARN_AVAILABLE:
        return _extract_palette_by_frequency(pixels, num_colors)
    
    if pixels.shape[0] == 0:
        return [], [], {}
    
    try:
        # Convert to target color space
        if color_space.lower() == "hsv":
            converted_pixels = np.array([rgb_to_hsv(pixel) for pixel in pixels])
            # Normalize HSV for clustering
            converted_pixels[:, 0] /= 360.0  # Hue to 0-1
        elif color_space.lower() == "lab":
            converted_pixels = np.array([rgb_to_lab(pixel) for pixel in pixels])
            # Normalize LAB for clustering
            converted_pixels[:, 0] /= 100.0  # L to 0-1
            converted_pixels[:, 1] = (converted_pixels[:, 1] + 127) / 254.0  # a to 0-1
            converted_pixels[:, 2] = (converted_pixels[:, 2] + 127) / 254.0  # b to 0-1
        else:  # RGB
            converted_pixels = pixels.astype(float) / 255.0
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=min(num_colors, len(np.unique(converted_pixels, axis=0))), 
                       random_state=42, n_init=10)
        labels = kmeans.fit_predict(converted_pixels)
        centers = kmeans.cluster_centers_
        
        # Convert centers back to RGB
        rgb_centers = []
        for center in centers:
            if color_space.lower() == "hsv":
                # Denormalize HSV and convert to RGB
                hsv = (center[0] * 360, center[1], center[2])
                rgb_centers.append(hsv_to_rgb(hsv))
            elif color_space.lower() == "lab":
                # Denormalize LAB and convert to RGB
                lab = (center[0] * 100, center[1] * 254 - 127, center[2] * 254 - 127)
                rgb_centers.append(lab_to_rgb(lab))
            else:  # RGB
                rgb_centers.append(tuple((center * 255).astype(int)))
        
        # Calculate weights based on cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_pixels = len(labels)
        weights = [counts[i] / total_pixels for i in range(len(unique_labels))]
        
        # Convert to hex colors
        hex_colors = [rgb_to_hex(rgb) for rgb in rgb_centers]
        
        # Calculate quality metrics
        silhouette_avg = _calculate_silhouette_score(converted_pixels, labels)
        inertia = kmeans.inertia_
        
        metrics = {
            "method": "enhanced_kmeans",
            "color_space": color_space,
            "silhouette_score": silhouette_avg,
            "inertia": float(inertia),
            "clusters_found": len(unique_labels)
        }
        
        return hex_colors, weights, metrics
        
    except Exception as e:
        print(f"K-means clustering error: {e}")
        return _extract_palette_by_frequency(pixels, num_colors)

def _extract_palette_by_frequency(
    pixels: np.ndarray, 
    num_colors: int
) -> Tuple[List[str], List[float], Dict]:
    """Enhanced frequency-based extraction with color quantization."""
    if pixels.shape[0] == 0:
        return [], [], {}

    # Quantize colors to reduce noise (group similar colors)
    quantized_pixels = (pixels // 8) * 8  # Reduce precision to group similar colors
    
    # Count occurrences
    pixel_tuples = [tuple(p) for p in quantized_pixels]
    color_counts = Counter(pixel_tuples)
    
    # Get the most common colors
    most_common = color_counts.most_common(num_colors)
    colors = [color for color, count in most_common]
    counts = [count for color, count in most_common]
    
    # Calculate weights
    total_pixels = len(pixel_tuples)
    weights = [count / total_pixels for count in counts]
    
    # Convert to hex
    hex_colors = [rgb_to_hex(color) for color in colors]
    
    metrics = {
        "method": "frequency_based",
        "total_unique_colors": len(color_counts),
        "quantization_level": 8,
        "coverage": sum(weights)
    }
    
    return hex_colors, weights, metrics

def _extract_dominant_hues(
    pixels: np.ndarray, 
    num_colors: int
) -> Tuple[List[str], List[float], Dict]:
    """Extract colors based on dominant hues in HSV space."""
    if pixels.shape[0] == 0:
        return [], [], {}
    
    try:
        # Convert to HSV
        hsv_pixels = np.array([rgb_to_hsv(pixel) for pixel in pixels])
        
        # Extract hues (ignore very low saturation pixels - grays)
        saturated_mask = hsv_pixels[:, 1] > 0.1  # Saturation > 10%
        
        if not np.any(saturated_mask):
            # All pixels are unsaturated, fall back to frequency method
            return _extract_palette_by_frequency(pixels, num_colors)
        
        saturated_hues = hsv_pixels[saturated_mask, 0]
        saturated_pixels = pixels[saturated_mask]
        
        # Quantize hues to 12 segments (30° each)
        hue_segments = (saturated_hues // 30).astype(int)
        
        # Count pixels in each hue segment
        segment_counts = Counter(hue_segments)
        most_common_segments = segment_counts.most_common(min(num_colors, len(segment_counts)))
        
        # For each dominant hue segment, find the most representative color
        colors = []
        weights = []
        total_saturated_pixels = len(saturated_pixels)
        
        for segment, count in most_common_segments:
            # Find pixels in this hue segment
            segment_mask = hue_segments == segment
            segment_pixels = saturated_pixels[segment_mask]
            
            if len(segment_pixels) > 0:
                # Use median color in this segment
                median_color = np.median(segment_pixels, axis=0).astype(int)
                colors.append(rgb_to_hex(tuple(median_color)))
                weights.append(count / total_saturated_pixels)
        
        # Add unsaturated colors if we need more colors
        if len(colors) < num_colors:
            unsaturated_pixels = pixels[~saturated_mask]
            if len(unsaturated_pixels) > 0:
                # Add the most common unsaturated color
                unsaturated_counts = Counter(tuple(p) for p in unsaturated_pixels)
                for color, count in unsaturated_counts.most_common(num_colors - len(colors)):
                    colors.append(rgb_to_hex(color))
                    weights.append(count / len(pixels))
        
        metrics = {
            "method": "dominant_hues",
            "saturated_pixels": int(np.sum(saturated_mask)),
            "hue_segments_found": len(segment_counts),
            "total_pixels": len(pixels)
        }
        
        return colors, weights, metrics
        
    except Exception as e:
        print(f"Dominant hues extraction error: {e}")
        return _extract_palette_by_frequency(pixels, num_colors)

def _extract_color_harmony_palette(
    pixels: np.ndarray, 
    num_colors: int
) -> Tuple[List[str], List[float], Dict]:
    """Extract colors based on color harmony relationships."""
    if pixels.shape[0] == 0:
        return [], [], {}
    
    # First, get dominant colors using k-means
    base_colors, base_weights, _ = _extract_enhanced_kmeans_colors(pixels, min(num_colors * 2, 10))
    
    if len(base_colors) < 2:
        return base_colors, base_weights, {"method": "color_harmony", "harmony_type": "insufficient_colors"}
    
    # Convert to HSV for harmony analysis
    base_rgb = [hex_to_rgb(color) for color in base_colors]
    base_hsv = [rgb_to_hsv(rgb) for rgb in base_rgb]
    
    # Find harmonious combinations
    harmonious_colors = []
    harmony_weights = []
    
    # Start with the most dominant color
    primary_color = base_colors[0]
    primary_hsv = base_hsv[0]
    harmonious_colors.append(primary_color)
    harmony_weights.append(base_weights[0])
    
    # Find complementary, triadic, or analogous colors
    for i, (color, hsv, weight) in enumerate(zip(base_colors[1:], base_hsv[1:], base_weights[1:]), 1):
        hue_diff = abs(hsv[0] - primary_hsv[0])
        hue_diff = min(hue_diff, 360 - hue_diff)  # Handle circular hue space
        
        # Check for color harmony relationships
        is_complementary = 150 <= hue_diff <= 210  # ~180 degrees
        is_triadic = 110 <= hue_diff <= 130 or 230 <= hue_diff <= 250  # ~120 or ~240 degrees
        is_analogous = hue_diff <= 60  # Adjacent colors
        is_split_complementary = 140 <= hue_diff <= 160 or 200 <= hue_diff <= 220
        
        if is_complementary or is_triadic or is_analogous or is_split_complementary:
            harmonious_colors.append(color)
            harmony_weights.append(weight)
        
        if len(harmonious_colors) >= num_colors:
            break
    
    # If we don't have enough harmonious colors, fill with remaining dominant colors
    remaining_needed = num_colors - len(harmonious_colors)
    if remaining_needed > 0:
        for i, color in enumerate(base_colors):
            if color not in harmonious_colors:
                harmonious_colors.append(color)
                harmony_weights.append(base_weights[i])
                remaining_needed -= 1
                if remaining_needed <= 0:
                    break
    
    # Normalize weights
    total_weight = sum(harmony_weights)
    if total_weight > 0:
        harmony_weights = [w / total_weight for w in harmony_weights]
    
    metrics = {
        "method": "color_harmony",
        "base_colors_analyzed": len(base_colors),
        "harmonious_found": len(harmonious_colors)
    }
    
    return harmonious_colors[:num_colors], harmony_weights[:num_colors], metrics

def _calculate_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Calculate silhouette score with error handling."""
    try:
        if len(set(labels)) > 1 and len(X) > 1:
            return float(silhouette_score(X, labels))
        return 1.0
    except ImportError:
        # Fallback if sklearn.metrics not available
        return 0.5
    except Exception:
        return 0.0

# ============= CLUSTERING FEATURE EXTRACTION =============

def get_color_features_for_clustering(
    image: Image.Image, 
    num_colors: int = 5,
    strategy: str = "enhanced_kmeans",
    color_space: str = "rgb",
    include_harmony_features: bool = True,
    return_colors: bool = False
) -> Union[List[float], Tuple[List[float], List[str]]]:
    """
    Extract enhanced feature vector for clustering, optionally including color weights and harmony metrics.
    NOW SUPPORTS returning both features and hex colors to avoid double extraction.
    
    Args:
        image: PIL Image object
        num_colors: Number of dominant colors to extract
        strategy: Color extraction strategy
        color_space: Color space for analysis
        include_harmony_features: Whether to include color harmony features
        return_colors: If True, return both features and hex colors
    
    Returns:
        If return_colors=False: List of float features for clustering
        If return_colors=True: Tuple of (features, hex_colors)
    """
    try:
        # Get dominant colors with weights and metrics
        pixels = preprocess_image_for_analysis(image)
        
        if strategy == "enhanced_kmeans":
            colors_hex, weights, metrics = _extract_enhanced_kmeans_colors(pixels, num_colors, color_space)
        elif strategy == "dominant_hues":
            colors_hex, weights, metrics = _extract_dominant_hues(pixels, num_colors)
        elif strategy == "color_harmony":
            colors_hex, weights, metrics = _extract_color_harmony_palette(pixels, num_colors)
        else:
            colors_hex, weights, metrics = _extract_enhanced_kmeans_colors(pixels, num_colors, color_space)
        
        features = []
        
        # Add color values (normalized to target color space)
        for i, hex_color in enumerate(colors_hex):
            rgb = hex_to_rgb(hex_color)
            
            if color_space.lower() == "hsv":
                h, s, v = rgb_to_hsv(rgb)
                features.extend([h/360.0, s, v])  # Normalize hue to 0-1
            elif color_space.lower() == "lab":
                l, a, b = rgb_to_lab(rgb)
                features.extend([l/100.0, (a+127)/254.0, (b+127)/254.0])  # Normalize to 0-1
            else:  # RGB
                features.extend([c/255.0 for c in rgb])
            
            # Add color weight
            weight = weights[i] if i < len(weights) else 0.0
            features.append(weight)
        
        # Pad features if we have fewer colors than expected
        expected_color_features = num_colors * 4  # 3 color channels + 1 weight per color
        while len(features) < expected_color_features:
            features.extend([0.0, 0.0, 0.0, 0.0])  # Add zero color and weight
        
        # Truncate if we have too many
        features = features[:expected_color_features]
        
        # Add harmony features if requested
        if include_harmony_features:
            if len(colors_hex) > 1:
                try:
                    rgb_colors = [hex_to_rgb(color) for color in colors_hex]
                    
                    # Color temperature features
                    from utils.image_utils import calculate_color_temperature
                    temperatures = [calculate_color_temperature(rgb) for rgb in rgb_colors]
                    features.append(np.mean(temperatures))  # Average temperature
                    features.append(np.var(temperatures))   # Temperature variance
                    
                    # Color contrast
                    from utils.image_utils import calculate_color_contrast
                    features.append(calculate_color_contrast(rgb_colors))
                    
                    # Color spread (how diverse the colors are)
                    if len(rgb_colors) > 1:
                        color_distances = []
                        for i in range(len(rgb_colors)):
                            for j in range(i+1, len(rgb_colors)):
                                dist = np.linalg.norm(np.array(rgb_colors[i]) - np.array(rgb_colors[j]))
                                color_distances.append(dist)
                        features.append(np.mean(color_distances) / 441.673)  # Normalize by max RGB distance
                    else:
                        features.append(0.0)
                    
                    # Dominant hue concentration
                    hsv_colors = [rgb_to_hsv(rgb) for rgb in rgb_colors]
                    hues = [hsv[0] for hsv in hsv_colors if hsv[1] > 0.1]  # Only saturated colors
                    if len(hues) > 1:
                        hue_var = np.var(hues)
                        features.append(1.0 / (1.0 + hue_var/10000.0))  # Concentration measure
                    else:
                        features.append(1.0)  # High concentration
                        
                except ImportError:
                    # Harmony features not available
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                except Exception as e:
                    print(f"Error calculating harmony features: {e}")
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        if return_colors:
            return features, colors_hex
        else:
            return features
            
    except Exception as e:
        print(f"Error extracting color features: {e}")
        if return_colors:
            return [], []
        else:
            return []

# ============= CONVENIENCE FUNCTIONS =============

def extract_colors_with_psychology(
    image,
    num_colors: int = 5, 
    strategy: str = "enhanced_kmeans",
    cultural_context: str = "universal"
) -> Dict:
    """
    Convenience function for extracting colors with psychology
    Returns detailed response with both colors and psychology
    """
    return extract_dominant_colors_from_image(
        image=image,
        num_colors=num_colors,
        strategy=strategy,
        include_psychology=True,
        cultural_context=cultural_context,
        return_detailed=True
    )

# Export main functions
__all__ = [
    "extract_dominant_colors_from_image",
    "get_color_features_for_clustering", 
    "extract_colors_with_psychology"
]