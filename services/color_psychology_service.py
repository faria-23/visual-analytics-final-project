# services/color_psychology_service.py
"""
Color Psychology Service v2.0
Loads color psychology data from datastore/color_psychology_data.json
Provides psychology associations for colors across different cultural contexts
NOW WITH COLOR MATCHING: Convert hex colors to psychology data
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import webcolors

# Import color matching utilities
try:
    from utils.color_matching_utils import (
        hex_to_hsv, is_in_hue_range, is_in_range, 
        calculate_color_distance, get_color_brightness_category,
        get_color_saturation_category
    )
    from utils.image_utils import hex_to_rgb
    COLOR_MATCHING_AVAILABLE = True
except ImportError:
    print("⚠️ Color matching utilities not available. Install color matching utils for hex color support.")
    COLOR_MATCHING_AVAILABLE = False

# Handle webcolors CSS3_NAMES_TO_HEX compatibility
CSS3_NAMES_TO_HEX = getattr(webcolors, 'CSS3_NAMES_TO_HEX', None)
if CSS3_NAMES_TO_HEX is None:
    # Minimal fallback for linter, not production
    CSS3_NAMES_TO_HEX = {
        'black': '#000000', 'white': '#ffffff', 'red': '#ff0000', 'green': '#008000', 'blue': '#0000ff',
        'gray': '#808080', 'grey': '#808080', 'yellow': '#ffff00', 'orange': '#ffa500', 'brown': '#a52a2a',
        'purple': '#800080', 'pink': '#ffc0cb', 'cyan': '#00ffff', 'magenta': '#ff00ff', 'lime': '#00ff00',
        'navy': '#000080', 'teal': '#008080', 'maroon': '#800000', 'olive': '#808000', 'silver': '#c0c0c0'
    }

class ColorPsychologyService:
    """Service for color psychology analysis and cultural associations"""
    
    def __init__(self):
        self.psychology_data = self._load_psychology_data()
        self.color_mapping = self.psychology_data.get("colorMapping", {})
        self.cultural_contexts = self.psychology_data.get("culturalContexts", {})
        self.config = self.psychology_data.get("config", {})
        
        colors_count = len(self.psychology_data.get("colorPsychology", {}))
        contexts_count = len(self.cultural_contexts)
        
        print(f"ColorPsychologyService initialized:")
        print(f"  🎨 Colors loaded: {colors_count}")
        print(f"  🌍 Cultural contexts: {contexts_count}")
        print(f"  ⚙️  Default context: {self.config.get('defaultCulturalContext', 'universal')}")
        print(f"  🔧 Color matching: {'✅ Available' if COLOR_MATCHING_AVAILABLE else '❌ Unavailable'}")
    
    def _load_psychology_data(self) -> Dict:
        """Load color psychology data from datastore/color_psychology_data.json"""
        try:
            psychology_file = Path(__file__).parent.parent / "datastore" / "color_psychology_data.json"
            
            if not psychology_file.exists():
                print(f"⚠️  Color psychology data not found: {psychology_file}")
                print("Using minimal fallback data")
                return self._minimal_psychology_fallback()
            
            with open(psychology_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ Loaded color psychology data from: {psychology_file}")
            return data
            
        except Exception as e:
            print(f"❌ Error loading color psychology data: {e}")
            print("Using minimal fallback data")
            return self._minimal_psychology_fallback()
    
    def _minimal_psychology_fallback(self) -> Dict:
        """Minimal fallback psychology data if main file can't be loaded"""
        return {
            "metadata": {
                "version": "fallback",
                "totalColors": 5,
                "culturalContexts": ["universal"]
            },
            "colorPsychology": {
                "red": {"universal": ["Energy", "Passion", "Attention"]},
                "blue": {"universal": ["Calm", "Trust", "Stability"]},
                "green": {"universal": ["Nature", "Growth", "Harmony"]},
                "yellow": {"universal": ["Happiness", "Optimism", "Energy"]},
                "brown": {"universal": ["Earth", "Stability", "Natural"]}
            },
            "culturalContexts": {
                "universal": {"name": "Universal", "code": "universal"}
            },
            "config": {"defaultCulturalContext": "universal"}
        }
    
    def get_psychology_for_color(self, color_name: str, cultural_context: str = "universal") -> List[str]:
        """
        Get psychology associations for a specific color and cultural context
        
        Args:
            color_name: Color name (e.g., "red", "blue", "brown")
            cultural_context: Cultural context ("universal", "english", "chinese")
            
        Returns:
            List of psychology associations
        """
        try:
            if not color_name:
                return []
            
            color_name_lower = color_name.lower()
            color_psychology = self.psychology_data.get("colorPsychology", {})
            
            if color_name_lower not in color_psychology:
                print(f"⚠️  Color '{color_name}' not found in psychology data")
                return []
            
            color_data = color_psychology[color_name_lower]
            
            # Get associations for the requested cultural context
            if cultural_context in color_data:
                return color_data[cultural_context]
            
            # Fallback to universal if specific context not found
            if "universal" in color_data:
                print(f"⚠️  Cultural context '{cultural_context}' not found for '{color_name}', using universal")
                return color_data["universal"]
            
            print(f"⚠️  No psychology data found for '{color_name}' in context '{cultural_context}'")
            return []
            
        except Exception as e:
            print(f"❌ Error getting psychology for color '{color_name}': {e}")
            return []
    
    def get_all_psychology_for_color(self, color_name: str) -> Dict[str, List[str]]:
        """
        Get psychology associations for a color across all cultural contexts
        
        Args:
            color_name: Color name (e.g., "red", "blue", "brown")
            
        Returns:
            Dict with cultural contexts as keys and psychology lists as values
        """
        try:
            if not color_name:
                return {}
            
            color_name_lower = color_name.lower()
            color_psychology = self.psychology_data.get("colorPsychology", {})
            
            if color_name_lower not in color_psychology:
                return {}
            
            return color_psychology[color_name_lower]
            
        except Exception as e:
            print(f"❌ Error getting all psychology for color '{color_name}': {e}")
            return {}
    
    def get_available_colors(self) -> List[str]:
        """Get list of all available color names"""
        try:
            color_psychology = self.psychology_data.get("colorPsychology", {})
            return list(color_psychology.keys())
        except Exception as e:
            print(f"❌ Error getting available colors: {e}")
            return []
    
    def get_cultural_contexts(self) -> List[str]:
        """Get list of available cultural contexts"""
        try:
            return list(self.cultural_contexts.keys())
        except Exception as e:
            print(f"❌ Error getting cultural contexts: {e}")
            return ["universal"]
    
    def get_cultural_context_info(self, context: str) -> Dict:
        """Get detailed information about a cultural context"""
        try:
            return self.cultural_contexts.get(context, {})
        except Exception as e:
            print(f"❌ Error getting cultural context info for '{context}': {e}")
            return {}
    
    def compare_cultural_psychology(self, color_name: str) -> Dict:
        """
        Compare psychology associations for a color across all cultural contexts
        
        Args:
            color_name: Color name to compare
            
        Returns:
            Dict with comparison data and insights
        """
        try:
            all_psychology = self.get_all_psychology_for_color(color_name)
            
            if not all_psychology:
                return {"error": f"No psychology data found for color '{color_name}'"}
            
            # Find common associations across cultures
            all_associations = []
            for context_associations in all_psychology.values():
                all_associations.extend(context_associations)
            
            # Count frequency of each association
            association_counts = {}
            for association in all_associations:
                association_counts[association] = association_counts.get(association, 0) + 1
            
            # Find associations that appear in multiple contexts
            common_associations = [
                assoc for assoc, count in association_counts.items() 
                if count > 1
            ]
            
            # Find context-specific associations
            context_specific = {}
            for context, associations in all_psychology.items():
                unique_to_context = [
                    assoc for assoc in associations 
                    if association_counts.get(assoc, 0) == 1
                ]
                if unique_to_context:
                    context_specific[context] = unique_to_context
            
            return {
                "color": color_name,
                "allAssociations": all_psychology,
                "commonAcrossCultures": common_associations,
                "contextSpecific": context_specific,
                "totalAssociations": len(all_associations),
                "culturalVariation": len(context_specific) > 0
            }
            
        except Exception as e:
            print(f"❌ Error comparing cultural psychology for '{color_name}': {e}")
            return {"error": str(e)}
    
    def classify_hex_color(self, hex_color: str, tolerance: Optional[float] = None) -> Tuple[Optional[str], float]:
        """
        Classify a hex color to a color name using HSV matching and webcolors fallback
        """
        if not COLOR_MATCHING_AVAILABLE:
            print("⚠️ Color matching not available. Install color matching utilities.")
            return None, 0.0
        try:
            if not hex_color:
                print(f"[DEBUG] No hex color provided.")
                return None, 0.0
            # Use tolerance from config if not specified
            if tolerance is None:
                config_tolerance = self.config.get("colorMatchingTolerance", 15)
                if config_tolerance is None:
                    config_tolerance = 15.0
                tolerance = float(config_tolerance) / 360.0  # Convert degrees to 0-1
            else:
                tolerance = float(tolerance)
            # Convert hex to HSV
            h, s, v = hex_to_hsv(hex_color)
            print(f"[DEBUG] Classifying hex: {hex_color} -> HSV: H={h:.2f}, S={s:.3f}, V={v:.3f}")
            # Improved logic for black/gray/white classification
            if isinstance(v, float) and v < 0.15:
                print(f"[DEBUG] Classified as black (V={v:.3f} < 0.15)")
                return "black", 0.9
            if isinstance(s, float) and s < 0.12 and isinstance(v, float) and 0.15 <= v <= 0.85:
                print(f"[DEBUG] Classified as gray (S={s:.3f} < 0.12 and 0.15 <= V={v:.3f} <= 0.85)")
                return "gray", 0.8
            if isinstance(v, float) and v > 0.92 and isinstance(s, float) and s < 0.12:
                print(f"[DEBUG] Classified as white (V={v:.3f} > 0.92 and S={s:.3f} < 0.12)")
                return "white", 0.9
            # Try to match against each color's HSV ranges
            best_match = None
            best_confidence = 0.0
            for color_name, mapping_data in self.color_mapping.items():
                print(f"[DEBUG] Checking color: {color_name}")
                confidence = self._calculate_hsv_match_confidence(h, s, v, mapping_data)
                if isinstance(confidence, float) and isinstance(tolerance, float):
                    if confidence > best_confidence and confidence > tolerance:
                        best_match = color_name
                        best_confidence = confidence
                        print(f"[DEBUG] New best match: {color_name} (confidence: {confidence:.2f})")
            if best_match is not None and isinstance(best_confidence, float) and best_confidence > 0.0:
                print(f"[DEBUG] HSV matched color: {best_match} (confidence: {best_confidence:.2f})")
                return best_match, best_confidence
            # Fallback: Use webcolors to get the closest CSS3 color name
            try:
                rgb = hex_to_rgb(hex_color)
                css3_name = webcolors.rgb_to_name(rgb, spec='css3')
                css3_name = css3_name.lower()
                print(f"[DEBUG] webcolors exact match: {css3_name}")
                if css3_name in self.color_mapping:
                    return css3_name, 0.7  # Assign a moderate confidence
            except ValueError:
                # No exact match, find closest
                min_dist = float('inf')
                closest_name = None
                rgb = hex_to_rgb(hex_color)
                if isinstance(CSS3_NAMES_TO_HEX, dict):
                    for name, hex_val in CSS3_NAMES_TO_HEX.items():
                        candidate_rgb = webcolors.hex_to_rgb(hex_val)
                        dist = sum((a - b) ** 2 for a, b in zip(rgb, candidate_rgb))
                        if dist < min_dist:
                            min_dist = dist
                            closest_name = name
                    if closest_name:
                        closest_name = closest_name.lower()
                        print(f"[DEBUG] webcolors closest match: {closest_name}")
                        if closest_name in self.color_mapping:
                            return closest_name, 0.6  # Lower confidence for closest match
            print(f"[DEBUG] No match found for {hex_color}")
            # If all else fails, return None
            return None, 0.0
        except Exception as e:
            print(f"❌ Error classifying hex color '{hex_color}': {e}")
            return None, 0.0
    
    def _calculate_hsv_match_confidence(self, h: float, s: float, v: float, mapping_data: Dict) -> float:
        """
        Calculate confidence score for HSV values against color mapping data
        
        Args:
            h, s, v: HSV values to check
            mapping_data: Color mapping data from JSON
            
        Returns:
            Confidence score (0.0-1.0)
        """
        try:
            confidence = 0.0
            color_name = mapping_data.get("description", "unknown")
            
            # Check hue range
            hue_ranges = mapping_data.get("hue", [])
            if isinstance(hue_ranges, list) and len(hue_ranges) >= 2:
                print(f"[DEBUG] {color_name}: Checking hue {h:.1f} against ranges {hue_ranges}")
                if is_in_hue_range(h, hue_ranges):
                    confidence += 0.6  # Hue is most important
                    print(f"[DEBUG] {color_name}: Hue {h:.1f} matches range {hue_ranges} (+0.6)")
                else:
                    print(f"[DEBUG] {color_name}: Hue {h:.1f} does NOT match range {hue_ranges}")
            else:
                print(f"[DEBUG] {color_name}: No valid hue ranges found: {hue_ranges}")
            
            # Check saturation range
            sat_range = mapping_data.get("saturation", [0, 1])
            if is_in_range(s, sat_range):
                confidence += 0.2
                print(f"[DEBUG] {color_name}: Saturation {s:.3f} matches range {sat_range} (+0.2)")
            else:
                print(f"[DEBUG] {color_name}: Saturation {s:.3f} does NOT match range {sat_range}")
                # Penalize heavily for saturation mismatch - this is critical for color classification
                confidence *= 0.3  # Reduce confidence by 70% for saturation mismatch
            
            # Check value range  
            val_range = mapping_data.get("value", [0, 1])
            if is_in_range(v, val_range):
                confidence += 0.2
                print(f"[DEBUG] {color_name}: Value {v:.3f} matches range {val_range} (+0.2)")
            else:
                print(f"[DEBUG] {color_name}: Value {v:.3f} does NOT match range {val_range}")
                # Penalize for value mismatch
                confidence *= 0.7  # Reduce confidence by 30% for value mismatch
            
            print(f"[DEBUG] {color_name}: Total confidence = {confidence:.2f}")
            return confidence
            
        except Exception as e:
            print(f"❌ Error calculating HSV match confidence: {e}")
            return 0.0
    
    def get_psychology_for_hex_color(self, hex_color: str, cultural_context: str = "universal", 
                                   confidence_threshold: float = 0.5) -> Dict:
        """
        Get psychology associations directly from a hex color
        
        Args:
            hex_color: Hex color string (e.g., "#9c774b")
            cultural_context: Cultural context for psychology
            confidence_threshold: Minimum confidence required for classification
            
        Returns:
            Dict with psychology data and classification info
        """
        try:
            # Classify the hex color
            color_name, confidence = self.classify_hex_color(hex_color)
            
            if not color_name or confidence < confidence_threshold:
                return {
                    "hexColor": hex_color,
                    "classifiedAs": color_name,
                    "confidence": confidence,
                    "psychology": [],
                    "status": "low_confidence" if color_name else "no_match",
                    "threshold": confidence_threshold
                }
            
            # Get psychology associations
            psychology = self.get_psychology_for_color(color_name, cultural_context)
            
            # Get HSV for additional context
            h, s, v = hex_to_hsv(hex_color) if COLOR_MATCHING_AVAILABLE else (0, 0, 0)
            
            return {
                "hexColor": hex_color,
                "classifiedAs": color_name,
                "confidence": confidence,
                "psychology": psychology,
                "culturalContext": cultural_context,
                "hsv": {"hue": round(h, 1), "saturation": round(s, 2), "value": round(v, 2)},
                "brightness": get_color_brightness_category((h, s, v)) if COLOR_MATCHING_AVAILABLE else "unknown",
                "saturationLevel": get_color_saturation_category((h, s, v)) if COLOR_MATCHING_AVAILABLE else "unknown",
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Error getting psychology for hex color '{hex_color}': {e}")
            return {
                "hexColor": hex_color,
                "error": str(e),
                "psychology": [],
                "status": "error"
            }
    
    def analyze_color_palette_psychology(self, hex_colors: List[str], cultural_context: str = "universal") -> Dict:
        """
        Analyze psychology of multiple hex colors (color palette)
        
        Args:
            hex_colors: List of hex color strings
            cultural_context: Cultural context for psychology
            
        Returns:
            Dict with palette psychology analysis
        """
        try:
            if not hex_colors:
                return {"error": "No colors provided"}
            
            # Analyze each color
            color_analyses = []
            all_psychology = []
            successful_classifications = 0
            
            for hex_color in hex_colors:
                analysis = self.get_psychology_for_hex_color(hex_color, cultural_context)
                color_analyses.append(analysis)
                
                if analysis.get("status") == "success":
                    successful_classifications += 1
                    all_psychology.extend(analysis.get("psychology", []))
            
            # Find common psychological themes
            psychology_counts = {}
            for psych in all_psychology:
                psychology_counts[psych] = psychology_counts.get(psych, 0) + 1
            
            # Sort by frequency
            common_themes = sorted(psychology_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate palette metrics
            palette_metrics = {
                "totalColors": len(hex_colors),
                "successfulClassifications": successful_classifications,
                "classificationRate": round(successful_classifications / len(hex_colors), 2),
                "totalPsychologyAssociations": len(all_psychology),
                "uniquePsychologyThemes": len(psychology_counts),
                "dominantThemes": [theme for theme, count in common_themes[:5]]
            }
            
            return {
                "palette": hex_colors,
                "culturalContext": cultural_context,
                "colorAnalyses": color_analyses,
                "commonPsychologyThemes": dict(common_themes),
                "dominantPsychologyThemes": palette_metrics["dominantThemes"],
                "paletteMetrics": palette_metrics,
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Error analyzing color palette psychology: {e}")
            return {"error": str(e), "status": "error"}
    
    def find_similar_colors(self, hex_color: str, max_distance: float = 0.3) -> List[Dict]:
        """
        Find colors psychologically similar to the given hex color
        
        Args:
            hex_color: Reference hex color
            max_distance: Maximum HSV distance for similarity
            
        Returns:
            List of similar colors with psychology data
        """
        if not COLOR_MATCHING_AVAILABLE:
            return []
        
        try:
            reference_hsv = hex_to_hsv(hex_color)
            similar_colors = []
            
            # Check each color in our database
            for color_name in self.get_available_colors():
                # Get representative HSV for this color (using mapping data)
                mapping_data = self.color_mapping.get(color_name, {})
                
                if not mapping_data:
                    continue
                
                # Calculate approximate HSV for this color category
                hue_ranges = mapping_data.get("hue", [])
                if len(hue_ranges) >= 2:
                    avg_hue = (hue_ranges[0] + hue_ranges[1]) / 2
                else:
                    continue
                
                sat_range = mapping_data.get("saturation", [0.5, 0.8])
                val_range = mapping_data.get("value", [0.5, 0.8])
                
                avg_sat = (sat_range[0] + sat_range[1]) / 2
                avg_val = (val_range[0] + val_range[1]) / 2
                
                color_hsv = (avg_hue, avg_sat, avg_val)
                
                # Calculate distance
                distance = calculate_color_distance(reference_hsv, color_hsv)
                
                if distance <= max_distance:
                    psychology = self.get_psychology_for_color(color_name, "universal")
                    similar_colors.append({
                        "colorName": color_name,
                        "distance": round(distance, 3),
                        "psychology": psychology,
                        "estimatedHSV": {
                            "hue": round(avg_hue, 1),
                            "saturation": round(avg_sat, 2),
                            "value": round(avg_val, 2)
                        }
                    })
            
            # Sort by distance (most similar first)
            similar_colors.sort(key=lambda x: x["distance"])
            
            return similar_colors
            
        except Exception as e:
            print(f"❌ Error finding similar colors for '{hex_color}': {e}")
            return []
    
    def validate_cultural_context(self, context: str) -> bool:
        """Validate if a cultural context is available"""
        return context in self.cultural_contexts
    
    def get_psychology_summary(self) -> Dict:
        """Get summary statistics about the psychology data"""
        try:
            color_psychology = self.psychology_data.get("colorPsychology", {})
            
            total_colors = len(color_psychology)
            total_contexts = len(self.cultural_contexts)
            
            # Count total associations
            total_associations = 0
            for color_data in color_psychology.values():
                for context_associations in color_data.values():
                    total_associations += len(context_associations)
            
            # Get color families
            color_families = self.psychology_data.get("colorFamilies", {})
            
            return {
                "totalColors": total_colors,
                "totalCulturalContexts": total_contexts,
                "totalAssociations": total_associations,
                "averageAssociationsPerColor": round(total_associations / max(total_colors, 1), 2),
                "availableColorFamilies": list(color_families.keys()),
                "colorMatchingEnabled": COLOR_MATCHING_AVAILABLE,
                "colorMappingRules": len(self.color_mapping),
                "metadata": self.psychology_data.get("metadata", {}),
                "dataSource": "datastore/color_psychology_data.json",
                "serviceStatus": "initialized",
                "generatedAt": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error generating psychology summary: {e}")
            return {"error": str(e)}

# Export for easy importing
__all__ = ["ColorPsychologyService"]