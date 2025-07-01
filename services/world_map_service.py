# services/world_map_service.py
"""
Enhanced WorldMapService with Color Psychology Integration
Builds world map data with cultural insights and psychology analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter, defaultdict
import numpy as np
from datetime import datetime

# Import color utilities
try:
    from ..utils.image_utils import hex_to_rgb, rgb_to_hsv, rgb_to_lab, calculate_color_temperature
except ImportError:
    # For direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.image_utils import hex_to_rgb, rgb_to_hsv, rgb_to_lab, calculate_color_temperature

class WorldMapService:
    """Service for generating world map visualization data from clustering results."""
    
    def __init__(self, country_service, psychology_service=None):
        self.country_service = country_service
        self.psychology_service = psychology_service  # 🆕 NEW: Psychology service integration
        self.world_masterdata = self._load_world_masterdata()
        self.svg_coordinates = self._load_svg_coordinates()
        
        # 🆕 NEW: Psychology integration status
        self.psychology_enabled = psychology_service is not None
        if self.psychology_enabled:
            print("🧠 WorldMapService: Psychology integration enabled")
        else:
            print("⚠️ WorldMapService: Psychology service not available")
    
    def _load_world_masterdata(self) -> Dict:
        """Load world countries master data."""
        try:
            # CHANGED: Use datastore folder instead of data folder
            world_file = Path(__file__).parent.parent / "datastore" / "world_countries_masterdata.json"
            with open(world_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"WorldMapService: Loaded masterdata from datastore ({len(data.get('countries', {}))} countries)")
            return data
        except Exception as e:
            print(f"Error loading world masterdata from datastore: {e}")
            return {"countries": {}, "continents": {}, "metadata": {}}
    
    def _load_svg_coordinates(self) -> Dict:
        """Load SVG map coordinates."""
        try:
            # CHANGED: Use datastore folder instead of data folder
            svg_file = Path(__file__).parent.parent / "datastore" / "country_svg_coordinates.json"
            with open(svg_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"WorldMapService: Loaded SVG coordinates from datastore ({len(data.get('countryPositions', {}))} positions)")
            return data
        except Exception as e:
            print(f"Error loading SVG coordinates from datastore: {e}")
            return {"mapProjection": {}, "countryPositions": {}, "continentRegions": {}}
    
    def build_lightweight_summary(
        self, 
        clusters, # Can be List[Dict] or List[ClusterOutputItem]
        url_to_country: Dict[str, str],
        image_colors_list: List[List[str]],
        include_psychology: bool = False,  # 🆕 NEW: Psychology flag
        cultural_context: str = "universal"  # 🆕 NEW: Cultural context
    ) -> Dict:
        """
        🆕 ENHANCED: Build lightweight world map data with optional psychology analysis.
        
        Args:
            clusters: Clustering results from SOM/K-means
            url_to_country: Mapping of image URLs to country names
            image_colors_list: List of dominant colors per image (REUSED from existing extraction)
            include_psychology: Whether to include psychology analysis
            cultural_context: Cultural context for psychology analysis
        
        Returns:
            Enhanced world map summary with optional psychology insights
        """
        try:
            print(f"🗺️ Building enhanced world map summary. Psychology: {'✅' if include_psychology else '❌'}")
            
            # 🎯 REUSE: Extract existing country-color mapping from clusters
            country_to_colors = self._extract_country_colors_from_clusters(
                clusters, url_to_country, image_colors_list
            )
            
            # 🆕 NEW: Add psychology analysis if enabled
            country_psychology_data = {}
            if include_psychology and self.psychology_enabled:
                country_psychology_data = self._analyze_country_psychology(
                    country_to_colors, cultural_context
                )
            
            # Build enhanced country colors dictionary
            country_colors = {}
            country_image_counts = {}
            available_countries = []
            
            for country_code, color_data in country_to_colors.items():
                country_name = self.country_service.get_country_name(country_code)
                if not country_name or country_name.startswith("Unknown"):
                    continue
                
                available_countries.append(country_name)
                colors = color_data["colors"]
                image_count = color_data["image_count"]
                
                if colors:
                    # 🎯 REUSE: Calculate color metrics using existing utilities
                    dominant_color = colors[0]  # Most frequent color
                    color_family = self._classify_color_family(dominant_color)
                    temperature = self._calculate_color_temperature_safe(dominant_color)
                    
                    # 🆕 NEW: Enhanced country color data with psychology
                    enhanced_color_data = {
                        "dominant": dominant_color,
                        "family": color_family,
                        "totalColors": len(colors),
                        "temperature": temperature,
                        "temperatureDescription": self._get_temperature_description(temperature),
                        "palette": colors[:5],  # Top 5 colors
                        "imageCount": image_count
                    }
                    
                    # 🆕 NEW: Add psychology data if available
                    if country_code in country_psychology_data:
                        psychology_profile = country_psychology_data[country_code]
                        enhanced_color_data.update({
                            "psychologyProfile": psychology_profile["aggregated_themes"],
                            "dominantThemes": psychology_profile["top_themes"],
                            "culturalInsights": psychology_profile["cultural_insights"],
                            "psychologyConfidence": psychology_profile["confidence"]
                        })
                    
                    country_colors[country_code] = enhanced_color_data
                    country_image_counts[country_code] = image_count
            
            # Calculate continent distribution
            continent_distribution = self._calculate_continent_distribution(available_countries)
            
            # 🆕 NEW: Build psychology-based country groups if psychology enabled
            psychology_groups = []
            if include_psychology and country_psychology_data:
                psychology_groups = self._create_psychology_country_groups(country_psychology_data)
            
            # 🆕 NEW: Enhanced summary with psychology integration
            enhanced_summary = {
                "availableCountries": available_countries,
                "countryColors": country_colors,
                "continentDistribution": continent_distribution,
                "countryImageCounts": country_image_counts,
                "totalCountriesWithData": len(available_countries),
                "lastUpdated": datetime.now().isoformat(),
                "hasWorldMapData": len(available_countries) > 0,
                
                # 🆕 NEW: Psychology insights
                "psychologyEnabled": include_psychology and self.psychology_enabled,
                "culturalContext": cultural_context if include_psychology else None,
                "psychologyGroups": psychology_groups,
                "psychologyStats": self._calculate_psychology_stats(country_psychology_data) if country_psychology_data else {}
            }
            
            success_msg = f"✅ Enhanced world map: {len(available_countries)} countries"
            if include_psychology and country_psychology_data:
                success_msg += f", {len(psychology_groups)} psychology groups"
            print(success_msg)
            
            return enhanced_summary
            
        except Exception as e:
            print(f"❌ Error building enhanced world map summary: {e}")
            import traceback
            traceback.print_exc()
            return {
                "availableCountries": [],
                "countryColors": {},
                "continentDistribution": {},
                "countryImageCounts": {},
                "totalCountriesWithData": 0,
                "hasWorldMapData": False,
                "psychologyEnabled": False,
                "error": str(e)
            }
    
    # def _extract_country_colors_from_clusters(
    #     self, 
    #     clusters, 
    #     url_to_country: Dict[str, str], 
    #     image_colors_list: List[List[str]]
    # ) -> Dict[str, Dict]:
    #     """
    #     🎯 REUSE: Extract country-to-colors mapping from existing cluster data.
    #     This reuses the already-extracted colors from K-means clustering.
    #     """
    #     country_to_colors = defaultdict(lambda: {"colors": [], "image_count": 0})
        
    #     try:
    #         print(f"🔍 DEBUG: Processing {len(clusters)} clusters")
    #         print(f"🔍 DEBUG: url_to_country has {len(url_to_country)} entries")
            
    #         # Handle both dict and object cluster formats
    #         for i, cluster in enumerate(clusters):
    #             if hasattr(cluster, 'imageUrls'):  # ClusterOutputItem object
    #                 image_urls = cluster.imageUrls
    #                 cluster_colors = cluster.dominantColors
    #             elif isinstance(cluster, dict):  # Dict format
    #                 image_urls = cluster.get('imageUrls', [])
    #                 cluster_colors = cluster.get('dominantColors', [])
    #             else:
    #                 print(f"🔍 DEBUG: Unknown cluster format: {type(cluster)}")
    #                 continue
                
    #             print(f"🔍 DEBUG: Cluster {i}: {len(image_urls)} images, {len(cluster_colors)} colors")
                
    #             # Map images to countries and collect colors
    #             for j, image_url in enumerate(image_urls):
    #                 country_name = url_to_country.get(image_url, "Unknown")
    #                 # print(f"🔍 DEBUG: Image {j}: {image_url[:50]}... -> {country_name}")
                    
    #                 if country_name and not country_name.startswith("Unknown"):
    #                     # Get country code
    #                     country_code = self.country_service.get_country_code(country_name)
    #                     print(f"🔍 DEBUG: Country name '{country_name}' -> code '{country_code}'")
                        
    #                     if country_code:
    #                         # Add cluster colors to country
    #                         country_to_colors[country_code]["colors"].extend(cluster_colors)
    #                         country_to_colors[country_code]["image_count"] += 1
    #                         print(f"🔍 DEBUG: Added colors to {country_code}")
            
    #         # 🎯 REUSE: Process and deduplicate colors for each country
    #         for country_code in country_to_colors:
    #             colors = country_to_colors[country_code]["colors"]
    #             if colors:
    #                 # Get most frequent colors (weighted by occurrence)
    #                 color_counts = Counter(colors)
    #                 # Get top 8 most frequent colors
    #                 top_colors = [color for color, count in color_counts.most_common(8)]
    #                 country_to_colors[country_code]["colors"] = top_colors
    #                 print(f"🔍 DEBUG: {country_code} final colors: {top_colors}")
            
    #         print(f"🎨 Extracted colors for {len(country_to_colors)} countries from existing cluster data")
    #         return dict(country_to_colors)
            
    #     except Exception as e:
    #         print(f"❌ Error extracting country colors: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return {}


    def _extract_country_colors_from_clusters(
        self, 
        clusters, 
        url_to_country: Dict[str, str], 
        image_colors_list: List[List[str]]
    ) -> Dict[str, Dict]:
        """
        🎯 REUSE: Extract country-to-colors mapping from existing cluster data.
        This reuses the already-extracted colors from K-means clustering.
        """
        country_to_colors = defaultdict(lambda: {"colors": [], "image_count": 0})
        
        try:
            print(f"🔍 DEBUG: Processing {len(clusters)} clusters")
            print(f"🔍 DEBUG: url_to_country has {len(url_to_country)} entries")
            
            # 🆕 SAFE: Show country distribution without URLs
            if url_to_country:
                country_counts = {}
                for url, country in url_to_country.items():
                    country_counts[country] = country_counts.get(country, 0) + 1
                print(f"🔍 DEBUG: Available countries: {country_counts}")
            
            # Handle both dict and object cluster formats
            for i, cluster in enumerate(clusters):
                if hasattr(cluster, 'imageUrls'):  # ClusterOutputItem object
                    image_urls = cluster.imageUrls
                    cluster_colors = cluster.dominantColors
                elif isinstance(cluster, dict):  # Dict format
                    image_urls = cluster.get('imageUrls', [])
                    cluster_colors = cluster.get('dominantColors', [])
                else:
                    print(f"🔍 DEBUG: Unknown cluster format: {type(cluster)}")
                    continue
                
                print(f"🔍 DEBUG: Cluster {i}: {len(image_urls)} images, {len(cluster_colors)} colors")
                
                # Map images to countries and collect colors
                for j, image_url in enumerate(image_urls):
                    country_name = url_to_country.get(image_url, "Unknown")
                    print(f"🔍 DEBUG: Image {j}: -> '{country_name}'")  # 🆕 SAFE: No URL
                    
                    if country_name and not country_name.startswith("Unknown"):
                        # Get country code
                        country_code = self.country_service.get_country_code_from_name(country_name)
                        print(f"🔍 DEBUG: '{country_name}' -> code '{country_code}'")
                        
                        if country_code:
                            # Add cluster colors to country
                            country_to_colors[country_code]["colors"].extend(cluster_colors)
                            country_to_colors[country_code]["image_count"] += 1
                            print(f"🔍 DEBUG: ✅ Added colors to {country_code}")
                        else:
                            print(f"🔍 DEBUG: ❌ No country code found for '{country_name}'")
                    else:
                        print(f"🔍 DEBUG: ❌ Skipping unknown: '{country_name}'")
            
            # 🎯 REUSE: Process and deduplicate colors for each country
            for country_code in country_to_colors:
                colors = country_to_colors[country_code]["colors"]
                if colors:
                    # Get most frequent colors (weighted by occurrence)
                    color_counts = Counter(colors)
                    # Get top 8 most frequent colors
                    top_colors = [color for color, count in color_counts.most_common(8)]
                    country_to_colors[country_code]["colors"] = top_colors
                    print(f"🔍 DEBUG: {country_code} final: {len(top_colors)} colors")
            
            print(f"🎨 DEBUG SUMMARY: Extracted colors for {len(country_to_colors)} countries")
            if country_to_colors:
                print(f"🏛️ DEBUG: Countries with colors: {list(country_to_colors.keys())}")
            else:
                print("❌ DEBUG: NO COUNTRIES EXTRACTED - Check country mapping logic")
            
            print(f"🎨 Extracted colors for {len(country_to_colors)} countries from existing cluster data")
            return dict(country_to_colors)
            
        except Exception as e:
            print(f"❌ Error extracting country colors: {e}")
            import traceback
            traceback.print_exc()
            return {}

    
    def _analyze_country_psychology(
        self, 
        country_to_colors: Dict[str, Dict], 
        cultural_context: str = "universal"
    ) -> Dict[str, Dict]:
        """
        🆕 NEW: Analyze psychology for each country based on their dominant colors.
        Uses existing ColorPsychologyService.
        """
        if not self.psychology_service:
            return {}
        
        country_psychology_data = {}
        
        try:
            print(f"🧠 Analyzing psychology for {len(country_to_colors)} countries")
            
            for country_code, color_data in country_to_colors.items():
                colors = color_data["colors"]
                image_count = color_data["image_count"]
                
                if not colors:
                    continue
                
                # Analyze psychology for each color
                color_psychology_results = []
                for color in colors[:5]:  # Analyze top 5 colors
                    try:
                        psych_result = self.psychology_service.get_psychology_for_hex_color(
                            hex_color=color,
                            cultural_context=cultural_context,
                            confidence_threshold=0.5
                        )
                        if psych_result.get("status") == "success":
                            color_psychology_results.append({
                                "color": color,
                                "psychology": psych_result.get("psychology", []),
                                "classifiedAs": psych_result.get("classifiedAs"),
                                "confidence": psych_result.get("confidence", 0.0)
                            })
                    except Exception as e:
                        print(f"Error analyzing color {color}: {e}")
                        continue
                
                # Aggregate psychology themes
                if color_psychology_results:
                    aggregated_analysis = self._aggregate_color_psychology(
                        color_psychology_results, image_count
                    )
                    country_psychology_data[country_code] = aggregated_analysis
            
            print(f"✅ Psychology analysis completed for {len(country_psychology_data)} countries")
            return country_psychology_data
            
        except Exception as e:
            print(f"❌ Error in country psychology analysis: {e}")
            return {}
    
    def _aggregate_color_psychology(
        self, 
        color_psychology_results: List[Dict], 
        image_count: int
    ) -> Dict:
        """
        🆕 NEW: Aggregate psychology themes with frequency and prominence weighting.
        Implements the requirement: "Most frequent + weighted by prominence + top 3-5 themes"
        """
        try:
            all_themes = []
            theme_color_mapping = defaultdict(list)
            confidence_scores = []
            
            # Collect all psychology themes
            for result in color_psychology_results:
                psychology = result.get("psychology", [])
                color = result.get("color")
                confidence = result.get("confidence", 0.0)
                
                for theme in psychology:
                    all_themes.append(theme)
                    theme_color_mapping[theme].append(color)
                    confidence_scores.append(confidence)
            
            if not all_themes:
                return {
                    "aggregated_themes": [],
                    "top_themes": [],
                    "cultural_insights": [],
                    "confidence": 0.0
                }
            
            # Count frequency of themes
            theme_counts = Counter(all_themes)
            
            # Calculate weighted scores (frequency + prominence)
            theme_scores = {}
            for theme, count in theme_counts.items():
                frequency_weight = count / len(all_themes)  # Frequency weight
                prominence_weight = min(count / len(color_psychology_results), 1.0)  # Prominence weight
                combined_score = (frequency_weight * 0.6) + (prominence_weight * 0.4)
                theme_scores[theme] = combined_score
            
            # Get top themes sorted by combined score
            sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Extract top 3-5 themes
            top_themes = [theme for theme, score in sorted_themes[:5]]
            aggregated_themes = [theme for theme, score in sorted_themes[:8]]
            
            # Generate cultural insights
            cultural_insights = self._generate_cultural_insights(
                top_themes, theme_color_mapping, image_count
            )
            
            # Calculate average confidence
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            return {
                "aggregated_themes": aggregated_themes,
                "top_themes": top_themes,
                "cultural_insights": cultural_insights,
                "confidence": round(avg_confidence, 2),
                "theme_scores": dict(sorted_themes[:5]),  # Top 5 with scores
                "total_colors_analyzed": len(color_psychology_results)
            }
            
        except Exception as e:
            print(f"Error aggregating psychology: {e}")
            return {
                "aggregated_themes": [],
                "top_themes": [],
                "cultural_insights": [],
                "confidence": 0.0
            }
    
    def _generate_cultural_insights(
        self, 
        top_themes: List[str], 
        theme_color_mapping: Dict[str, List[str]], 
        image_count: int
    ) -> List[str]:
        """🆕 NEW: Generate cultural insights based on psychology themes."""
        insights = []
        
        try:
            # Emotional patterns
            emotional_themes = ["Happiness", "Calmness", "Energy", "Enthusiasm", "Love"]
            found_emotional = [t for t in top_themes if t in emotional_themes]
            if found_emotional:
                insights.append(f"Emotional expression through {', '.join(found_emotional[:2])}")
            
            # Natural themes
            natural_themes = ["Nature", "Earth", "Sky", "Water", "Sunshine"]
            found_natural = [t for t in top_themes if t in natural_themes]
            if found_natural:
                insights.append(f"Connection to nature via {', '.join(found_natural[:2])}")
            
            # Cultural themes
            cultural_themes = ["Royalty", "Tradition", "Stability", "Balance"]
            found_cultural = [t for t in top_themes if t in cultural_themes]
            if found_cultural:
                insights.append(f"Cultural values: {', '.join(found_cultural[:2])}")
            
            # Add data quality insight
            if image_count > 1:
                insights.append(f"Analysis based on {image_count} postcards")
            
            return insights[:3]  # Return top 3 insights
            
        except Exception as e:
            print(f"Error generating cultural insights: {e}")
            return []
    
    def _create_psychology_country_groups(
        self, 
        country_psychology_data: Dict[str, Dict]
    ) -> List[Dict]:
        """
        🆕 NEW: Group countries by shared psychology themes.
        Implements requirement: "Countries sharing 2+ psychology terms"
        """
        try:
            psychology_groups = []
            
            # Create theme-to-countries mapping
            theme_to_countries = defaultdict(list)
            for country_code, psych_data in country_psychology_data.items():
                top_themes = psych_data.get("top_themes", [])
                for theme in top_themes:
                    theme_to_countries[theme].append(country_code)
            
            # Find countries that share multiple themes
            country_pairs_with_shared_themes = defaultdict(set)
            
            for theme, countries in theme_to_countries.items():
                if len(countries) >= 2:  # Theme shared by 2+ countries
                    for i, country1 in enumerate(countries):
                        for country2 in countries[i+1:]:
                            country_pairs_with_shared_themes[(country1, country2)].add(theme)
            
            # Create groups for countries with 2+ shared themes
            processed_countries = set()
            
            for (country1, country2), shared_themes in country_pairs_with_shared_themes.items():
                if len(shared_themes) >= 2:  # 2+ shared themes requirement
                    if country1 not in processed_countries and country2 not in processed_countries:
                        # Find all countries that share themes with this pair
                        group_countries = {country1, country2}
                        group_themes = shared_themes.copy()
                        
                        # Expand group with similar countries
                        for other_country, other_psych in country_psychology_data.items():
                            if other_country not in group_countries:
                                other_themes = set(other_psych.get("top_themes", []))
                                if len(other_themes.intersection(group_themes)) >= 2:
                                    group_countries.add(other_country)
                        
                        # Create psychology group
                        psychology_groups.append({
                            "groupId": f"psych_group_{len(psychology_groups) + 1}",
                            "sharedThemes": list(group_themes),
                            "countries": list(group_countries),
                            "countryNames": [
                                self.country_service.get_country_name(code) 
                                for code in group_countries
                            ],
                            "groupSize": len(group_countries),
                            "themeCount": len(group_themes)
                        })
                        
                        # Mark countries as processed
                        processed_countries.update(group_countries)
            
            # Sort groups by size (largest first)
            psychology_groups.sort(key=lambda x: x["groupSize"], reverse=True)
            
            print(f"🧠 Created {len(psychology_groups)} psychology-based country groups")
            return psychology_groups[:10]  # Return top 10 groups
            
        except Exception as e:
            print(f"Error creating psychology country groups: {e}")
            return []
    
    def _calculate_psychology_stats(self, country_psychology_data: Dict) -> Dict:
        """🆕 NEW: Calculate statistics about psychology analysis."""
        try:
            if not country_psychology_data:
                return {}
            
            all_themes = []
            confidence_scores = []
            
            for psych_data in country_psychology_data.values():
                all_themes.extend(psych_data.get("aggregated_themes", []))
                confidence_scores.append(psych_data.get("confidence", 0.0))
            
            theme_counts = Counter(all_themes)
            
            return {
                "totalCountriesAnalyzed": len(country_psychology_data),
                "averageConfidence": round(sum(confidence_scores) / len(confidence_scores), 2) if confidence_scores else 0.0,
                "totalUniqueThemes": len(theme_counts),
                "mostCommonThemes": [theme for theme, count in theme_counts.most_common(5)],
                "analysisCompleteness": len(country_psychology_data) / max(len(country_psychology_data), 1) * 100
            }
            
        except Exception as e:
            print(f"Error calculating psychology stats: {e}")
            return {}
    
    # 🎯 EXISTING METHODS (UNCHANGED) - Keeping all existing functionality
    
    def _classify_color_family(self, hex_color: str) -> str:
        """Classify color into a family based on HSV values."""
        try:
            rgb = hex_to_rgb(hex_color)
            h, s, v = rgb_to_hsv(rgb)
            
            # Low saturation = grayscale
            if s < 0.2:
                return "Grays" if v > 0.3 else "Blacks"
            
            # High saturation colors by hue
            if h < 15 or h >= 345:
                return "Reds"
            elif 15 <= h < 45:
                return "Oranges"
            elif 45 <= h < 75:
                return "Yellows"
            elif 75 <= h < 165:
                return "Greens"
            elif 165 <= h < 195:
                return "Cyans"
            elif 195 <= h < 255:
                return "Blues"
            elif 255 <= h < 285:
                return "Purples"
            elif 285 <= h < 345:
                return "Magentas"
            else:
                return "Others"
                
        except Exception as e:
            print(f"Error classifying color {hex_color}: {e}")
            return "Others"
    
    def _calculate_color_temperature_safe(self, hex_color: str) -> float:
        """Safely calculate color temperature."""
        try:
            rgb = hex_to_rgb(hex_color)
            return calculate_color_temperature(rgb)
        except Exception as e:
            print(f"Error calculating temperature for {hex_color}: {e}")
            return 0.5  # Neutral temperature
    
    def _get_temperature_description(self, temperature: float) -> str:
        """Get human-readable temperature description."""
        if temperature < 0.3:
            return "Cool"
        elif temperature > 0.7:
            return "Warm"
        else:
            return "Neutral"
    
    def _calculate_continent_distribution(self, countries: List[str]) -> Dict[str, int]:
        """Calculate distribution of countries per continent."""
        continent_counts = Counter()
        
        for country_name in countries:
            country_code = self.country_service.get_country_code_from_name(country_name)
            if country_code:
                country_info = self.world_masterdata.get("countries", {}).get(country_code, {})
                continent = country_info.get("continent", "Unknown")
                if continent != "Unknown":
                    continent_counts[continent] += 1
        
        return dict(continent_counts)
    
    # Keep all other existing methods unchanged...
    def build_complete_world_map_data(self) -> Dict:
        """Build complete world map data for detailed visualization."""
        try:
            all_countries = {}
            available_countries = list(self.world_masterdata.get("countries", {}).keys())
            
            for country_code in available_countries:
                country_data = self._build_country_data(country_code)
                if country_data:
                    all_countries[country_code] = country_data
            
            cultural_insights = self._generate_complete_cultural_insights(all_countries)
            global_statistics = self._calculate_global_statistics(all_countries)
            
            return {
                "countries": all_countries,
                "culturalInsights": cultural_insights,
                "globalStatistics": global_statistics,
                "mapProjection": self.svg_coordinates.get("mapProjection", {}),
                "continentRegions": self.svg_coordinates.get("continentRegions", {}),
                "metadata": {
                    "totalCountries": len(all_countries),
                    "lastUpdated": datetime.now().isoformat(),
                    "dataVersion": self.world_masterdata.get("metadata", {}).get("version", "1.0"),
                    "psychologyEnabled": self.psychology_enabled
                }
            }
            
        except Exception as e:
            print(f"Error building complete world map data: {e}")
            return {"countries": {}, "culturalInsights": [], "globalStatistics": {}, "error": str(e)}
    
    def _build_country_data(self, country_code: str) -> Optional[Dict]:
        """Build complete data for a single country."""
        try:
            country_info = self.world_masterdata.get("countries", {}).get(country_code, {})
            if not country_info:
                return None
            
            svg_coords = self.svg_coordinates.get("countryPositions", {}).get(country_code, {})
            
            result = {
                "fullName": country_info.get("fullName", ""),
                "countryCode": country_code,
                "continent": country_info.get("continent", ""),
                "region": country_info.get("region", ""),
                "coordinates": country_info.get("coordinates", {}),
                "mapPosition": {"x": svg_coords.get("x", 0), "y": svg_coords.get("y", 0)},
                "labelOffset": svg_coords.get("labelOffset", {"x": 0, "y": -5}),
                "circleRadius": svg_coords.get("circleRadius", 6),
                "timezone": country_info.get("timezone", ""),
                "culturalTags": country_info.get("culturalTags", []),
                "geographicFeatures": country_info.get("geographicFeatures", []),
                "neighbors": country_info.get("neighbors", []),
                "dominantColor": self._get_default_continent_color(country_info.get("continent", "")),
                "colorPalette": [],
                "colorFamily": "Others",
                "totalImages": 0,
                "dataQuality": 0.0,
                "hasData": False
            }
            
            return result
            
        except Exception as e:
            print(f"Error building country data for {country_code}: {e}")
            return None
    
    def _get_default_continent_color(self, continent: str) -> str:
        """Get default color for continent."""
        continent_colors = {
            "Europe": "#4A90E2", "Asia": "#9013FE", "North America": "#D0021B",
            "South America": "#7ED321", "Africa": "#F5A623", "Oceania": "#92400E"
        }
        return continent_colors.get(continent, "#9CA3AF")
    
    def _generate_complete_cultural_insights(self, countries_data: Dict) -> List[Dict]:
        """Generate cultural insights from country data."""
        try:
            insights = []
            continent_colors = defaultdict(list)
            
            for data in countries_data.values():
                continent = data.get("continent", "")
                color = data.get("dominantColor", "")
                if continent and color:
                    continent_colors[continent].append(color)
            
            for continent, colors in continent_colors.items():
                unique_colors = len(set(colors))
                insights.append({
                    "type": "continental_pattern",
                    "continent": continent,
                    "description": f"{continent} shows {unique_colors} distinct color patterns",
                    "countries": len(colors)
                })
            
            return insights[:5]
            
        except Exception as e:
            print(f"Error generating cultural insights: {e}")
            return []
    
    def _calculate_global_statistics(self, countries_data: Dict) -> Dict:
        """Calculate global statistics from countries data."""
        try:
            continent_stats = defaultdict(lambda: {"countries": 0, "avgImages": 0.0})
            continent_totals = defaultdict(int)
            
            for data in countries_data.values():
                continent = data.get("continent", "")
                if continent:
                    continent_stats[continent]["countries"] += 1
                    continent_totals[continent] += data.get("totalImages", 0)
            
            for continent in continent_stats:
                if continent_stats[continent]["countries"] > 0:
                    continent_stats[continent]["avgImages"] = (
                        continent_totals[continent] / continent_stats[continent]["countries"]
                    )
            
            color_families = Counter()
            for data in countries_data.values():
                family = data.get("colorFamily", "Others")
                color_families[family] += 1
            
            return {
                "totalCountries": len(countries_data),
                "continentStatistics": dict(continent_stats),
                "colorFamilyDistribution": dict(color_families),
                "totalContinents": len(continent_stats),
                "dataCompleteness": {
                    "withImages": sum(1 for d in countries_data.values() if d.get("totalImages", 0) > 0),
                    "withoutImages": sum(1 for d in countries_data.values() if d.get("totalImages", 0) == 0)
                }
            }
            
        except Exception as e:
            print(f"Error calculating global statistics: {e}")
            return {}