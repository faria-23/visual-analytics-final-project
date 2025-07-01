# import json
# import re
# from pathlib import Path
# from typing import Dict, Optional, List, Tuple, Union
# from urllib.parse import unquote
# from collections import Counter, defaultdict

# class CountryService:
#     """Service for managing country detection and mapping with world map support."""
    
#     def __init__(self):
#         self.country_mapping = self._load_country_codes()
#         self.world_masterdata = self._load_world_masterdata()
#         self.reverse_mapping = {v: k for k, v in self.country_mapping.items()}
    
#     def _load_country_codes(self) -> Dict[str, str]:
#         """Load country codes from JSON file and create code->name mapping."""
#         try:
#             # CHANGED: Use datastore folder instead of data folder
#             country_file = Path(__file__).parent.parent / "datastore" / "country_codes.json"
#             with open(country_file, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
            
#             # Create mapping from code to name
#             mapping = {}
#             for country in data.get("countries", []):
#                 mapping[country["code"].upper()] = country["name"]
            
#             print(f"Loaded {len(mapping)} country codes from datastore")
#             return mapping
            
#         except Exception as e:
#             print(f"Error loading country codes from datastore: {e}")
#             # Fallback with some common countries
#             return {
#                 "US": "United States of America",
#                 "DE": "Germany", 
#                 "FR": "France",
#                 "GB": "United Kingdom",
#                 "IT": "Italy",
#                 "ES": "Spain",
#                 "CA": "Canada",
#                 "AU": "Australia",
#                 "JP": "Japan",
#                 "CN": "China",
#                 "NL": "Netherlands",
#                 "CH": "Switzerland",
#                 "AT": "Austria",
#                 "BE": "Belgium",
#                 "DK": "Denmark",
#                 "SE": "Sweden",
#                 "NO": "Norway",
#                 "FI": "Finland"
#             }
    
#     def _load_world_masterdata(self) -> Dict:
#         """Load world countries master data for enhanced country information."""
#         try:
#             # CHANGED: Use datastore folder instead of data folder
#             world_file = Path(__file__).parent.parent / "datastore" / "world_countries_masterdata.json"
#             with open(world_file, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
            
#             print(f"Loaded world masterdata for {len(data.get('countries', {}))} countries from datastore")
#             return data
            
#         except Exception as e:
#             print(f"Error loading world masterdata from datastore: {e}")
#             return {"countries": {}, "continents": {}, "metadata": {}}
    
#     def extract_country_from_filename(self, filename: str) -> Optional[str]:
#         """
#         Extract country name from filename format: 'CC-XX.ext' where CC is country code.
        
#         Args:
#             filename: The image filename (e.g., 'DE-39.jpeg')
            
#         Returns:
#             Country name if found, None otherwise
#         """
#         try:
#             # Handle URL-encoded filenames
#             filename = unquote(filename)
            
#             # Extract just the filename without path
#             filename = Path(filename).name
            
#             # Pattern: CC-XX.ext where CC is 2-letter country code
#             pattern = r'^([A-Z]{2})-.*\.'
#             match = re.match(pattern, filename.upper())
            
#             if match:
#                 country_code = match.group(1)
#                 country_name = self.country_mapping.get(country_code)
                
#                 if country_name:
#                     print(f"Detected country: {country_code} -> {country_name} from filename: {filename}")
#                     return country_name
#                 else:
#                     print(f"Unknown country code: {country_code} in filename: {filename}")
#                     return f"Unknown ({country_code})"
#             else:
#                 print(f"No country pattern found in filename: {filename}")
#                 return None
                
#         except Exception as e:
#             print(f"Error extracting country from filename '{filename}': {e}")
#             return None
    
#     def extract_country_from_data_uri(self, data_uri: str) -> Optional[str]:
#         """
#         Extract country from data URI if it contains filename information.
#         This is a fallback method for data URIs that might contain filename hints.
#         """
#         try:
#             # Some data URIs might have filename in the header
#             if 'filename=' in data_uri:
#                 filename_match = re.search(r'filename=([^;,]+)', data_uri)
#                 if filename_match:
#                     filename = filename_match.group(1)
#                     return self.extract_country_from_filename(filename)
            
#             return None
            
#         except Exception as e:
#             print(f"Error extracting country from data URI: {e}")
#             return None
    
#     def get_country_code_from_name(self, country_name: str) -> Optional[str]:
#         """Get country code from country name."""
#         return self.reverse_mapping.get(country_name)
    
#     def get_country_statistics(self, countries: List[str]) -> Dict[str, int]:
#         """Get statistics about countries in the dataset."""
#         country_counts = {}
#         for country in countries:
#             if country and not country.startswith("Unknown"):
#                 country_counts[country] = country_counts.get(country, 0) + 1
        
#         return country_counts
    
#     def validate_country_code(self, code: str) -> bool:
#         """Validate if a country code exists."""
#         return code.upper() in self.country_mapping
    
#     def get_all_countries(self) -> Dict[str, str]:
#         """Get all available country codes and names."""
#         return self.country_mapping.copy()
    
#     # NEW: World Map Support Methods
    
#     def get_country_world_data(self, country_code: str) -> Dict:
#         """
#         Get comprehensive world data for a country.
        
#         Args:
#             country_code: Two-letter country code
            
#         Returns:
#             Dictionary with geographic, cultural, and mapping data
#         """
#         try:
#             country_info = self.world_masterdata.get("countries", {}).get(country_code.upper(), {})
#             if not country_info:
#                 return {}
            
#             # Merge basic country mapping with world data
#             result = {
#                 "countryCode": country_code.upper(),
#                 "countryName": self.country_mapping.get(country_code.upper(), ""),
#                 **country_info
#             }
            
#             return result
            
#         except Exception as e:
#             print(f"Error getting world data for {country_code}: {e}")
#             return {}

######
# services/country_service.py - ENHANCED VERSION
"""
Enhanced Country Service with Color Psychology Integration
Builds on existing country detection and adds psychology analysis
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union
from urllib.parse import unquote
from collections import Counter, defaultdict

class CountryService:
    """Enhanced service for managing country detection and psychology mapping."""
    
    def __init__(self):
        self.country_mapping = self._load_country_codes()
        self.world_masterdata = self._load_world_masterdata()
        self.reverse_mapping = {v: k for k, v in self.country_mapping.items()}
        
        # 🆕 NEW: Psychology service integration
        self.psychology_service = None
        self._init_psychology_service()
        
        # 🆕 NEW: Country psychology cache (in-memory)
        self.country_psychology_cache = {}
    
    def _init_psychology_service(self):
        """Initialize psychology service if available"""
        try:
            from .color_psychology_service import ColorPsychologyService
            self.psychology_service = ColorPsychologyService()
            print("✅ Country service: Psychology integration enabled")
        except ImportError:
            print("⚠️ Country service: Psychology integration not available")
            self.psychology_service = None
    
    def _load_country_codes(self) -> Dict[str, str]:
        """Load country codes from JSON file and create code->name mapping."""
        try:
            country_file = Path(__file__).parent.parent / "datastore" / "country_codes.json"
            with open(country_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            mapping = {}
            for country in data.get("countries", []):
                mapping[country["code"].upper()] = country["name"]
            
            print(f"Loaded {len(mapping)} country codes from datastore")
            return mapping
            
        except Exception as e:
            print(f"Error loading country codes from datastore: {e}")
            return {
                "US": "United States of America", "DE": "Germany", "FR": "France",
                "GB": "United Kingdom", "IT": "Italy", "ES": "Spain", "CA": "Canada",
                "AU": "Australia", "JP": "Japan", "CN": "China", "NL": "Netherlands",
                "CH": "Switzerland", "AT": "Austria", "BE": "Belgium", "DK": "Denmark",
                "SE": "Sweden", "NO": "Norway", "FI": "Finland"
            }
    
    def _load_world_masterdata(self) -> Dict:
        """Load world countries master data for enhanced country information."""
        try:
            world_file = Path(__file__).parent.parent / "datastore" / "world_countries_masterdata.json"
            with open(world_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Loaded world masterdata for {len(data.get('countries', {}))} countries from datastore")
            return data
            
        except Exception as e:
            print(f"Error loading world masterdata from datastore: {e}")
            return {"countries": {}, "continents": {}, "metadata": {}}
    
    # ===== EXISTING METHODS (unchanged) =====
    
    def extract_country_from_filename(self, filename: str) -> Optional[str]:
        """Extract country name from filename format: 'CC-XX.ext' where CC is country code."""
        try:
            filename = unquote(filename)
            filename = Path(filename).name
            pattern = r'^([A-Z]{2})-.*\.'
            match = re.match(pattern, filename.upper())
            
            if match:
                country_code = match.group(1)
                country_name = self.country_mapping.get(country_code)
                
                if country_name:
                    print(f"Detected country: {country_code} -> {country_name} from filename: {filename}")
                    return country_name
                else:
                    print(f"Unknown country code: {country_code} in filename: {filename}")
                    return f"Unknown ({country_code})"
            else:
                print(f"No country pattern found in filename: {filename}")
                return None
                
        except Exception as e:
            print(f"Error extracting country from filename '{filename}': {e}")
            return None
    
    def extract_country_from_data_uri(self, data_uri: str) -> Optional[str]:
        """Extract country from data URI if it contains filename information."""
        try:
            if 'filename=' in data_uri:
                filename_match = re.search(r'filename=([^;,]+)', data_uri)
                if filename_match:
                    filename = filename_match.group(1)
                    return self.extract_country_from_filename(filename)
            return None
        except Exception as e:
            print(f"Error extracting country from data URI: {e}")
            return None
    
    def get_country_code_from_name(self, country_name: str) -> Optional[str]:
        """Get country code from country name."""
        return self.reverse_mapping.get(country_name)
    
    def get_country_statistics(self, countries: List[str]) -> Dict[str, int]:
        """Get statistics about countries in the dataset."""
        country_counts = {}
        for country in countries:
            if country and not country.startswith("Unknown"):
                country_counts[country] = country_counts.get(country, 0) + 1
        return country_counts
    
    def validate_country_code(self, code: str) -> bool:
        """Validate if a country code exists."""
        return code.upper() in self.country_mapping
    
    def get_all_countries(self) -> Dict[str, str]:
        """Get all available country codes and names."""
        return self.country_mapping.copy()
    
    # ===== 🆕 NEW: PSYCHOLOGY INTEGRATION METHODS =====
    
    def analyze_country_colors_psychology(
        self, 
        country_to_colors: Dict[str, List[str]], 
        cultural_context: str = "universal"
    ) -> Dict[str, Dict]:
        """
        Analyze psychology for each country based on their dominant colors.
        
        Args:
            country_to_colors: Dict mapping country names to lists of hex colors
            cultural_context: Cultural context for psychology analysis
            
        Returns:
            Dict mapping country codes to psychology analysis
        """
        if not self.psychology_service:
            return {}
        
        try:
            country_psychology = {}
            
            for country_name, colors in country_to_colors.items():
                if not colors or country_name.startswith("Unknown"):
                    continue
                
                country_code = self.get_country_code_from_name(country_name)
                if not country_code:
                    continue
                
                # Analyze psychology for this country's colors
                psychology_analysis = self._analyze_country_psychology(
                    country_code=country_code,
                    country_name=country_name,
                    colors=colors,
                    cultural_context=cultural_context
                )
                
                if psychology_analysis:
                    country_psychology[country_code] = psychology_analysis
            
            print(f"✅ Analyzed psychology for {len(country_psychology)} countries")
            return country_psychology
            
        except Exception as e:
            print(f"❌ Error analyzing country colors psychology: {e}")
            return {}
    
    def _analyze_country_psychology(
        self, 
        country_code: str, 
        country_name: str, 
        colors: List[str], 
        cultural_context: str
    ) -> Dict:
        """Analyze psychology for a single country."""
        try:
            # Use cache if available
            cache_key = f"{country_code}_{cultural_context}_{hash(tuple(sorted(colors)))}"
            if cache_key in self.country_psychology_cache:
                return self.country_psychology_cache[cache_key]
            
            # Count color frequency (weighted by prominence)
            color_counts = Counter(colors)
            total_colors = len(colors)
            
            # Analyze each unique color
            color_psychology_data = []
            all_psychology_themes = []
            successful_classifications = 0
            
            for color, frequency in color_counts.items():
                weight = frequency / total_colors
                
                # Get psychology for this color
                color_analysis = self.psychology_service.get_psychology_for_hex_color(
                    hex_color=color,
                    cultural_context=cultural_context,
                    confidence_threshold=0.5
                )
                
                if color_analysis.get("status") == "success":
                    successful_classifications += 1
                    psychology_themes = color_analysis.get("psychology", [])
                    
                    # Weight psychology themes by color frequency
                    weighted_themes = [(theme, weight) for theme in psychology_themes]
                    all_psychology_themes.extend(weighted_themes)
                    
                    color_psychology_data.append({
                        "color": color,
                        "classifiedAs": color_analysis.get("classifiedAs"),
                        "psychology": psychology_themes,
                        "confidence": color_analysis.get("confidence", 0),
                        "frequency": frequency,
                        "weight": weight
                    })
            
            # Aggregate weighted psychology themes
            theme_weights = defaultdict(float)
            for theme, weight in all_psychology_themes:
                theme_weights[theme] += weight
            
            # Sort by weighted frequency
            dominant_themes = sorted(theme_weights.items(), key=lambda x: x[1], reverse=True)
            top_themes = [theme for theme, weight in dominant_themes[:5]]
            
            # Create country psychology profile
            analysis = {
                "countryCode": country_code,
                "countryName": country_name,
                "culturalContext": cultural_context,
                "dominantPsychologyThemes": top_themes,
                "themeWeights": dict(dominant_themes),
                "colorBreakdown": color_psychology_data,
                "metrics": {
                    "totalColors": total_colors,
                    "uniqueColors": len(color_counts),
                    "successfulClassifications": successful_classifications,
                    "classificationRate": successful_classifications / len(color_counts) if color_counts else 0,
                    "psychologyDiversity": len(theme_weights)
                },
                "culturalInsights": self._generate_country_cultural_insights(
                    country_code, top_themes, cultural_context
                )
            }
            
            # Cache the result
            self.country_psychology_cache[cache_key] = analysis
            return analysis
            
        except Exception as e:
            print(f"❌ Error analyzing psychology for {country_code}: {e}")
            return {}
    
    def _generate_country_cultural_insights(
        self, 
        country_code: str, 
        psychology_themes: List[str], 
        cultural_context: str
    ) -> List[Dict]:
        """Generate cultural insights for a country based on psychology themes."""
        try:
            insights = []
            
            # Get country metadata
            country_data = self.world_masterdata.get("countries", {}).get(country_code, {})
            cultural_tags = country_data.get("culturalTags", [])
            geographic_features = country_data.get("geographicFeatures", [])
            continent = country_data.get("continent", "")
            
            # 1. Geography-Psychology Alignment
            geo_psychology_matches = []
            if "Coastal" in geographic_features and any(theme in ["Water", "Calmness", "Blue"] for theme in psychology_themes):
                geo_psychology_matches.append("coastal-calm alignment")
            if "Mountains" in geographic_features and any(theme in ["Stability", "Earth", "Natural"] for theme in psychology_themes):
                geo_psychology_matches.append("mountain-earth alignment")
            if "Desert" in geographic_features and any(theme in ["Warmth", "Earth", "Energy"] for theme in psychology_themes):
                geo_psychology_matches.append("desert-warm alignment")
            
            if geo_psychology_matches:
                insights.append({
                    "type": "geography_psychology_alignment",
                    "title": "Visual themes match geographic features",
                    "description": f"Color psychology aligns with {country_data.get('fullName', country_code)}'s geographic characteristics",
                    "details": geo_psychology_matches,
                    "confidence": 0.8
                })
            
            # 2. Cultural Tag Alignment
            cultural_psychology_matches = []
            if "Historic" in cultural_tags and any(theme in ["Tradition", "Stability", "Earth"] for theme in psychology_themes):
                cultural_psychology_matches.append("historic-traditional alignment")
            if "Maritime" in cultural_tags and any(theme in ["Water", "Blue", "Calmness"] for theme in psychology_themes):
                cultural_psychology_matches.append("maritime-water alignment")
            if "Industrial" in cultural_tags and any(theme in ["Energy", "Movement", "Modern"] for theme in psychology_themes):
                cultural_psychology_matches.append("industrial-energy alignment")
            
            if cultural_psychology_matches:
                insights.append({
                    "type": "cultural_psychology_alignment",
                    "title": "Visual themes reflect cultural characteristics",
                    "description": f"Color psychology matches {country_data.get('fullName', country_code)}'s cultural identity",
                    "details": cultural_psychology_matches,
                    "confidence": 0.75
                })
            
            # 3. Continental Patterns
            if continent:
                continent_insights = self._get_continental_psychology_insights(continent, psychology_themes)
                if continent_insights:
                    insights.extend(continent_insights)
            
            return insights
            
        except Exception as e:
            print(f"❌ Error generating cultural insights for {country_code}: {e}")
            return []
    
    def _get_continental_psychology_insights(self, continent: str, psychology_themes: List[str]) -> List[Dict]:
        """Get continent-specific psychology insights."""
        insights = []
        
        continental_patterns = {
            "Europe": {
                "expected_themes": ["Historic", "Culture", "Sophistication", "Tradition"],
                "description": "European cultural heritage"
            },
            "Asia": {
                "expected_themes": ["Tradition", "Harmony", "Balance", "Spirituality"],
                "description": "Asian philosophical traditions"
            },
            "Africa": {
                "expected_themes": ["Earth", "Natural", "Energy", "Warmth"],
                "description": "African natural landscapes"
            },
            "North America": {
                "expected_themes": ["Energy", "Modern", "Freedom", "Innovation"],
                "description": "North American dynamism"
            },
            "South America": {
                "expected_themes": ["Energy", "Warmth", "Natural", "Passion"],
                "description": "South American vibrancy"
            },
            "Oceania": {
                "expected_themes": ["Natural", "Water", "Calmness", "Freedom"],
                "description": "Oceanic island characteristics"
            }
        }
        
        if continent in continental_patterns:
            pattern = continental_patterns[continent]
            matches = [theme for theme in psychology_themes if theme in pattern["expected_themes"]]
            
            if matches:
                insights.append({
                    "type": "continental_pattern",
                    "title": f"Typical {continent} characteristics",
                    "description": f"Color psychology reflects {pattern['description']}",
                    "details": {
                        "continent": continent,
                        "matching_themes": matches,
                        "pattern_strength": len(matches) / len(pattern["expected_themes"])
                    },
                    "confidence": 0.7
                })
        
        return insights
    
    def group_countries_by_psychology_similarity(
        self, 
        country_psychology: Dict[str, Dict],
        similarity_threshold: int = 2
    ) -> List[Dict]:
        """
        Group countries by psychology similarity.
        
        Args:
            country_psychology: Country psychology analysis data
            similarity_threshold: Minimum shared psychology themes for grouping
            
        Returns:
            List of country groups with shared psychology characteristics
        """
        try:
            if not country_psychology:
                return []
            
            # Extract psychology themes for each country
            country_themes = {}
            for country_code, analysis in country_psychology.items():
                themes = set(analysis.get("dominantPsychologyThemes", []))
                if themes:
                    country_themes[country_code] = themes
            
            # Find country pairs with shared themes
            country_groups = []
            processed_countries = set()
            
            for country1, themes1 in country_themes.items():
                if country1 in processed_countries:
                    continue
                
                # Find similar countries
                similar_countries = [country1]
                shared_themes = themes1.copy()
                
                for country2, themes2 in country_themes.items():
                    if country2 == country1 or country2 in processed_countries:
                        continue
                    
                    # Calculate shared themes
                    overlap = themes1 & themes2
                    if len(overlap) >= similarity_threshold:
                        similar_countries.append(country2)
                        shared_themes &= themes2  # Keep only common themes
                
                # Create group if we have multiple countries
                if len(similar_countries) > 1:
                    group_data = {
                        "groupId": f"psychology_group_{len(country_groups) + 1}",
                        "countries": similar_countries,
                        "countryNames": [
                            country_psychology[code].get("countryName", code)
                            for code in similar_countries
                        ],
                        "sharedPsychologyThemes": list(shared_themes),
                        "groupSize": len(similar_countries),
                        "groupType": "psychology_similarity",
                        "similarityLevel": len(shared_themes),
                        "description": f"Countries sharing {len(shared_themes)} psychology themes: {', '.join(list(shared_themes)[:3])}"
                    }
                    
                    country_groups.append(group_data)
                    processed_countries.update(similar_countries)
            
            print(f"✅ Found {len(country_groups)} psychology-based country groups")
            return country_groups
            
        except Exception as e:
            print(f"❌ Error grouping countries by psychology: {e}")
            return []
    
    def get_enhanced_country_statistics_with_psychology(
        self, 
        countries: List[str],
        country_to_colors: Dict[str, List[str]] = None,
        cultural_context: str = "universal"
    ) -> Dict:
        """
        Enhanced country statistics with psychology analysis.
        
        Args:
            countries: List of country names
            country_to_colors: Optional mapping of countries to their colors
            cultural_context: Cultural context for psychology
            
        Returns:
            Enhanced statistics with psychology insights
        """
        try:
            # Basic statistics
            basic_stats = self.get_country_statistics(countries)
            
            result = {
                "countryStatistics": basic_stats,
                "totalCountries": len(basic_stats),
                "totalImages": sum(basic_stats.values()),
                "psychologyAnalysis": None,
                "countryGroups": [],
                "culturalInsights": []
            }
            
            # Add psychology analysis if colors provided
            if country_to_colors and self.psychology_service:
                print("🧠 Adding psychology analysis to country statistics...")
                
                # Analyze country psychology
                country_psychology = self.analyze_country_colors_psychology(
                    country_to_colors, cultural_context
                )
                
                if country_psychology:
                    # Group similar countries
                    country_groups = self.group_countries_by_psychology_similarity(
                        country_psychology, similarity_threshold=2
                    )
                    
                    # Generate overall insights
                    cultural_insights = self._generate_overall_cultural_insights(
                        country_psychology, country_groups
                    )
                    
                    result.update({
                        "psychologyAnalysis": {
                            "enabled": True,
                            "culturalContext": cultural_context,
                            "countriesAnalyzed": len(country_psychology),
                            "countryPsychologyProfiles": country_psychology,
                            "analysisTimestamp": __import__('datetime').datetime.now().isoformat()
                        },
                        "countryGroups": country_groups,
                        "culturalInsights": cultural_insights
                    })
            
            return result
            
        except Exception as e:
            print(f"❌ Error generating enhanced country statistics: {e}")
            return {
                "countryStatistics": self.get_country_statistics(countries),
                "totalCountries": len(set(countries)),
                "error": str(e)
            }
    
    def _generate_overall_cultural_insights(
        self, 
        country_psychology: Dict[str, Dict], 
        country_groups: List[Dict]
    ) -> List[Dict]:
        """Generate overall cultural insights from country psychology data."""
        insights = []
        
        try:
            # 1. Most common psychology themes across all countries
            all_themes = []
            for analysis in country_psychology.values():
                themes = analysis.get("dominantPsychologyThemes", [])
                all_themes.extend(themes)
            
            if all_themes:
                theme_counts = Counter(all_themes)
                most_common_theme, frequency = theme_counts.most_common(1)[0]
                
                insights.append({
                    "type": "global_pattern",
                    "title": f"'{most_common_theme}' is the dominant global theme",
                    "description": f"The '{most_common_theme}' psychology theme appears in {frequency} countries",
                    "data": {
                        "theme": most_common_theme,
                        "frequency": frequency,
                        "percentage": frequency / len(country_psychology)
                    },
                    "confidence": 0.8
                })
            
            # 2. Country grouping insights
            if country_groups:
                largest_group = max(country_groups, key=lambda g: g["groupSize"])
                insights.append({
                    "type": "country_grouping",
                    "title": f"Psychology-based country clusters found",
                    "description": f"Found {len(country_groups)} groups of countries with similar color psychology",
                    "data": {
                        "total_groups": len(country_groups),
                        "largest_group": largest_group["countryNames"],
                        "largest_group_themes": largest_group["sharedPsychologyThemes"]
                    },
                    "confidence": 0.75
                })
            
            # 3. Continental analysis
            continental_themes = defaultdict(list)
            for country_code, analysis in country_psychology.items():
                country_data = self.world_masterdata.get("countries", {}).get(country_code, {})
                continent = country_data.get("continent", "Unknown")
                themes = analysis.get("dominantPsychologyThemes", [])
                continental_themes[continent].extend(themes)
            
            for continent, themes in continental_themes.items():
                if continent != "Unknown" and len(themes) > 1:
                    theme_counts = Counter(themes)
                    dominant_theme = theme_counts.most_common(1)[0][0]
                    
                    insights.append({
                        "type": "continental_pattern",
                        "title": f"{continent} shows '{dominant_theme}' characteristics",
                        "description": f"Countries in {continent} predominantly show '{dominant_theme}' psychology themes",
                        "data": {
                            "continent": continent,
                            "dominant_theme": dominant_theme,
                            "countries_count": len(set(country_code for country_code, analysis in country_psychology.items() 
                                                     if self.world_masterdata.get("countries", {}).get(country_code, {}).get("continent") == continent))
                        },
                        "confidence": 0.7
                    })
            
            return insights
            
        except Exception as e:
            print(f"❌ Error generating overall cultural insights: {e}")
            return []
    
    def get_country_psychology_profile(self, country_code: str, cultural_context: str = "universal") -> Dict:
        """
        Get detailed psychology profile for a specific country.
        
        Args:
            country_code: Two-letter country code
            cultural_context: Cultural context for analysis
            
        Returns:
            Detailed psychology profile
        """
        try:
            # Check if we have cached data
            cache_key = f"{country_code}_{cultural_context}"
            for cache_key_full, cached_data in self.country_psychology_cache.items():
                if cache_key_full.startswith(cache_key):
                    return cached_data
            
            # If no cached data, return basic country info
            country_data = self.get_country_world_data(country_code)
            return {
                "countryCode": country_code,
                "countryName": country_data.get("countryName", ""),
                "culturalContext": cultural_context,
                "psychologyAnalysis": None,
                "message": "No psychology analysis available. Country needs to be analyzed with image data first."
            }
            
        except Exception as e:
            print(f"❌ Error getting country psychology profile: {e}")
            return {"error": str(e)}
    
    # ===== EXISTING WORLD MAP METHODS (unchanged) =====
    
    def get_country_world_data(self, country_code: str) -> Dict:
        """Get comprehensive world data for a country."""
        try:
            country_info = self.world_masterdata.get("countries", {}).get(country_code.upper(), {})
            if not country_info:
                return {}
            
            result = {
                "countryCode": country_code.upper(),
                "countryName": self.country_mapping.get(country_code.upper(), ""),
                **country_info
            }
            
            return result
            
        except Exception as e:
            print(f"Error getting world data for {country_code}: {e}")
            return {}

######
    
    def get_continent_countries(self, continent: str) -> List[str]:
        """Get all country codes for a specific continent."""
        try:
            continent_info = self.world_masterdata.get("continents", {}).get(continent, {})
            return continent_info.get("countries", [])
        except Exception as e:
            print(f"Error getting countries for continent {continent}: {e}")
            return []
    
    def get_country_neighbors(self, country_code: str) -> List[Dict]:
        """
        Get neighboring countries with their details.
        
        Args:
            country_code: Two-letter country code
            
        Returns:
            List of neighbor country data
        """
        try:
            country_info = self.world_masterdata.get("countries", {}).get(country_code.upper(), {})
            neighbor_codes = country_info.get("neighbors", [])
            
            neighbors = []
            for neighbor_code in neighbor_codes:
                neighbor_data = self.get_country_world_data(neighbor_code)
                if neighbor_data:
                    neighbors.append(neighbor_data)
            
            return neighbors
            
        except Exception as e:
            print(f"Error getting neighbors for {country_code}: {e}")
            return []
    
    def calculate_geographic_distance(self, country_code1: str, country_code2: str) -> float:
        """
        Calculate approximate geographic distance between two countries.
        Uses simple Euclidean distance on lat/lng coordinates.
        
        Args:
            country_code1: First country code
            country_code2: Second country code
            
        Returns:
            Distance as float (0.0 if calculation fails)
        """
        try:
            country1_data = self.world_masterdata.get("countries", {}).get(country_code1.upper(), {})
            country2_data = self.world_masterdata.get("countries", {}).get(country_code2.upper(), {})
            
            coords1 = country1_data.get("coordinates", {})
            coords2 = country2_data.get("coordinates", {})
            
            if not (coords1 and coords2):
                return 0.0
            
            lat1, lng1 = coords1.get("lat", 0), coords1.get("lng", 0)
            lat2, lng2 = coords2.get("lat", 0), coords2.get("lng", 0)
            
            # Simple Euclidean distance (not perfect for spherical coordinates, but sufficient for clustering)
            distance = ((lat2 - lat1) ** 2 + (lng2 - lng1) ** 2) ** 0.5
            return float(distance)
            
        except Exception as e:
            print(f"Error calculating distance between {country_code1} and {country_code2}: {e}")
            return 0.0
    
    def find_similar_countries_by_geography(self, country_code: str, max_distance: float = 10.0) -> List[Tuple[str, float]]:
        """
        Find countries geographically similar to the given country.
        
        Args:
            country_code: Target country code
            max_distance: Maximum distance threshold
            
        Returns:
            List of (country_code, distance) tuples, sorted by distance
        """
        try:
            similar_countries = []
            all_countries = self.world_masterdata.get("countries", {}).keys()
            
            for other_country in all_countries:
                if other_country != country_code.upper():
                    distance = self.calculate_geographic_distance(country_code, other_country)
                    if 0 < distance <= max_distance:
                        similar_countries.append((other_country, distance))
            
            # Sort by distance
            similar_countries.sort(key=lambda x: x[1])
            return similar_countries
            
        except Exception as e:
            print(f"Error finding similar countries for {country_code}: {e}")
            return []
    
    def get_cultural_cluster_countries(self, cultural_tags: List[str]) -> List[str]:
        """
        Find countries that share similar cultural tags.
        
        Args:
            cultural_tags: List of cultural tags to match
            
        Returns:
            List of country codes with matching cultural characteristics
        """
        try:
            matching_countries = []
            all_countries = self.world_masterdata.get("countries", {})
            
            for country_code, country_info in all_countries.items():
                country_tags = set(country_info.get("culturalTags", []))
                overlap = len(set(cultural_tags) & country_tags)
                
                if overlap > 0:
                    matching_countries.append((country_code, overlap))
            
            # Sort by number of matching tags
            matching_countries.sort(key=lambda x: x[1], reverse=True)
            return [country_code for country_code, _ in matching_countries]
            
        except Exception as e:
            print(f"Error finding cultural cluster countries: {e}")
            return []
    
    def build_country_relationship_matrix(self, country_codes: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Build a relationship matrix between countries based on geographic and cultural similarity.
        
        Args:
            country_codes: List of country codes to analyze
            
        Returns:
            Dictionary of country relationships
        """
        try:
            relationship_matrix = {}
            
            for country1 in country_codes:
                relationship_matrix[country1] = {}
                
                for country2 in country_codes:
                    if country1 == country2:
                        relationship_matrix[country1][country2] = 1.0
                    else:
                        # Calculate combined similarity score
                        geo_distance = self.calculate_geographic_distance(country1, country2)
                        geo_similarity = max(0.0, 1.0 - (geo_distance / 50.0))  # Normalize to 0-1
                        
                        # Cultural similarity
                        country1_data = self.world_masterdata.get("countries", {}).get(country1, {})
                        country2_data = self.world_masterdata.get("countries", {}).get(country2, {})
                        
                        tags1 = set(country1_data.get("culturalTags", []))
                        tags2 = set(country2_data.get("culturalTags", []))
                        
                        if tags1 and tags2:
                            cultural_similarity = len(tags1 & tags2) / len(tags1 | tags2)
                        else:
                            cultural_similarity = 0.0
                        
                        # Combine similarities (weighted)
                        combined_similarity = (geo_similarity * 0.6) + (cultural_similarity * 0.4)
                        relationship_matrix[country1][country2] = float(combined_similarity)
            
            return relationship_matrix
            
        except Exception as e:
            print(f"Error building relationship matrix: {e}")
            return {}
    
    def get_continent_statistics(self) -> Dict[str, Dict]:
        """Get comprehensive statistics for all continents."""
        try:
            continent_stats = {}
            continents = self.world_masterdata.get("continents", {})
            
            for continent, continent_info in continents.items():
                country_codes = continent_info.get("countries", [])
                
                continent_stats[continent] = {
                    "name": continent,
                    "color": continent_info.get("color", "#9CA3AF"),
                    "totalCountries": len(country_codes),
                    "availableCountries": [code for code in country_codes if code in self.country_mapping],
                    "culturalCharacteristics": continent_info.get("culturalCharacteristics", [])
                }
            
            return continent_stats
            
        except Exception as e:
            print(f"Error getting continent statistics: {e}")
            return {}
    
    def analyze_country_clustering_patterns(self, country_image_mapping: Dict[str, List[str]]) -> Dict:
        """
        Analyze clustering patterns across countries.
        
        Args:
            country_image_mapping: Mapping of country codes to lists of image colors
            
        Returns:
            Analysis of geographic vs visual clustering patterns
        """
        try:
            analysis = {
                "geographic_clusters": {},
                "visual_similarity_matrix": {},
                "continent_coherence": {},
                "anomalies": []
            }
            
            # Group by continent
            continent_groups = defaultdict(list)
            for country_code in country_image_mapping.keys():
                country_data = self.world_masterdata.get("countries", {}).get(country_code, {})
                continent = country_data.get("continent", "Unknown")
                if continent != "Unknown":
                    continent_groups[continent].append(country_code)
            
            # Analyze continent coherence
            for continent, countries in continent_groups.items():
                if len(countries) >= 2:
                    # Calculate visual similarity within continent
                    similarities = []
                    for i, country1 in enumerate(countries):
                        for country2 in countries[i+1:]:
                            # This would use actual color similarity calculation
                            # For now, use geographic similarity as placeholder
                            similarity = 1.0 - self.calculate_geographic_distance(country1, country2) / 50.0
                            similarities.append(max(0.0, similarity))
                    
                    analysis["continent_coherence"][continent] = {
                        "average_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
                        "country_count": len(countries),
                        "countries": countries
                    }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing clustering patterns: {e}")
            return {}
    
    def get_country_cultural_profile(self, country_code: str) -> Dict:
        """
        Get detailed cultural profile for a country.
        
        Args:
            country_code: Two-letter country code
            
        Returns:
            Cultural profile with tags, characteristics, and relationships
        """
        try:
            country_data = self.world_masterdata.get("countries", {}).get(country_code.upper(), {})
            if not country_data:
                return {}
            
            # Get cultural neighbors (countries with similar tags)
            cultural_tags = country_data.get("culturalTags", [])
            cultural_neighbors = self.get_cultural_cluster_countries(cultural_tags)
            
            # Remove self from neighbors
            cultural_neighbors = [c for c in cultural_neighbors if c != country_code.upper()]
            
            # Get geographic neighbors
            neighbors = country_data.get("neighbors", [])
            
            return {
                "countryCode": country_code.upper(),
                "fullName": country_data.get("fullName", ""),
                "culturalTags": cultural_tags,
                "geographicFeatures": country_data.get("geographicFeatures", []),
                "culturalNeighbors": cultural_neighbors[:5],  # Top 5 culturally similar
                "geographicNeighbors": neighbors,
                "continent": country_data.get("continent", ""),
                "region": country_data.get("region", ""),
                "culturalInfluences": self._analyze_cultural_influences(country_code)
            }
            
        except Exception as e:
            print(f"Error getting cultural profile for {country_code}: {e}")
            return {}
    
    def _analyze_cultural_influences(self, country_code: str) -> Dict:
        """Analyze cultural influences for a country based on neighbors and tags."""
        try:
            country_data = self.world_masterdata.get("countries", {}).get(country_code.upper(), {})
            if not country_data:
                return {}
            
            # Analyze neighbor influence
            neighbors = country_data.get("neighbors", [])
            neighbor_tags = []
            
            for neighbor_code in neighbors:
                neighbor_data = self.world_masterdata.get("countries", {}).get(neighbor_code, {})
                neighbor_tags.extend(neighbor_data.get("culturalTags", []))
            
            # Count tag frequencies among neighbors
            tag_influence = Counter(neighbor_tags)
            
            # Get own tags
            own_tags = set(country_data.get("culturalTags", []))
            
            # Calculate shared vs unique cultural characteristics
            shared_influences = {tag: count for tag, count in tag_influence.items() if tag in own_tags}
            regional_patterns = {tag: count for tag, count in tag_influence.items() if tag not in own_tags}
            
            return {
                "sharedWithNeighbors": shared_influences,
                "regionalPatterns": regional_patterns,
                "uniqueCharacteristics": list(own_tags - set(neighbor_tags)),
                "culturalDiversity": len(own_tags) / max(len(set(neighbor_tags)), 1)
            }
            
        except Exception as e:
            print(f"Error analyzing cultural influences for {country_code}: {e}")
            return {}
    
    def get_regional_analysis(self, region: str) -> Dict:
        """
        Get comprehensive analysis for a specific region.
        
        Args:
            region: Region name (e.g., "Western Europe", "Southeast Asia")
            
        Returns:
            Regional analysis with countries, patterns, and characteristics
        """
        try:
            regional_countries = []
            all_countries = self.world_masterdata.get("countries", {})
            
            # Find countries in this region
            for country_code, country_data in all_countries.items():
                if country_data.get("region", "") == region:
                    regional_countries.append(country_code)
            
            if not regional_countries:
                return {}
            
            # Analyze regional patterns
            all_cultural_tags = []
            all_geographic_features = []
            
            for country_code in regional_countries:
                country_data = all_countries[country_code]
                all_cultural_tags.extend(country_data.get("culturalTags", []))
                all_geographic_features.extend(country_data.get("geographicFeatures", []))
            
            # Calculate regional characteristics
            cultural_frequency = Counter(all_cultural_tags)
            geographic_frequency = Counter(all_geographic_features)
            
            # Build relationship matrix for region
            relationship_matrix = self.build_country_relationship_matrix(regional_countries)
            
            # Calculate regional coherence
            if len(regional_countries) > 1:
                similarities = []
                for country1 in regional_countries:
                    for country2 in regional_countries:
                        if country1 != country2 and country1 in relationship_matrix:
                            similarity = relationship_matrix[country1].get(country2, 0.0)
                            similarities.append(similarity)
                
                regional_coherence = sum(similarities) / len(similarities) if similarities else 0.0
            else:
                regional_coherence = 1.0
            
            return {
                "region": region,
                "countries": regional_countries,
                "totalCountries": len(regional_countries),
                "dominantCulturalTags": dict(cultural_frequency.most_common(5)),
                "dominantGeographicFeatures": dict(geographic_frequency.most_common(5)),
                "regionalCoherence": regional_coherence,
                "relationshipMatrix": relationship_matrix,
                "countryDetails": {
                    code: self.get_country_world_data(code) 
                    for code in regional_countries
                }
            }
            
        except Exception as e:
            print(f"Error getting regional analysis for {region}: {e}")
            return {}
    
    def find_cultural_outliers(self, country_codes: List[str]) -> List[Dict]:
        """
        Find countries that are cultural outliers within a given set.
        
        Args:
            country_codes: List of country codes to analyze
            
        Returns:
            List of outlier analysis results
        """
        try:
            outliers = []
            
            if len(country_codes) < 3:
                return outliers  # Need at least 3 countries to detect outliers
            
            # Calculate average similarity for each country with all others
            relationship_matrix = self.build_country_relationship_matrix(country_codes)
            
            for country_code in country_codes:
                if country_code in relationship_matrix:
                    similarities = []
                    for other_country in country_codes:
                        if other_country != country_code:
                            similarity = relationship_matrix[country_code].get(other_country, 0.0)
                            similarities.append(similarity)
                    
                    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                    
                    # Consider outlier if average similarity is significantly below average
                    overall_avg = sum(
                        sum(row.values()) / len(row) 
                        for row in relationship_matrix.values()
                    ) / len(relationship_matrix)
                    
                    if avg_similarity < overall_avg * 0.7:  # 30% below average
                        country_data = self.get_country_world_data(country_code)
                        outliers.append({
                            "countryCode": country_code,
                            "countryName": country_data.get("fullName", ""),
                            "averageSimilarity": avg_similarity,
                            "outlierScore": (overall_avg - avg_similarity) / overall_avg,
                            "uniqueCharacteristics": country_data.get("culturalTags", []),
                            "explanation": f"Shows low cultural similarity ({avg_similarity:.2f}) compared to group average ({overall_avg:.2f})"
                        })
            
            # Sort by outlier score (most outlying first)
            outliers.sort(key=lambda x: x["outlierScore"], reverse=True)
            return outliers
            
        except Exception as e:
            print(f"Error finding cultural outliers: {e}")
            return []

    def get_country_code(self, country_name: str) -> Optional[str]:
        """Alias for get_country_code_from_name for compatibility."""
        return self.get_country_code_from_name(country_name)

    def get_country_name(self, country_code: str) -> Optional[str]:
        """Get country name from country code."""
        return self.country_mapping.get(country_code.upper())