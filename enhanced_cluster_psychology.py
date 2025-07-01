# Enhanced Clustering Psychology Analysis Functions
"""
Functions to add psychology analysis to clustering results
To be integrated into existing clustering pipeline
"""

from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Union, Optional

def analyze_cluster_psychology_enhanced(
    clustered_results: Dict[Tuple[int, int], List[Tuple[str, int]]],
    image_colors_list: List[List[str]],
    original_indices: List[int],
    url_to_country: Dict[str, str],
    psychology_service,
    cultural_context: str = "universal",
    confidence_threshold: float = 0.5
) -> Dict:
    """
    🆕 Enhanced cluster psychology analysis that leverages existing color data.
    
    Args:
        clustered_results: Dict mapping grid positions to (url, index) pairs
        image_colors_list: List of dominant colors for each image (from existing extraction)
        original_indices: Mapping of processed images to original indices
        url_to_country: Mapping of image URLs to country names
        psychology_service: ColorPsychologyService instance
        cultural_context: Cultural context for psychology analysis
        confidence_threshold: Minimum confidence for color classification
        
    Returns:
        Dict with comprehensive cluster psychology analysis
    """
    
    if not psychology_service:
        return {"enabled": False, "error": "Psychology service not available"}
    
    try:
        print(f"🧠 Analyzing cluster psychology for {len(clustered_results)} clusters")
        
        cluster_psychology_data = {}
        all_cluster_themes = []
        country_psychology_mapping = defaultdict(list)
        
        # Analyze each cluster
        for (x, y), url_index_pairs in clustered_results.items():
            cluster_key = f"{x}_{y}"
            
            # Get colors for this cluster
            cluster_colors = []
            cluster_countries = []
            cluster_images = []
            
            for url, processed_idx in url_index_pairs:
                # Get colors for this image
                if processed_idx < len(image_colors_list):
                    image_colors = image_colors_list[processed_idx]
                    cluster_colors.extend(image_colors)
                    cluster_images.append({
                        "url": url,
                        "colors": image_colors,
                        "country": url_to_country.get(url, "Unknown")
                    })
                
                # Get country for this image
                country = url_to_country.get(url, "Unknown")
                if country != "Unknown":
                    cluster_countries.append(country)
            
            # Analyze cluster psychology
            if cluster_colors:
                cluster_analysis = _analyze_single_cluster_psychology(
                    cluster_colors=cluster_colors,
                    cluster_countries=cluster_countries,
                    cluster_images=cluster_images,
                    psychology_service=psychology_service,
                    cultural_context=cultural_context,
                    confidence_threshold=confidence_threshold
                )
                
                cluster_psychology_data[cluster_key] = cluster_analysis
                
                # Collect themes for global analysis
                themes = cluster_analysis.get("dominantPsychologyThemes", [])
                all_cluster_themes.extend(themes)
                
                # Map psychology to countries
                for country in set(cluster_countries):
                    if country != "Unknown":
                        country_psychology_mapping[country].extend(themes)
        
        # Generate cross-cluster insights
        cross_cluster_insights = _generate_cross_cluster_insights(
            cluster_psychology_data=cluster_psychology_data,
            all_themes=all_cluster_themes,
            country_psychology_mapping=dict(country_psychology_mapping)
        )
        
        # Calculate cluster psychology relationships
        cluster_relationships = _calculate_cluster_psychology_relationships(
            cluster_psychology_data
        )
        
        return {
            "enabled": True,
            "culturalContext": cultural_context,
            "clusterPsychologyProfiles": cluster_psychology_data,
            "crossClusterInsights": cross_cluster_insights,
            "clusterRelationships": cluster_relationships,
            "countryPsychologyDistribution": dict(country_psychology_mapping),
            "processingMetrics": {
                "clustersAnalyzed": len(cluster_psychology_data),
                "totalThemes": len(all_cluster_themes),
                "uniqueThemes": len(set(all_cluster_themes)),
                "countriesInvolved": len(country_psychology_mapping),
                "analysisLevel": "cluster_enhanced"
            }
        }
        
    except Exception as e:
        print(f"❌ Error in enhanced cluster psychology analysis: {e}")
        return {
            "enabled": False,
            "error": str(e),
            "culturalContext": cultural_context
        }

def _analyze_single_cluster_psychology(
    cluster_colors: List[str],
    cluster_countries: List[str],
    cluster_images: List[Dict],
    psychology_service,
    cultural_context: str,
    confidence_threshold: float
) -> Dict:
    """Analyze psychology for a single cluster."""
    
    try:
        # Count color frequency (weighted by prominence)
        color_counts = Counter(cluster_colors)
        total_colors = len(cluster_colors)
        
        # Analyze each unique color
        color_analyses = []
        weighted_themes = []
        successful_classifications = 0
        
        for color, frequency in color_counts.items():
            weight = frequency / total_colors
            
            # Get psychology for this color
            color_analysis = psychology_service.get_psychology_for_hex_color(
                hex_color=color,
                cultural_context=cultural_context,
                confidence_threshold=confidence_threshold
            )
            
            if color_analysis.get("status") == "success":
                successful_classifications += 1
                psychology_themes = color_analysis.get("psychology", [])
                
                color_analyses.append({
                    "color": color,
                    "classifiedAs": color_analysis.get("classifiedAs"),
                    "psychology": psychology_themes,
                    "confidence": color_analysis.get("confidence", 0),
                    "frequency": frequency,
                    "weight": weight
                })
                
                # Weight themes by color frequency
                for theme in psychology_themes:
                    weighted_themes.append((theme, weight))
        
        # Aggregate weighted themes
        theme_weights = defaultdict(float)
        for theme, weight in weighted_themes:
            theme_weights[theme] += weight
        
        # Sort by weighted frequency
        dominant_themes_weighted = sorted(theme_weights.items(), key=lambda x: x[1], reverse=True)
        dominant_themes = [theme for theme, weight in dominant_themes_weighted[:5]]
        
        # Analyze country coherence in cluster
        country_coherence = _analyze_cluster_country_coherence(
            cluster_countries, dominant_themes
        )
        
        # Generate cluster-specific insights
        cluster_insights = _generate_cluster_specific_insights(
            dominant_themes=dominant_themes,
            countries=cluster_countries,
            color_analyses=color_analyses,
            cultural_context=cultural_context
        )
        
        return {
            "dominantPsychologyThemes": dominant_themes,
            "themeWeights": dict(dominant_themes_weighted),
            "colorBreakdown": color_analyses,
            "countryCoherence": country_coherence,
            "clusterInsights": cluster_insights,
            "metrics": {
                "totalColors": total_colors,
                "uniqueColors": len(color_counts),
                "uniqueCountries": len(set(cluster_countries)),
                "imagesInCluster": len(cluster_images),
                "successfulClassifications": successful_classifications,
                "classificationRate": successful_classifications / len(color_counts) if color_counts else 0,
                "psychologyDiversity": len(theme_weights),
                "dominantThemeStrength": dominant_themes_weighted[0][1] if dominant_themes_weighted else 0
            }
        }
        
    except Exception as e:
        print(f"❌ Error analyzing single cluster psychology: {e}")
        return {"error": str(e), "totalColors": len(cluster_colors)}

def _analyze_cluster_country_coherence(cluster_countries: List[str], psychology_themes: List[str]) -> Dict:
    """Analyze how well countries in a cluster align psychologically."""
    
    try:
        unique_countries = list(set([c for c in cluster_countries if c != "Unknown"]))
        
        if len(unique_countries) <= 1:
            return {
                "type": "single_country" if unique_countries else "no_countries",
                "countries": unique_countries,
                "coherence_score": 1.0 if unique_countries else 0.0,
                "description": "Single country cluster" if unique_countries else "No countries identified"
            }
        
        # Multi-country cluster
        country_counts = Counter(cluster_countries)
        total_images = len(cluster_countries)
        
        # Calculate country distribution
        country_distribution = {
            country: count / total_images 
            for country, count in country_counts.items() 
            if country != "Unknown"
        }
        
        # Determine coherence type
        if len(unique_countries) == 2:
            coherence_type = "bilateral_cluster"
            description = f"Bilateral cluster: {' + '.join(unique_countries)}"
        elif len(unique_countries) <= 4:
            coherence_type = "regional_cluster"
            description = f"Regional cluster: {len(unique_countries)} countries"
        else:
            coherence_type = "diverse_cluster"
            description = f"Diverse cluster: {len(unique_countries)} countries"
        
        # Calculate coherence score (inverse of diversity)
        if len(unique_countries) > 0:
            coherence_score = 1.0 / len(unique_countries)
        else:
            coherence_score = 0.0
        
        return {
            "type": coherence_type,
            "countries": unique_countries,
            "countryDistribution": country_distribution,
            "coherence_score": coherence_score,
            "description": description,
            "psychologyAlignment": psychology_themes[:3]  # Top 3 themes for this cluster
        }
        
    except Exception as e:
        return {"error": str(e), "countries": unique_countries if 'unique_countries' in locals() else []}

def _generate_cluster_specific_insights(
    dominant_themes: List[str],
    countries: List[str],
    color_analyses: List[Dict],
    cultural_context: str
) -> List[Dict]:
    """Generate insights specific to this cluster."""
    
    insights = []
    
    try:
        unique_countries = list(set([c for c in countries if c != "Unknown"]))
        
        # 1. Strong theme coherence
        if len(dominant_themes) >= 1:
            primary_theme = dominant_themes[0]
            
            if len(color_analyses) > 2:  # Multiple colors support the theme
                insights.append({
                    "type": "strong_theme_coherence",
                    "title": f"Strong '{primary_theme}' characteristics",
                    "description": f"Multiple colors in this cluster support '{primary_theme}' psychology theme",
                    "confidence": 0.8,
                    "data": {
                        "primary_theme": primary_theme,
                        "supporting_colors": len([c for c in color_analyses if primary_theme in c.get("psychology", [])])
                    }
                })
        
        # 2. Geographic-psychology alignment
        if len(unique_countries) == 1:
            country = unique_countries[0]
            
            # Check for geographic alignment patterns
            geographic_themes = ["Earth", "Natural", "Water", "Mountains", "Coastal"]
            theme_matches = [theme for theme in dominant_themes if any(geo in theme for geo in geographic_themes)]
            
            if theme_matches:
                insights.append({
                    "type": "geographic_psychology_match",
                    "title": f"{country} shows characteristic geography-psychology alignment",
                    "description": f"Color psychology themes align with {country}'s natural characteristics",
                    "confidence": 0.75,
                    "data": {
                        "country": country,
                        "geographic_themes": theme_matches
                    }
                })
        
        # 3. Cross-cultural patterns
        elif len(unique_countries) > 1:
            # Find common cultural themes
            cultural_themes = ["Historic", "Traditional", "Modern", "Industrial"]
            matching_themes = [theme for theme in dominant_themes if any(cult in theme for cult in cultural_themes)]
            
            if matching_themes:
                insights.append({
                    "type": "cross_cultural_pattern",
                    "title": "Cross-cultural psychology alignment",
                    "description": f"Countries {', '.join(unique_countries)} share similar cultural psychology patterns",
                    "confidence": 0.7,
                    "data": {
                        "countries": unique_countries,
                        "shared_themes": matching_themes
                    }
                })
        
        # 4. Color harmony insights
        if len(color_analyses) > 1:
            color_families = {}
            for analysis in color_analyses:
                classified_as = analysis.get("classifiedAs", "unknown")
                if classified_as != "unknown":
                    if classified_as not in color_families:
                        color_families[classified_as] = 0
                    color_families[classified_as] += analysis.get("weight", 0)
            
            if len(color_families) >= 2:
                dominant_family = max(color_families.items(), key=lambda x: x[1])
                insights.append({
                    "type": "color_harmony_pattern",
                    "title": f"Color harmony: {dominant_family[0]}-dominant palette",
                    "description": f"Cluster shows harmonious color relationships with {dominant_family[0]} tones",
                    "confidence": 0.65,
                    "data": {
                        "color_families": color_families,
                        "dominant_family": dominant_family[0]
                    }
                })
        
        return insights
        
    except Exception as e:
        print(f"❌ Error generating cluster insights: {e}")
        return []

def _generate_cross_cluster_insights(
    cluster_psychology_data: Dict[str, Dict],
    all_themes: List[str],
    country_psychology_mapping: Dict[str, List[str]]
) -> List[Dict]:
    """Generate insights across all clusters."""
    
    insights = []
    
    try:
        # 1. Most common theme across clusters
        if all_themes:
            theme_counts = Counter(all_themes)
            most_common_theme, frequency = theme_counts.most_common(1)[0]
            
            # Count how many clusters have this theme
            clusters_with_theme = 0
            for cluster_data in cluster_psychology_data.values():
                if most_common_theme in cluster_data.get("dominantPsychologyThemes", []):
                    clusters_with_theme += 1
            
            insights.append({
                "type": "global_theme_dominance",
                "title": f"'{most_common_theme}' dominates across clusters",
                "description": f"The '{most_common_theme}' theme appears in {clusters_with_theme} clusters",
                "confidence": 0.8,
                "data": {
                    "theme": most_common_theme,
                    "frequency": frequency,
                    "clusters_affected": clusters_with_theme,
                    "percentage": clusters_with_theme / len(cluster_psychology_data) if cluster_psychology_data else 0
                }
            })
        
        # 2. Country psychology consistency
        consistent_countries = []
        for country, themes in country_psychology_mapping.items():
            if len(themes) > 1:
                theme_counts = Counter(themes)
                # Check if one theme dominates for this country across clusters
                most_common = theme_counts.most_common(1)[0]
                if most_common[1] / len(themes) > 0.6:  # 60% consistency
                    consistent_countries.append({
                        "country": country,
                        "dominant_theme": most_common[0],
                        "consistency": most_common[1] / len(themes)
                    })
        
        if consistent_countries:
            insights.append({
                "type": "country_psychology_consistency",
                "title": "Countries show consistent psychology patterns",
                "description": f"{len(consistent_countries)} countries maintain consistent themes across multiple clusters",
                "confidence": 0.75,
                "data": {
                    "consistent_countries": consistent_countries,
                    "count": len(consistent_countries)
                }
            })
        
        # 3. Cluster psychology diversity
        cluster_theme_diversity = []
        for cluster_key, cluster_data in cluster_psychology_data.items():
            themes = cluster_data.get("dominantPsychologyThemes", [])
            diversity_score = len(themes) / 5.0  # Normalized by max themes (5)
            cluster_theme_diversity.append((cluster_key, diversity_score, themes))
        
        if cluster_theme_diversity:
            avg_diversity = sum(score for _, score, _ in cluster_theme_diversity) / len(cluster_theme_diversity)
            
            if avg_diversity > 0.6:
                insights.append({
                    "type": "high_psychology_diversity",
                    "title": "High psychology theme diversity across clusters",
                    "description": f"Clusters show diverse psychology patterns (avg diversity: {avg_diversity:.2f})",
                    "confidence": 0.7,
                    "data": {
                        "average_diversity": avg_diversity,
                        "clusters_analyzed": len(cluster_theme_diversity)
                    }
                })
            elif avg_diversity < 0.3:
                insights.append({
                    "type": "low_psychology_diversity",
                    "title": "Focused psychology themes across clusters",
                    "description": f"Clusters show focused psychology patterns (avg diversity: {avg_diversity:.2f})",
                    "confidence": 0.7,
                    "data": {
                        "average_diversity": avg_diversity,
                        "clusters_analyzed": len(cluster_theme_diversity)
                    }
                })
        
        return insights
        
    except Exception as e:
        print(f"❌ Error generating cross-cluster insights: {e}")
        return []

def _calculate_cluster_psychology_relationships(cluster_psychology_data: Dict[str, Dict]) -> Dict:
    """Calculate relationships between clusters based on psychology similarity."""
    
    try:
        relationships = {}
        cluster_keys = list(cluster_psychology_data.keys())
        
        for i, cluster1_key in enumerate(cluster_keys):
            cluster1_themes = set(cluster_psychology_data[cluster1_key].get("dominantPsychologyThemes", []))
            relationships[cluster1_key] = {}
            
            for j, cluster2_key in enumerate(cluster_keys):
                if i != j:
                    cluster2_themes = set(cluster_psychology_data[cluster2_key].get("dominantPsychologyThemes", []))
                    
                    # Calculate similarity
                    if cluster1_themes and cluster2_themes:
                        intersection = cluster1_themes & cluster2_themes
                        union = cluster1_themes | cluster2_themes
                        similarity = len(intersection) / len(union) if union else 0
                    else:
                        similarity = 0
                    
                    relationships[cluster1_key][cluster2_key] = {
                        "similarity": similarity,
                        "shared_themes": list(intersection) if 'intersection' in locals() else [],
                        "relationship_strength": "high" if similarity > 0.5 else "medium" if similarity > 0.2 else "low"
                    }
        
        return relationships
        
    except Exception as e:
        print(f"❌ Error calculating cluster relationships: {e}")
        return {}

def build_country_to_colors_mapping(
    url_to_country: Dict[str, str],
    image_colors_list: List[List[str]],
    payload_image_urls: List[str],
    original_indices: List[int]
) -> Dict[str, List[str]]:
    """
    Build mapping of countries to their dominant colors using existing extracted data.
    
    Args:
        url_to_country: Mapping of image URLs to country names
        image_colors_list: List of extracted colors for each processed image
        payload_image_urls: Original list of image URLs from request
        original_indices: Mapping of processed images to original indices
        
    Returns:
        Dict mapping country names to lists of their dominant colors
    """
    try:
        country_to_colors = defaultdict(list)
        
        # Map processed images back to original URLs and extract colors by country
        for processed_idx, original_idx in enumerate(original_indices):
            if processed_idx < len(image_colors_list) and original_idx < len(payload_image_urls):
                image_url = payload_image_urls[original_idx]
                image_colors = image_colors_list[processed_idx]
                country = url_to_country.get(image_url, "Unknown")
                
                if country != "Unknown" and not country.startswith("Unknown"):
                    country_to_colors[country].extend(image_colors)
        
        print(f"✅ Built country-to-colors mapping for {len(country_to_colors)} countries")
        return dict(country_to_colors)
        
    except Exception as e:
        print(f"❌ Error building country-to-colors mapping: {e}")
        return {}