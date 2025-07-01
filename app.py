import sys
import os
from pathlib import Path
from PIL import Image
import json
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import base64
from PIL import Image
from io import BytesIO

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import base64
from io import BytesIO
from collections import Counter
from sklearn.cluster import KMeans

from enhanced_cluster_psychology import (
    analyze_cluster_psychology_enhanced, 
    build_country_to_colors_mapping
)

# Enhanced imports with psychology integration
try:
    from services.color_extraction import extract_dominant_colors_from_image, extract_colors_with_psychology
    from services.color_psychology_service import ColorPsychologyService
    COLOR_PSYCHOLOGY_AVAILABLE = True
except ImportError:
    print("⚠️ Color psychology service not available")
    COLOR_PSYCHOLOGY_AVAILABLE = False

# Handle both relative and absolute imports
try:
    from .utils.image_utils import data_uri_to_pil_image, pil_image_to_data_uri, hex_to_rgb, rgb_to_hsv, rgb_to_lab, rgb_to_hex, lab_to_rgb
    from .services.image_clustering import (
        get_image_features, cluster_image_features_som, 
        analyze_cluster_quality, get_cluster_representative_features,
        cluster_image_features_hybrid_pipeline
    )
    from .services.country_service import CountryService
    from .services.world_map_service import WorldMapService
except ImportError:
    from utils.image_utils import data_uri_to_pil_image, pil_image_to_data_uri, hex_to_rgb, rgb_to_hsv, rgb_to_lab, rgb_to_hex, lab_to_rgb
    from services.image_clustering import (
        get_image_features, cluster_image_features_som, 
        analyze_cluster_quality, get_cluster_representative_features,
        cluster_image_features_hybrid_pipeline
    )
    from services.country_service import CountryService
    from services.world_map_service import WorldMapService

app = FastAPI(title="Enhanced Image Analysis API with Color Psychology & Geographic World Map")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
country_service = CountryService()
# Initialize psychology service with comprehensive error handling
psychology_service = None
psychology_available = False

try:
    if COLOR_PSYCHOLOGY_AVAILABLE:
        from services.color_psychology_service import ColorPsychologyService
        psychology_service = ColorPsychologyService()
        psychology_available = True
        print("✅ Color psychology service initialized successfully")
    else:
        print("⚠️ COLOR_PSYCHOLOGY_AVAILABLE is False")
except ImportError as e:
    print(f"⚠️ Cannot import ColorPsychologyService: {e}")
    psychology_service = None
    psychology_available = False
except Exception as e:
    print(f"⚠️ Failed to initialize psychology service: {e}")
    psychology_service = None  
    psychology_available = False

# Initialize world map service with safe psychology integration
try:
    if psychology_service is not None and psychology_available:
        world_map_service = WorldMapService(country_service, psychology_service)
        print("🗺️ WorldMapService initialized WITH psychology integration")
    else:
        world_map_service = WorldMapService(country_service)
        print("🗺️ WorldMapService initialized WITHOUT psychology integration")
except Exception as e:
    print(f"❌ Failed to initialize WorldMapService: {e}")
    # Fallback initialization without psychology
    world_map_service = WorldMapService(country_service)
    print("🗺️ WorldMapService initialized with fallback (no psychology)")

# --- Enhanced Pydantic Models ---

class ExtractColorsInput(BaseModel):
    """Enhanced input model for color extraction with optional psychology"""
    photoDataUri: str = Field(..., description="Image data URI")
    numberOfColors: int = Field(default=5, ge=1, le=8, description="Number of colors to extract")
    extractionStrategy: str = Field(default="enhanced_kmeans", description="Color extraction strategy")
    strategyOption: Optional[str] = Field(default=None, description="Strategy-specific option")
    colorSpace: str = Field(default="rgb", description="Color space for extraction")
    
    # 🆕 Psychology enhancement parameters (all optional for backward compatibility)
    includePsychology: bool = Field(default=False, description="Include color psychology analysis")
    culturalContext: str = Field(default="universal", description="Cultural context for psychology")
    psychologyConfidenceThreshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence for color classification")
    returnDetailed: bool = Field(default=False, description="Return detailed response with metrics")

class ImageMetadata(BaseModel):
    """Metadata for an image including country information."""
    dataUri: str = Field(..., description="Data URI of the image")
    filename: Optional[str] = Field(default=None, description="Original filename")
    countryCode: Optional[str] = Field(default=None, description="Two-letter country code")
    countryName: Optional[str] = Field(default=None, description="Full country name")

class ClusterImagesInput(BaseModel):
    """🆕 ENHANCED: Clustering input with optional psychology parameters"""
    imageUrls: List[str] = Field(..., description="Array of postcard image data URIs to cluster.")
    gridSize: int = Field(default=5, ge=2, le=15, description="The size of the SOM grid.")
    clusteringStrategy: str = Field(default="enhanced_features", description="Strategy for feature extraction in clustering.")
    colorSpace: str = Field(default="lab", description="Color space for clustering (rgb, hsv, lab).")
    numberOfColors: int = Field(default=5, ge=1, le=8, description="Number of dominant colors to extract per image")
    clusteringMode: str = Field(default="som", description="Clustering algorithm mode: 'som', 'kmeans', or 'hybrid'.")

    # 🆕 NEW: Optional metadata for improved country detection
    imageMetadata: Optional[List[ImageMetadata]] = Field(default=None, description="Optional metadata for images including country info")
    
    # 🆕 NEW: Psychology integration parameters (Phase 1)
    includePsychology: bool = Field(default=False, description="Include color psychology analysis")
    culturalContext: str = Field(default="universal", description="Cultural context for psychology analysis")
    psychologyConfidenceThreshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence for psychology analysis")

    # Clustering mode parameters
  
    numClusters: int = Field(default=0, ge=0, le=50, description="Number of K-means clusters (0 = auto-select)")
    kSelectionMethod: str = Field(default="auto", description="K selection method")
    
    # Metadata support
    imageMetadata: Optional[List[ImageMetadata]] = Field(
        default=None, 
        description="Optional metadata for each image including country information"
    )
    
    # 🆕 NEW: Psychology Integration Parameters (All Optional - Backward Compatible)
    psychologyConfidenceThreshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence for color classification")
    includeColorCulture: bool = Field(default=False, description="Include color-culture associations")
    culturalInsights: bool = Field(default=False, description="Generate cultural pattern insights")

class ClusterGridPosition(BaseModel):
    x: int
    y: int

class ClusterOutputItem(BaseModel):
    gridPosition: ClusterGridPosition
    imageUrls: List[str] 
    dominantColors: List[str] = Field(default=[], description="Representative colors for this cluster")
    clusterMetrics: Dict[str, Union[str, float, int]] = Field(default={}, description="Cluster quality metrics")
    countries: List[str] = Field(default=[], description="Countries represented in this cluster")
    countryMapping: Dict[str, str] = Field(default={}, description="Mapping of image URLs to country names")
    
    # 🆕 NEW: Optional psychology data per cluster (only included when requested)
    psychology: Optional[Dict] = Field(default=None, description="Psychology analysis for this cluster")

class WorldMapSummary(BaseModel):
    availableCountries: List[str] = Field(default=[], description="Countries with clustering data")
    countryColors: Dict[str, Dict] = Field(default={}, description="Dominant colors per country")
    continentDistribution: Dict[str, int] = Field(default={}, description="Number of countries per continent")
    countryImageCounts: Dict[str, int] = Field(default={}, description="Number of images per country")
    totalCountriesWithData: int = Field(default=0, description="Total countries with data")
    lastUpdated: str = Field(default="", description="Last update timestamp")
    hasWorldMapData: bool = Field(default=False, description="Whether world map data is available")

class ClusterImagesOutput(BaseModel):
    """🆕 ENHANCED: Clustering output with optional psychology data"""
    clusters: List[ClusterOutputItem]
    gridSize: int
    totalImages: int
    clusteringMetrics: Dict[str, Union[str, float, int]] = Field(default={}, description="Overall clustering quality metrics")
    countryStatistics: Dict[str, int] = Field(default={}, description="Statistics about countries in the dataset")
    totalCountries: int = Field(default=0, description="Number of unique countries detected")
    worldMapSummary: WorldMapSummary = Field(default_factory=WorldMapSummary, description="Lightweight world map data")
    
    # 🆕 NEW: Optional psychology analysis (only included when requested)
    psychologyAnalysis: Optional[Dict] = Field(default=None, description="Psychology analysis results")

class ExtractColorsOutput(BaseModel):
    """Enhanced output model with optional psychology data"""
    colors: List[str]
    extraction_metrics: Optional[Dict] = None
    psychology: Optional[Dict] = None

class BackendImageInfo(BaseModel):
    id: str
    dataUri: str
    previewUrl: str
    countryCode: Optional[str] = Field(default=None, description="Two-letter country code")
    countryName: Optional[str] = Field(default=None, description="Full country name")
    filename: Optional[str] = Field(default=None, description="Original filename")

# --- Psycho-Asscociation Functions --- #
# 🆕 NEW: Enhanced country statistics endpoint with psychology
@app.get("/country-statistics-enhanced")
async def get_enhanced_country_statistics():
    """
    🆕 NEW: Enhanced country statistics with psychology analysis.
    Returns basic stats + psychology insights if image data is available.
    """
    try:
        backend_dir = Path(__file__).parent
        data_folder = backend_dir / "data"
        
        if not data_folder.exists():
            return {"error": "Data folder not found"}
        
        # Get basic country statistics
        countries = []
        image_colors_by_country = defaultdict(list)
        supported_extensions = ['.png', '.jpg', '.jpeg', '.webp']
        total_files = 0
        
        # Process backend images to get countries and colors
        for item in data_folder.iterdir():
            if item.is_file() and item.suffix.lower() in supported_extensions:
                total_files += 1
                country_name = country_service.extract_country_from_filename(item.name)
                
                if country_name and country_name != "Unknown" and not country_name.startswith("Unknown"):
                    countries.append(country_name)
                    
                    # Extract colors for this image
                    try:
                        with Image.open(item) as img:
                            # Extract dominant colors using existing function
                            colors = extract_dominant_colors_from_image(
                                image=img,
                                num_colors=5,
                                strategy="enhanced_kmeans"
                            )
                            image_colors_by_country[country_name].extend(colors)
                    except Exception as e:
                        print(f"Error extracting colors from {item.name}: {e}")
        
        # Get enhanced statistics with psychology
        enhanced_stats = country_service.get_enhanced_country_statistics_with_psychology(
            countries=countries,
            country_to_colors=dict(image_colors_by_country),
            cultural_context="universal"
        )
        
        return enhanced_stats
        
    except Exception as e:
        print(f"Error generating enhanced country statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# 🆕 NEW: Individual country psychology endpoint
@app.get("/countries/{country_code}/psychology")
async def get_country_psychology_profile(country_code: str, cultural_context: str = "universal"):
    """
    🆕 NEW: Get detailed psychology profile for a specific country.
    
    Args:
        country_code: Two-letter country code (e.g., 'DE', 'FR')
        cultural_context: Cultural context for analysis ('universal', 'english', 'chinese')
    """
    try:
        if not country_service.validate_country_code(country_code):
            raise HTTPException(status_code=404, detail=f"Country code '{country_code}' not found")
        
        # Get psychology profile
        psychology_profile = country_service.get_country_psychology_profile(
            country_code=country_code.upper(),
            cultural_context=cultural_context
        )
        
        # Get basic country data
        country_data = country_service.get_country_world_data(country_code.upper())
        
        # Combine data
        result = {
            **country_data,
            "psychologyProfile": psychology_profile,
            "culturalContext": cultural_context,
            "analysisTimestamp": datetime.now().isoformat()
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting psychology profile for {country_code}: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# 🆕 NEW: Country similarity grouping endpoint
@app.get("/countries/psychology-groups")
async def get_country_psychology_groups(cultural_context: str = "universal", similarity_threshold: int = 2):
    """
    🆕 NEW: Get countries grouped by psychology similarity.
    
    Args:
        cultural_context: Cultural context for analysis
        similarity_threshold: Minimum shared psychology themes for grouping
    """
    try:
        # This would typically require existing analysis data
        # For now, return a message indicating analysis is needed
        return {
            "message": "Country psychology grouping requires clustering analysis first",
            "instructions": [
                "1. Run clustering analysis with psychology enabled",
                "2. Psychology groups will be included in clustering response",
                "3. Or use /country-statistics-enhanced for backend image analysis"
            ],
            "culturalContext": cultural_context,
            "similarityThreshold": similarity_threshold
        }
        
    except Exception as e:
        print(f"Error getting country psychology groups: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")




# --- Enhanced API Endpoints ---

@app.get("/psychology-capabilities")
async def get_psychology_capabilities():
    """
    🆕 NEW: Get psychology service capabilities for frontend discovery.
    This helps your frontend determine what psychology features are available.
    """
    if not COLOR_PSYCHOLOGY_AVAILABLE:
        return {
            "available": False,
            "supportedCultures": [],
            "availableColors": [],
            "features": ["Service not available"]
        }
    
    try:
        summary = psychology_service.get_psychology_summary()
        
        return {
            "available": True,
            "supportedCultures": psychology_service.get_cultural_contexts(),
            "availableColors": psychology_service.get_available_colors(),
            "totalColors": summary.get("totalColors", 0),
            "colorMatchingEnabled": summary.get("colorMatchingEnabled", False),
            "version": summary.get("metadata", {}).get("version", "1.0"),
            "features": [
                "Color classification from hex",
                "Cultural color psychology",
                "Palette analysis", 
                "Color-culture associations",
                "Cultural pattern detection",
                "Cluster psychology analysis"
            ]
        }
    except Exception as e:
        return {
            "available": True,
            "error": str(e),
            "features": [f"Error: {str(e)}"]
        }

@app.post("/extract-dominant-colors", response_model=Union[List[str], ExtractColorsOutput])
async def extract_colors_enhanced(payload: ExtractColorsInput):
    """
    Enhanced color extraction endpoint with optional psychology analysis.
    
    BACKWARD COMPATIBLE: Returns List[str] by default, Dict when psychology enabled.
    """
    try:
        image = data_uri_to_pil_image(payload.photoDataUri)
        
        result = extract_dominant_colors_from_image(
            image=image,
            num_colors=payload.numberOfColors,
            strategy=payload.extractionStrategy,
            strategy_option=payload.strategyOption,
            color_space=payload.colorSpace,
            include_psychology=payload.includePsychology,
            cultural_context=payload.culturalContext,
            psychology_confidence_threshold=payload.psychologyConfidenceThreshold,
            return_detailed=payload.returnDetailed or payload.includePsychology
        )
        
        # BACKWARD COMPATIBILITY: Return List[str] if no psychology requested
        if not payload.includePsychology and not payload.returnDetailed:
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return result.get("colors", [])
        
        # Return detailed response for psychology or detailed requests
        if isinstance(result, dict):
            return ExtractColorsOutput(**result)
        else:
            return ExtractColorsOutput(
                colors=result,
                extraction_metrics={
                    "strategy": payload.extractionStrategy,
                    "num_colors_extracted": len(result)
                }
            )
            
    except Exception as e:
        print(f"❌ Error in enhanced color extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Color extraction failed: {str(e)}")


### PA ###

async def smart_country_detection_for_clustering(image_urls: List[str]) -> Dict[str, str]:
    """
    Smart country detection that works for both backend and uploaded images.
    Returns mapping of data_uri -> country_name.
    """
    try:
        # Get current backend images with country info
        backend_images = await list_backend_images()
        
        # Create a set of backend data URIs for quick lookup
        backend_data_uris = {img.dataUri for img in backend_images}
        
        # Check if this clustering request is using backend images
        using_backend_images = any(url in backend_data_uris for url in image_urls)
        
        url_to_country = {}
        
        if using_backend_images:
            print("🎯 Detected backend images in clustering request - using direct country mapping")
            
            # Create mapping for backend images
            backend_mapping = {img.dataUri: img.countryName for img in backend_images 
                             if img.countryName and img.countryName != "Unknown"}
            
            for data_uri in image_urls:
                if data_uri in backend_mapping:
                    country_name = backend_mapping[data_uri]
                    print(f"✅ Backend image: {country_name}")
                else:
                    country_name = "Unknown"
                    print(f"⚠️ Backend image not found in mapping")
                
                url_to_country[data_uri] = country_name
        else:
            print("📤 Processing uploaded images - using fallback country detection")
            
            # For uploaded images, try to extract from data URI
            for data_uri in image_urls:
                country_name = country_service.extract_country_from_data_uri(data_uri)
                if not country_name:
                    country_name = "Unknown"
                url_to_country[data_uri] = country_name
        
        detected_countries = [c for c in url_to_country.values() if c != "Unknown"]
        print(f"🌍 Smart detection result: {len(detected_countries)} countries detected")
        
        return url_to_country
        
    except Exception as e:
        print(f"Error in smart country detection: {e}")
        return {url: "Unknown" for url in image_urls}



# 🆕 NEW: Enhanced clustering endpoint modification
# REPLACE the existing cluster_images_enhanced function with this enhanced version

@app.post("/cluster-postcard-images", response_model=ClusterImagesOutput)
async def cluster_images_enhanced_with_psychology(payload: ClusterImagesInput):
    """
    🆕 ENHANCED: Clustering endpoint with comprehensive psychology integration.
    
    🎯 NEW FEATURES:
    - Leverages existing color extraction for psychology analysis
    - Country psychology profiling and grouping
    - Enhanced cluster psychology insights
    - Cultural rationale for frontend display
    - Psychology-based country similarity grouping
    """
    if not payload.imageUrls:
        return ClusterImagesOutput(
            clusters=[], 
            gridSize=payload.gridSize, 
            totalImages=0,
            clusteringMetrics={},
            countryStatistics={},
            totalCountries=0,
            worldMapSummary=WorldMapSummary()
        )

    print(f"🔄 Enhanced clustering with psychology: enabled={payload.includePsychology}")

    # ===== EXISTING CLUSTERING LOGIC (unchanged) =====
    image_features_list = []
    image_colors_list = []  # 🎯 This already contains our color data!
    original_indices_of_valid_images = []
    image_objects = [] 

    for idx, data_uri in enumerate(payload.imageUrls):
        try:
            img = data_uri_to_pil_image(data_uri)
            features, colors = get_image_features(
                img, 
                payload.numberOfColors,
                strategy=payload.clusteringStrategy,
                color_space=payload.colorSpace
            )
            if not features:
                print(f"Warning: No features extracted for image at index {idx}. Skipping.")
                continue
            
            image_features_list.append(features)
            image_colors_list.append(colors)  # 🎯 Colors extracted here!
            original_indices_of_valid_images.append(idx)
            image_objects.append(img)
            
        except ValueError as e: 
            print(f"Skipping image at index {idx} due to invalid data URI: {str(e)[:100]}...")
        except Exception as e:
            print(f"Error processing image at index {idx} for feature extraction: {e}")
    
    if not image_features_list:
        raise HTTPException(status_code=400, detail="No valid images could be processed for clustering.")

    # ===== EXISTING CLUSTERING (unchanged) =====
    try:
        if payload.clusteringMode == "hybrid":
            winner_coordinates, som_metrics = cluster_image_features_hybrid_pipeline(
                image_features_list,
                grid_size=payload.gridSize,
                num_features=len(image_features_list[0]),
                num_clusters=payload.numClusters if payload.numClusters > 0 else 0,
                k_selection_method=payload.kSelectionMethod
            )
        else:
            winner_coordinates, som_metrics = cluster_image_features_som(
                image_features_list, 
                payload.gridSize,
                len(image_features_list[0]) 
            )
            
    except Exception as e:
        print(f"Error during clustering: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing clustering: {type(e).__name__}")

    # ===== EXISTING COUNTRY MAPPING (unchanged) =====
    url_to_country = await smart_country_detection_for_clustering(payload.imageUrls)
    all_countries = [country for country in url_to_country.values() 
                        if country != "Unknown" and not country.startswith("Unknown")]
    
    # ===== FIXED COUNTRY MAPPING =====
    if payload.imageMetadata and len(payload.imageMetadata) == len(payload.imageUrls):
        print("Using provided metadata for country detection")
        url_to_country = {}
        for i, metadata in enumerate(payload.imageMetadata):
            data_uri = payload.imageUrls[i]
            
            if metadata.countryName and metadata.countryName != "Unknown":
                country_name = metadata.countryName
            elif metadata.countryCode:
                country_name = country_service.country_mapping.get(metadata.countryCode.upper())
                if not country_name:
                    country_name = f"Unknown ({metadata.countryCode})"
            elif metadata.filename:
                country_name = country_service.extract_country_from_filename(metadata.filename)
            else:
                country_name = "Unknown"
            
            url_to_country[data_uri] = country_name or "Unknown"
    else:
        print("🎯 Using smart country detection for clustering")
        # Use the smart detection function - DON'T override it!
        url_to_country = await smart_country_detection_for_clustering(payload.imageUrls)

    # Extract all valid countries
    all_countries = [country for country in url_to_country.values() 
                    if country != "Unknown" and not country.startswith("Unknown")]

    # ===== EXISTING CLUSTER BUILDING (unchanged) =====
    clustered_results: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
    for i, coords_tuple in enumerate(winner_coordinates): 
        original_image_idx = original_indices_of_valid_images[i]
        original_image_url = payload.imageUrls[original_image_idx]
        
        pos_tuple = (coords_tuple[0], coords_tuple[1]) 
        if pos_tuple not in clustered_results:
            clustered_results[pos_tuple] = []
        clustered_results[pos_tuple].append((original_image_url, i))

    # 🆕 NEW: ENHANCED PSYCHOLOGY ANALYSIS using existing color data
    psychology_analysis = None
    country_psychology_groups = []
    country_to_colors = {}
    
    if payload.includePsychology and COLOR_PSYCHOLOGY_AVAILABLE:
        try:
            print("🧠 Running enhanced psychology analysis...")
            
            # 🎯 Build country-to-colors mapping using existing extracted colors
            country_to_colors = build_country_to_colors_mapping(
                url_to_country=url_to_country,
                image_colors_list=image_colors_list,
                payload_image_urls=payload.imageUrls,
                original_indices=original_indices_of_valid_images
            )
            
            # 🎯 Enhanced cluster psychology analysis
            psychology_analysis = analyze_cluster_psychology_enhanced(
                clustered_results=clustered_results,
                image_colors_list=image_colors_list,
                original_indices=original_indices_of_valid_images,
                url_to_country=url_to_country,
                psychology_service=psychology_service,
                cultural_context=payload.culturalContext,
                confidence_threshold=payload.psychologyConfidenceThreshold
            )
            
            # 🎯 Country psychology analysis and grouping
            if country_to_colors:
                country_psychology_data = country_service.analyze_country_colors_psychology(
                    country_to_colors=country_to_colors,
                    cultural_context=payload.culturalContext
                )
                
                if country_psychology_data:
                    country_psychology_groups = country_service.group_countries_by_psychology_similarity(
                        country_psychology=country_psychology_data,
                        similarity_threshold=2
                    )
                    
                    # Add country psychology to the analysis
                    if psychology_analysis and psychology_analysis.get("enabled"):
                        psychology_analysis["countryPsychologyProfiles"] = country_psychology_data
                        psychology_analysis["countryPsychologyGroups"] = country_psychology_groups
            
        except Exception as e:
            print(f"Enhanced psychology analysis failed: {e}")
            psychology_analysis = None

    # ===== CREATE OUTPUT CLUSTERS (enhanced) =====
    output_clusters = []
    for (x, y), url_index_pairs in clustered_results.items():
        urls = [url for url, _ in url_index_pairs]
        image_indices = [idx for _, idx in url_index_pairs]
        
        # Get countries for this cluster
        cluster_countries = [url_to_country.get(url, "Unknown") for url in urls]
        unique_countries = list(set([c for c in cluster_countries if c != "Unknown" and not c.startswith("Unknown")]))
        
        # Create country mapping for this cluster
        cluster_country_mapping = {url: url_to_country.get(url, "Unknown") for url in urls}
        
        # Get representative colors for cluster
        cluster_colors = [image_colors_list[idx] for idx in image_indices]
        representative_colors = _get_cluster_representative_colors_perceptual(
            cluster_colors, 
            payload.numberOfColors,
            payload.colorSpace
        )
        
        # Calculate cluster metrics
        cluster_features = [image_features_list[idx] for idx in image_indices]
        current_cluster_metrics = _calculate_cluster_metrics(cluster_features)
        
        # 🆕 NEW: Add enhanced psychology data to cluster
        cluster_psychology = None
        if psychology_analysis and psychology_analysis.get("enabled"):
            cluster_key = f"{x}_{y}"
            cluster_psychology_profiles = psychology_analysis.get("clusterPsychologyProfiles", {})
            if cluster_key in cluster_psychology_profiles:
                cluster_psychology = cluster_psychology_profiles[cluster_key]
                
                # 🎯 Add color rationale for frontend
                cluster_psychology["colorRationale"] = _build_color_rationale(
                    representative_colors=representative_colors,
                    cluster_psychology=cluster_psychology,
                    countries=unique_countries
                )
        
        output_clusters.append(
            ClusterOutputItem(
                gridPosition=ClusterGridPosition(x=x, y=y),
                imageUrls=urls,
                dominantColors=representative_colors,
                clusterMetrics=current_cluster_metrics,
                countries=unique_countries,
                countryMapping=cluster_country_mapping,
                psychology=cluster_psychology  # 🆕 Enhanced with rationale
            )
        )
    
    # ===== EXISTING METRICS CALCULATION (enhanced) =====
    country_stats = country_service.get_country_statistics(all_countries)
    cluster_quality = analyze_cluster_quality(image_features_list, winner_coordinates)
    
    overall_metrics = {
        "num_clusters": len(output_clusters),
        "average_cluster_size": len(payload.imageUrls) / max(len(output_clusters), 1),
        "som_quantization_error": som_metrics.get("quantization_error", 0.0),
        "som_topographic_error": som_metrics.get("topographic_error", 0.0),
        "som_coverage": som_metrics.get("coverage", 0.0),
        "clustering_separation_score": cluster_quality.get("separation_score", 0.0),
        "cluster_size_variance": cluster_quality.get("cluster_size_variance", 0.0),
        "clustering_mode": payload.clusteringMode,
        "number_of_colors": payload.numberOfColors,
        "feature_dimensions": payload.numberOfColors * 4 + 5,
        "used_metadata": payload.imageMetadata is not None,
        "psychology_enabled": payload.includePsychology,  # 🆕 Enhanced tracking
        "psychology_analysis_level": "cluster" if payload.includePsychology else "none",
        "countries_with_psychology": len(country_to_colors) if country_to_colors else 0,
        "psychology_groups_found": len(country_psychology_groups)
    }
    
    # Add hybrid-specific metrics if using hybrid mode
    if payload.clusteringMode == "hybrid":
        overall_metrics.update({
            "k_used": som_metrics.get("k_used", 0),
            "k_selection_method": som_metrics.get("k_selection_method", "unknown"),
            "kmeans_silhouette_score": som_metrics.get("kmeans_silhouette_score", 0.0),
            "kmeans_inertia": som_metrics.get("kmeans_inertia", 0.0)
        })
    
    # ===== ENHANCED WORLD MAP SUMMARY =====
    # ===== 🆕 ENHANCED WORLD MAP SUMMARY WITH PSYCHOLOGY =====
    try:
        world_summary = world_map_service.build_lightweight_summary(
            output_clusters, 
            url_to_country,
            image_colors_list,
            include_psychology=payload.includePsychology,  # 🆕 NEW: Psychology flag
            cultural_context=payload.culturalContext        # 🆕 NEW: Cultural context
        )
        
        success_msg = f"Generated enhanced world map summary with {world_summary.get('totalCountriesWithData', 0)} countries"
        if payload.includePsychology and world_summary.get('psychologyEnabled'):
            psychology_groups = len(world_summary.get('psychologyGroups', []))
            success_msg += f", psychology: {psychology_groups} groups"
        print(success_msg)
        
    except Exception as e:
        print(f"Error generating enhanced world map summary: {e}")
        world_summary = WorldMapSummary().dict()
    
    # 🆕 NEW: Enhanced response with psychology integration
    response = ClusterImagesOutput(
        clusters=output_clusters,
        gridSize=payload.gridSize,
        totalImages=len(payload.imageUrls),
        clusteringMetrics=overall_metrics,
        countryStatistics=country_stats,
        totalCountries=len(country_stats),
        worldMapSummary=WorldMapSummary(**world_summary),
        psychologyAnalysis=psychology_analysis  # 🆕 Comprehensive psychology data
    )
    
    success_msg = "✅ Enhanced clustering completed"
    if psychology_analysis:
        success_msg += f" with psychology analysis ({len(country_to_colors)} countries)"
    if country_psychology_groups:
        success_msg += f" and {len(country_psychology_groups)} psychology groups"
    print(success_msg)
    
    return response

# 🆕 NEW: Color rationale endpoint for frontend visualization
@app.post("/color-rationale")
async def get_color_rationale(payload: dict):
    """
    🆕 NEW: Get detailed color rationale for frontend visualization.
    
    Usage: POST {"colors": ["#9c774b", "#ff0000"], "context": "cluster"}
    """
    if not COLOR_PSYCHOLOGY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Color psychology service not available")
    
    try:
        colors = payload.get("colors", [])
        context = payload.get("context", "general")
        cultural_context = payload.get("culturalContext", "universal")
        
        if not colors:
            raise HTTPException(status_code=400, detail="No colors provided")
        
        rationale_data = []
        
        for color in colors:
            color_analysis = psychology_service.get_psychology_for_hex_color(
                hex_color=color,
                cultural_context=cultural_context,
                confidence_threshold=0.5
            )
            
            if color_analysis.get("status") == "success":
                rationale_data.append({
                    "color": color,
                    "colorName": color_analysis.get("classifiedAs"),
                    "psychology": color_analysis.get("psychology", []),
                    "confidence": color_analysis.get("confidence", 0),
                    "culturalContext": cultural_context,
                    "explanation": _generate_color_explanation(color_analysis, context)
                })
        
        return {
            "colors": rationale_data,
            "context": context,
            "culturalContext": cultural_context,
            "summary": _generate_rationale_summary(rationale_data, context)
        }
        
    except Exception as e:
        print(f"❌ Error generating color rationale: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ===== 🆕 NEW HELPER FUNCTIONS =====

def _build_color_rationale(
    representative_colors: List[str],
    cluster_psychology: Dict,
    countries: List[str]
) -> Dict:
    """Build detailed color rationale for frontend display."""
    try:
        if not COLOR_PSYCHOLOGY_AVAILABLE:
            return {"available": False}
        
        color_explanations = []
        dominant_themes = cluster_psychology.get("dominantPsychologyThemes", [])
        
        for color in representative_colors:
            color_analysis = psychology_service.get_psychology_for_hex_color(
                hex_color=color,
                cultural_context="universal",
                confidence_threshold=0.5
            )
            
            if color_analysis.get("status") == "success":
                explanation = {
                    "color": color,
                    "colorName": color_analysis.get("classifiedAs"),
                    "psychology": color_analysis.get("psychology", []),
                    "confidence": color_analysis.get("confidence", 0),
                    "reasoning": f"This {color_analysis.get('classifiedAs', 'color')} contributes {', '.join(color_analysis.get('psychology', [])[:2])} characteristics to the cluster"
                }
                color_explanations.append(explanation)
        
        return {
            "available": True,
            "colorExplanations": color_explanations,
            "clusterThemes": dominant_themes,
            "countries": countries,
            "overallReasoning": f"This cluster is characterized by {', '.join(dominant_themes[:3])} themes" + (f" from {', '.join(countries)}" if countries else ""),
            "culturalAlignment": len([country for country in countries if country in ["Germany", "France", "Italy"]]) > 0  # Example check
        }
        
    except Exception as e:
        print(f"❌ Error building color rationale: {e}")
        return {"available": False, "error": str(e)}

def _extract_global_psychology_themes(psychology_analysis: Dict) -> List[str]:
    """Extract global psychology themes from analysis."""
    try:
        cross_cluster_insights = psychology_analysis.get("crossClusterInsights", [])
        global_themes = []
        
        for insight in cross_cluster_insights:
            if insight.get("type") == "global_theme_dominance":
                theme = insight.get("data", {}).get("theme")
                if theme:
                    global_themes.append(theme)
        
        return global_themes[:5]  # Top 5 global themes
        
    except Exception as e:
        print(f"❌ Error extracting global themes: {e}")
        return []

def _generate_color_explanation(color_analysis: Dict, context: str) -> str:
    """Generate human-readable explanation for a color."""
    try:
        color_name = color_analysis.get("classifiedAs", "this color")
        psychology = color_analysis.get("psychology", [])
        confidence = color_analysis.get("confidence", 0)
        
        if not psychology:
            return f"The {color_name} in this {context} could not be psychologically classified."
        
        explanation = f"The {color_name} in this {context} evokes {', '.join(psychology[:2])}"
        if len(psychology) > 2:
            explanation += f" and {psychology[2]}"
        
        if confidence > 0.8:
            explanation += " with high confidence"
        elif confidence > 0.6:
            explanation += " with good confidence"
        
        explanation += f" (confidence: {confidence:.1%})"
        
        return explanation
        
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def _generate_rationale_summary(rationale_data: List[Dict], context: str) -> str:
    """Generate overall rationale summary."""
    try:
        if not rationale_data:
            return f"No color psychology analysis available for this {context}."
        
        all_themes = []
        for color_data in rationale_data:
            all_themes.extend(color_data.get("psychology", []))
        
        if not all_themes:
            return f"Colors in this {context} could not be psychologically classified."
        
        theme_counts = Counter(all_themes)
        dominant_theme = theme_counts.most_common(1)[0][0]
        
        return f"This {context} is psychologically characterized by {dominant_theme} themes, with {len(rationale_data)} colors contributing to the overall impression."
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# 🆕 NEW: Comparison endpoint for cultural contexts
@app.post("/compare-cultural-color-meanings")
async def compare_cultural_color_meanings(payload: dict):
    """
    🆕 NEW: Compare color meanings across cultural contexts.
    
    Usage: POST {"hexColor": "#ff0000"}
    """
    if not COLOR_PSYCHOLOGY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Color psychology service not available")
    
    try:
        hex_color = payload.get("hexColor", "")
        if not hex_color:
            raise HTTPException(status_code=400, detail="No hex color provided")
        
        # Get psychology across all cultures
        color_name, _ = psychology_service.classify_hex_color(hex_color)
        
        if not color_name:
            return {
                "hexColor": hex_color,
                "classified": False,
                "message": "Color could not be classified"
            }
        
        cultural_comparison = psychology_service.compare_cultural_psychology(color_name)
        
        return {
            "hexColor": hex_color,
            "colorName": color_name,
            "classified": True,
            **cultural_comparison
        }
        
    except Exception as e:
        print(f"❌ Error comparing cultural meanings: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# @app.post("/cluster-postcard-images", response_model=ClusterImagesOutput)
# async def cluster_images_enhanced(payload: ClusterImagesInput):
#     """
#     🆕 ENHANCED: Clustering endpoint with optional psychology analysis.
    
#     BACKWARD COMPATIBLE: Works exactly like before when psychology parameters are omitted.
    
#     New Psychology Features:
#     - includePsychology: Enable psychology analysis
#     - culturalContext: Cultural context for analysis
#     - psychologyAnalysisLevel: Detail level ('colors', 'cluster', 'full')
#     - includeColorCulture: Add color-culture associations
#     - culturalInsights: Generate cultural pattern insights
#     """
#     if not payload.imageUrls:
#         return ClusterImagesOutput(
#             clusters=[], 
#             gridSize=payload.gridSize, 
#             totalImages=0,
#             clusteringMetrics={},
#             countryStatistics={},
#             totalCountries=0,
#             worldMapSummary=WorldMapSummary()
#         )

#     print(f"🔄 Enhanced clustering: psychology={payload.includePsychology}, level={payload.psychologyAnalysisLevel}")

#     # Extract features and colors from images
#     image_features_list = []
#     image_colors_list = []
#     original_indices_of_valid_images = []
#     image_objects = [] 

#     for idx, data_uri in enumerate(payload.imageUrls):
#         try:
#             img = data_uri_to_pil_image(data_uri)
#             features, colors = get_image_features(
#                 img, 
#                 payload.numberOfColors,
#                 strategy=payload.clusteringStrategy,
#                 color_space=payload.colorSpace
#             )
#             if not features:
#                 print(f"Warning: No features extracted for image at index {idx}. Skipping.")
#                 continue
            
#             image_features_list.append(features)
#             image_colors_list.append(colors)
#             original_indices_of_valid_images.append(idx)
#             image_objects.append(img)
            
#         except ValueError as e: 
#             print(f"Skipping image at index {idx} due to invalid data URI: {str(e)[:100]}...")
#         except Exception as e:
#             print(f"Error processing image at index {idx} for feature extraction: {e}")
    
#     if not image_features_list:
#         raise HTTPException(status_code=400, detail="No valid images could be processed for clustering.")

#     # Perform clustering
#     try:
#         if payload.clusteringMode == "hybrid":
#             winner_coordinates, som_metrics = cluster_image_features_hybrid_pipeline(
#                 image_features_list,
#                 grid_size=payload.gridSize,
#                 num_features=len(image_features_list[0]),
#                 num_clusters=payload.numClusters if payload.numClusters > 0 else 0,
#                 k_selection_method=payload.kSelectionMethod
#             )
#         else:
#             winner_coordinates, som_metrics = cluster_image_features_som(
#                 image_features_list, 
#                 payload.gridSize,
#                 len(image_features_list[0]) 
#             )
            
#     except Exception as e:
#         print(f"Error during clustering: {e}")
#         raise HTTPException(status_code=500, detail=f"Error performing clustering: {type(e).__name__}")

#     # Create URL to country mapping
#     url_to_country = {}
#     all_countries = []
    
#     if payload.imageMetadata and len(payload.imageMetadata) == len(payload.imageUrls):
#         print("Using provided metadata for country detection")
#         for i, metadata in enumerate(payload.imageMetadata):
#             data_uri = payload.imageUrls[i]
            
#             if metadata.countryName and metadata.countryName != "Unknown":
#                 country_name = metadata.countryName
#             elif metadata.countryCode:
#                 country_name = country_service.country_mapping.get(metadata.countryCode.upper())
#                 if not country_name:
#                     country_name = f"Unknown ({metadata.countryCode})"
#             elif metadata.filename:
#                 country_name = country_service.extract_country_from_filename(metadata.filename)
#             else:
#                 country_name = "Unknown"
            
#             url_to_country[data_uri] = country_name or "Unknown"
#             if country_name and country_name != "Unknown" and not country_name.startswith("Unknown"):
#                 all_countries.append(country_name)
#     else:
#         print("No metadata provided, using fallback country detection")
#         for data_uri in payload.imageUrls:
#             country_name = country_service.extract_country_from_data_uri(data_uri)
#             if not country_name:
#                 country_name = "Unknown"
            
#             url_to_country[data_uri] = country_name
#             if country_name != "Unknown" and not country_name.startswith("Unknown"):
#                 all_countries.append(country_name)

#     # Build cluster results
#     clustered_results: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
#     for i, coords_tuple in enumerate(winner_coordinates): 
#         original_image_idx = original_indices_of_valid_images[i]
#         original_image_url = payload.imageUrls[original_image_idx]
        
#         pos_tuple = (coords_tuple[0], coords_tuple[1]) 
#         if pos_tuple not in clustered_results:
#             clustered_results[pos_tuple] = []
#         clustered_results[pos_tuple].append((original_image_url, i))
    
#     # 🆕 NEW: Psychology analysis
#     psychology_analysis = None
#     if payload.includePsychology and COLOR_PSYCHOLOGY_AVAILABLE:
#         try:
#             psychology_analysis = _analyze_clustering_psychology(
#                 clustered_results=clustered_results,
#                 image_colors_list=image_colors_list,
#                 original_indices=original_indices_of_valid_images,
#                 cultural_context=payload.culturalContext,
#                 analysis_level=payload.psychologyAnalysisLevel,
#                 confidence_threshold=payload.psychologyConfidenceThreshold,
#                 include_color_culture=payload.includeColorCulture,
#                 cultural_insights=payload.culturalInsights,
#                 url_to_country=url_to_country
#             )
#         except Exception as e:
#             print(f"Psychology analysis failed: {e}")
#             psychology_analysis = None

#     # Create output clusters
#     output_clusters = []
#     for (x, y), url_index_pairs in clustered_results.items():
#         urls = [url for url, _ in url_index_pairs]
#         image_indices = [idx for _, idx in url_index_pairs]
        
#         # Get countries for this cluster
#         cluster_countries = [url_to_country.get(url, "Unknown") for url in urls]
#         unique_countries = list(set([c for c in cluster_countries if c != "Unknown" and not c.startswith("Unknown")]))
        
#         # Create country mapping for this cluster
#         cluster_country_mapping = {url: url_to_country.get(url, "Unknown") for url in urls}
        
#         # Get representative colors for cluster
#         cluster_colors = [image_colors_list[idx] for idx in image_indices]
#         representative_colors = _get_cluster_representative_colors_perceptual(
#             cluster_colors, 
#             payload.numberOfColors,
#             payload.colorSpace
#         )
        
#         # Calculate cluster metrics
#         cluster_features = [image_features_list[idx] for idx in image_indices]
#         current_cluster_metrics = _calculate_cluster_metrics(cluster_features)
        
#         # 🆕 NEW: Add psychology data to cluster if available
#         cluster_psychology = None
#         if psychology_analysis and psychology_analysis.get("enabled"):
#             cluster_key = f"{x}_{y}"
#             cluster_psychology = psychology_analysis.get("clusterPsychology", {}).get(cluster_key, {})
        
#         output_clusters.append(
#             ClusterOutputItem(
#                 gridPosition=ClusterGridPosition(x=x, y=y),
#                 imageUrls=urls,
#                 dominantColors=representative_colors,
#                 clusterMetrics=current_cluster_metrics,
#                 countries=unique_countries,
#                 countryMapping=cluster_country_mapping,
#                 psychology=cluster_psychology  # 🆕 NEW: Psychology data
#             )
#         )
    
#     # Calculate country statistics and overall metrics
#     country_stats = country_service.get_country_statistics(all_countries)
#     cluster_quality = analyze_cluster_quality(image_features_list, winner_coordinates)
    
#     overall_metrics = {
#         "num_clusters": len(output_clusters),
#         "average_cluster_size": len(payload.imageUrls) / max(len(output_clusters), 1),
#         "som_quantization_error": som_metrics.get("quantization_error", 0.0),
#         "som_topographic_error": som_metrics.get("topographic_error", 0.0),
#         "som_coverage": som_metrics.get("coverage", 0.0),
#         "clustering_separation_score": cluster_quality.get("separation_score", 0.0),
#         "cluster_size_variance": cluster_quality.get("cluster_size_variance", 0.0),
#         "clustering_mode": payload.clusteringMode,
#         "number_of_colors": payload.numberOfColors,
#         "feature_dimensions": payload.numberOfColors * 4 + 5,
#         "used_metadata": payload.imageMetadata is not None,
#         "psychology_enabled": payload.includePsychology  # 🆕 NEW
#     }
    
#     # Add hybrid-specific metrics if using hybrid mode
#     if payload.clusteringMode == "hybrid":
#         overall_metrics.update({
#             "k_used": som_metrics.get("k_used", 0),
#             "k_selection_method": som_metrics.get("k_selection_method", "unknown"),
#             "k_selection_algorithm": som_metrics.get("k_selection_algorithm", "unknown"),
#             "kmeans_silhouette_score": som_metrics.get("kmeans_silhouette_score", 0.0),
#             "kmeans_inertia": som_metrics.get("kmeans_inertia", 0.0)
#         })
    
#     # Generate world map summary data
#     try:
#         world_summary = world_map_service.build_lightweight_summary(
#             output_clusters, 
#             url_to_country,
#             image_colors_list
#         )
#         print(f"Generated world map summary with {world_summary.get('totalCountriesWithData', 0)} countries")
#     except Exception as e:
#         print(f"Error generating world map summary: {e}")
#         world_summary = WorldMapSummary().dict()
    
#     # 🆕 NEW: Create response with optional psychology analysis
#     response = ClusterImagesOutput(
#         clusters=output_clusters,
#         gridSize=payload.gridSize,
#         totalImages=len(payload.imageUrls),
#         clusteringMetrics=overall_metrics,
#         countryStatistics=country_stats,
#         totalCountries=len(country_stats),
#         worldMapSummary=WorldMapSummary(**world_summary),
#         psychologyAnalysis=psychology_analysis  # 🆕 NEW: Only included when requested
#     )
    
#     print(f"✅ Enhanced clustering completed. Psychology: {'✅' if psychology_analysis else '❌'}")
#     return response

# 🆕 NEW: Quick color psychology endpoint (simplified)
@app.post("/analyze-color-psychology")
async def analyze_single_color_psychology(payload: dict):
    """
    Quick single color psychology analysis.
    Usage: POST {"hexColor": "#9c774b", "culturalContext": "universal"}
    """
    if not COLOR_PSYCHOLOGY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Color psychology service not available")
    
    try:
        hex_color = payload.get("hexColor", "")
        cultural_context = payload.get("culturalContext", "universal")
        confidence_threshold = payload.get("confidenceThreshold", 0.5)
        
        result = psychology_service.get_psychology_for_hex_color(
            hex_color=hex_color,
            cultural_context=cultural_context,
            confidence_threshold=confidence_threshold
        )
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing color psychology: {e}")
        raise HTTPException(status_code=500, detail=f"Psychology analysis failed: {str(e)}")

# Keep all existing endpoints for full backward compatibility
@app.get("/list-backend-images", response_model=List[BackendImageInfo])
async def list_backend_images():
    """List available backend images with country detection."""
    backend_dir = Path(__file__).parent
    data_folder = backend_dir / "data"
    images_info = []

    if not data_folder.exists() or not data_folder.is_dir():
        return []

    supported_extensions = ['.png', '.jpg', '.jpeg', '.webp']

    for item in data_folder.iterdir():
        if item.is_file() and item.suffix.lower() in supported_extensions:
            try:
                with Image.open(item) as img:
                    fmt = img.format if img.format else 'PNG'
                    if fmt.upper() == 'JPEG': 
                        mime_type = 'image/jpeg'
                    elif fmt.upper() == 'WEBP': 
                        mime_type = 'image/webp'
                    else: 
                        mime_type = f'image/{fmt.lower()}'
                    
                    img_data_uri = pil_image_to_data_uri(img, fmt=fmt)
                    
                    # Extract country information
                    country_name = country_service.extract_country_from_filename(item.name)
                    country_code = None
                    if country_name and country_name != "Unknown" and not country_name.startswith("Unknown"):
                        country_code = country_service.get_country_code_from_name(country_name)

                    images_info.append(
                        BackendImageInfo(
                            id=item.name,
                            dataUri=img_data_uri,
                            previewUrl=img_data_uri,
                            countryCode=country_code,
                            countryName=country_name,
                            filename=item.name
                        )
                    )
            except Exception as e:
                print(f"Error processing backend image {item.name}: {e}")

    return images_info

@app.get("/countries")
async def get_countries():
    """Get all available country codes and names."""
    return {
        "countries": country_service.get_all_countries(),
        "totalCountries": len(country_service.get_all_countries())
    }

@app.get("/country-statistics")
async def get_enhanced_country_statistics():
    """🆕 ENHANCED: Get statistics about countries with optional psychology insights."""
    backend_dir = Path(__file__).parent
    data_folder = backend_dir / "data"
    
    if not data_folder.exists():
        return {"error": "Data folder not found"}
    
    countries = []
    supported_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    total_files = 0
    
    for item in data_folder.iterdir():
        if item.is_file() and item.suffix.lower() in supported_extensions:
            total_files += 1
            country_name = country_service.extract_country_from_filename(item.name)
            if country_name and country_name != "Unknown" and not country_name.startswith("Unknown"):
                countries.append(country_name)
    
    stats = country_service.get_country_statistics(countries)
    
    # 🆕 NEW: Enhanced response with psychology readiness
    enhanced_response = {
        "countryStatistics": stats,
        "totalCountries": len(stats),
        "totalImages": total_files,
        "identifiedCountryImages": len(countries),
        "unknownImages": total_files - len(countries),
        
        # 🆕 NEW: Psychology integration status
        "psychologyIntegration": {
            "available": COLOR_PSYCHOLOGY_AVAILABLE,
            "worldMapPsychologyEnabled": world_map_service.psychology_enabled,
            "supportedCulturalContexts": ["universal", "english", "chinese"] if COLOR_PSYCHOLOGY_AVAILABLE else []
        }
    }
    
    return enhanced_response

@app.get("/countries/{country_code}/psychology")
async def get_country_psychology(
    country_code: str,
    cultural_context: str = Query(default="universal", description="Cultural context for analysis"),
    confidence_threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
):
    """
    🆕 NEW: Get detailed psychology analysis for a specific country.
    This endpoint provides the psychological profile based on the country's dominant colors.
    """
    if not COLOR_PSYCHOLOGY_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Color psychology service not available"
        )
    
    try:
        # Validate country code
        country_name = country_service.get_country_name(country_code)
        if not country_name:
            raise HTTPException(
                status_code=404, 
                detail=f"Country code '{country_code}' not found"
            )
        
        # Check if we have actual clustering data for this country
        # For now, return a structured response indicating data availability
        
        response = {
            "countryCode": country_code,
            "countryName": country_name,
            "culturalContext": cultural_context,
            "psychologyAnalysis": {
                "status": "ready",
                "message": "Psychology analysis will be available after clustering postcards for this country",
                "dataSource": "clustering_based",
                "confidence": None,
                "dominantThemes": [],
                "culturalInsights": [],
                "colorPsychology": []
            },
            "metadata": {
                "analysisTimestamp": datetime.now().isoformat(),
                "confidenceThreshold": confidence_threshold,
                "psychologyServiceAvailable": True,
                "requiresClusteringData": True
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in country psychology analysis: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error analyzing country psychology: {str(e)}"
        )

@app.get("/visualization-data/world-map-detailed")
async def get_detailed_world_map():
    """Get detailed world map data with cultural insights."""
    try:
        print("Generating detailed world map data...")
        detailed_data = world_map_service.build_complete_world_map_data()
        print(f"Generated detailed data for {len(detailed_data.get('countries', {}))} countries")
        return detailed_data
    except Exception as e:
        print(f"Error generating detailed world map data: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating world map data: {str(e)}"
        )

# --- Psychology Helper Functions ---

def _analyze_clustering_psychology(
    clustered_results: Dict,
    image_colors_list: List[List[str]],
    original_indices: List[int],
    cultural_context: str,
    analysis_level: str,
    confidence_threshold: float,
    include_color_culture: bool,
    cultural_insights: bool,
    url_to_country: Dict[str, str]
) -> Dict:
    """
    🆕 NEW: Analyze psychology for clustering results.
    This is the core psychology integration function.
    """
    try:
        print(f"🧠 Analyzing clustering psychology (level: {analysis_level}, culture: {cultural_context})")
        
        # Collect all colors from all clusters
        all_cluster_colors = []
        cluster_psychology_data = {}
        
        for (x, y), url_index_pairs in clustered_results.items():
            cluster_key = f"{x}_{y}"
            image_indices = [idx for _, idx in url_index_pairs]
            
            # Get colors for this cluster
            cluster_colors = []
            for idx in image_indices:
                if idx < len(image_colors_list):
                    cluster_colors.extend(image_colors_list[idx])
            
            all_cluster_colors.extend(cluster_colors)
            
            # Analyze psychology for this cluster if requested
            if analysis_level in ["cluster", "full"] and cluster_colors:
                cluster_psychology = _analyze_cluster_colors_psychology(
                    cluster_colors=cluster_colors,
                    cultural_context=cultural_context,
                    confidence_threshold=confidence_threshold
                )
                cluster_psychology_data[cluster_key] = cluster_psychology
        
        # Overall palette psychology
        overall_psychology = {}
        if analysis_level == "full" and all_cluster_colors:
            try:
                unique_colors = list(set(all_cluster_colors))
                overall_psychology = psychology_service.analyze_color_palette_psychology(
                    hex_colors=unique_colors,
                    cultural_context=cultural_context
                )
            except Exception as e:
                print(f"Error in overall palette analysis: {e}")
                overall_psychology = {"error": str(e)}
        
        # Cultural patterns and color-culture mapping
        cultural_patterns = []
        color_culture_mapping = {}
        
        if cultural_insights and analysis_level == "full":
            cultural_patterns = _generate_cultural_clustering_insights(
                clustered_results=clustered_results,
                url_to_country=url_to_country,
                cluster_psychology_data=cluster_psychology_data,
                cultural_context=cultural_context
            )
        
        if include_color_culture and all_cluster_colors:
            # Create color to culture associations (limit to prevent overload)
            unique_colors = list(set(all_cluster_colors))[:10]
            for color in unique_colors:
                try:
                    color_analysis = psychology_service.get_psychology_for_hex_color(
                        hex_color=color,
                        cultural_context=cultural_context,
                        confidence_threshold=confidence_threshold
                    )
                    if color_analysis.get("status") == "success":
                        color_culture_mapping[color] = {
                            "classifiedAs": color_analysis.get("classifiedAs"),
                            "psychology": color_analysis.get("psychology", []),
                            "confidence": color_analysis.get("confidence", 0.0)
                        }
                except Exception as e:
                    print(f"Error analyzing color {color}: {e}")
        # Add this line before the return statement
        psychology_patterns = _generate_psychology_patterns(cluster_psychology_data, cultural_context)
        cultural_patterns.extend(psychology_patterns)  # Merge psychology patterns

        return {
            "enabled": True,
            "culturalContext": cultural_context,
            "clusterPsychology": cluster_psychology_data,
            "overallPalettePsychology": overall_psychology,
            "culturalPatterns": cultural_patterns,
            "colorCultureMapping": color_culture_mapping,
            "processingMetrics": {
                "totalColors": len(all_cluster_colors),
                "uniqueColors": len(set(all_cluster_colors)),
                "clustersAnalyzed": len(cluster_psychology_data),
                "analysisLevel": analysis_level,
                "processingTime": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        print(f"❌ Error in clustering psychology analysis: {e}")
        return {
            "enabled": False,
            "error": str(e),
            "processingMetrics": {"error": str(e)}
        }

def _analyze_cluster_colors_psychology(
    cluster_colors: List[str],
    cultural_context: str,
    confidence_threshold: float
) -> Dict:
    """Analyze psychology for a single cluster's colors."""
    try:
        if not cluster_colors:
            return {"error": "No colors in cluster"}
        
        # Get psychology for the cluster's colors
        cluster_analysis = psychology_service.analyze_color_palette_psychology(
            hex_colors=cluster_colors,
            cultural_context=cultural_context
        )
        
        # Extract key insights
        if cluster_analysis.get("status") == "success":
            palette_metrics = cluster_analysis.get("paletteMetrics", {})
            return {
                "dominantThemes": palette_metrics.get("dominantThemes", []),
                "overallMood": cluster_analysis.get("dominantPsychologyThemes", [])[:3],  # Top 3
                "classificationRate": palette_metrics.get("classificationRate", 0.0),
                "totalColors": len(cluster_colors),
                "uniqueColors": palette_metrics.get("uniquePsychologyThemes", 0),
                "culturalContext": cultural_context
            }
        else:
            return {"error": "Analysis failed", "totalColors": len(cluster_colors)}
            
    except Exception as e:
        return {"error": str(e), "totalColors": len(cluster_colors)}

def _generate_cultural_clustering_insights(
    clustered_results: Dict,
    url_to_country: Dict[str, str],
    cluster_psychology_data: Dict,
    cultural_context: str
) -> List[Dict]:
    """Generate cultural insights from clustering patterns."""
    try:
        insights = []
        
        # Analyze geographic clustering patterns
        country_cluster_mapping = {}
        for (x, y), url_index_pairs in clustered_results.items():
            urls = [url for url, _ in url_index_pairs]
            countries = [url_to_country.get(url, "Unknown") for url in urls]
            countries = [c for c in countries if c != "Unknown"]
            
            if countries:
                for country in set(countries):
                    if country not in country_cluster_mapping:
                        country_cluster_mapping[country] = []
                    country_cluster_mapping[country].append((x, y))
        
        # Find countries that cluster together
        if len(country_cluster_mapping) > 1:
            for country, clusters in country_cluster_mapping.items():
                if len(clusters) > 1:
                    insights.append({
                        "type": "geographic_clustering",
                        "title": f"{country} shows consistent visual patterns",
                        "description": f"{country} images cluster across {len(clusters)} different regions",
                        "confidence": 0.8,
                        "data": {"country": country, "clusters": len(clusters)}
                    })
        
        # Find psychology patterns
        if cluster_psychology_data:
            common_themes = []
            for cluster_data in cluster_psychology_data.values():
                themes = cluster_data.get("dominantThemes", [])
                common_themes.extend(themes)
            
            if common_themes:
                theme_counts = Counter(common_themes)
                most_common_theme = theme_counts.most_common(1)
                
                if most_common_theme and most_common_theme[0][1] > 1:
                    theme, count = most_common_theme[0]
                    insights.append({
                        "type": "psychological_pattern",
                        "title": f"Common {theme.lower()} theme across clusters",
                        "description": f"The '{theme}' psychology theme appears in {count} clusters",
                        "confidence": 0.7,
                        "data": {"theme": theme, "frequency": count}
                    })
        
        return insights
        
    except Exception as e:
        print(f"Error generating cultural insights: {e}")
        return [{"error": str(e)}]
    
def _generate_psychology_patterns(cluster_psychology_data: Dict, cultural_context: str) -> List[Dict]:
    """Generate psychology-specific cultural patterns (MISSING IMPLEMENTATION)."""
    patterns = []
    
    if not cluster_psychology_data:
        return patterns
    
    # Extract all psychology themes
    all_themes = []
    mood_data = []
    
    for cluster_data in cluster_psychology_data.values():
        themes = cluster_data.get("dominantThemes", [])
        all_themes.extend(themes)
        
        mood = cluster_data.get("overallMood", [])
        if mood:
            mood_data.append(mood[0])
    
    # Common theme patterns
    if all_themes:
        from collections import Counter
        theme_counts = Counter(all_themes)
        
        for theme, count in theme_counts.most_common(2):
            if count >= 2:
                patterns.append({
                    "type": "psychological_pattern",
                    "pattern": "psychological_pattern", 
                    "title": f"Common {theme.lower()} theme across clusters",
                    "description": f"The '{theme}' psychology theme appears in {count} clusters",
                    "confidence": 0.7,
                    "data": {"theme": theme, "frequency": count}
                })
    
    # Mood coherence
    if mood_data and len(mood_data) > 1:
        mood_counts = Counter(mood_data)
        dominant_mood, freq = mood_counts.most_common(1)[0]
        
        if freq / len(mood_data) > 0.5:
            patterns.append({
                "type": "psychological_pattern",
                "pattern": "mood_coherence",
                "title": f"Consistent {dominant_mood.lower()} mood pattern",
                "description": f"{freq}/{len(mood_data)} clusters show '{dominant_mood}' psychological characteristics", 
                "confidence": 0.72,
                "data": {"mood": dominant_mood, "frequency": freq}
            })
    
    return patterns

# Keep existing helper functions
def _get_cluster_representative_colors_perceptual(
    stored_colors_list: List[List[str]], 
    num_colors: int = 3,
    color_space: str = "rgb"
) -> List[str]:
    """Get representative colors for a cluster using perceptually-aware grouping."""
    if not stored_colors_list:
        return []
    
    try:
        all_hex_colors = []
        for color_list in stored_colors_list:
            all_hex_colors.extend(color_list)
        
        if not all_hex_colors:
            return []
        
        unique_colors = list(dict.fromkeys(all_hex_colors))
        if len(unique_colors) <= num_colors:
            return unique_colors
        
        if color_space.lower() in ["lab", "hsv"]:
            return _select_perceptual_representatives(unique_colors, num_colors, color_space)
        else:
            color_counts = Counter(all_hex_colors)
            most_common = [color for color, count in color_counts.most_common(num_colors)]
            return most_common
        
    except Exception as e:
        print(f"Error calculating perceptual representative colors: {e}")
        all_colors = []
        for color_list in stored_colors_list:
            all_colors.extend(color_list)
        
        if all_colors:
            color_counts = Counter(all_colors)
            return [color for color, count in color_counts.most_common(num_colors)]
        return []

def _select_perceptual_representatives(hex_colors: List[str], num_colors: int, color_space: str) -> List[str]:
    """Select representative colors using perceptual clustering."""
    try:
        if color_space.lower() == "lab":
            color_vectors = []
            for hex_color in hex_colors:
                try:
                    rgb = hex_to_rgb(hex_color)
                    lab = rgb_to_lab(rgb)
                    color_vectors.append(lab)
                except Exception:
                    continue
            
            if not color_vectors:
                color_vectors = [[*hex_to_rgb(color)] for color in hex_colors]
            color_vectors = np.array(color_vectors)
            
        elif color_space.lower() == "hsv":
            color_vectors = []
            for hex_color in hex_colors:
                try:
                    rgb = hex_to_rgb(hex_color)
                    hsv = rgb_to_hsv(rgb)
                    normalized_hsv = [hsv[0]/360.0, hsv[1], hsv[2]]
                    color_vectors.append(normalized_hsv)
                except Exception:
                    continue
            
            if not color_vectors:
                color_vectors = [[*hex_to_rgb(color)] for color in hex_colors]
                color_vectors = np.array(color_vectors) / 255.0
            else:
                color_vectors = np.array(color_vectors)
        else:
            color_vectors = []
            for hex_color in hex_colors:
                try:
                    rgb = hex_to_rgb(hex_color)
                    color_vectors.append([rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0])
                except Exception:
                    continue
            color_vectors = np.array(color_vectors)
        
        if len(color_vectors) == 0:
            color_counts = Counter(hex_colors)
            return [color for color, count in color_counts.most_common(num_colors)]
        
        n_clusters = min(num_colors, len(color_vectors))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(color_vectors)
        
        representative_colors = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_colors = color_vectors[cluster_mask]
            cluster_hex_colors = [hex_colors[i] for i, mask in enumerate(cluster_mask) if mask]
            
            if len(cluster_colors) > 0:
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = [np.linalg.norm(color - centroid) for color in cluster_colors]
                closest_idx = np.argmin(distances)
                selected_color = cluster_hex_colors[closest_idx]
                representative_colors.append(selected_color)
        
        return representative_colors[:num_colors]
        
    except Exception as e:
        print(f"Error in perceptual color selection: {e}")
        color_counts = Counter(hex_colors)
        return [color for color, count in color_counts.most_common(num_colors)]



def _calculate_cluster_metrics(cluster_features: List[List[float]]) -> Dict[str, Union[str, float, int]]:
    """Calculate metrics for a cluster."""
    if len(cluster_features) <= 1:
        return {"coherence": 1.0, "size": float(len(cluster_features))}
    
    try:
        features_array = np.array(cluster_features)
        variance = np.var(features_array, axis=0).mean()
        coherence = 1.0 / (1.0 + variance) 
        
        try:
            from scipy.spatial.distance import pdist
            distances = pdist(features_array)
            avg_distance = np.mean(distances) if len(distances) > 0 else 0.0
        except ImportError:
            avg_distance = 0.0
            count = 0
            for i in range(len(features_array)):
                for j in range(i + 1, len(features_array)):
                    dist = np.linalg.norm(features_array[i] - features_array[j])
                    avg_distance += dist
                    count += 1
            avg_distance = avg_distance / count if count > 0 else 0.0
        
        return {
            "coherence": float(coherence),
            "size": float(len(cluster_features)),
            "average_distance": float(avg_distance),
            "variance": float(variance)
        }
    except Exception as e:
        print(f"Error calculating cluster metrics: {e}")
        return {"coherence": 0.0, "size": float(len(cluster_features))}

if __name__ == "__main__":
    import uvicorn
    
    # Create directories
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    datastore_dir = Path(__file__).parent / "datastore"
    datastore_dir.mkdir(exist_ok=True)
    
    # Create sample files if needed
    country_codes_file = datastore_dir / "country_codes.json"
    if not country_codes_file.exists():
        sample_country_data = {
            "countries": [
                {"code": "US", "name": "United States of America"},
                {"code": "DE", "name": "Germany"},
                {"code": "FR", "name": "France"},
                {"code": "GB", "name": "United Kingdom"},
                {"code": "IT", "name": "Italy"},
                {"code": "ES", "name": "Spain"},
                {"code": "CA", "name": "Canada"},
                {"code": "AU", "name": "Australia"},
                {"code": "JP", "name": "Japan"},
                {"code": "CN", "name": "China"}
            ]
        }
        with open(country_codes_file, 'w', encoding='utf-8') as f:
            json.dump(sample_country_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Created sample country_codes.json")
    
    print(f"\n🎨 ENHANCED API WITH PSYCHOLOGY INTEGRATION")
    print(f"=" * 60)
    print(f"✨ Psychology Features: {'✅ Available' if COLOR_PSYCHOLOGY_AVAILABLE else '❌ Unavailable'}")
    
    print(f"\n📡 ENHANCED ENDPOINTS:")
    print(f"   📊 GET  /psychology-capabilities")
    print(f"   🎨 POST /extract-dominant-colors (enhanced)")
    print(f"   🧩 POST /cluster-postcard-images (enhanced)")
    print(f"   💡 POST /analyze-color-psychology")
    
    print(f"\n⚙️  BACKWARD COMPATIBILITY:")
    print(f"   ✅ All existing API calls work unchanged")
    print(f"   ✅ Add 'includePsychology: true' for psychology features")
    print(f"   ✅ Optional parameters for enhanced functionality")
    print(f"   ✅ Progressive enhancement possible")
    
    print(f"\n🔧 INTEGRATION LEVELS:")
    print(f"   Level 1: includePsychology: true")
    print(f"   Level 2: psychologyAnalysisLevel: 'cluster'")
    print(f"   Level 3: psychologyAnalysisLevel: 'full', culturalInsights: true")
    
    uvicorn.run("app:app", host="0.0.0.0", port=8008, reload=True)