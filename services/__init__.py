# services/__init__.py
"""
Enhanced Image Analysis API - Services Module
All service classes for image analysis, clustering, and psychology
"""

# Core services - should always be available
try:
    from .color_extraction import extract_dominant_colors_from_image
    COLOR_EXTRACTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import color extraction service: {e}")
    COLOR_EXTRACTION_AVAILABLE = False

try:
    from .image_clustering import (
        get_image_features, 
        cluster_image_features_som, 
        analyze_cluster_quality, 
        get_cluster_representative_features
    )
    IMAGE_CLUSTERING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import image clustering service: {e}")
    IMAGE_CLUSTERING_AVAILABLE = False

try:
    from .country_service import CountryService
    COUNTRY_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import country service: {e}")
    COUNTRY_SERVICE_AVAILABLE = False

try:
    from .world_map_service import WorldMapService
    WORLD_MAP_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import world map service: {e}")
    WORLD_MAP_SERVICE_AVAILABLE = False

# NEW: Color psychology service
try:
    from .color_psychology_service import ColorPsychologyService
    COLOR_PSYCHOLOGY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import color psychology service: {e}")
    COLOR_PSYCHOLOGY_AVAILABLE = False

# Optional advanced services
try:
    from .reference_clustering_service import ReferenceClustering
    REFERENCE_CLUSTERING_AVAILABLE = True
except ImportError as e:
    print(f"Info: Reference clustering service not available: {e}")
    REFERENCE_CLUSTERING_AVAILABLE = False

try:
    from .image_metadata_service import ImageMetadataService
    IMAGE_METADATA_AVAILABLE = True
except ImportError as e:
    print(f"Info: Image metadata service not available: {e}")
    IMAGE_METADATA_AVAILABLE = False

try:
    from .gpu_processor import GPUProcessor
    GPU_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Info: GPU processor not available: {e}")
    GPU_PROCESSOR_AVAILABLE = False

__version__ = "3.1.0"

# Define what should be available when importing from services
__all__ = [
    # Core functions
    "extract_dominant_colors_from_image",
    "get_image_features",
    "cluster_image_features_som",
    "analyze_cluster_quality",
    "get_cluster_representative_features",
    
    # Service classes
    "CountryService",
    "WorldMapService",
    "ColorPsychologyService",
    
    # Optional services (if available)
    "ReferenceClustering",
    "ImageMetadataService", 
    "GPUProcessor",
    
    # Availability flags
    "COLOR_EXTRACTION_AVAILABLE",
    "IMAGE_CLUSTERING_AVAILABLE", 
    "COUNTRY_SERVICE_AVAILABLE",
    "WORLD_MAP_SERVICE_AVAILABLE",
    "COLOR_PSYCHOLOGY_AVAILABLE",
    "REFERENCE_CLUSTERING_AVAILABLE",
    "IMAGE_METADATA_AVAILABLE",
    "GPU_PROCESSOR_AVAILABLE"
]

# Print service availability status
def print_service_status():
    """Print the status of all services"""
    print("\n🔧 SERVICES MODULE STATUS:")
    print(f"   • Color Extraction: {'✅' if COLOR_EXTRACTION_AVAILABLE else '❌'}")
    print(f"   • Image Clustering: {'✅' if IMAGE_CLUSTERING_AVAILABLE else '❌'}")
    print(f"   • Country Service: {'✅' if COUNTRY_SERVICE_AVAILABLE else '❌'}")
    print(f"   • World Map Service: {'✅' if WORLD_MAP_SERVICE_AVAILABLE else '❌'}")
    print(f"   • Color Psychology: {'✅' if COLOR_PSYCHOLOGY_AVAILABLE else '❌'} (NEW)")
    print(f"   • Reference Clustering: {'✅' if REFERENCE_CLUSTERING_AVAILABLE else '⚪'}")
    print(f"   • Image Metadata: {'✅' if IMAGE_METADATA_AVAILABLE else '⚪'}")
    print(f"   • GPU Processor: {'✅' if GPU_PROCESSOR_AVAILABLE else '⚪'}")

# Count available services
core_services = [
    COLOR_EXTRACTION_AVAILABLE,
    IMAGE_CLUSTERING_AVAILABLE,
    COUNTRY_SERVICE_AVAILABLE,
    WORLD_MAP_SERVICE_AVAILABLE,
    COLOR_PSYCHOLOGY_AVAILABLE
]

optional_services = [
    REFERENCE_CLUSTERING_AVAILABLE,
    IMAGE_METADATA_AVAILABLE,
    GPU_PROCESSOR_AVAILABLE
]

CORE_SERVICES_COUNT = sum(core_services)
OPTIONAL_SERVICES_COUNT = sum(optional_services)
TOTAL_SERVICES_COUNT = CORE_SERVICES_COUNT + OPTIONAL_SERVICES_COUNT

# Auto-print status on import
print(f"📦 Services module loaded: {CORE_SERVICES_COUNT}/5 core services, {OPTIONAL_SERVICES_COUNT}/3 optional services")

if COLOR_PSYCHOLOGY_AVAILABLE:
    print("🎨 Color Psychology service is ready for Step 2 testing!")