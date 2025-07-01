#!/usr/bin/env python3
"""
Test Import Script
Verify that all __init__.py updates work correctly
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_utils_imports():
    """Test utils module imports"""
    print("🧪 TESTING UTILS MODULE IMPORTS")
    print("=" * 40)
    
    try:
        # Test package-level import
        import utils
        print("✅ utils package imported successfully")
        
        # Test availability flags
        print(f"   • Image utils available: {getattr(utils, 'IMAGE_UTILS_AVAILABLE', False)}")
        print(f"   • Color matching available: {getattr(utils, 'COLOR_MATCHING_AVAILABLE', False)}")
        
        # Test specific imports
        try:
            from utils import hex_to_hsv, get_color_brightness_category
            print("✅ Color matching functions imported successfully")
        except ImportError as e:
            print(f"❌ Color matching import failed: {e}")
        
        try:
            from utils import data_uri_to_pil_image, calculate_color_temperature, hex_to_rgb
            print("✅ Original image utils imported successfully")
            print("   • hex_to_rgb comes from image_utils (no conflict)")
        except ImportError as e:
            print(f"❌ Original utils import failed: {e}")
            
    except ImportError as e:
        print(f"❌ Utils package import failed: {e}")
        return False
    
    return True

def test_services_imports():
    """Test services module imports"""
    print("\n🧪 TESTING SERVICES MODULE IMPORTS")
    print("=" * 40)
    
    try:
        # Test package-level import
        import services
        print("✅ services package imported successfully")
        
        # Print service status
        if hasattr(services, 'print_service_status'):
            services.print_service_status()
        
        # Test ColorPsychologyService import
        try:
            from services import ColorPsychologyService
            print("✅ ColorPsychologyService imported successfully")
            
            # Quick initialization test
            psychology_service = ColorPsychologyService()
            print("✅ ColorPsychologyService initialized successfully")
            
        except ImportError as e:
            print(f"❌ ColorPsychologyService import failed: {e}")
            return False
        
        # Test other core services
        try:
            from services import CountryService, WorldMapService
            print("✅ Core services (Country, WorldMap) imported successfully")
        except ImportError as e:
            print(f"❌ Core services import failed: {e}")
        
        # Test extraction functions
        try:
            from services import extract_dominant_colors_from_image
            print("✅ Color extraction function imported successfully")
        except ImportError as e:
            print(f"❌ Color extraction import failed: {e}")
            
    except ImportError as e:
        print(f"❌ Services package import failed: {e}")
        return False
    
    return True

def test_integrated_functionality():
    """Test that the integrated functionality works"""
    print("\n🧪 TESTING INTEGRATED FUNCTIONALITY")
    print("=" * 40)
    
    try:
        # Test integrated import
        from services import ColorPsychologyService
        from utils import hex_to_hsv
        
        # Initialize service
        psychology_service = ColorPsychologyService()
        
        # Test color conversion
        hsv = hex_to_hsv("#9c774b")
        print(f"✅ Color conversion: #9c774b → HSV {hsv}")
        
        # Test psychology analysis
        result = psychology_service.get_psychology_for_hex_color("#9c774b")
        print(f"✅ Psychology analysis: {result.get('classifiedAs')} → {result.get('psychology', [])}")
        
        print("\n🎉 All integrated functionality working!")
        return True
        
    except Exception as e:
        print(f"❌ Integrated functionality test failed: {e}")
        return False

def test_backward_compatibility():
    """Test that existing imports still work"""
    print("\n🧪 TESTING BACKWARD COMPATIBILITY")
    print("=" * 40)
    
    try:
        # Test direct service imports (old style)
        from services.color_psychology_service import ColorPsychologyService
        print("✅ Direct service import still works")
        
        # Test direct utils imports (old style)
        from utils.color_matching_utils import hex_to_hsv
        print("✅ Direct utils import still works")
        
        print("✅ Backward compatibility maintained")
        return True
        
    except ImportError as e:
        print(f"❌ Backward compatibility issue: {e}")
        return False

if __name__ == "__main__":
    print("🔧 TESTING __INIT__.PY UPDATES")
    print("=" * 50)
    
    # Run all tests
    utils_ok = test_utils_imports()
    services_ok = test_services_imports() 
    integrated_ok = test_integrated_functionality()
    compat_ok = test_backward_compatibility()
    
    # Summary
    print(f"\n📊 TEST RESULTS SUMMARY:")
    print(f"   • Utils imports: {'✅' if utils_ok else '❌'}")
    print(f"   • Services imports: {'✅' if services_ok else '❌'}")
    print(f"   • Integrated functionality: {'✅' if integrated_ok else '❌'}")
    print(f"   • Backward compatibility: {'✅' if compat_ok else '❌'}")
    
    if all([utils_ok, services_ok, integrated_ok, compat_ok]):
        print(f"\n🎉 ALL IMPORT TESTS PASSED!")
        print(f"✅ __init__.py updates are working correctly")
        print(f"🚀 Ready to proceed with Step 2 testing!")
    else:
        print(f"\n❌ Some import tests failed. Check the errors above.")
        print(f"💡 Make sure both __init__.py files are updated correctly.")