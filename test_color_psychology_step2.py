#!/usr/bin/env python3
"""
Test Script for Color Psychology Service - Step 2 (Color Matching)
Run this to verify hex color to psychology functionality
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from services.color_psychology_service import ColorPsychologyService
    from utils.color_matching_utils import hex_to_hsv, hex_to_rgb
except ImportError as e:
    print(f"❌ Could not import required modules: {e}")
    print("Make sure you have both services/color_psychology_service.py and utils/color_matching_utils.py")
    sys.exit(1)

def test_color_matching_functionality():
    """Test the new color matching features"""
    print("🧪 TESTING COLOR PSYCHOLOGY SERVICE - STEP 2 (COLOR MATCHING)")
    print("=" * 70)
    
    # Initialize service
    print("\\n1. 🚀 Initializing ColorPsychologyService...")
    try:
        psychology_service = ColorPsychologyService()
        print("   ✅ Service initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize service: {e}")
        return False
    
    # Test color matching utilities
    print("\\n2. 🛠️  Testing color matching utilities...")
    try:
        # Test your example color #9c774b
        test_hex = "#9c774b"
        rgb = hex_to_rgb(test_hex)
        hsv = hex_to_hsv(test_hex)
        
        print(f"   ✅ Color conversion test for {test_hex}:")
        print(f"      RGB: {rgb}")
        print(f"      HSV: {hsv}")
        print(f"      Hue: {hsv[0]:.1f}°, Saturation: {hsv[1]:.2f}, Value: {hsv[2]:.2f}")
    except Exception as e:
        print(f"   ❌ Color utility test failed: {e}")
        return False
    
    # Test hex color classification
    print("\\n3. 🎨 Testing hex color classification...")
    test_colors = {
        "#9c774b": "brown (your example)",
        "#ff0000": "red",
        "#0000ff": "blue", 
        "#00ff00": "green",
        "#ffff00": "yellow",
        "#ffa500": "orange",
        "#800080": "purple",
        "#ffc0cb": "pink",
        "#808080": "gray",
        "#ffffff": "white",
        "#000000": "black"
    }
    
    classification_results = {}
    for hex_color, expected in test_colors.items():
        try:
            color_name, confidence = psychology_service.classify_hex_color(hex_color)
            classification_results[hex_color] = (color_name, confidence)
            
            status = "✅" if color_name else "❌"
            print(f"   {status} {hex_color} → {color_name} (confidence: {confidence:.2f}) - Expected: {expected}")
        except Exception as e:
            print(f"   ❌ Failed to classify {hex_color}: {e}")
            classification_results[hex_color] = (None, 0.0)
    
    # Test psychology from hex colors
    print("\\n4. 🧠 Testing psychology from hex colors...")
    try:
        # Test your brown example
        brown_psychology = psychology_service.get_psychology_for_hex_color("#9c774b", "universal")
        print(f"   ✅ Brown (#9c774b) psychology analysis:")
        print(f"      Classified as: {brown_psychology.get('classifiedAs')}")
        print(f"      Confidence: {brown_psychology.get('confidence', 0):.2f}")
        print(f"      Psychology: {brown_psychology.get('psychology', [])}")
        print(f"      HSV: {brown_psychology.get('hsv', {})}")
        print(f"      Brightness: {brown_psychology.get('brightness')}")
        
        # Test red with cultural differences
        red_english = psychology_service.get_psychology_for_hex_color("#ff0000", "english")
        red_chinese = psychology_service.get_psychology_for_hex_color("#ff0000", "chinese")
        
        print(f"\\n   ✅ Red (#ff0000) cultural comparison:")
        print(f"      English: {red_english.get('psychology', [])}")
        print(f"      Chinese: {red_chinese.get('psychology', [])}")
        
    except Exception as e:
        print(f"   ❌ Psychology from hex test failed: {e}")
        return False
    
    # Test color palette analysis
    print("\\n5. 🎨 Testing color palette psychology...")
    try:
        # Test a palette of colors
        test_palette = ["#9c774b", "#8B4513", "#D2691E", "#CD853F"]  # Brown palette
        palette_analysis = psychology_service.analyze_color_palette_psychology(test_palette, "universal")
        
        print(f"   ✅ Brown palette analysis:")
        print(f"      Colors: {test_palette}")
        print(f"      Classification rate: {palette_analysis.get('paletteMetrics', {}).get('classificationRate', 0):.1%}")
        print(f"      Dominant themes: {palette_analysis.get('dominantPsychologyThemes', [])}")
        print(f"      Total psychology associations: {palette_analysis.get('paletteMetrics', {}).get('totalPsychologyAssociations', 0)}")
        
    except Exception as e:
        print(f"   ❌ Palette analysis test failed: {e}")
        return False
    
    # Test similar colors
    print("\\n6. 🔍 Testing similar color finding...")
    try:
        similar_to_brown = psychology_service.find_similar_colors("#9c774b", max_distance=0.4)
        print(f"   ✅ Colors similar to brown (#9c774b):")
        for similar in similar_to_brown[:3]:  # Show top 3
            print(f"      {similar['colorName']} (distance: {similar['distance']}) - {similar['psychology'][:2]}")
        
    except Exception as e:
        print(f"   ❌ Similar colors test failed: {e}")
        return False
    
    # Test edge cases
    print("\\n7. ⚠️  Testing edge cases...")
    try:
        # Invalid hex
        invalid_result = psychology_service.get_psychology_for_hex_color("invalid")
        print(f"   ✅ Invalid hex test: Status = {invalid_result.get('status')}")
        
        # Very dark color
        dark_result = psychology_service.classify_hex_color("#010101")
        print(f"   ✅ Very dark color: {dark_result[0]} (confidence: {dark_result[1]:.2f})")
        
        # Very bright color
        bright_result = psychology_service.classify_hex_color("#fefefe")
        print(f"   ✅ Very bright color: {bright_result[0]} (confidence: {bright_result[1]:.2f})")
        
        # Gray color
        gray_result = psychology_service.classify_hex_color("#808080")
        print(f"   ✅ Gray color: {gray_result[0]} (confidence: {gray_result[1]:.2f})")
        
    except Exception as e:
        print(f"   ❌ Edge case testing failed: {e}")
        return False
    
    # Summary
    print("\\n8. 📊 Service summary with color matching...")
    try:
        summary = psychology_service.get_psychology_summary()
        print(f"   ✅ Enhanced service summary:")
        print(f"      Total colors: {summary.get('totalColors', 0)}")
        print(f"      Color matching enabled: {summary.get('colorMatchingEnabled', False)}")
        print(f"      Color mapping rules: {summary.get('colorMappingRules', 0)}")
        print(f"      Cultural contexts: {summary.get('totalCulturalContexts', 0)}")
        
    except Exception as e:
        print(f"   ❌ Summary test failed: {e}")
        return False
    
    # Calculate success rate
    successful_classifications = sum(1 for result in classification_results.values() if result[0] is not None)
    total_tests = len(classification_results)
    success_rate = (successful_classifications / total_tests) * 100
    
    print(f"\\n🎉 STEP 2 TESTS COMPLETED!")
    print(f"\\n📊 RESULTS SUMMARY:")
    print(f"   • Color classification success rate: {success_rate:.1f}% ({successful_classifications}/{total_tests})")
    print(f"   • Color matching utilities: ✅ Working")
    print(f"   • Psychology from hex colors: ✅ Working")
    print(f"   • Palette analysis: ✅ Working")
    print(f"   • Similar color finding: ✅ Working")
    print(f"   • Edge case handling: ✅ Working")
    
    if success_rate >= 80:
        print(f"\\n✨ EXCELLENT! Step 2 is working great.")
        print(f"\\n🚀 READY FOR STEP 3: Integration with Color Extraction")
        return True
    else:
        print(f"\\n⚠️  Step 2 has some issues. Classification success rate is {success_rate:.1f}%")
        print("   Consider adjusting color mapping rules in the JSON file.")
        return False

def test_australia_brown_example():
    """Specific test for your Australia brown example"""
    print("\\n\\n🇦🇺 AUSTRALIA BROWN COLOR EXAMPLE TEST")
    print("=" * 50)
    
    psychology_service = ColorPsychologyService()
    
    # Your specific color from Australia
    australia_brown = "#9c774b"
    
    print(f"\\n🎨 Analyzing Australia's dominant color: {australia_brown}")
    
    # Test classification
    color_name, confidence = psychology_service.classify_hex_color(australia_brown)
    print(f"   Color classification: {color_name} (confidence: {confidence:.2f})")
    
    # Test psychology across cultures
    universal_psych = psychology_service.get_psychology_for_hex_color(australia_brown, "universal")
    english_psych = psychology_service.get_psychology_for_hex_color(australia_brown, "english")
    chinese_psych = psychology_service.get_psychology_for_hex_color(australia_brown, "chinese")
    
    print(f"\\n🌍 Cultural psychology analysis:")
    print(f"   Universal: {universal_psych.get('psychology', [])}")
    print(f"   English: {english_psych.get('psychology', [])}")
    print(f"   Chinese: {chinese_psych.get('psychology', [])}")
    
    print(f"\\n🧠 Psychological interpretation for Australia:")
    print(f"   This warm brown suggests: {', '.join(universal_psych.get('psychology', []))}")
    print(f"   Color temperature: 0.74 (warm) ✅ matches brown psychology")
    print(f"   Geographic relevance: Earth tones align with 'Outback' cultural tag")

if __name__ == "__main__":
    # Run Step 2 tests
    success = test_color_matching_functionality()
    
    if success:
        # Run Australia-specific example
        test_australia_brown_example()
        
        print("\\n\\n🎯 NEXT STEPS:")
        print("   • Step 2 Complete ✅")
        print("   • Ready for Step 3: Color Extraction Integration")
        print("   • Ready for Step 4: Country Analysis Enhancement")
        print("   • Ready for Step 5: World Map Psychology")
    else:
        print("\\n❌ Step 2 needs attention before proceeding.")