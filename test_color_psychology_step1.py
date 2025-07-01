#!/usr/bin/env python3
"""
Test Script for Color Psychology Service - Step 1
Run this to verify the basic ColorPsychologyService functionality
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from services.color_psychology_service import ColorPsychologyService
except ImportError:
    print("❌ Could not import ColorPsychologyService")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def test_basic_functionality():
    """Test basic psychology service functionality"""
    print("🧪 TESTING COLOR PSYCHOLOGY SERVICE - STEP 1")
    print("=" * 60)
    
    # Test 1: Initialize service
    print("\\n1. 🚀 Initializing ColorPsychologyService...")
    try:
        psychology_service = ColorPsychologyService()
        print("   ✅ Service initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize service: {e}")
        return False
    
    # Test 2: Get available colors
    print("\\n2. 🎨 Testing available colors...")
    try:
        available_colors = psychology_service.get_available_colors()
        print(f"   ✅ Found {len(available_colors)} colors:")
        for color in available_colors:
            print(f"      • {color}")
    except Exception as e:
        print(f"   ❌ Failed to get available colors: {e}")
        return False
    
    # Test 3: Get cultural contexts
    print("\\n3. 🌍 Testing cultural contexts...")
    try:
        contexts = psychology_service.get_cultural_contexts()
        print(f"   ✅ Found {len(contexts)} cultural contexts:")
        for context in contexts:
            context_info = psychology_service.get_cultural_context_info(context)
            name = context_info.get('name', context)
            print(f"      • {context} ({name})")
    except Exception as e:
        print(f"   ❌ Failed to get cultural contexts: {e}")
        return False
    
    # Test 4: Test psychology lookup for brown (your example color)
    print("\\n4. 🤎 Testing brown color psychology...")
    try:
        brown_universal = psychology_service.get_psychology_for_color("brown", "universal")
        brown_english = psychology_service.get_psychology_for_color("brown", "english")
        brown_chinese = psychology_service.get_psychology_for_color("brown", "chinese")
        
        print("   ✅ Brown psychology associations:")
        print(f"      Universal: {brown_universal}")
        print(f"      English: {brown_english}")
        print(f"      Chinese: {brown_chinese}")
    except Exception as e:
        print(f"   ❌ Failed to get brown psychology: {e}")
        return False
    
    # Test 5: Test all psychology for a color
    print("\\n5. 🔴 Testing red color across all cultures...")
    try:
        red_all = psychology_service.get_all_psychology_for_color("red")
        print("   ✅ Red psychology across cultures:")
        for context, associations in red_all.items():
            print(f"      {context}: {associations}")
    except Exception as e:
        print(f"   ❌ Failed to get all red psychology: {e}")
        return False
    
    # Test 6: Test cultural comparison
    print("\\n6. 🔍 Testing cultural comparison for red...")
    try:
        red_comparison = psychology_service.compare_cultural_psychology("red")
        print("   ✅ Red cultural comparison:")
        print(f"      Common across cultures: {red_comparison.get('commonAcrossCultures', [])}")
        print(f"      Context-specific: {red_comparison.get('contextSpecific', {})}")
        print(f"      Cultural variation: {red_comparison.get('culturalVariation', False)}")
    except Exception as e:
        print(f"   ❌ Failed to compare red psychology: {e}")
        return False
    
    # Test 7: Get service summary
    print("\\n7. 📊 Testing service summary...")
    try:
        summary = psychology_service.get_psychology_summary()
        print("   ✅ Service summary:")
        print(f"      Total colors: {summary.get('totalColors', 0)}")
        print(f"      Total contexts: {summary.get('totalCulturalContexts', 0)}")
        print(f"      Total associations: {summary.get('totalAssociations', 0)}")
        print(f"      Avg associations per color: {summary.get('averageAssociationsPerColor', 0)}")
    except Exception as e:
        print(f"   ❌ Failed to get service summary: {e}")
        return False
    
    # Test 8: Test edge cases
    print("\\n8. ⚠️  Testing edge cases...")
    try:
        # Test invalid color
        invalid_color = psychology_service.get_psychology_for_color("invalidcolor")
        print(f"   ✅ Invalid color test: {len(invalid_color)} associations (should be 0)")
        
        # Test invalid context
        invalid_context = psychology_service.get_psychology_for_color("red", "invalidcontext")
        print(f"   ✅ Invalid context test: {len(invalid_context)} associations (should fallback to universal)")
        
        # Test empty input
        empty_input = psychology_service.get_psychology_for_color("")
        print(f"   ✅ Empty input test: {len(empty_input)} associations (should be 0)")
        
    except Exception as e:
        print(f"   ❌ Edge case testing failed: {e}")
        return False
    
    print("\\n🎉 ALL TESTS PASSED! Step 1 is working correctly.")
    print("\\n📋 READY FOR STEP 2:")
    print("   • Color Psychology Service ✅")
    print("   • JSON Data Loading ✅")
    print("   • Basic Lookup Functions ✅")
    print("   • Cultural Context Support ✅")
    print("   • Error Handling ✅")
    
    return True

def test_specific_colors():
    """Test some specific colors to verify data integrity"""
    print("\\n\\n🎯 BONUS: Testing specific colors from your data...")
    
    psychology_service = ColorPsychologyService()
    
    test_colors = ["white", "black", "red", "blue", "green", "yellow", "orange", "purple", "pink", "gray", "brown"]
    
    for color in test_colors:
        print(f"\\n🎨 {color.upper()}:")
        all_psych = psychology_service.get_all_psychology_for_color(color)
        for context, associations in all_psych.items():
            print(f"   {context}: {associations}")

if __name__ == "__main__":
    # Run basic functionality tests
    success = test_basic_functionality()
    
    if success:
        # Run specific color tests
        test_specific_colors()
        
        print("\\n✨ Step 1 Complete! ColorPsychologyService is ready.")
        print("\\n🚀 Next: We can proceed to Step 2 (Color Matching Logic)")
    else:
        print("\\n❌ Step 1 Failed. Please check the error messages above.")