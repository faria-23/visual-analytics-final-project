#!/usr/bin/env python3
"""
Test Script for Enhanced Color Extraction - Step 3 (Simplified Single Service)
Tests the enhanced color extraction with psychology integration
"""

import sys
import json
import requests
import base64
import io
from pathlib import Path
from PIL import Image
import numpy as np

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

class EnhancedColorExtractionTester:
    def __init__(self, host="localhost", port=8008):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
    
    def create_test_data_uri(self, dominant_color="#9c774b", size=(150, 150)) -> str:
        """Create a test image with a dominant color"""
        try:
            # Parse hex color
            rgb = tuple(int(dominant_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            
            # Create image with variations
            img_array = np.full((*size, 3), rgb, dtype=np.uint8)
            
            # Add color variations for realism
            for i in range(size[0]):
                for j in range(size[1]):
                    variation = (i + j) % 7
                    if variation == 0:
                        img_array[i, j] = [min(255, c + 15) for c in rgb]
                    elif variation == 1:
                        img_array[i, j] = [max(0, c - 15) for c in rgb]
            
            # Convert to data URI
            img = Image.fromarray(img_array, 'RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            print(f"Error creating test image: {e}")
            return ""
    
    def test_server_availability(self) -> bool:
        """Check if API server is running"""
        try:
            response = self.session.get(f"{self.base_url}/docs", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def test_direct_function_import(self):
        """Test importing and using the enhanced function directly"""
        print("\\n1. 🔧 Testing Direct Function Import...")
        
        try:
            from services.color_extraction import extract_dominant_colors_from_image
            print("   ✅ Successfully imported enhanced function")
            
            # Create test image
            test_image = Image.new('RGB', (100, 100), color='#9c774b')
            
            # Test 1: Basic backward compatible call
            basic_result = extract_dominant_colors_from_image(
                image=test_image,
                num_colors=5
            )
            
            if isinstance(basic_result, list):
                print(f"   ✅ Backward compatible call: {len(basic_result)} colors")
                print(f"      Colors: {basic_result}")
            else:
                print(f"   ❌ Expected list, got: {type(basic_result)}")
                return False
            
            # Test 2: Enhanced call with psychology
            enhanced_result = extract_dominant_colors_from_image(
                image=test_image,
                num_colors=5,
                include_psychology=True,
                cultural_context="universal"
            )
            
            if isinstance(enhanced_result, dict):
                print(f"   ✅ Enhanced call successful")
                print(f"      Colors: {enhanced_result.get('colors', [])}")
                
                psychology = enhanced_result.get('psychology', {})
                if psychology.get('enabled'):
                    print(f"      Psychology enabled: {psychology.get('culturalContext')}")
                    
                    color_analyses = psychology.get('colorAnalyses', [])
                    print(f"      Color analyses: {len(color_analyses)}")
                    
                    if color_analyses:
                        first_analysis = color_analyses[0]
                        print(f"      First color: {first_analysis.get('hexColor')} → {first_analysis.get('classifiedAs')}")
                        print(f"      Psychology: {first_analysis.get('psychology', [])}")
                else:
                    print(f"      Psychology disabled: {psychology.get('error', 'Unknown reason')}")
            else:
                print(f"   ❌ Expected dict for enhanced call, got: {type(enhanced_result)}")
                return False
            
            return True
            
        except ImportError as e:
            print(f"   ❌ Import failed: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Function test failed: {e}")
            return False
    
    def test_api_backward_compatibility(self):
        """Test that existing API calls still work"""
        print("\\n2. 🔄 Testing API Backward Compatibility...")
        
        test_data_uri = self.create_test_data_uri("#9c774b")
        
        # Original API call format
        payload = {
            "photoDataUri": test_data_uri,
            "numberOfColors": 5,
            "extractionStrategy": "enhanced_kmeans"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/extract-dominant-colors",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if "colors" in result and isinstance(result["colors"], list):
                    colors = result["colors"]
                    print(f"   ✅ Backward compatibility: {len(colors)} colors extracted")
                    print(f"      Colors: {colors}")
                    
                    # Should not have psychology data
                    if "psychology" not in result or not result.get("psychology", {}).get("enabled"):
                        print(f"   ✅ No psychology data (correct for basic call)")
                        return True, colors
                    else:
                        print(f"   ⚠️  Unexpected psychology data in basic call")
                        return True, colors
                else:
                    print(f"   ❌ Invalid response format")
                    return False, []
            else:
                print(f"   ❌ API call failed: {response.status_code}")
                print(f"      Error: {response.text}")
                return False, []
                
        except Exception as e:
            print(f"   ❌ Backward compatibility test failed: {e}")
            return False, []
    
    def test_psychology_enhancement_api(self):
        """Test the psychology enhancement through API"""
        print("\\n3. 🧠 Testing Psychology Enhancement API...")
        
        test_data_uri = self.create_test_data_uri("#9c774b")
        
        # Enhanced API call with psychology
        payload = {
            "photoDataUri": test_data_uri,
            "numberOfColors": 5,
            "extractionStrategy": "enhanced_kmeans",
            "includePsychology": True,
            "culturalContext": "universal",
            "psychologyConfidenceThreshold": 0.5
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/extract-dominant-colors",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"   ✅ Enhanced API call successful")
                
                # Check colors
                colors = result.get("colors", [])
                print(f"      Colors extracted: {len(colors)}")
                print(f"      Colors: {colors}")
                
                # Check psychology data
                psychology = result.get("psychology", {})
                if psychology and psychology.get("enabled"):
                    print(f"      Psychology analysis: ✅ Enabled")
                    print(f"      Cultural context: {psychology.get('culturalContext')}")
                    
                    # Check color analyses
                    color_analyses = psychology.get("colorAnalyses", [])
                    print(f"      Color analyses: {len(color_analyses)}")
                    
                    for i, analysis in enumerate(color_analyses[:2]):
                        hex_color = analysis.get("hexColor")
                        classified_as = analysis.get("classifiedAs")
                        psychology_themes = analysis.get("psychology", [])
                        confidence = analysis.get("confidence", 0)
                        
                        print(f"        {i+1}. {hex_color} → {classified_as} ({confidence:.2f})")
                        print(f"           Psychology: {psychology_themes}")
                    
                    # Check palette analysis
                    palette_analysis = psychology.get("paletteAnalysis", {})
                    if palette_analysis:
                        print(f"      Palette analysis:")
                        print(f"        Dominant themes: {palette_analysis.get('dominantPsychologyThemes', [])}")
                        print(f"        Overall mood: {palette_analysis.get('overallMood')}")
                        print(f"        Color family: {palette_analysis.get('colorFamily')}")
                    
                    return True, result
                else:
                    error = psychology.get("error", "Unknown error")
                    print(f"      Psychology analysis: ❌ Failed - {error}")
                    return False, result
            else:
                print(f"   ❌ Enhanced API call failed: {response.status_code}")
                print(f"      Error: {response.text}")
                return False, {}
                
        except Exception as e:
            print(f"   ❌ Psychology enhancement test failed: {e}")
            return False, {}
    
    def test_cultural_contexts_api(self):
        """Test different cultural contexts through API"""
        print("\\n4. 🌍 Testing Cultural Contexts...")
        
        test_data_uri = self.create_test_data_uri("#ff0000")  # Red for cultural differences
        contexts = ["universal", "english", "chinese"]
        results = {}
        
        for context in contexts:
            payload = {
                "photoDataUri": test_data_uri,
                "numberOfColors": 3,
                "includePsychology": True,
                "culturalContext": context
            }
            
            try:
                response = self.session.post(
                    f"{self.base_url}/extract-dominant-colors",
                    json=payload,
                    timeout=20
                )
                
                if response.status_code == 200:
                    result = response.json()
                    psychology = result.get("psychology", {})
                    
                    if psychology.get("enabled"):
                        color_analyses = psychology.get("colorAnalyses", [])
                        if color_analyses:
                            red_analysis = color_analyses[0]
                            psychology_themes = red_analysis.get("psychology", [])
                            results[context] = psychology_themes
                            print(f"   • {context.capitalize()}: {psychology_themes}")
                        else:
                            print(f"   • {context.capitalize()}: No analyses")
                    else:
                        error = psychology.get("error", "Unknown")
                        print(f"   • {context.capitalize()}: Psychology disabled - {error}")
                else:
                    print(f"   • {context.capitalize()}: API failed ({response.status_code})")
                    
            except Exception as e:
                print(f"   • {context.capitalize()}: Error - {e}")
        
        # Show cultural differences
        if len(results) > 1:
            print(f"\\n   🔍 Cultural differences for red:")
            contexts_list = list(results.keys())
            for i in range(len(contexts_list)):
                for j in range(i + 1, len(contexts_list)):
                    ctx1, ctx2 = contexts_list[i], contexts_list[j]
                    themes1 = set(results[ctx1])
                    themes2 = set(results[ctx2])
                    
                    different1 = themes1 - themes2
                    different2 = themes2 - themes1
                    
                    if different1 or different2:
                        print(f"      {ctx1} vs {ctx2}:")
                        if different1:
                            print(f"        {ctx1} unique: {list(different1)}")
                        if different2:
                            print(f"        {ctx2} unique: {list(different2)}")
        
        return len(results) > 0
    
    def test_australia_brown_example(self):
        """Test the specific Australia brown example"""
        print("\\n5. 🇦🇺 Testing Australia Brown Example...")
        
        test_data_uri = self.create_test_data_uri("#9c774b")
        
        payload = {
            "photoDataUri": test_data_uri,
            "numberOfColors": 5,
            "extractionStrategy": "enhanced_kmeans",
            "includePsychology": True,
            "culturalContext": "universal"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/extract-dominant-colors",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                psychology = result.get("psychology", {})
                
                if psychology.get("enabled"):
                    print(f"   ✅ Australia brown analysis:")
                    
                    # Find brown color analysis
                    color_analyses = psychology.get("colorAnalyses", [])
                    brown_analysis = None
                    
                    for analysis in color_analyses:
                        hex_color = analysis.get("hexColor", "").upper()
                        if hex_color == "#9C774B" or analysis.get("classifiedAs") == "brown":
                            brown_analysis = analysis
                            break
                    
                    if brown_analysis:
                        psychology_themes = brown_analysis.get("psychology", [])
                        confidence = brown_analysis.get("confidence", 0)
                        
                        print(f"      Color: {brown_analysis.get('hexColor')}")
                        print(f"      Classified: {brown_analysis.get('classifiedAs')}")
                        print(f"      Psychology: {psychology_themes}")
                        print(f"      Confidence: {confidence:.2f}")
                        
                        # Check for expected themes
                        expected = ["Earth", "Stability", "Natural"]
                        found = [theme for theme in expected if theme in psychology_themes]
                        
                        print(f"      Expected themes found: {found}")
                        
                        if len(found) >= 2:
                            print(f"      🎯 Great match for Australia's Outback theme!")
                        else:
                            print(f"      ⚠️  Partial match for Outback theme")
                    
                    # Check overall palette
                    palette = psychology.get("paletteAnalysis", {})
                    if palette:
                        mood = palette.get("overallMood", "")
                        family = palette.get("colorFamily", "")
                        print(f"      Overall mood: {mood}")
                        print(f"      Color family: {family}")
                        
                        if "earth" in family.lower() or "grounded" in mood.lower():
                            print(f"      🇦🇺 Perfect match for Australia's earthy character!")
                    
                    return True
                else:
                    error = psychology.get("error", "Unknown")
                    print(f"   ❌ Psychology failed: {error}")
                    return False
            else:
                print(f"   ❌ API call failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Australia test failed: {e}")
            return False
    
    def test_new_endpoints(self):
        """Test the new psychology endpoints"""
        print("\\n6. 🆕 Testing New Psychology Endpoints...")
        
        # Test psychology analysis endpoint
        print("\\n   6a. Testing single color psychology analysis...")
        try:
            payload = {"hexColor": "#9c774b", "culturalContext": "universal"}
            response = self.session.post(f"{self.base_url}/analyze-color-psychology", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"      ✅ Color: {result.get('hexColor')}")
                print(f"         Classified: {result.get('classifiedAs')}")
                print(f"         Psychology: {result.get('psychology', [])}")
            else:
                print(f"      ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"      ❌ Error: {e}")
        
        # Test cultural comparison endpoint
        print("\\n   6b. Testing cultural comparison...")
        try:
            payload = {"hexColor": "#ff0000"}
            response = self.session.post(f"{self.base_url}/compare-cultural-color-meanings", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"      ✅ Color: {result.get('hexColor')}")
                
                all_assoc = result.get('allAssociations', {})
                for context, themes in all_assoc.items():
                    print(f"         {context}: {themes}")
                
                conflicts = result.get('culturalConflicts', [])
                if conflicts:
                    print(f"         Conflicts: {conflicts}")
            else:
                print(f"      ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    def run_complete_test(self):
        """Run all enhanced color extraction tests"""
        print("🧪 TESTING ENHANCED COLOR EXTRACTION - STEP 3 (SIMPLIFIED)")
        print("=" * 70)
        
        # Test 1: Server availability
        print("\\n0. 🌐 Checking server availability...")
        if not self.test_server_availability():
            print("❌ Server not available. Make sure API is running on localhost:8008")
            return False
        print("✅ Server is running")
        
        # Test 2: Direct function import
        func_success = self.test_direct_function_import()
        
        # Test 3: API backward compatibility
        compat_success, basic_colors = self.test_api_backward_compatibility()
        
        # Test 4: Psychology enhancement
        psych_success, enhanced_result = self.test_psychology_enhancement_api()
        
        # Test 5: Cultural contexts
        cultural_success = self.test_cultural_contexts_api()
        
        # Test 6: Australia example
        australia_success = self.test_australia_brown_example()
        
        # Test 7: New endpoints
        self.test_new_endpoints()
        
        # Calculate success rate
        tests = [func_success, compat_success, psych_success, cultural_success, australia_success]
        success_rate = sum(tests) / len(tests) * 100
        
        # Summary
        print(f"\\n📊 STEP 3 TEST RESULTS:")
        print(f"   • Direct function: {'✅' if func_success else '❌'}")
        print(f"   • Backward compatibility: {'✅' if compat_success else '❌'}")
        print(f"   • Psychology enhancement: {'✅' if psych_success else '❌'}")
        print(f"   • Cultural contexts: {'✅' if cultural_success else '❌'}")
        print(f"   • Australia example: {'✅' if australia_success else '❌'}")
        print(f"\\n🎯 Success rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print(f"\\n🎉 STEP 3 SUCCESSFUL!")
            print(f"✅ Enhanced color extraction working")
            print(f"✅ Psychology integration complete")
            print(f"✅ Single service approach working")
            print(f"🚀 Ready for Step 4: Country Analysis Enhancement")
            return True
        else:
            print(f"\\n⚠️  Step 3 needs attention ({success_rate:.1f}% success)")
            return False

if __name__ == "__main__":
    tester = EnhancedColorExtractionTester()
    success = tester.run_complete_test()
    
    if success:
        print("\\n🎯 NEXT STEPS:")
        print("   • Step 3 Complete ✅")
        print("   • Enhanced single service working")
        print("   • Psychology integrated into color extraction")
        print("   • API endpoints enhanced")
        print("   • Ready for Step 4: Country Analysis Enhancement")
    else:
        print("\\n❌ Step 3 needs fixes. Check error messages above.")