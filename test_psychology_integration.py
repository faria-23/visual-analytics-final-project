#!/usr/bin/env python3
"""
Phase 1 Psychology Integration Test Script
Tests enhanced world map service with color psychology analysis

Usage:
    python test_phase1_psychology.py

Tests:
    1. Enhanced world map service initialization
    2. Clustering with psychology enabled/disabled  
    3. Country psychology endpoints
    4. Psychology group generation
    5. Backward compatibility
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
import base64
from PIL import Image
from io import BytesIO

class Phase1PsychologyTester:
    """Comprehensive tester for Phase 1 psychology integration"""
    
    def __init__(self, base_url: str = "http://localhost:8008"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {
            "initialization": {},
            "clustering_psychology": {},
            "country_endpoints": {},
            "backward_compatibility": {},
            "psychology_groups": {},
            "summary": {}
        }
        
    def run_all_tests(self) -> Dict:
        """Run complete Phase 1 test suite"""
        print("🧪 PHASE 1 PSYCHOLOGY INTEGRATION TEST SUITE")
        print("=" * 60)
        
        try:
            # Test 1: Service initialization and status
            print("\n1️⃣ Testing Service Initialization...")
            self.test_service_initialization()
            
            # Test 2: Enhanced country statistics  
            print("\n2️⃣ Testing Enhanced Country Statistics...")
            self.test_enhanced_country_statistics()
            
            # Test 3: Backend images and setup
            print("\n3️⃣ Testing Backend Images Setup...")
            backend_images = self.test_backend_images()
            
            if not backend_images:
                print("⚠️ No backend images found. Creating test images...")
                backend_images = self.create_test_images()
            
            # Test 4: Clustering without psychology (backward compatibility)
            print("\n4️⃣ Testing Clustering WITHOUT Psychology...")
            self.test_clustering_without_psychology(backend_images)
            
            # Test 5: Clustering with psychology enabled
            print("\n5️⃣ Testing Clustering WITH Psychology...")
            self.test_clustering_with_psychology(backend_images)
            
            # Test 6: Country psychology endpoints
            print("\n6️⃣ Testing Country Psychology Endpoints...")
            self.test_country_psychology_endpoints()
            
            # Test 7: Psychology groups generation
            print("\n7️⃣ Testing Psychology Groups Generation...")
            self.test_psychology_groups_generation(backend_images)
            
            # Generate final report
            print("\n📊 Generating Test Report...")
            self.generate_test_report()
            
            return self.test_results
            
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "results": self.test_results}
    
    def test_service_initialization(self):
        """Test service initialization and psychology availability"""
        try:
            # Check if server is running
            response = self.session.get(f"{self.base_url}/docs")
            if response.status_code != 200:
                raise Exception(f"Server not running on {self.base_url}")
            
            print("✅ Server is running")
            
            # Test country statistics to check psychology integration status
            response = self.session.get(f"{self.base_url}/country-statistics")
            if response.status_code == 200:
                data = response.json()
                psychology_integration = data.get("psychologyIntegration", {})
                
                self.test_results["initialization"] = {
                    "server_running": True,
                    "psychology_available": psychology_integration.get("available", False),
                    "world_map_psychology_enabled": psychology_integration.get("worldMapPsychologyEnabled", False),
                    "supported_contexts": psychology_integration.get("supportedCulturalContexts", [])
                }
                
                print(f"✅ Psychology Service Available: {psychology_integration.get('available', False)}")
                print(f"✅ World Map Psychology: {psychology_integration.get('worldMapPsychologyEnabled', False)}")
                print(f"✅ Cultural Contexts: {psychology_integration.get('supportedCulturalContexts', [])}")
                
            else:
                raise Exception(f"Country statistics endpoint failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Service initialization test failed: {e}")
            self.test_results["initialization"] = {"error": str(e)}
    
    def test_enhanced_country_statistics(self):
        """Test enhanced country statistics endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/country-statistics")
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for enhanced fields
                required_fields = [
                    "countryStatistics", "totalCountries", "totalImages",
                    "identifiedCountryImages", "unknownImages", "psychologyIntegration"
                ]
                
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    raise Exception(f"Missing enhanced fields: {missing_fields}")
                
                # Check psychology integration details
                psych_integration = data["psychologyIntegration"]
                required_psych_fields = ["available", "worldMapPsychologyEnabled", "supportedCulturalContexts"]
                
                missing_psych_fields = [field for field in required_psych_fields if field not in psych_integration]
                
                if missing_psych_fields:
                    raise Exception(f"Missing psychology integration fields: {missing_psych_fields}")
                
                self.test_results["country_endpoints"]["enhanced_statistics"] = {
                    "success": True,
                    "total_countries": data["totalCountries"],
                    "total_images": data["totalImages"],
                    "psychology_available": psych_integration["available"],
                    "supported_contexts": len(psych_integration["supportedCulturalContexts"])
                }
                
                print(f"✅ Enhanced country statistics working")
                print(f"   Countries: {data['totalCountries']}, Images: {data['totalImages']}")
                print(f"   Psychology: {psych_integration['available']}")
                
            else:
                raise Exception(f"Enhanced country statistics failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Enhanced country statistics test failed: {e}")
            self.test_results["country_endpoints"]["enhanced_statistics"] = {"error": str(e)}
    
    def test_backend_images(self) -> Optional[List[Dict]]:
        """Test backend images availability"""
        try:
            response = self.session.get(f"{self.base_url}/list-backend-images")
            
            if response.status_code == 200:
                backend_images = response.json()
                
                if len(backend_images) == 0:
                    print("⚠️ No backend images found")
                    return None
                
                print(f"✅ Found {len(backend_images)} backend images")
                
                # Show country distribution
                countries = [img.get("countryCode", "Unknown") for img in backend_images]
                country_counts = {}
                for country in countries:
                    country_counts[country] = country_counts.get(country, 0) + 1
                
                print("📊 Country distribution:")
                for country, count in sorted(country_counts.items()):
                    print(f"   {country}: {count} images")
                
                return backend_images[:10]  # Use first 10 images for testing
                
            else:
                print(f"❌ Backend images endpoint failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Backend images test failed: {e}")
            return None
    
    def create_test_images(self) -> List[Dict]:
        """Create test images if no backend images available"""
        try:
            print("🎨 Creating test images with different colors...")
            
            test_images = []
            test_colors = [
                ("#FF5733", "red", "US"),      # Red for US
                ("#33FF57", "green", "BR"),    # Green for Brazil  
                ("#3357FF", "blue", "DE"),     # Blue for Germany
                ("#FFD700", "yellow", "AU"),   # Gold for Australia
                ("#9966CC", "purple", "JP")    # Purple for Japan
            ]
            
            for i, (color, color_name, country) in enumerate(test_colors):
                # Create simple colored image
                img = Image.new('RGB', (100, 100), color)
                
                # Convert to data URI
                buffer = BytesIO()
                img.save(buffer, format='JPEG')
                img_data = base64.b64encode(buffer.getvalue()).decode()
                data_uri = f"data:image/jpeg;base64,{img_data}"
                
                test_images.append({
                    "id": f"test_{country}_{i}",
                    "dataUri": data_uri,
                    "countryCode": country,
                    "countryName": f"Test Country {country}",
                    "filename": f"test_{country}_{color_name}.jpg"
                })
            
            print(f"✅ Created {len(test_images)} test images")
            return test_images
            
        except Exception as e:
            print(f"❌ Failed to create test images: {e}")
            return []
    
    def test_clustering_without_psychology(self, images: List[Dict]):
        """Test clustering without psychology (backward compatibility)"""
        try:
            if not images:
                print("⚠️ No images available for clustering test")
                return
            
            clustering_request = {
                "imageUrls": [img["dataUri"] for img in images[:5]],
                "gridSize": 3,
                "clusteringStrategy": "enhanced_features",
                "numberOfColors": 5,
                "includePsychology": False  # Explicitly disable psychology
            }
            
            print(f"🔄 Testing clustering WITHOUT psychology ({len(clustering_request['imageUrls'])} images)...")
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/cluster-postcard-images",
                json=clustering_request
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                world_summary = data.get("worldMapSummary", {})
                
                self.test_results["backward_compatibility"] = {
                    "success": True,
                    "response_time": round(end_time - start_time, 2),
                    "clusters_generated": len(data.get("clusters", [])),
                    "countries_detected": world_summary.get("totalCountriesWithData", 0),
                    "psychology_enabled": world_summary.get("psychologyEnabled", False),
                    "psychology_groups": len(world_summary.get("psychologyGroups", []))
                }
                
                print(f"✅ Clustering without psychology completed in {end_time - start_time:.2f}s")
                print(f"   Clusters: {len(data.get('clusters', []))}")
                print(f"   Countries: {world_summary.get('totalCountriesWithData', 0)}")
                print(f"   Psychology enabled: {world_summary.get('psychologyEnabled', False)}")
                
                # Verify psychology is disabled
                if world_summary.get("psychologyEnabled", False):
                    print("⚠️ WARNING: Psychology should be disabled but appears enabled")
                else:
                    print("✅ Psychology correctly disabled")
                
            else:
                raise Exception(f"Clustering without psychology failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Clustering without psychology test failed: {e}")
            self.test_results["backward_compatibility"] = {"error": str(e)}
    
    def test_clustering_with_psychology(self, images: List[Dict]):
        """Test clustering with psychology enabled"""
        try:
            if not images:
                print("⚠️ No images available for psychology clustering test")
                return
            
            clustering_request = {
                "imageUrls": [img["dataUri"] for img in images],
                "gridSize": 3,
                "clusteringStrategy": "enhanced_features", 
                "numberOfColors": 5,
                "includePsychology": True,  # Enable psychology
                "culturalContext": "universal",
                "psychologyConfidenceThreshold": 0.5
            }
            
            print(f"🔄 Testing clustering WITH psychology ({len(clustering_request['imageUrls'])} images)...")
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/cluster-postcard-images",
                json=clustering_request
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                world_summary = data.get("worldMapSummary", {})
                clusters = data.get("clusters", [])
                
                # Analyze psychology integration
                psychology_enabled = world_summary.get("psychologyEnabled", False)
                psychology_groups = world_summary.get("psychologyGroups", [])
                country_colors = world_summary.get("countryColors", {})
                
                # Check for psychology data in country colors
                countries_with_psychology = 0
                for country_code, color_data in country_colors.items():
                    if "psychologyProfile" in color_data or "dominantThemes" in color_data:
                        countries_with_psychology += 1
                
                self.test_results["clustering_psychology"] = {
                    "success": True,
                    "response_time": round(end_time - start_time, 2),
                    "clusters_generated": len(clusters),
                    "countries_detected": world_summary.get("totalCountriesWithData", 0),
                    "psychology_enabled": psychology_enabled,
                    "psychology_groups": len(psychology_groups),
                    "countries_with_psychology": countries_with_psychology,
                    "cultural_context": clustering_request["culturalContext"]
                }
                
                print(f"✅ Clustering with psychology completed in {end_time - start_time:.2f}s")
                print(f"   Clusters: {len(clusters)}")
                print(f"   Countries: {world_summary.get('totalCountriesWithData', 0)}")
                print(f"   Psychology enabled: {psychology_enabled}")
                print(f"   Psychology groups: {len(psychology_groups)}")
                print(f"   Countries with psychology: {countries_with_psychology}")
                
                # Show psychology groups details
                if psychology_groups:
                    print(f"\n🧠 Psychology Groups Generated:")
                    for i, group in enumerate(psychology_groups[:3]):  # Show first 3 groups
                        print(f"   Group {i+1}: {group.get('sharedThemes', [])} -> {group.get('countries', [])}")
                
                # Show sample country psychology
                self.show_sample_country_psychology(country_colors)
                
            else:
                raise Exception(f"Clustering with psychology failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Clustering with psychology test failed: {e}")
            self.test_results["clustering_psychology"] = {"error": str(e)}
    
    def show_sample_country_psychology(self, country_colors: Dict):
        """Show sample country psychology data"""
        try:
            print(f"\n🎨 Sample Country Psychology Analysis:")
            
            sample_count = 0
            for country_code, color_data in country_colors.items():
                if sample_count >= 3:  # Show max 3 samples
                    break
                
                psychology_profile = color_data.get("psychologyProfile", [])
                dominant_themes = color_data.get("dominantThemes", [])
                cultural_insights = color_data.get("culturalInsights", [])
                
                if psychology_profile or dominant_themes:
                    print(f"\n   🏛️ {country_code}:")
                    print(f"      Dominant Color: {color_data.get('dominant', 'N/A')}")
                    print(f"      Color Family: {color_data.get('family', 'N/A')}")
                    print(f"      Temperature: {color_data.get('temperatureDescription', 'N/A')}")
                    
                    if dominant_themes:
                        print(f"      Psychology Themes: {', '.join(dominant_themes[:3])}")
                    
                    if cultural_insights:
                        print(f"      Cultural Insights: {cultural_insights[0] if cultural_insights else 'None'}")
                    
                    sample_count += 1
            
            if sample_count == 0:
                print("   ⚠️ No countries with psychology data found")
                
        except Exception as e:
            print(f"   ❌ Error showing sample psychology: {e}")
    
    def test_country_psychology_endpoints(self):
        """Test individual country psychology endpoints"""
        try:
            test_countries = ["US", "DE", "AU", "BR", "JP"]
            
            results = {}
            for country_code in test_countries:
                try:
                    response = self.session.get(
                        f"{self.base_url}/countries/{country_code}/psychology",
                        params={"cultural_context": "universal", "confidence_threshold": 0.5}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        results[country_code] = {
                            "success": True,
                            "country_name": data.get("countryName"),
                            "analysis_status": data.get("psychologyAnalysis", {}).get("status"),
                            "requires_clustering": data.get("metadata", {}).get("requiresClusteringData", False)
                        }
                        
                        print(f"✅ {country_code} psychology endpoint working")
                        
                    elif response.status_code == 404:
                        results[country_code] = {"success": False, "error": "Country not found"}
                        print(f"⚠️ {country_code} not found (expected)")
                        
                    elif response.status_code == 503:
                        results[country_code] = {"success": False, "error": "Psychology service unavailable"}
                        print(f"⚠️ {country_code} psychology service unavailable")
                        
                    else:
                        results[country_code] = {"success": False, "error": f"HTTP {response.status_code}"}
                        print(f"❌ {country_code} failed: {response.status_code}")
                        
                except Exception as e:
                    results[country_code] = {"success": False, "error": str(e)}
                    print(f"❌ {country_code} exception: {e}")
            
            self.test_results["country_endpoints"]["individual_psychology"] = results
            
            successful_tests = sum(1 for result in results.values() if result.get("success", False))
            print(f"\n📊 Country psychology endpoints: {successful_tests}/{len(test_countries)} successful")
            
        except Exception as e:
            print(f"❌ Country psychology endpoints test failed: {e}")
            self.test_results["country_endpoints"]["individual_psychology"] = {"error": str(e)}
    
    def test_psychology_groups_generation(self, images: List[Dict]):
        """Test psychology groups generation with detailed analysis"""
        try:
            if not images:
                print("⚠️ No images for psychology groups test")
                return
            
            # Run clustering with psychology to generate groups
            clustering_request = {
                "imageUrls": [img["dataUri"] for img in images],
                "gridSize": 4,
                "includePsychology": True,
                "culturalContext": "universal",
                "psychologyConfidenceThreshold": 0.4  # Lower threshold for more groups
            }
            
            print(f"🔄 Testing psychology groups generation...")
            
            response = self.session.post(
                f"{self.base_url}/cluster-postcard-images",
                json=clustering_request
            )
            
            if response.status_code == 200:
                data = response.json()
                world_summary = data.get("worldMapSummary", {})
                psychology_groups = world_summary.get("psychologyGroups", [])
                psychology_stats = world_summary.get("psychologyStats", {})
                
                self.test_results["psychology_groups"] = {
                    "success": True,
                    "groups_generated": len(psychology_groups),
                    "countries_analyzed": psychology_stats.get("totalCountriesAnalyzed", 0),
                    "average_confidence": psychology_stats.get("averageConfidence", 0.0),
                    "unique_themes": psychology_stats.get("totalUniqueThemes", 0),
                    "most_common_themes": psychology_stats.get("mostCommonThemes", [])
                }
                
                print(f"✅ Psychology groups analysis completed")
                print(f"   Groups generated: {len(psychology_groups)}")
                print(f"   Countries analyzed: {psychology_stats.get('totalCountriesAnalyzed', 0)}")
                print(f"   Average confidence: {psychology_stats.get('averageConfidence', 0.0):.2f}")
                print(f"   Unique themes: {psychology_stats.get('totalUniqueThemes', 0)}")
                
                if psychology_stats.get("mostCommonThemes"):
                    print(f"   Most common themes: {', '.join(psychology_stats.get('mostCommonThemes', [])[:5])}")
                
                # Detailed group analysis
                if psychology_groups:
                    print(f"\n🧠 Detailed Psychology Groups:")
                    for i, group in enumerate(psychology_groups):
                        print(f"   Group {i+1}: {group.get('groupId', 'unknown')}")
                        print(f"      Shared themes: {group.get('sharedThemes', [])}")
                        print(f"      Countries: {group.get('countries', [])} ({group.get('groupSize', 0)} total)")
                        print(f"      Theme count: {group.get('themeCount', 0)}")
                        print()
                
            else:
                raise Exception(f"Psychology groups test failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Psychology groups test failed: {e}")
            self.test_results["psychology_groups"] = {"error": str(e)}
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        try:
            print("\n" + "="*60)
            print("📊 PHASE 1 PSYCHOLOGY INTEGRATION TEST REPORT")
            print("="*60)
            
            # Summary statistics
            total_tests = 0
            passed_tests = 0
            
            for category, results in self.test_results.items():
                if category == "summary":
                    continue
                    
                if isinstance(results, dict):
                    if "error" not in results:
                        for test_name, test_result in results.items():
                            total_tests += 1
                            if isinstance(test_result, dict) and test_result.get("success", False):
                                passed_tests += 1
                            elif not isinstance(test_result, dict):
                                passed_tests += 1
                    else:
                        total_tests += 1
            
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            print(f"\n📈 OVERALL RESULTS:")
            print(f"   Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
            print(f"   Phase 1 Status: {'✅ PASSED' if success_rate >= 70 else '❌ NEEDS WORK'}")
            
            # Detailed results by category
            print(f"\n📋 DETAILED RESULTS:")
            
            for category, results in self.test_results.items():
                if category == "summary":
                    continue
                    
                category_name = category.replace("_", " ").title()
                print(f"\n   🔸 {category_name}:")
                
                if isinstance(results, dict) and "error" in results:
                    print(f"      ❌ {results['error']}")
                elif isinstance(results, dict):
                    for test_name, test_result in results.items():
                        if isinstance(test_result, dict):
                            status = "✅" if test_result.get("success", False) else "❌"
                            print(f"      {status} {test_name.replace('_', ' ').title()}")
                            
                            if test_result.get("success", False):
                                # Show key metrics
                                if "response_time" in test_result:
                                    print(f"         Response time: {test_result['response_time']}s")
                                if "psychology_enabled" in test_result:
                                    print(f"         Psychology enabled: {test_result['psychology_enabled']}")
                                if "countries_with_psychology" in test_result:
                                    print(f"         Countries with psychology: {test_result['countries_with_psychology']}")
                        else:
                            print(f"      ✅ {test_name.replace('_', ' ').title()}")
            
            # Recommendations
            print(f"\n🎯 RECOMMENDATIONS:")
            
            if success_rate >= 90:
                print("   🌟 Excellent! Phase 1 is working perfectly.")
                print("   🚀 Ready to proceed to Phase 2 (Cluster Psychology Enhancement)")
                
            elif success_rate >= 70:
                print("   👍 Good! Phase 1 is mostly working.")
                print("   🔧 Minor issues to address before Phase 2")
                
            else:
                print("   ⚠️ Phase 1 needs attention before proceeding.")
                print("   🛠️ Review error messages and fix core issues")
            
            # Save detailed results
            self.test_results["summary"] = {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": success_rate,
                "timestamp": time.time(),
                "phase1_status": "PASSED" if success_rate >= 70 else "NEEDS_WORK"
            }
            
            print(f"\n💾 Detailed results saved to test_results dictionary")
            
        except Exception as e:
            print(f"❌ Error generating test report: {e}")

def main():
    """Main test function"""
    print("🚀 Starting Phase 1 Psychology Integration Tests...")
    
    tester = Phase1PsychologyTester()
    
    try:
        results = tester.run_all_tests()
        
        # Save results to file
        results_file = Path("phase1_test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Test results saved to: {results_file}")
        
        # Quick summary
        summary = results.get("summary", {})
        if summary.get("success_rate", 0) >= 70:
            print(f"\n🎉 PHASE 1 TESTING COMPLETE - SUCCESS!")
            print(f"   Ready for Phase 2 implementation")
        else:
            print(f"\n⚠️ PHASE 1 TESTING COMPLETE - NEEDS WORK")
            print(f"   Review errors before proceeding to Phase 2")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Testing interrupted by user")
        return None
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the tests
    test_results = main()
    
    # Optional: Print final status
    if test_results and test_results.get("summary", {}).get("success_rate", 0) >= 70:
        print("\n✨ Phase 1 Psychology Integration: READY FOR PRODUCTION! ✨")
    else:
        print("\n🔧 Phase 1 Psychology Integration: NEEDS DEBUGGING")