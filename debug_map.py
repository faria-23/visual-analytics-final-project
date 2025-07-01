# #!/usr/bin/env python3
# """
# Updated Complete Debug Script - Tests both backend images scenarios
# Usage: python debug_complete_updated.py
# """

# import requests
# import json
# import base64
# import io
# from PIL import Image
# import time
# from collections import Counter

# class UpdatedWorldMapDebugger:
#     def __init__(self, base_url="http://localhost:8008"):
#         self.base_url = base_url
#         self.session = requests.Session()
#         self.session.timeout = 60
        
#     def test_server_connectivity(self):
#         """Test if the server is running"""
#         print("🔗 Testing server connectivity...")
        
#         try:
#             response = self.session.get(f"{self.base_url}/docs", timeout=5)
#             if response.status_code == 200:
#                 print("✅ Server is running and accessible")
#                 return True
#             else:
#                 print(f"⚠️  Server responded with status {response.status_code}")
#                 return False
#         except requests.exceptions.RequestException as e:
#             print(f"❌ Cannot connect to server: {e}")
#             print("💡 Make sure to run: python app.py")
#             return False
    
#     def test_backend_images_endpoint(self):
#         """Test the backend images endpoint and analyze country detection"""
#         print("\n📸 Testing backend images endpoint...")
        
#         try:
#             response = self.session.get(f"{self.base_url}/list-backend-images")
            
#             if response.status_code == 200:
#                 images = response.json()
#                 print(f"✅ Found {len(images)} backend images")
                
#                 if len(images) == 0:
#                     print("⚠️  No backend images found")
#                     return None
                
#                 # Analyze country detection in backend images
#                 countries_detected = 0
#                 country_distribution = Counter()
                
#                 print(f"\n📋 Backend images analysis:")
#                 for i, img in enumerate(images[:10]):  # Show first 10
#                     filename = img.get('filename', 'Unknown')
#                     country_name = img.get('countryName', 'None')
#                     country_code = img.get('countryCode', 'None')
                    
#                     if country_name and country_name != 'Unknown' and not country_name.startswith('Unknown'):
#                         countries_detected += 1
#                         country_distribution[country_name] += 1
#                         status = "✅"
#                     else:
#                         status = "❌"
                    
#                     print(f"   {i+1:2d}. {filename} → {country_name} ({country_code}) {status}")
                
#                 print(f"\n📊 Country distribution in backend images:")
#                 for country, count in country_distribution.most_common():
#                     print(f"   {country}: {count} images")
                
#                 print(f"\n📈 Summary:")
#                 print(f"   Total images: {len(images)}")
#                 print(f"   With countries: {countries_detected}")
#                 print(f"   Without countries: {len(images) - countries_detected}")
#                 print(f"   Success rate: {countries_detected/len(images)*100:.1f}%")
                
#                 return images
                
#             else:
#                 print(f"❌ Backend images endpoint failed: {response.status_code}")
#                 return None
                
#         except Exception as e:
#             print(f"❌ Error testing backend images: {e}")
#             return None
    
#     def test_clustering_without_metadata(self, backend_images):
#         """Test clustering with backend images WITHOUT metadata (current broken method)"""
#         print(f"\n🧪 TEST 1: Clustering WITHOUT metadata (current method)")
#         print("-" * 50)
        
#         if not backend_images or len(backend_images) == 0:
#             print("❌ No backend images available")
#             return None
        
#         # Select sample images
#         sample_size = min(15, len(backend_images))
#         sample_images = backend_images[:sample_size]
        
#         print(f"📝 Using {sample_size} backend images:")
#         expected_countries = []
#         for i, img in enumerate(sample_images):
#             filename = img.get('filename', 'Unknown')
#             country = img.get('countryName', 'Unknown')
#             expected_countries.append(country)
#             print(f"   {i+1:2d}. {filename} → {country}")
        
#         # Create clustering request - ONLY data URIs (like current frontend)
#         clustering_request = {
#             "imageUrls": [img['dataUri'] for img in sample_images],
#             "gridSize": 5,
#             "clusteringStrategy": "enhanced_features",
#             "numberOfColors": 5,
#             "clusteringMode": "hybrid",
#             "numClusters": 0  # Auto
#         }
        
#         print(f"\n🔄 Running clustering WITHOUT metadata...")
#         print(f"   Request contains: {len(clustering_request['imageUrls'])} data URIs only")
#         print(f"   Expected countries: {len(set([c for c in expected_countries if c != 'Unknown']))}")
        
#         try:
#             start_time = time.time()
#             response = self.session.post(
#                 f"{self.base_url}/cluster-postcard-images",
#                 json=clustering_request
#             )
#             end_time = time.time()
            
#             if response.status_code == 200:
#                 data = response.json()
#                 world_summary = data.get('worldMapSummary', {})
                
#                 print(f"✅ Clustering completed in {end_time - start_time:.2f}s")
#                 print(f"\n📊 Results WITHOUT metadata:")
#                 print(f"   Countries detected: {world_summary.get('totalCountriesWithData', 0)}")
#                 print(f"   Available countries: {world_summary.get('availableCountries', [])}")
#                 print(f"   Country colors: {len(world_summary.get('countryColors', {}))}")
#                 print(f"   Has world map data: {world_summary.get('hasWorldMapData', False)}")
                
#                 if world_summary.get('totalCountriesWithData', 0) == 0:
#                     print(f"❌ PROBLEM: No countries detected despite using backend images!")
#                     print(f"   This is why your frontend shows 'Images Analyzed: 0'")
#                 else:
#                     print(f"✅ Countries detected successfully")
                
#                 return data
#             else:
#                 print(f"❌ Clustering failed: {response.status_code}")
#                 print(f"Response: {response.text[:300]}")
#                 return None
                
#         except Exception as e:
#             print(f"❌ Error during clustering: {e}")
#             return None
    
#     def test_clustering_with_metadata(self, backend_images):
#         """Test clustering with backend images WITH metadata (fixed method)"""
#         print(f"\n🧪 TEST 2: Clustering WITH metadata (fixed method)")
#         print("-" * 50)
        
#         if not backend_images or len(backend_images) == 0:
#             print("❌ No backend images available")
#             return None
        
#         # Select sample images
#         sample_size = min(15, len(backend_images))
#         sample_images = backend_images[:sample_size]
        
#         print(f"📝 Using {sample_size} backend images with metadata:")
#         for i, img in enumerate(sample_images):
#             filename = img.get('filename', 'Unknown')
#             country = img.get('countryName', 'Unknown')
#             code = img.get('countryCode', 'None')
#             print(f"   {i+1:2d}. {filename} → {country} ({code})")
        
#         # Create clustering request WITH metadata
#         clustering_request = {
#             "imageUrls": [img['dataUri'] for img in sample_images],
#             "imageMetadata": [
#                 {
#                     "dataUri": img['dataUri'],
#                     "filename": img.get('filename', ''),
#                     "countryCode": img.get('countryCode', ''),
#                     "countryName": img.get('countryName', '')
#                 }
#                 for img in sample_images
#             ],
#             "gridSize": 5,
#             "clusteringStrategy": "enhanced_features",
#             "numberOfColors": 5,
#             "clusteringMode": "hybrid",
#             "numClusters": 0
#         }
        
#         print(f"\n🔄 Running clustering WITH metadata...")
#         print(f"   Request contains: {len(clustering_request['imageUrls'])} data URIs + metadata")
#         print(f"   Metadata includes: filename, countryCode, countryName for each image")
        
#         try:
#             start_time = time.time()
#             response = self.session.post(
#                 f"{self.base_url}/cluster-postcard-images",
#                 json=clustering_request
#             )
#             end_time = time.time()
            
#             if response.status_code == 200:
#                 data = response.json()
#                 world_summary = data.get('worldMapSummary', {})
                
#                 print(f"✅ Clustering completed in {end_time - start_time:.2f}s")
#                 print(f"\n📊 Results WITH metadata:")
#                 print(f"   Countries detected: {world_summary.get('totalCountriesWithData', 0)}")
#                 print(f"   Available countries: {world_summary.get('availableCountries', [])}")
#                 print(f"   Country colors: {len(world_summary.get('countryColors', {}))}")
#                 print(f"   Has world map data: {world_summary.get('hasWorldMapData', False)}")
                
#                 # Show detailed country colors
#                 country_colors = world_summary.get('countryColors', {})
#                 if country_colors:
#                     print(f"\n🎨 Country colors detected:")
#                     for country_code, colors in country_colors.items():
#                         dominant = colors.get('dominant', 'N/A')
#                         family = colors.get('family', 'N/A')
#                         total_colors = colors.get('totalColors', 0)
#                         temp = colors.get('temperature', 0)
#                         temp_desc = "Cool" if temp < 0.4 else "Warm" if temp > 0.6 else "Neutral"
                        
#                         print(f"     {country_code}: {dominant} ({family}, {temp_desc})")
#                         print(f"        Images: {world_summary.get('countryImageCounts', {}).get(country_code, 0)}")
                
#                 if world_summary.get('totalCountriesWithData', 0) > 0:
#                     print(f"🎉 SUCCESS: Countries detected with metadata!")
#                     print(f"   This should fix your frontend 'Images Analyzed: 0' issue")
#                 else:
#                     print(f"❌ Still no countries detected - check backend metadata handling")
                
#                 return data
#             else:
#                 print(f"❌ Clustering with metadata failed: {response.status_code}")
#                 print(f"Response: {response.text[:300]}")
#                 return None
                
#         except Exception as e:
#             print(f"❌ Error during clustering with metadata: {e}")
#             return None
    
#     def test_detailed_world_map_endpoint(self):
#         """Test the detailed world map endpoint"""
#         print(f"\n🌍 Testing detailed world map endpoint...")
        
#         try:
#             response = self.session.get(f"{self.base_url}/visualization-data/world-map-detailed")
            
#             if response.status_code == 200:
#                 data = response.json()
#                 countries = data.get('countries', {})
#                 insights = data.get('culturalInsights', [])
                
#                 print(f"✅ Detailed world map endpoint working")
#                 print(f"   Total countries: {len(countries)}")
#                 print(f"   Cultural insights: {len(insights)}")
                
#                 # Show sample countries
#                 sample_countries = ['DE', 'FR', 'IT', 'ES', 'AT']
#                 available_countries = [c for c in sample_countries if c in countries]
                
#                 if available_countries:
#                     print(f"\n🏳️ Sample countries in detailed data:")
#                     for country_code in available_countries[:5]:
#                         country = countries[country_code]
#                         print(f"   {country_code}: {country.get('fullName', 'Unknown')}")
#                         print(f"      Continent: {country.get('continent', 'Unknown')}")
#                         print(f"      Map position: ({country.get('mapPosition', {}).get('x', '?')}, {country.get('mapPosition', {}).get('y', '?')})")
                
#                 return data
#             else:
#                 print(f"❌ Detailed world map failed: {response.status_code}")
#                 return None
                
#         except Exception as e:
#             print(f"❌ Error testing detailed world map: {e}")
#             return None
    
#     def compare_results(self, result_without_metadata, result_with_metadata):
#         """Compare results between the two methods"""
#         print(f"\n⚔️ COMPARISON: Without vs With Metadata")
#         print("="*60)
        
#         if result_without_metadata:
#             without_summary = result_without_metadata.get('worldMapSummary', {})
#             without_countries = without_summary.get('totalCountriesWithData', 0)
#             without_colors = len(without_summary.get('countryColors', {}))
#         else:
#             without_countries = 0
#             without_colors = 0
        
#         if result_with_metadata:
#             with_summary = result_with_metadata.get('worldMapSummary', {})
#             with_countries = with_summary.get('totalCountriesWithData', 0)
#             with_colors = len(with_summary.get('countryColors', {}))
#         else:
#             with_countries = 0
#             with_colors = 0
        
#         print(f"📊 Results Comparison:")
#         print(f"   Method                    | Countries | Colors")
#         print(f"   -------------------------|-----------|--------")
#         print(f"   Without Metadata (broken) |    {without_countries:2d}     |   {without_colors:2d}")
#         print(f"   With Metadata (fixed)     |    {with_countries:2d}     |   {with_colors:2d}")
        
#         if with_countries > without_countries:
#             print(f"\n✅ SUCCESS: Metadata method detects {with_countries - without_countries} more countries!")
#             print(f"   This proves the fix works")
#         elif with_countries == without_countries == 0:
#             print(f"\n❌ PROBLEM: Both methods detect 0 countries")
#             print(f"   Check backend imageMetadata handling")
#         else:
#             print(f"\n⚠️ Unexpected result - investigate further")
    
#     def generate_frontend_fix_code(self, backend_images):
#         """Generate the exact frontend code needed to fix the issue"""
#         print(f"\n💻 FRONTEND FIX CODE")
#         print("="*60)
        
#         if not backend_images:
#             print("❌ No backend images available to generate fix code")
#             return
        
#         print("🔧 Replace your current frontend code with this:")
#         print()
#         print("```javascript")
#         print("// OLD CODE (broken):")
#         print("// const backendImages = await fetch('/list-backend-images').then(r => r.json());")
#         print("// const imageUrls = backendImages.map(img => img.dataUri);")
#         print("// const clusteringRequest = { imageUrls, gridSize: 5 };")
#         print("")
#         print("// NEW CODE (fixed):")
#         print("async function loadBackendImagesWithMetadata() {")
#         print("  const backendImages = await fetch('/list-backend-images').then(r => r.json());")
#         print("  ")
#         print("  console.log(`Loaded ${backendImages.length} backend images`);")
#         print("  ")
#         print("  // Show what countries are available")
#         print("  const countries = backendImages")
#         print("    .filter(img => img.countryName && img.countryName !== 'Unknown')")
#         print("    .map(img => img.countryName);")
#         print("  console.log('Available countries:', [...new Set(countries)]);")
#         print("  ")
#         print("  return backendImages;")
#         print("}")
#         print("")
#         print("async function clusterWithBackendImages() {")
#         print("  const backendImages = await loadBackendImagesWithMetadata();")
#         print("  ")
#         print("  // Use first 20 images for faster clustering")
#         print("  const sampleImages = backendImages.slice(0, 20);")
#         print("  ")
#         print("  const clusteringRequest = {")
#         print("    imageUrls: sampleImages.map(img => img.dataUri),")
#         print("    ")
#         print("    // ✅ CRITICAL: Include metadata for country detection")
#         print("    imageMetadata: sampleImages.map(img => ({")
#         print("      dataUri: img.dataUri,")
#         print("      filename: img.filename,")
#         print("      countryCode: img.countryCode,")
#         print("      countryName: img.countryName")
#         print("    })),")
#         print("    ")
#         print("    gridSize: 5,")
#         print("    clusteringStrategy: 'enhanced_features',")
#         print("    numberOfColors: 5,")
#         print("    clusteringMode: 'hybrid'")
#         print("  };")
#         print("  ")
#         print("  console.log('Sending clustering request with metadata...');")
#         print("  ")
#         print("  const response = await fetch('/cluster-postcard-images', {")
#         print("    method: 'POST',")
#         print("    headers: { 'Content-Type': 'application/json' },")
#         print("    body: JSON.stringify(clusteringRequest)")
#         print("  });")
#         print("  ")
#         print("  const data = await response.json();")
#         print("  ")
#         print("  // Check world map results")
#         print("  const worldMap = data.worldMapSummary;")
#         print("  console.log('🗺️ World Map Results:');")
#         print("  console.log(`Countries detected: ${worldMap.totalCountriesWithData}`);")
#         print("  console.log('Country colors:', worldMap.countryColors);")
#         print("  ")
#         print("  return data;")
#         print("}")
#         print("")
#         print("// Usage:")
#         print("clusterWithBackendImages().then(result => {")
#         print("  console.log('Clustering complete!', result);")
#         print("});")
#         print("```")
        
#         # Show sample data structure
#         sample_image = backend_images[0] if backend_images else None
#         if sample_image:
#             print(f"\n📋 Sample backend image structure:")
#             print("```json")
#             print(json.dumps({
#                 "filename": sample_image.get('filename', 'Unknown'),
#                 "countryCode": sample_image.get('countryCode', 'None'),
#                 "countryName": sample_image.get('countryName', 'None'),
#                 "dataUri": "[TRUNCATED]",
#                 "dataUriLength": len(sample_image.get('dataUri', ''))            }, indent=2))
#             print("```")
    
#     def run_complete_diagnosis(self):
#         """Run the complete updated diagnosis"""
#         print("🔧 UPDATED WORLD MAP COMPLETE DIAGNOSTIC")
#         print("="*60)
#         print("Testing backend images country detection in clustering")
#         print("="*60)
        
#         # Test 1: Server connectivity
#         if not self.test_server_connectivity():
#             return {"error": "Server not running"}
        
#         # Test 2: Backend images endpoint
#         backend_images = self.test_backend_images_endpoint()
#         if not backend_images:
#             return {"error": "No backend images available"}
        
#         # Test 3: Clustering without metadata (current broken method)
#         result_without_metadata = self.test_clustering_without_metadata(backend_images)
        
#         # Test 4: Clustering with metadata (fixed method)
#         result_with_metadata = self.test_clustering_with_metadata(backend_images)
        
#         # Test 5: Detailed world map
#         detailed_result = self.test_detailed_world_map_endpoint()
        
#         # Test 6: Compare results
#         self.compare_results(result_without_metadata, result_with_metadata)
        
#         # Test 7: Generate frontend fix
#         self.generate_frontend_fix_code(backend_images)
        
#         # Final diagnosis
#         print(f"\n" + "="*60)
#         print("🎯 FINAL DIAGNOSIS")
#         print("="*60)
        
#         if result_with_metadata:
#             with_summary = result_with_metadata.get('worldMapSummary', {})
#             countries_detected = with_summary.get('totalCountriesWithData', 0)
            
#             if countries_detected > 0:
#                 print("✅ SOLUTION CONFIRMED: Backend and world map system working correctly")
#                 print(f"✅ Countries detected with metadata: {countries_detected}")
#                 print("✅ Frontend fix: Send imageMetadata with clustering requests")
#                 print()
#                 print("🔧 Next steps:")
#                 print("1. Update your frontend to include imageMetadata in clustering requests")
#                 print("2. Your frontend should show 'Images Analyzed: X' instead of 'Images Analyzed: 0'")
#                 print("3. World map visualization should display country data correctly")
#             else:
#                 print("❌ PROBLEM: Backend not handling imageMetadata correctly")
#                 print("🔧 Next steps:")
#                 print("1. Check if backend imageMetadata support is properly implemented")
#                 print("2. Verify app.py has updated ClusterImagesInput model")
#                 print("3. Check console logs for metadata processing errors")
#         else:
#             print("❌ PROBLEM: Clustering with metadata failed")
#             print("🔧 Next steps:")
#             print("1. Check server logs for errors")
#             print("2. Verify backend supports imageMetadata parameter")
#             print("3. Test individual endpoints separately")
        
#         print("="*60)
        
#         return {
#             "backend_images_count": len(backend_images) if backend_images else 0,
#             "without_metadata_countries": result_without_metadata.get('worldMapSummary', {}).get('totalCountriesWithData', 0) if result_without_metadata else 0,
#             "with_metadata_countries": result_with_metadata.get('worldMapSummary', {}).get('totalCountriesWithData', 0) if result_with_metadata else 0,
#             "fix_works": (result_with_metadata.get('worldMapSummary', {}).get('totalCountriesWithData', 0) > 0) if result_with_metadata else False
#         }

# def main():
#     """Main function"""
#     debugger = UpdatedWorldMapDebugger()
    
#     try:
#         results = debugger.run_complete_diagnosis()
        
#         print(f"\n✅ DIAGNOSIS COMPLETE")
#         print("Copy the frontend fix code above and implement it to resolve the issue.")
        
#         return results
        
#     except KeyboardInterrupt:
#         print("\n⚠️ Diagnosis interrupted by user")
#         return None
#     except Exception as e:
#         print(f"\n❌ Unexpected error during diagnosis: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# if __name__ == "__main__":
#     main()