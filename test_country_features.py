#!/usr/bin/env python3
"""
Simple Country Features Test Script
Usage: python test_country_features.py
"""

import requests
import json
import time
from typing import Dict, List

class CountryClusteringTester:
    def __init__(self, base_url="http://localhost:8008"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
    
    def test_server_connection(self):
        """Test if the API server is running"""
        print("🔗 Testing server connection...")
        try:
            response = self.session.get(f"{self.base_url}/docs", timeout=5)
            if response.status_code == 200:
                print("✅ Server is running")
                return True
            else:
                print(f"⚠️ Server responded with status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            print("💡 Make sure to run: python app.py")
            return False
    
    def test_backend_images_with_countries(self):
        """Test backend images endpoint for country detection"""
        print("\n📸 Testing backend images with country detection...")
        
        try:
            response = self.session.get(f"{self.base_url}/list-backend-images")
            
            if response.status_code == 200:
                images = response.json()
                print(f"✅ Found {len(images)} backend images")
                
                if not images:
                    print("⚠️ No backend images found. Add some images to data/ folder.")
                    return None
                
                # Analyze country detection
                countries_detected = 0
                countries = {}
                
                print("\n📋 Country detection results:")
                for i, img in enumerate(images[:10]):  # Show first 10
                    filename = img.get('filename', 'Unknown')
                    country_name = img.get('countryName', 'None')
                    country_code = img.get('countryCode', 'None')
                    
                    if country_name and country_name != 'Unknown':
                        countries_detected += 1
                        countries[country_code] = country_name
                        status = "✅"
                    else:
                        status = "❌"
                    
                    print(f"   {i+1:2d}. {filename} → {country_name} ({country_code}) {status}")
                
                print(f"\n📊 Summary:")
                print(f"   Total images: {len(images)}")
                print(f"   Countries detected: {countries_detected}")
                print(f"   Unique countries: {list(countries.values())}")
                
                return images
            else:
                print(f"❌ Backend images failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def test_country_statistics(self):
        """Test country statistics endpoint"""
        print("\n📊 Testing country statistics...")
        
        try:
            response = self.session.get(f"{self.base_url}/country-statistics")
            
            if response.status_code == 200:
                stats = response.json()
                print("✅ Country statistics retrieved")
                print(f"   Total countries: {stats.get('totalCountries', 0)}")
                print(f"   Total images: {stats.get('totalImages', 0)}")
                print(f"   Identified images: {stats.get('identifiedCountryImages', 0)}")
                
                country_stats = stats.get('countryStatistics', {})
                if country_stats:
                    print("   Country breakdown:")
                    for country, count in country_stats.items():
                        print(f"     {country}: {count} images")
                
                return stats
            else:
                print(f"❌ Country statistics failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def test_clustering_with_countries(self, backend_images):
        """Test clustering with country metadata"""
        print("\n🗺️ Testing clustering with country detection...")
        
        if not backend_images:
            print("❌ No backend images available for clustering test")
            return None
        
        # Use first 8 images for faster testing
        sample_size = min(8, len(backend_images))
        sample_images = backend_images[:sample_size]
        
        print(f"📝 Using {sample_size} images for clustering test:")
        for i, img in enumerate(sample_images):
            country = img.get('countryName', 'Unknown')
            print(f"   {i+1}. {img.get('filename', 'Unknown')} → {country}")
        
        # Create clustering request with metadata
        clustering_request = {
            "imageUrls": [img['dataUri'] for img in sample_images],
            "imageMetadata": [
                {
                    "dataUri": img['dataUri'],
                    "filename": img.get('filename'),
                    "countryCode": img.get('countryCode'),
                    "countryName": img.get('countryName')
                }
                for img in sample_images
            ],
            "gridSize": 3,  # Small grid for faster testing
            "clusteringStrategy": "enhanced_features",
            "numberOfColors": 5,
            "clusteringMode": "som"
        }
        
        print(f"\n🔄 Running clustering...")
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/cluster-postcard-images",
                json=clustering_request
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                world_map = result.get('worldMapSummary', {})
                
                print(f"✅ Clustering completed in {end_time - start_time:.2f}s")
                print(f"\n🌍 World Map Results:")
                print(f"   Countries detected: {world_map.get('totalCountriesWithData', 0)}")
                print(f"   Available countries: {world_map.get('availableCountries', [])}")
                print(f"   Has world map data: {world_map.get('hasWorldMapData', False)}")
                
                # Show country colors
                country_colors = world_map.get('countryColors', {})
                if country_colors:
                    print(f"\n🎨 Country Colors:")
                    for country_code, colors in country_colors.items():
                        dominant = colors.get('dominant', 'N/A')
                        family = colors.get('family', 'N/A')
                        temp_val = colors.get('temperature', 0)
                        temp_desc = "Cool" if temp_val < 0.4 else "Warm" if temp_val > 0.6 else "Neutral"
                        
                        print(f"     {country_code}: {dominant} ({family}, {temp_desc})")
                
                # Show country image counts
                country_counts = world_map.get('countryImageCounts', {})
                if country_counts:
                    print(f"\n📈 Images per Country:")
                    for country_code, count in country_counts.items():
                        print(f"     {country_code}: {count} images")
                
                # Show continent distribution
                continent_dist = world_map.get('continentDistribution', {})
                if continent_dist:
                    print(f"\n🌎 Continent Distribution:")
                    for continent, count in continent_dist.items():
                        print(f"     {continent}: {count} countries")
                
                return result
            else:
                print(f"❌ Clustering failed: {response.status_code}")
                print(f"Response: {response.text[:300]}")
                return None
                
        except Exception as e:
            print(f"❌ Error during clustering: {e}")
            return None
    
    def test_world_map_endpoints(self):
        """Test world map visualization endpoints"""
        print("\n🌍 Testing world map endpoints...")
        
        # Test detailed world map
        try:
            response = self.session.get(f"{self.base_url}/visualization-data/world-map-detailed")
            if response.status_code == 200:
                data = response.json()
                countries = data.get('countries', {})
                insights = data.get('culturalInsights', [])
                print(f"✅ Detailed world map: {len(countries)} countries, {len(insights)} insights")
            else:
                print(f"⚠️ Detailed world map limited: {response.status_code}")
        except Exception as e:
            print(f"❌ World map error: {e}")
        
        # Test specific country
        try:
            response = self.session.get(f"{self.base_url}/visualization-data/country/DE")
            if response.status_code == 200:
                print("✅ Country-specific data working (tested Germany)")
            else:
                print(f"⚠️ Country data limited: {response.status_code}")
        except Exception as e:
            print(f"❌ Country data error: {e}")
        
        # Test continent analysis
        try:
            response = self.session.get(f"{self.base_url}/visualization-data/continent-analysis")
            if response.status_code == 200:
                data = response.json()
                continents = data.get('continentStatistics', {})
                print(f"✅ Continent analysis: {len(continents)} continents")
            else:
                print(f"⚠️ Continent analysis limited: {response.status_code}")
        except Exception as e:
            print(f"❌ Continent analysis error: {e}")
    
    def test_filename_country_extraction(self):
        """Test country extraction from filenames"""
        print("\n🏷️ Testing filename country extraction...")
        
        test_filenames = [
            "DE-39.jpeg",
            "FR-12.png", 
            "IT-05.jpg",
            "US-01.webp",
            "invalid-filename.jpg"
        ]
        
        for filename in test_filenames:
            try:
                response = self.session.post(
                    f"{self.base_url}/extract-country-from-filename",
                    json={"filename": filename}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    detected = "✅" if result['detected'] else "❌"
                    country = result.get('countryName', 'None')
                    print(f"   {filename} → {country} {detected}")
                else:
                    print(f"   {filename} → Error {response.status_code}")
                    
            except Exception as e:
                print(f"   {filename} → Error: {e}")
    
    def run_complete_test(self):
        """Run all country clustering tests"""
        print("🧪 COUNTRY CLUSTERING FEATURES TEST")
        print("=" * 60)
        
        # Test 1: Server connection
        if not self.test_server_connection():
            return
        
        # Test 2: Backend images with country detection
        backend_images = self.test_backend_images_with_countries()
        
        # Test 3: Country statistics
        self.test_country_statistics()
        
        # Test 4: Filename extraction
        self.test_filename_country_extraction()
        
        # Test 5: Clustering with countries
        if backend_images:
            clustering_result = self.test_clustering_with_countries(backend_images)
            
            if clustering_result:
                world_map = clustering_result.get('worldMapSummary', {})
                countries_detected = world_map.get('totalCountriesWithData', 0)
                
                if countries_detected > 0:
                    print(f"\n🎉 SUCCESS: Country clustering working!")
                    print(f"   {countries_detected} countries detected in clustering")
                    print(f"   World map integration: {'✅' if world_map.get('hasWorldMapData') else '❌'}")
                else:
                    print(f"\n⚠️ PARTIAL: Clustering works but no countries detected")
                    print(f"   Check image filenames follow CC-XX.ext format")
            else:
                print(f"\n❌ ISSUE: Clustering failed")
        else:
            print(f"\n⚠️ PARTIAL: No backend images to test clustering")
        
        # Test 6: World map endpoints
        self.test_world_map_endpoints()
        
        # Final summary
        print(f"\n" + "=" * 60)
        print("🎯 TESTING COMPLETE")
        print("=" * 60)
        
        if backend_images:
            countries_in_images = len([img for img in backend_images if img.get('countryName') and img['countryName'] != 'Unknown'])
            print(f"📊 Results Summary:")
            print(f"   Backend images: {len(backend_images)}")
            print(f"   With countries: {countries_in_images}")
            print(f"   Country detection rate: {countries_in_images/len(backend_images)*100:.1f}%")
            
            if countries_in_images > 0:
                print(f"\n✅ Country clustering features are working!")
                print(f"💡 Try the clustering endpoint with your frontend")
            else:
                print(f"\n⚠️ Add images with country codes to data/ folder")
                print(f"💡 Use format: CC-XX.ext (e.g., DE-39.jpeg, FR-12.png)")
        else:
            print(f"📁 No images found in data/ folder")
            print(f"💡 Add some postcard images to test country features")

def main():
    """Main function"""
    tester = CountryClusteringTester()
    tester.run_complete_test()

if __name__ == "__main__":
    main()