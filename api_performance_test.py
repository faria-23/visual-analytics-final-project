#!/usr/bin/env python3
"""
API Performance Test - Test the actual FastAPI endpoints
Usage: python api_performance_test.py [--host localhost] [--port 8008]
"""

import requests
import time
import base64
import io
import numpy as np
from PIL import Image
import json
import argparse
from typing import List, Dict

class APIPerformanceTester:
    def __init__(self, host="localhost", port=8008):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
    
    def create_test_data_uri(self, complexity="medium", size=(150, 150)) -> str:
        """Create a test image as data URI"""
        if complexity == "simple":
            img_array = np.full((*size, 3), [120, 80, 200], dtype=np.uint8)
        elif complexity == "medium":
            x, y = np.meshgrid(np.linspace(0, 255, size[0]), np.linspace(0, 255, size[1]))
            r = np.sin(x/20) * 127 + 128
            g = np.cos(y/20) * 127 + 128
            b = np.sin((x+y)/30) * 127 + 128
            img_array = np.stack([r, g, b], axis=2).astype(np.uint8)
        else:  # complex
            img_array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            # Add some colored shapes
            center_x, center_y = size[0]//2, size[1]//2
            y_coords, x_coords = np.ogrid[:size[1], :size[0]]
            mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= (size[0]//4)**2
            img_array[mask] = [255, 100, 100]
        
        img = Image.fromarray(img_array, 'RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def test_server_availability(self) -> bool:
        """Test if the API server is running"""
        try:
            response = self.session.get(f"{self.base_url}/docs", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def benchmark_color_extraction_endpoint(self) -> Dict:
        """Benchmark /extract-dominant-colors endpoint"""
        print("\\n=== Color Extraction Endpoint Benchmark ===")
        
        # Test different configurations
        test_configs = [
            {"strategy": "enhanced_kmeans", "colors": 5},
            {"strategy": "frequency_based", "colors": 5},
            {"strategy": "dominant_hues", "colors": 5},
            {"strategy": "enhanced_kmeans", "colors": 3},
            {"strategy": "enhanced_kmeans", "colors": 8},
        ]
        
        complexities = ["simple", "medium", "complex"]
        results = {}
        
        for complexity in complexities:
            results[complexity] = {}
            data_uri = self.create_test_data_uri(complexity)
            
            print(f"\\nTesting {complexity} images:")
            
            for config in test_configs:
                config_name = f"{config['strategy']}_{config['colors']}colors"
                
                payload = {
                    "photoDataUri": data_uri,
                    "numberOfColors": config["colors"],
                    "extractionStrategy": config["strategy"]
                }
                
                # Run multiple times for average
                times = []
                colors_extracted = 0
                
                for _ in range(3):
                    start_time = time.time()
                    try:
                        response = self.session.post(
                            f"{self.base_url}/extract-dominant-colors",
                            json=payload,
                            timeout=30
                        )
                        end_time = time.time()
                        
                        if response.status_code == 200:
                            times.append(end_time - start_time)
                            result = response.json()
                            colors_extracted = len(result.get("colors", []))
                        else:
                            print(f"  Error {response.status_code}: {response.text}")
                    
                    except requests.exceptions.RequestException as e:
                        print(f"  Request error: {e}")
                
                if times:
                    avg_time = sum(times) / len(times)
                    results[complexity][config_name] = {
                        "avg_time_ms": avg_time * 1000,
                        "colors_extracted": colors_extracted
                    }
                    print(f"  {config_name}: {avg_time*1000:.1f}ms -> {colors_extracted} colors")
        
        return results
    
    def benchmark_clustering_endpoint(self) -> Dict:
        """Benchmark /cluster-postcard-images endpoint"""
        print("\\n=== Clustering Endpoint Benchmark ===")
        
        # Test with different numbers of images and grid sizes
        test_configs = [
            {"images": 5, "grid": 3, "strategy": "enhanced_features"},
            {"images": 10, "grid": 5, "strategy": "enhanced_features"},
            {"images": 15, "grid": 5, "strategy": "harmony_focused"},
            {"images": 10, "grid": 7, "strategy": "enhanced_features"},
        ]
        
        results = {}
        
        for config in test_configs:
            config_name = f"{config['images']}imgs_{config['grid']}x{config['grid']}_grid"
            print(f"\\nTesting {config_name}:")
            
            # Create test images
            image_uris = []
            for i in range(config["images"]):
                complexity = ["simple", "medium", "complex"][i % 3]
                data_uri = self.create_test_data_uri(complexity)
                image_uris.append(data_uri)
            
            payload = {
                "imageUrls": image_uris,
                "gridSize": config["grid"],
                "clusteringStrategy": config["strategy"],
                "colorSpace": "lab"
            }
            
            start_time = time.time()
            try:
                response = self.session.post(
                    f"{self.base_url}/cluster-postcard-images",
                    json=payload,
                    timeout=60  # Longer timeout for clustering
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    total_time = end_time - start_time
                    
                    results[config_name] = {
                        "total_time_ms": total_time * 1000,
                        "time_per_image_ms": (total_time * 1000) / config["images"],
                        "num_clusters": len(result.get("clusters", [])),
                        "clustering_metrics": result.get("clusteringMetrics", {})
                    }
                    
                    print(f"  Total time: {total_time*1000:.1f}ms")
                    print(f"  Time per image: {total_time*1000/config['images']:.1f}ms")
                    print(f"  Clusters formed: {len(result.get('clusters', []))}")
                    
                    metrics = result.get("clusteringMetrics", {})
                    if "som_quantization_error" in metrics:
                        print(f"  SOM QE: {metrics['som_quantization_error']:.4f}")
                
                else:
                    print(f"  Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"  Request error: {e}")
        
        return results
    
    def benchmark_backend_images_endpoint(self) -> Dict:
        """Benchmark /list-backend-images endpoint"""
        print("\\n=== Backend Images Endpoint Benchmark ===")
        
        times = []
        
        for _ in range(3):
            start_time = time.time()
            try:
                response = self.session.get(f"{self.base_url}/list-backend-images", timeout=30)
                end_time = time.time()
                
                if response.status_code == 200:
                    times.append(end_time - start_time)
                    result = response.json()
                    num_images = len(result)
                    print(f"  Found {num_images} backend images")
                else:
                    print(f"  Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"  Request error: {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"  Average response time: {avg_time*1000:.1f}ms")
            return {"avg_time_ms": avg_time * 1000, "num_images": num_images}
        
        return {}
    
    def benchmark_country_endpoints(self) -> Dict:
        """Benchmark country-related endpoints"""
        print("\\n=== Country Endpoints Benchmark ===")
        
        results = {}
        
        # Test /countries endpoint
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/countries", timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                results["countries"] = {
                    "time_ms": (end_time - start_time) * 1000,
                    "total_countries": result.get("totalCountries", 0)
                }
                print(f"  /countries: {(end_time - start_time)*1000:.1f}ms, {result.get('totalCountries', 0)} countries")
            
        except requests.exceptions.RequestException as e:
            print(f"  /countries error: {e}")
        
        # Test /country-statistics endpoint
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/country-statistics", timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                results["country_statistics"] = {
                    "time_ms": (end_time - start_time) * 1000,
                    "total_images": result.get("totalImages", 0),
                    "identified_countries": result.get("totalCountries", 0)
                }
                print(f"  /country-statistics: {(end_time - start_time)*1000:.1f}ms")
                print(f"    Total images: {result.get('totalImages', 0)}")
                print(f"    Identified countries: {result.get('totalCountries', 0)}")
            
        except requests.exceptions.RequestException as e:
            print(f"  /country-statistics error: {e}")
        
        return results
    
    def run_stress_test(self, num_requests=10) -> Dict:
        """Run stress test with concurrent requests"""
        print(f"\\n=== Stress Test ({num_requests} requests) ===")
        
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            data_uri = self.create_test_data_uri("medium")
            payload = {
                "photoDataUri": data_uri,
                "numberOfColors": 5,
                "extractionStrategy": "enhanced_kmeans"
            }
            
            start_time = time.time()
            try:
                response = self.session.post(
                    f"{self.base_url}/extract-dominant-colors",
                    json=payload,
                    timeout=30
                )
                end_time = time.time()
                
                results.put({
                    "success": response.status_code == 200,
                    "time_ms": (end_time - start_time) * 1000,
                    "status_code": response.status_code
                })
                
            except Exception as e:
                results.put({
                    "success": False,
                    "time_ms": 0,
                    "error": str(e)
                })
        
        # Launch threads
        threads = []
        start_time = time.time()
        
        for _ in range(num_requests):
            thread = threading.Thread(target=make_request)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        all_results = []
        while not results.empty():
            all_results.append(results.get())
        
        successes = [r for r in all_results if r["success"]]
        failures = [r for r in all_results if not r["success"]]
        
        if successes:
            avg_response_time = sum(r["time_ms"] for r in successes) / len(successes)
            print(f"  Total time: {total_time*1000:.1f}ms")
            print(f"  Successful requests: {len(successes)}/{num_requests}")
            print(f"  Average response time: {avg_response_time:.1f}ms")
            print(f"  Requests per second: {len(successes)/total_time:.1f}")
            
            if failures:
                print(f"  Failed requests: {len(failures)}")
        
        return {
            "total_time_ms": total_time * 1000,
            "successful_requests": len(successes),
            "failed_requests": len(failures),
            "avg_response_time_ms": avg_response_time if successes else 0,
            "requests_per_second": len(successes) / total_time if total_time > 0 else 0
        }
    
    def run_complete_benchmark(self) -> Dict:
        """Run complete API benchmark suite"""
        print("API PERFORMANCE BENCHMARK")
        print("=" * 50)
        print(f"Testing API at: {self.base_url}")
        
        if not self.test_server_availability():
            print("\\nERROR: API server is not available!")
            print("Please start the server with: python app.py")
            return {}
        
        print("\\nAPI server is running ✓")
        
        results = {
            "color_extraction": self.benchmark_color_extraction_endpoint(),
            "clustering": self.benchmark_clustering_endpoint(),
            "backend_images": self.benchmark_backend_images_endpoint(),
            "country_endpoints": self.benchmark_country_endpoints(),
            "stress_test": self.run_stress_test(10)
        }
        
        return results
    
    def save_results(self, results: Dict, filename: str = "api_performance_results.json"):
        """Save benchmark results to file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\\nResults saved to {filename}")
    
    def print_summary(self, results: Dict):
        """Print summary of benchmark results"""
        print("\\n" + "=" * 50)
        print("API PERFORMANCE SUMMARY")
        print("=" * 50)
        
        # Color extraction summary
        if "color_extraction" in results:
            print("\\nColor Extraction Performance:")
            for complexity, data in results["color_extraction"].items():
                if data:
                    times = [metrics["avg_time_ms"] for metrics in data.values()]
                    if times:
                        print(f"  {complexity.capitalize()} images: {min(times):.0f}-{max(times):.0f}ms range")
        
        # Clustering summary
        if "clustering" in results:
            print("\\nClustering Performance:")
            for config, data in results["clustering"].items():
                if data:
                    print(f"  {config}: {data['time_per_image_ms']:.1f}ms/image")
        
        # Stress test summary
        if "stress_test" in results and results["stress_test"]:
            stress = results["stress_test"]
            print(f"\\nStress Test:")
            print(f"  Success rate: {stress['successful_requests']}/{stress['successful_requests'] + stress['failed_requests']}")
            print(f"  Throughput: {stress['requests_per_second']:.1f} requests/second")

def main():
    parser = argparse.ArgumentParser(description="API Performance Testing")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8008, help="API port")
    parser.add_argument("--output", default="api_performance_results.json", help="Output file")
    parser.add_argument("--stress-requests", type=int, default=10, help="Number of stress test requests")
    args = parser.parse_args()
    
    tester = APIPerformanceTester(args.host, args.port)
    
    try:
        results = tester.run_complete_benchmark()
        
        if results:
            tester.print_summary(results)
            tester.save_results(results, args.output)
            
            print("\\n" + "=" * 50)
            print("API benchmarking complete!")
            print(f"Results saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\\nBenchmarking interrupted by user")
    except Exception as e:
        print(f"\\nError during benchmarking: {e}")

if __name__ == "__main__":
    main()