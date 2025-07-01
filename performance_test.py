#!/usr/bin/env python3
"""
Performance Testing Script for Enhanced Image Analysis API
Run this script to benchmark your implementation with actual data.
"""

import time
import numpy as np
import statistics
from pathlib import Path
import json
import sys
from typing import List, Dict, Tuple
from PIL import Image
import io
import base64

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from utils.image_utils import data_uri_to_pil_image, preprocess_image_for_analysis
    from services.color_extraction import extract_dominant_colors_from_image
    from services.image_clustering import get_image_features, cluster_image_features_som
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

class PerformanceTester:
    def __init__(self):
        self.results = {}
        self.test_images = []
        
    def create_synthetic_images(self, count: int = 10) -> List[Image.Image]:
        """Create synthetic images for testing when real data isn't available"""
        images = []
        print(f"Creating {count} synthetic test images...")
        
        for i in range(count):
            # Create images with varying complexity
            if i % 3 == 0:  # Simple images
                img_array = np.full((150, 150, 3), [120 + i*10, 80 + i*5, 200 - i*8], dtype=np.uint8)
                noise = np.random.randint(-5, 5, (150, 150, 3))
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            elif i % 3 == 1:  # Medium complexity
                x, y = np.meshgrid(np.linspace(0, 255, 150), np.linspace(0, 255, 150))
                r = np.sin(x/20 + i) * 127 + 128
                g = np.cos(y/20 + i) * 127 + 128
                b = np.sin((x+y)/30 + i) * 127 + 128
                img_array = np.stack([r, g, b], axis=2).astype(np.uint8)
            else:  # Complex images
                img_array = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
                # Add some colored circles
                for j in range(3):
                    center_x, center_y = np.random.randint(30, 120), np.random.randint(30, 120)
                    radius = np.random.randint(10, 25)
                    color = np.random.randint(0, 256, 3)
                    
                    y_coords, x_coords = np.ogrid[:150, :150]
                    mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
                    img_array[mask] = color
            
            images.append(Image.fromarray(img_array, 'RGB'))
        
        return images
    
    def load_real_images(self, data_dir: str = "data") -> List[Image.Image]:
        """Load real images from data directory"""
        images = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"Data directory {data_dir} not found, using synthetic images")
            return self.create_synthetic_images()
        
        supported_extensions = ['.png', '.jpg', '.jpeg', '.webp']
        image_files = []
        
        for ext in supported_extensions:
            image_files.extend(data_path.glob(f"*{ext}"))
            image_files.extend(data_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {data_dir}, using synthetic images")
            return self.create_synthetic_images()
        
        print(f"Loading {len(image_files)} real images from {data_dir}")
        
        for img_path in image_files[:20]:  # Limit to 20 images for testing
            try:
                with Image.open(img_path) as img:
                    images.append(img.copy())
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        return images
    
    def benchmark_color_extraction(self, images: List[Image.Image]) -> Dict:
        """Benchmark color extraction strategies"""
        print("\\n=== Color Extraction Performance ===")
        
        strategies = [
            "enhanced_kmeans",
            "frequency_based", 
            "dominant_hues",
            "color_harmony",
            "adaptive"
        ]
        
        color_spaces = ["rgb", "hsv", "lab"]
        num_colors_options = [3, 5, 8]
        
        results = {}
        
        for strategy in strategies:
            results[strategy] = {}
            print(f"\\nTesting strategy: {strategy}")
            
            for color_space in color_spaces:
                for num_colors in num_colors_options:
                    test_key = f"{color_space}_{num_colors}colors"
                    times = []
                    
                    # Test with first 5 images for each configuration
                    test_images = images[:5]
                    
                    for img in test_images:
                        try:
                            start_time = time.time()
                            colors = extract_dominant_colors_from_image(
                                img, 
                                num_colors=num_colors,
                                strategy=strategy,
                                color_space=color_space
                            )
                            end_time = time.time()
                            
                            if colors:  # Only count successful extractions
                                times.append(end_time - start_time)
                                
                        except Exception as e:
                            print(f"Error with {strategy}/{color_space}/{num_colors}: {e}")
                    
                    if times:
                        avg_time = statistics.mean(times)
                        std_time = statistics.stdev(times) if len(times) > 1 else 0
                        results[strategy][test_key] = {
                            "avg_time_ms": avg_time * 1000,
                            "std_time_ms": std_time * 1000,
                            "samples": len(times)
                        }
                        print(f"  {test_key}: {avg_time*1000:.1f}ms ±{std_time*1000:.1f}ms")
        
        return results
    
    def benchmark_image_preprocessing(self, images: List[Image.Image]) -> Dict:
        """Benchmark image preprocessing operations"""
        print("\\n=== Image Preprocessing Performance ===")
        
        times = []
        sizes_before = []
        sizes_after = []
        
        for img in images[:10]:  # Test first 10 images
            # Original size
            sizes_before.append(img.width * img.height)
            
            start_time = time.time()
            try:
                pixels = preprocess_image_for_analysis(img)
                end_time = time.time()
                
                times.append(end_time - start_time)
                sizes_after.append(len(pixels))
                
            except Exception as e:
                print(f"Preprocessing error: {e}")
        
        if times:
            avg_time = statistics.mean(times)
            avg_size_before = statistics.mean(sizes_before)
            avg_size_after = statistics.mean(sizes_after)
            
            print(f"Average preprocessing time: {avg_time*1000:.1f}ms")
            print(f"Average size reduction: {avg_size_before:.0f} -> {avg_size_after:.0f} pixels")
            print(f"Compression ratio: {avg_size_before/avg_size_after:.1f}x")
            
            return {
                "avg_time_ms": avg_time * 1000,
                "avg_pixels_before": avg_size_before,
                "avg_pixels_after": avg_size_after,
                "compression_ratio": avg_size_before / avg_size_after
            }
        
        return {}
    
    def benchmark_feature_extraction(self, images: List[Image.Image]) -> Dict:
        """Benchmark feature extraction for clustering"""
        print("\\n=== Feature Extraction Performance ===")
        
        strategies = ["enhanced_features", "harmony_focused", "hue_based", "basic_colors"]
        color_spaces = ["rgb", "hsv", "lab"]
        
        results = {}
        
        for strategy in strategies:
            results[strategy] = {}
            print(f"\\nTesting feature strategy: {strategy}")
            
            for color_space in color_spaces:
                times = []
                feature_lengths = []
                
                for img in images[:5]:  # Test first 5 images
                    try:
                        start_time = time.time()
                        features = get_image_features(
                            img,
                            num_dominant_colors_for_features=5,
                            strategy=strategy,
                            color_space=color_space
                        )
                        end_time = time.time()
                        
                        if features:
                            times.append(end_time - start_time)
                            feature_lengths.append(len(features))
                    
                    except Exception as e:
                        print(f"Feature extraction error: {e}")
                
                if times:
                    avg_time = statistics.mean(times)
                    avg_length = statistics.mean(feature_lengths)
                    
                    results[strategy][color_space] = {
                        "avg_time_ms": avg_time * 1000,
                        "feature_length": avg_length,
                        "samples": len(times)
                    }
                    print(f"  {color_space}: {avg_time*1000:.1f}ms, {avg_length:.0f} features")
        
        return results
    
    def benchmark_som_clustering(self, images: List[Image.Image]) -> Dict:
        """Benchmark SOM clustering performance"""
        print("\\n=== SOM Clustering Performance ===")
        
        grid_sizes = [3, 5, 7]
        datasets = [5, 10, 15, 20]  # Number of images
        
        results = {}
        
        for grid_size in grid_sizes:
            results[grid_size] = {}
            
            for dataset_size in datasets:
                if dataset_size > len(images):
                    continue
                
                print(f"\\nTesting {grid_size}x{grid_size} grid with {dataset_size} images")
                
                # Extract features first
                try:
                    feature_times = []
                    features_list = []
                    
                    for img in images[:dataset_size]:
                        start_time = time.time()
                        features = get_image_features(img, strategy="enhanced_features")
                        feature_times.append(time.time() - start_time)
                        features_list.append(features)
                    
                    if not features_list or not all(features_list):
                        print(f"  Skipping - feature extraction failed")
                        continue
                    
                    # SOM clustering
                    start_time = time.time()
                    winner_coordinates, metrics = cluster_image_features_som(
                        features_list,
                        grid_size,
                        len(features_list[0])
                    )
                    clustering_time = time.time() - start_time
                    
                    avg_feature_time = statistics.mean(feature_times)
                    total_time = clustering_time + sum(feature_times)
                    
                    results[grid_size][dataset_size] = {
                        "feature_extraction_ms": avg_feature_time * 1000,
                        "clustering_time_ms": clustering_time * 1000,
                        "total_time_ms": total_time * 1000,
                        "time_per_image_ms": (total_time / dataset_size) * 1000,
                        "som_metrics": metrics
                    }
                    
                    print(f"  Feature extraction: {avg_feature_time*1000:.1f}ms/image")
                    print(f"  SOM clustering: {clustering_time*1000:.1f}ms")
                    print(f"  Total: {total_time*1000:.1f}ms ({total_time*1000/dataset_size:.1f}ms/image)")
                    print(f"  Quantization error: {metrics.get('quantization_error', 'N/A')}")
                
                except Exception as e:
                    print(f"  Error: {e}")
        
        return results
    
    def benchmark_memory_usage(self, images: List[Image.Image]) -> Dict:
        """Estimate memory usage patterns"""
        print("\\n=== Memory Usage Analysis ===")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        results = {
            "initial_memory_mb": initial_memory,
            "operations": {}
        }
        
        # Test memory usage for different operations
        operations = [
            ("Load 10 images", lambda: [img.copy() for img in images[:10]]),
            ("Preprocess 10 images", lambda: [preprocess_image_for_analysis(img) for img in images[:10]]),
            ("Extract features 10 images", lambda: [get_image_features(img) for img in images[:10]])
        ]
        
        for op_name, operation in operations:
            try:
                before_memory = process.memory_info().rss / 1024 / 1024
                start_time = time.time()
                
                result = operation()
                
                end_time = time.time()
                after_memory = process.memory_info().rss / 1024 / 1024
                
                results["operations"][op_name] = {
                    "memory_increase_mb": after_memory - before_memory,
                    "execution_time_ms": (end_time - start_time) * 1000,
                    "memory_after_mb": after_memory
                }
                
                print(f"{op_name}: +{after_memory - before_memory:.1f}MB, {(end_time - start_time)*1000:.1f}ms")
                
            except Exception as e:
                print(f"Error testing {op_name}: {e}")
        
        return results
    
    def run_full_benchmark(self, use_real_images: bool = True) -> Dict:
        """Run complete performance benchmark suite"""
        print("=" * 60)
        print("ENHANCED IMAGE ANALYSIS API - PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        # Load test images
        if use_real_images:
            self.test_images = self.load_real_images()
        else:
            self.test_images = self.create_synthetic_images(15)
        
        print(f"\\nTesting with {len(self.test_images)} images")
        
        # Run all benchmarks
        self.results = {
            "test_info": {
                "num_images": len(self.test_images),
                "image_source": "real" if use_real_images else "synthetic",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "preprocessing": self.benchmark_image_preprocessing(self.test_images),
            "color_extraction": self.benchmark_color_extraction(self.test_images),
            "feature_extraction": self.benchmark_feature_extraction(self.test_images),
            "som_clustering": self.benchmark_som_clustering(self.test_images),
            "memory_usage": self.benchmark_memory_usage(self.test_images)
        }
        
        return self.results
    
    def save_results(self, filename: str = "performance_results.json"):
        """Save benchmark results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\\nResults saved to {filename}")
    
    def print_summary(self):
        """Print a summary of benchmark results"""
        print("\\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        if not self.results:
            print("No results available. Run benchmark first.")
            return
        
        # Color extraction summary
        if "color_extraction" in self.results:
            print("\\nColor Extraction (avg time for 5 images):")
            for strategy, data in self.results["color_extraction"].items():
                if data:
                    avg_times = [metrics["avg_time_ms"] for metrics in data.values() if "avg_time_ms" in metrics]
                    if avg_times:
                        print(f"  {strategy}: {statistics.mean(avg_times):.1f}ms avg")
        
        # SOM clustering summary
        if "som_clustering" in self.results:
            print("\\nSOM Clustering (time per image):")
            for grid_size, data in self.results["som_clustering"].items():
                times_per_image = [metrics["time_per_image_ms"] for metrics in data.values()]
                if times_per_image:
                    print(f"  {grid_size}x{grid_size} grid: {statistics.mean(times_per_image):.1f}ms/image avg")
        
        # Memory usage summary
        if "memory_usage" in self.results and "operations" in self.results["memory_usage"]:
            print("\\nMemory Usage:")
            for op_name, metrics in self.results["memory_usage"]["operations"].items():
                print(f"  {op_name}: +{metrics['memory_increase_mb']:.1f}MB")


def main():
    """Main function to run performance tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance testing for Enhanced Image Analysis API")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic images instead of real data")
    parser.add_argument("--output", default="performance_results.json", help="Output file for results")
    args = parser.parse_args()
    
    tester = PerformanceTester()
    
    try:
        results = tester.run_full_benchmark(use_real_images=not args.synthetic)
        tester.print_summary()
        tester.save_results(args.output)
        
        print(f"\\nBenchmarking complete!")
        print(f"Run with --synthetic flag to test with synthetic images")
        print(f"Results saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\\nBenchmarking interrupted by user")
    except Exception as e:
        print(f"\\nError during benchmarking: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()