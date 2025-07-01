#!/usr/bin/env python3
"""
Performance Profiler - Identify bottlenecks in the image analysis pipeline
Usage: python performance_profiler.py
"""

import time
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import functools

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"  {func.__name__}: {(end-start)*1000:.1f}ms")
        return result
    return wrapper

class PerformanceProfiler:
    def __init__(self):
        self.timing_data = {}
    
    def profile_color_extraction_pipeline(self, image):
        """Profile the complete color extraction pipeline"""
        print("\\n=== Color Extraction Pipeline Profile ===")
        
        try:
            from utils.image_utils import preprocess_image_for_analysis, validate_image_quality
            from services.color_extraction import (
                _extract_enhanced_kmeans_colors,
                _extract_palette_by_frequency,
                _extract_dominant_hues
            )
            
            # Step 1: Image validation
            start = time.time()
            quality_metrics = validate_image_quality(image)
            validation_time = time.time() - start
            print(f"1. Image validation: {validation_time*1000:.1f}ms")
            
            # Step 2: Preprocessing
            start = time.time()
            pixels = preprocess_image_for_analysis(image)
            preprocessing_time = time.time() - start
            print(f"2. Preprocessing: {preprocessing_time*1000:.1f}ms ({len(pixels)} pixels)")
            
            # Step 3: Different extraction methods
            methods = [
                ("Enhanced K-means", lambda: _extract_enhanced_kmeans_colors(pixels, 5, "rgb")),
                ("Frequency-based", lambda: _extract_palette_by_frequency(pixels, 5)),
                ("Dominant hues", lambda: _extract_dominant_hues(pixels, 5))
            ]
            
            for method_name, method_func in methods:
                start = time.time()
                try:
                    colors, weights, metrics = method_func()
                    method_time = time.time() - start
                    print(f"3.{methods.index((method_name, method_func))+1} {method_name}: {method_time*1000:.1f}ms -> {len(colors)} colors")
                except Exception as e:
                    print(f"3.{methods.index((method_name, method_func))+1} {method_name}: FAILED ({e})")
            
        except ImportError as e:
            print(f"Import error in color extraction profiling: {e}")
    
    def profile_clustering_pipeline(self, images):
        """Profile the complete clustering pipeline"""
        print("\\n=== Clustering Pipeline Profile ===")
        
        try:
            from services.image_clustering import get_image_features, cluster_image_features_som
            
            print(f"Testing with {len(images)} images")
            
            # Step 1: Feature extraction for all images
            features_list = []
            total_feature_time = 0
            
            for i, img in enumerate(images):
                start = time.time()
                features = get_image_features(img, strategy="enhanced_features")
                feature_time = time.time() - start
                total_feature_time += feature_time
                
                if features:
                    features_list.append(features)
                    if i < 3:  # Show timing for first 3 images
                        print(f"1.{i+1} Feature extraction: {feature_time*1000:.1f}ms ({len(features)} features)")
            
            avg_feature_time = total_feature_time / len(images)
            print(f"1. Average feature extraction: {avg_feature_time*1000:.1f}ms/image")
            
            if not features_list:
                print("No features extracted, stopping clustering profile")
                return
            
            # Step 2: SOM parameter calculation
            start = time.time()
            from services.image_clustering import calculate_som_parameters
            som_params = calculate_som_parameters(len(features_list), 5)
            param_time = time.time() - start
            print(f"2. SOM parameter calculation: {param_time*1000:.1f}ms")
            print(f"   Parameters: {som_params}")
            
            # Step 3: SOM clustering
            start = time.time()
            winner_coords, metrics = cluster_image_features_som(
                features_list,
                grid_size=5,
                num_features=len(features_list[0])
            )
            clustering_time = time.time() - start
            print(f"3. SOM clustering: {clustering_time*1000:.1f}ms")
            print(f"   Metrics: QE={metrics.get('quantization_error', 'N/A'):.4f}")
            
            # Step 4: Quality analysis
            start = time.time()
            from services.image_clustering import analyze_cluster_quality
            quality_metrics = analyze_cluster_quality(features_list, winner_coords)
            quality_time = time.time() - start
            print(f"4. Quality analysis: {quality_time*1000:.1f}ms")
            
            # Total pipeline time
            total_time = total_feature_time + param_time + clustering_time + quality_time
            print(f"\\nTotal pipeline: {total_time*1000:.1f}ms ({total_time*1000/len(images):.1f}ms/image)")
            
        except ImportError as e:
            print(f"Import error in clustering profiling: {e}")
    
    def profile_gpu_vs_cpu(self, image):
        """Profile GPU vs CPU performance if available"""
        print("\\n=== GPU vs CPU Performance ===")
        
        try:
            from services.gpu_processor import GPUAcceleratedProcessor
            
            # This would need a mock GPU manager for testing
            print("GPU profiling requires actual GPU hardware")
            print("To test GPU performance:")
            print("1. Ensure GPU drivers are installed")
            print("2. Install torch-directml (Intel Arc) or torch (CUDA)")
            print("3. Run with actual GPU hardware")
            
        except ImportError:
            print("GPU acceleration modules not available")
    
    def profile_memory_efficiency(self, images):
        """Profile memory usage patterns"""
        print("\\n=== Memory Efficiency Profile ===")
        
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            
            initial_memory = process.memory_info().rss / 1024 / 1024
            print(f"Initial memory: {initial_memory:.1f}MB")
            
            # Test different batch sizes
            batch_sizes = [1, 5, 10, len(images)]
            
            for batch_size in batch_sizes:
                if batch_size > len(images):
                    continue
                
                # Memory before batch processing
                before_memory = process.memory_info().rss / 1024 / 1024
                
                # Process batch
                start = time.time()
                batch_images = images[:batch_size]
                
                try:
                    from utils.image_utils import preprocess_image_for_analysis
                    processed = [preprocess_image_for_analysis(img) for img in batch_images]
                    
                    processing_time = time.time() - start
                    after_memory = process.memory_info().rss / 1024 / 1024
                    
                    memory_per_image = (after_memory - before_memory) / batch_size
                    time_per_image = processing_time / batch_size
                    
                    print(f"Batch size {batch_size}: "
                          f"{memory_per_image:.1f}MB/image, "
                          f"{time_per_image*1000:.1f}ms/image")
                
                except ImportError:
                    print(f"Cannot test batch size {batch_size} - import error")
                
        except ImportError:
            print("psutil not available for memory profiling")
    
    def create_test_images(self, count=10):
        """Create test images with varying characteristics"""
        images = []
        
        for i in range(count):
            # Create images with different complexity levels
            if i % 3 == 0:  # Simple
                img_array = np.full((150, 150, 3), [100 + i*10, 150, 200], dtype=np.uint8)
            elif i % 3 == 1:  # Medium
                x, y = np.meshgrid(np.linspace(0, 255, 150), np.linspace(0, 255, 150))
                r = np.sin(x/30 + i) * 127 + 128
                g = np.cos(y/30 + i) * 127 + 128
                b = (r + g) / 2
                img_array = np.stack([r, g, b], axis=2).astype(np.uint8)
            else:  # Complex
                img_array = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
                # Add some structure
                center_x, center_y = 75, 75
                y_coords, x_coords = np.ogrid[:150, :150]
                mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= 30**2
                img_array[mask] = [255, 100, 100]
            
            images.append(Image.fromarray(img_array, 'RGB'))
        
        return images
    
    def run_complete_profile(self):
        """Run complete performance profiling"""
        print("PERFORMANCE PROFILER")
        print("=" * 50)
        
        # Create test images
        print("Creating test images...")
        images = self.create_test_images(10)
        test_image = images[0]
        
        # Run all profiles
        self.profile_color_extraction_pipeline(test_image)
        self.profile_clustering_pipeline(images[:8])  # Use 8 images for clustering
        self.profile_gpu_vs_cpu(test_image)
        self.profile_memory_efficiency(images[:5])  # Use 5 images for memory test
        
        print("\\n" + "=" * 50)
        print("Profiling complete!")
        print("\\nBottleneck Analysis:")
        print("- If preprocessing is slow: Check image sizes and formats")
        print("- If color extraction is slow: Try 'frequency_based' strategy")
        print("- If clustering is slow: Reduce grid size or use fewer features")
        print("- If memory usage is high: Process images in smaller batches")

def main():
    profiler = PerformanceProfiler()
    profiler.run_complete_profile()

if __name__ == "__main__":
    main()