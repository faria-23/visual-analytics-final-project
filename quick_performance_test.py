#!/usr/bin/env python3
"""
Quick Performance Test - Fast benchmark for key operations
Usage: python quick_performance_test.py
"""

import time
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def create_test_image(size=(150, 150)):
    """Create a simple test image"""
    img_array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    # Add some structure
    center_x, center_y = size[0]//2, size[1]//2
    y, x = np.ogrid[:size[1], :size[0]]
    mask = (x - center_x)**2 + (y - center_y)**2 <= (size[0]//4)**2
    img_array[mask] = [255, 100, 100]  # Red circle
    return Image.fromarray(img_array, 'RGB')

def benchmark_color_extraction():
    """Quick benchmark of color extraction"""
    print("=== Color Extraction Benchmark ===")
    
    try:
        from services.color_extraction import extract_dominant_colors_from_image
        
        # Create test image
        img = create_test_image()
        
        strategies = ["enhanced_kmeans", "frequency_based", "dominant_hues"]
        
        for strategy in strategies:
            times = []
            for i in range(3):  # 3 runs each
                start = time.time()
                colors = extract_dominant_colors_from_image(
                    img, 
                    num_colors=5, 
                    strategy=strategy
                )
                end = time.time()
                times.append(end - start)
            
            avg_time = sum(times) / len(times)
            print(f"{strategy}: {avg_time*1000:.1f}ms avg ({len(colors)} colors)")
            
    except ImportError as e:
        print(f"Import error: {e}")

def benchmark_clustering():
    """Quick benchmark of SOM clustering"""
    print("\\n=== SOM Clustering Benchmark ===")
    
    try:
        from services.image_clustering import get_image_features, cluster_image_features_som
        
        # Create test images
        images = [create_test_image() for _ in range(10)]
        
        # Extract features
        start = time.time()
        features_list = []
        for img in images:
            features = get_image_features(img, strategy="enhanced_features")
            features_list.append(features)
        feature_time = time.time() - start
        
        if not features_list or not all(features_list):
            print("Feature extraction failed")
            return
        
        # Clustering
        start = time.time()
        winner_coords, metrics = cluster_image_features_som(
            features_list, 
            grid_size=5, 
            num_features=len(features_list[0])
        )
        cluster_time = time.time() - start
        
        total_time = feature_time + cluster_time
        
        print(f"Feature extraction: {feature_time*1000:.1f}ms ({feature_time*1000/len(images):.1f}ms/image)")
        print(f"SOM clustering: {cluster_time*1000:.1f}ms")
        print(f"Total: {total_time*1000:.1f}ms for {len(images)} images")
        print(f"Quantization error: {metrics.get('quantization_error', 'N/A'):.4f}")
        
    except ImportError as e:
        print(f"Import error: {e}")

def benchmark_preprocessing():
    """Quick benchmark of image preprocessing"""
    print("\\n=== Image Preprocessing Benchmark ===")
    
    try:
        from utils.image_utils import preprocess_image_for_analysis
        
        # Test different image sizes
        sizes = [(50, 50), (150, 150), (300, 300)]
        
        for size in sizes:
            img = create_test_image(size)
            
            start = time.time()
            pixels = preprocess_image_for_analysis(img)
            end = time.time()
            
            original_pixels = size[0] * size[1]
            processed_pixels = len(pixels)
            
            print(f"{size[0]}x{size[1]}: {(end-start)*1000:.1f}ms, "
                  f"{original_pixels} -> {processed_pixels} pixels "
                  f"({original_pixels/processed_pixels:.1f}x reduction)")
            
    except ImportError as e:
        print(f"Import error: {e}")

def benchmark_memory():
    """Quick memory usage test"""
    print("\\n=== Memory Usage Test ===")
    
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        
        initial_mem = process.memory_info().rss / 1024 / 1024
        print(f"Initial memory: {initial_mem:.1f}MB")
        
        # Create and process 20 images
        images = [create_test_image() for _ in range(20)]
        
        after_creation = process.memory_info().rss / 1024 / 1024
        print(f"After creating 20 images: {after_creation:.1f}MB (+{after_creation-initial_mem:.1f}MB)")
        
        # Process them
        try:
            from utils.image_utils import preprocess_image_for_analysis
            processed = [preprocess_image_for_analysis(img) for img in images]
            
            after_processing = process.memory_info().rss / 1024 / 1024
            print(f"After preprocessing: {after_processing:.1f}MB (+{after_processing-after_creation:.1f}MB)")
            
        except ImportError:
            print("Could not test preprocessing memory usage")
            
    except ImportError:
        print("psutil not available for memory testing")

def main():
    print("QUICK PERFORMANCE TEST")
    print("=" * 50)
    
    benchmark_preprocessing()
    benchmark_color_extraction() 
    benchmark_clustering()
    benchmark_memory()
    
    print("\\n" + "=" * 50)
    print("Quick benchmark complete!")
    print("For detailed benchmarks, run: python performance_test.py")

if __name__ == "__main__":
    main()