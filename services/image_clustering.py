from typing import List, Tuple, Dict, Union
import numpy as np
from minisom import MiniSom
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Import enhanced color extraction
try:
    from .color_extraction import get_color_features_for_clustering
except ImportError:
    # For direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from color_extraction import get_color_features_for_clustering

def get_image_features(
    image: Image.Image, 
    num_dominant_colors_for_features: int = 5,
    strategy: str = "enhanced_features",
    color_space: str = "rgb"
) -> Tuple[List[float], List[str]]:
    """
    Enhanced feature extraction for image clustering with strategy awareness.
    Now returns BOTH features AND the extracted hex colors to avoid double extraction.
    
    Args:
        image: PIL Image object
        num_dominant_colors_for_features: Number of dominant colors to extract
        strategy: Feature extraction strategy
        color_space: Color space for analysis (rgb, hsv, lab)
    
    Returns:
        Tuple of (feature_vector, hex_colors_list)
    """
    try:
        if strategy == "enhanced_features":
            # Use the new enhanced feature extraction
            features, hex_colors = get_color_features_for_clustering(
                image, 
                num_dominant_colors_for_features, 
                "enhanced_kmeans",
                color_space,
                include_harmony_features=True,
                return_colors=True  # NEW: Also return hex colors
            )
        elif strategy == "harmony_focused":
            # Focus on color harmony relationships
            features, hex_colors = get_color_features_for_clustering(
                image,
                num_dominant_colors_for_features,
                "color_harmony",
                color_space,
                include_harmony_features=True,
                return_colors=True
            )
        elif strategy == "hue_based":
            # Focus on hue relationships
            features, hex_colors = get_color_features_for_clustering(
                image,
                num_dominant_colors_for_features,
                "dominant_hues",
                "hsv",  # Force HSV for hue-based analysis
                include_harmony_features=True,
                return_colors=True
            )
        elif strategy == "basic_colors":
            # Simple color extraction without harmony features
            features, hex_colors = get_color_features_for_clustering(
                image,
                num_dominant_colors_for_features,
                "enhanced_kmeans",
                color_space,
                include_harmony_features=False,
                return_colors=True
            )
        else:
            # Default to enhanced features
            features, hex_colors = get_color_features_for_clustering(
                image,
                num_dominant_colors_for_features,
                "enhanced_kmeans",
                color_space,
                include_harmony_features=True,
                return_colors=True
            )
        
        return features, hex_colors
            
    except Exception as e:
        print(f"Error in get_image_features: {e}")
        # Return zero vector with expected dimensions
        base_features = num_dominant_colors_for_features * 4  # 3 color channels + 1 weight
        harmony_features = 5 if strategy != "basic_colors" else 0
        empty_features = [0.0] * (base_features + harmony_features)
        empty_colors = []
        return empty_features, empty_colors

def calculate_som_parameters(data_size: int, grid_size: int) -> Dict[str, float]:
    """
    Calculate adaptive SOM parameters based on data size and grid size.
    
    Args:
        data_size: Number of data points
        grid_size: Size of the SOM grid
    
    Returns:
        Dictionary with SOM parameters
    """
    # Adaptive sigma (neighborhood radius)
    # Start with larger neighborhood for bigger grids and more data
    initial_sigma = max(grid_size / 3.0, 1.0)
    
    # Adaptive learning rate
    # Start higher for smaller datasets
    if data_size < 50:
        initial_learning_rate = 0.9
    elif data_size < 200:
        initial_learning_rate = 0.7
    else:
        initial_learning_rate = 0.5
    
    # Number of iterations
    # More iterations for larger datasets and grids
    base_iterations = max(1000, data_size * 10)
    grid_factor = (grid_size * grid_size) / 25.0  # Normalize to 5x5 grid
    num_iterations = int(base_iterations * min(grid_factor, 2.0))
    
    return {
        "sigma": initial_sigma,
        "learning_rate": initial_learning_rate,
        "num_iterations": num_iterations
    }

def calculate_som_quality_metrics(som: MiniSom, data: np.ndarray) -> Dict[str, float]:
    """
    Calculate quality metrics for the trained SOM.
    
    Args:
        som: Trained MiniSom object
        data: Training data
    
    Returns:
        Dictionary with quality metrics
    """
    try:
        # Quantization error (average distance to winner neuron)
        quantization_error = som.quantization_error(data)
        
        # Topographic error (percentage of data points for which the first and second
        # best matching units are not adjacent)
        topographic_error = som.topographic_error(data)
        
        # Calculate additional metrics
        winner_map = np.zeros((som.get_weights().shape[0], som.get_weights().shape[1]))
        for x in data:
            winner = som.winner(x)
            winner_map[winner] += 1
        
        # Coverage (percentage of neurons that are winners for at least one input)
        active_neurons = np.sum(winner_map > 0)
        total_neurons = som.get_weights().shape[0] * som.get_weights().shape[1]
        coverage = active_neurons / total_neurons
        
        # Uniformity (how evenly distributed the data is across neurons)
        if active_neurons > 0:
            winner_counts = winner_map[winner_map > 0]
            uniformity = 1.0 - (np.std(winner_counts) / np.mean(winner_counts))
        else:
            uniformity = 0.0
        
        return {
            "quantization_error": float(quantization_error),
            "topographic_error": float(topographic_error),
            "coverage": float(coverage),
            "uniformity": float(max(0.0, uniformity)),
            "active_neurons": int(active_neurons),
            "total_neurons": int(total_neurons)
        }
        
    except Exception as e:
        print(f"Error calculating SOM quality metrics: {e}")
        return {
            "quantization_error": 0.0,
            "topographic_error": 1.0,
            "coverage": 0.0,
            "uniformity": 0.0,
            "active_neurons": 0,
            "total_neurons": 0
        }

def cluster_image_features_som(
    image_features_list: List[List[float]],
    grid_size: int,
    num_features: int
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    """
    Enhanced SOM clustering with adaptive parameters and quality metrics.
    
    Args:
        image_features_list: List of feature vectors
        grid_size: Size of the SOM grid (creates grid_size x grid_size grid)
        num_features: Expected number of features per vector
    
    Returns:
        Tuple of (winner_coordinates, quality_metrics)
    """
    if not image_features_list:
        return [], {}

    # Convert to numpy array and validate
    data = np.array(image_features_list, dtype=np.float32)
    
    if data.shape[1] != num_features:
        raise ValueError(f"Feature vector length mismatch. Expected {num_features}, got {data.shape[1]}")
    
    # Handle edge cases
    if data.shape[0] == 1:
        # Only one image - place it at center of grid
        center = grid_size // 2
        return [(center, center)], {"single_image": True}
    
    # Calculate adaptive parameters
    som_params = calculate_som_parameters(data.shape[0], grid_size)
    
    print(f"SOM Parameters: sigma={som_params['sigma']:.2f}, "
          f"learning_rate={som_params['learning_rate']:.2f}, "
          f"iterations={som_params['num_iterations']}")
    
    # Create and initialize SOM
    som = MiniSom(
        x=grid_size, 
        y=grid_size, 
        input_len=num_features, 
        sigma=som_params['sigma'], 
        learning_rate=som_params['learning_rate'],
        decay_function=lambda x, t, max_iter: x / (1 + t / (max_iter / 2)),
        neighborhood_function='gaussian',
        random_seed=42  # For reproducibility
    )
    
    # Initialize weights
    try:
        # Try PCA initialization for better starting point
        som.pca_weights_init(data)
        print("SOM initialized with PCA")
    except Exception as e:
        print(f"PCA initialization failed ({e}), using random initialization")
        som.random_weights_init(data)
    
    # Train the SOM
    print(f"Training SOM with {data.shape[0]} samples...")
    
    # Use batch training for better stability
    som.train(data.tolist(), som_params['num_iterations'], verbose=False)
    
    # Get winner coordinates
    winner_coordinates = []
    for x in data:
        winner = som.winner(x)
        winner_coordinates.append(winner)
    
    # Calculate quality metrics
    quality_metrics = calculate_som_quality_metrics(som, data)
    
    print(f"SOM training completed. Quality metrics: "
          f"QE={quality_metrics['quantization_error']:.4f}, "
          f"TE={quality_metrics['topographic_error']:.4f}, "
          f"Coverage={quality_metrics['coverage']:.2f}")
    
    return winner_coordinates, quality_metrics

# NEW: Dynamic K Selection Functions

def find_optimal_k_silhouette(features_array: np.ndarray, k_range: range = None) -> Tuple[int, float]:
    """Find optimal K using silhouette analysis."""
    if k_range is None:
        max_k = min(10, len(features_array) // 2)
        k_range = range(2, max_k + 1)
    
    best_k = 2
    best_score = -1
    
    for k in k_range:
        if k >= len(features_array):
            break
            
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(features_array)
            
            # Skip if all points are in one cluster
            if len(set(cluster_labels)) < 2:
                continue
                
            score = silhouette_score(features_array, cluster_labels)
            
            if score > best_score:
                best_score = score
                best_k = k
                
        except Exception as e:
            print(f"Error testing K={k}: {e}")
            continue
    
    return best_k, best_score

def find_optimal_k_elbow(features_array: np.ndarray, k_range: range = None) -> Tuple[int, float]:
    """Find optimal K using elbow method."""
    if k_range is None:
        max_k = min(10, len(features_array) // 2)
        k_range = range(2, max_k + 1)
    
    inertias = []
    k_values = []
    
    for k in k_range:
        if k >= len(features_array):
            break
            
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(features_array)
            inertias.append(kmeans.inertia_)
            k_values.append(k)
        except Exception as e:
            print(f"Error testing K={k}: {e}")
            continue
    
    if len(inertias) < 3:
        return k_values[0] if k_values else 2, 0.0
    
    # Calculate rate of change to find elbow
    rates = []
    for i in range(1, len(inertias) - 1):
        rate = (inertias[i-1] - inertias[i]) - (inertias[i] - inertias[i+1])
        rates.append(rate)
    
    if rates:
        best_idx = np.argmax(rates) + 1  # +1 because we started from index 1
        best_k = k_values[best_idx] if best_idx < len(k_values) else k_values[-1]
        elbow_score = rates[best_idx - 1] if (best_idx - 1) < len(rates) else 0.0
    else:
        best_k = k_values[len(k_values) // 2]  # Middle value as fallback
        elbow_score = 0.0
    
    return best_k, elbow_score

def find_optimal_k_gap_statistic(features_array: np.ndarray, k_range: range = None) -> Tuple[int, float]:
    """Find optimal K using gap statistic (simplified version)."""
    if k_range is None:
        max_k = min(8, len(features_array) // 2)
        k_range = range(2, max_k + 1)
    
    n_refs = 5  # Number of reference datasets
    gaps = []
    k_values = []
    
    for k in k_range:
        if k >= len(features_array):
            break
            
        try:
            # Real data clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(features_array)
            real_inertia = kmeans.inertia_
            
            # Reference data clustering (random data)
            ref_inertias = []
            for _ in range(n_refs):
                # Generate random data with same bounds as real data
                random_data = np.random.uniform(
                    low=features_array.min(axis=0),
                    high=features_array.max(axis=0),
                    size=features_array.shape
                )
                
                ref_kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                ref_kmeans.fit(random_data)
                ref_inertias.append(ref_kmeans.inertia_)
            
            # Calculate gap
            ref_mean = np.mean(ref_inertias)
            gap = np.log(ref_mean) - np.log(real_inertia)
            gaps.append(gap)
            k_values.append(k)
            
        except Exception as e:
            print(f"Error testing K={k} for gap statistic: {e}")
            continue
    
    if not gaps:
        return 2, 0.0
    
    # Find K where gap is maximized
    best_idx = np.argmax(gaps)
    best_k = k_values[best_idx]
    best_gap = gaps[best_idx]
    
    return best_k, best_gap

def find_optimal_k_auto(features_array: np.ndarray) -> Tuple[int, Dict[str, float]]:
    """
    Find optimal K using combined approach with multiple methods.
    
    Args:
        features_array: Feature matrix for clustering
        
    Returns:
        Tuple of (optimal_k, method_scores)
    """
    max_k = min(10, len(features_array) // 2)
    k_range = range(2, max_k + 1)
    
    methods_results = {}
    
    # Test all methods
    try:
        k_sil, score_sil = find_optimal_k_silhouette(features_array, k_range)
        methods_results['silhouette'] = {'k': k_sil, 'score': score_sil}
    except Exception as e:
        print(f"Silhouette method failed: {e}")
        methods_results['silhouette'] = {'k': 3, 'score': 0.0}
    
    try:
        k_elbow, score_elbow = find_optimal_k_elbow(features_array, k_range)
        methods_results['elbow'] = {'k': k_elbow, 'score': score_elbow}
    except Exception as e:
        print(f"Elbow method failed: {e}")
        methods_results['elbow'] = {'k': 3, 'score': 0.0}
    
    try:
        k_gap, score_gap = find_optimal_k_gap_statistic(features_array, k_range)
        methods_results['gap'] = {'k': k_gap, 'score': score_gap}
    except Exception as e:
        print(f"Gap statistic failed: {e}")
        methods_results['gap'] = {'k': 3, 'score': 0.0}
    
    # Combine results using weighted voting
    k_votes = {}
    for method, result in methods_results.items():
        k = result['k']
        score = result['score']
        
        # Weight votes by score quality
        if method == 'silhouette':
            weight = max(0, score) * 2  # Silhouette scores are [-1, 1], prefer positive
        elif method == 'elbow':
            weight = max(0, score)
        elif method == 'gap':
            weight = max(0, score)
        else:
            weight = 1.0
        
        if k in k_votes:
            k_votes[k] += weight
        else:
            k_votes[k] = weight
    
    # Select K with highest weighted vote
    if k_votes:
        optimal_k = max(k_votes.items(), key=lambda x: x[1])[0]
    else:
        optimal_k = 3  # Fallback
    
    # Prepare scoring summary
    scoring_summary = {
        'silhouette_k': methods_results['silhouette']['k'],
        'silhouette_score': methods_results['silhouette']['score'],
        'elbow_k': methods_results['elbow']['k'],
        'elbow_score': methods_results['elbow']['score'],
        'gap_k': methods_results['gap']['k'],
        'gap_score': methods_results['gap']['score'],
        'combined_k': optimal_k,
        'vote_weights': k_votes
    }
    
    return optimal_k, scoring_summary

def find_optimal_k_dynamic(
    features_array: np.ndarray, 
    method: str = "auto"
) -> Tuple[int, Dict[str, float]]:
    """
    Dynamic K selection using specified method.
    
    Args:
        features_array: Feature matrix for clustering
        method: Selection method ('auto', 'silhouette', 'elbow', 'gap')
    
    Returns:
        Tuple of (optimal_k, metrics)
    """
    
    # Ensure we have enough data points
    if len(features_array) < 4:
        return 2, {"method": method, "reason": "insufficient_data", "data_points": len(features_array)}
    
    if method == "silhouette":
        k, score = find_optimal_k_silhouette(features_array)
        return k, {"method": "silhouette", "silhouette_score": score}
        
    elif method == "elbow":
        k, score = find_optimal_k_elbow(features_array)
        return k, {"method": "elbow", "elbow_score": score}
        
    elif method == "gap":
        k, score = find_optimal_k_gap_statistic(features_array)
        return k, {"method": "gap", "gap_score": score}
        
    else:  # method == "auto"
        k, scoring_summary = find_optimal_k_auto(features_array)
        scoring_summary["method"] = "auto"
        return k, scoring_summary

# NEW: Hybrid K-means + SOM Pipeline

def cluster_image_features_hybrid_pipeline(
    image_features_list: List[List[float]],
    grid_size: int,
    num_features: int,
    num_clusters: int = 0,  # 0 = auto-select K
    k_selection_method: str = "auto"
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    """
    Hybrid clustering pipeline: K-means for initial clustering + SOM for spatial organization.
    
    Args:
        image_features_list: List of feature vectors
        grid_size: Size of the SOM grid for final organization
        num_features: Expected number of features per vector
        num_clusters: Number of K-means clusters (0 = auto-select)
        k_selection_method: Method for K selection ('auto', 'silhouette', 'elbow', 'gap')
    
    Returns:
        Tuple of (winner_coordinates, combined_metrics)
    """
    if not image_features_list:
        return [], {}

    # Convert to numpy array and validate
    data = np.array(image_features_list, dtype=np.float32)
    
    if data.shape[1] != num_features:
        raise ValueError(f"Feature vector length mismatch. Expected {num_features}, got {data.shape[1]}")
    
    # Handle edge cases
    if data.shape[0] == 1:
        center = grid_size // 2
        return [(center, center)], {"single_image": True, "clustering_mode": "hybrid"}
    
    print(f"Starting hybrid K-means + SOM pipeline with {data.shape[0]} images")
    
    # STAGE 1: K-means clustering
    if num_clusters == 0:
        # Dynamic K selection
        print(f"Using dynamic K selection with method: {k_selection_method}")
        optimal_k, k_metrics = find_optimal_k_dynamic(data, k_selection_method)
        k_used = optimal_k
        k_selection_method_used = "dynamic"
    else:
        # User-specified K
        print(f"Using user-specified K: {num_clusters}")
        k_used = num_clusters
        k_metrics = {"method": "user_specified"}
        k_selection_method_used = "user_specified"
    
    print(f"Using K={k_used} for K-means clustering")
    
    # Perform K-means clustering
    try:
        kmeans = KMeans(n_clusters=k_used, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(data)
        kmeans_inertia = kmeans.inertia_
        
        # Calculate silhouette score for the chosen K
        if len(set(cluster_labels)) > 1:
            kmeans_silhouette = silhouette_score(data, cluster_labels)
        else:
            kmeans_silhouette = 0.0
            
    except Exception as e:
        print(f"K-means clustering failed: {e}")
        # Fallback to pure SOM
        return cluster_image_features_som(image_features_list, grid_size, num_features)
    
    print(f"K-means completed. Silhouette score: {kmeans_silhouette:.4f}")
    
    # STAGE 2: Create cluster representatives for SOM input
    cluster_representatives = []
    cluster_centers = kmeans.cluster_centers_
    
    # Use K-means centroids as representatives
    for center in cluster_centers:
        cluster_representatives.append(center.tolist())
    
    print(f"Created {len(cluster_representatives)} cluster representatives")
    
    # STAGE 3: SOM organization of cluster representatives
    if len(cluster_representatives) == 1:
        # Only one cluster, place at center
        center = grid_size // 2
        som_positions = [(center, center)]
        som_metrics = {"single_cluster": True}
    else:
        # Train SOM on cluster representatives
        try:
            som_positions, som_metrics = cluster_image_features_som(
                cluster_representatives, 
                grid_size,
                len(cluster_representatives[0])
            )
        except Exception as e:
            print(f"SOM organization failed: {e}")
            # Fallback: place clusters in grid order
            som_positions = []
            for i in range(len(cluster_representatives)):
                x = i % grid_size
                y = i // grid_size
                som_positions.append((x, y))
            som_metrics = {"fallback_grid_placement": True}
    
    # STAGE 4: Map original images to SOM grid positions
    winner_coordinates = []
    
    for i, cluster_label in enumerate(cluster_labels):
        # Each image gets the SOM position of its K-means cluster
        if cluster_label < len(som_positions):
            winner_coordinates.append(som_positions[cluster_label])
        else:
            # Fallback for any labeling issues
            center = grid_size // 2
            winner_coordinates.append((center, center))
    
    # STAGE 5: Combine metrics from both stages
    combined_metrics = {
        # Hybrid pipeline info
        "clustering_mode": "hybrid",
        "k_used": k_used,
        "k_selection_method": k_selection_method_used,
        "k_selection_algorithm": k_selection_method,
        
        # K-means metrics
        "kmeans_silhouette_score": float(kmeans_silhouette),
        "kmeans_inertia": float(kmeans_inertia),
        "kmeans_clusters": k_used,
        
        # SOM metrics (from cluster representatives)
        "som_quantization_error": som_metrics.get("quantization_error", 0.0),
        "som_topographic_error": som_metrics.get("topographic_error", 0.0),
        "som_coverage": som_metrics.get("coverage", 0.0),
        
        # Combined metrics
        "total_images": len(image_features_list),
        "cluster_representatives": len(cluster_representatives)
    }
    
    # Add K selection details if available
    if k_selection_method_used == "dynamic":
        combined_metrics.update(k_metrics)
    
    print(f"Hybrid pipeline completed. K-means: {k_used} clusters, "
          f"SOM: {grid_size}x{grid_size} grid, "
          f"Silhouette: {kmeans_silhouette:.4f}")
    
    return winner_coordinates, combined_metrics

def analyze_cluster_quality(
    image_features_list: List[List[float]], 
    winner_coordinates: List[Tuple[int, int]]
) -> Dict[str, float]:
    """
    Analyze the quality of clustering results.
    
    Args:
        image_features_list: Original feature vectors
        winner_coordinates: SOM winner coordinates for each image
    
    Returns:
        Dictionary with clustering quality metrics
    """
    if not image_features_list or not winner_coordinates:
        return {}
    
    try:
        # Group features by cluster
        clusters = {}
        for features, coords in zip(image_features_list, winner_coordinates):
            if coords not in clusters:
                clusters[coords] = []
            clusters[coords].append(features)
        
        # Calculate intra-cluster and inter-cluster distances
        intra_cluster_distances = []
        inter_cluster_distances = []
        
        cluster_centers = {}
        for coords, cluster_features in clusters.items():
            if len(cluster_features) > 1:
                cluster_array = np.array(cluster_features)
                center = np.mean(cluster_array, axis=0)
                cluster_centers[coords] = center
                
                # Calculate intra-cluster distances
                for features in cluster_features:
                    dist = np.linalg.norm(np.array(features) - center)
                    intra_cluster_distances.append(dist)
            elif len(cluster_features) == 1:
                cluster_centers[coords] = np.array(cluster_features[0])
        
        # Calculate inter-cluster distances
        center_coords = list(cluster_centers.keys())
        for i in range(len(center_coords)):
            for j in range(i + 1, len(center_coords)):
                center1 = cluster_centers[center_coords[i]]
                center2 = cluster_centers[center_coords[j]]
                dist = np.linalg.norm(center1 - center2)
                inter_cluster_distances.append(dist)
        
        # Calculate metrics
        avg_intra_distance = np.mean(intra_cluster_distances) if intra_cluster_distances else 0.0
        avg_inter_distance = np.mean(inter_cluster_distances) if inter_cluster_distances else 0.0
        
        # Silhouette-like score
        separation_score = avg_inter_distance / (avg_intra_distance + 1e-10)
        
        # Cluster size distribution
        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        size_variance = np.var(cluster_sizes) if len(cluster_sizes) > 1 else 0.0
        
        return {
            "num_clusters": len(clusters),
            "avg_intra_cluster_distance": float(avg_intra_distance),
            "avg_inter_cluster_distance": float(avg_inter_distance),
            "separation_score": float(separation_score),
            "cluster_size_variance": float(size_variance),
            "largest_cluster_size": int(max(cluster_sizes)) if cluster_sizes else 0,
            "smallest_cluster_size": int(min(cluster_sizes)) if cluster_sizes else 0
        }
        
    except Exception as e:
        print(f"Error analyzing cluster quality: {e}")
        return {"error": str(e)}

def get_cluster_representative_features(
    cluster_features: List[List[float]]
) -> List[float]:
    """
    Calculate representative features for a cluster.
    
    Args:
        cluster_features: List of feature vectors in the cluster
    
    Returns:
        Representative feature vector for the cluster
    """
    if not cluster_features:
        return []
    
    if len(cluster_features) == 1:
        return cluster_features[0]
    
    try:
        # Calculate mean features
        cluster_array = np.array(cluster_features)
        mean_features = np.mean(cluster_array, axis=0)
        
        # Optionally, find the feature vector closest to the mean
        distances = [np.linalg.norm(features - mean_features) for features in cluster_array]
        closest_idx = np.argmin(distances)
        
        # Return the closest actual feature vector rather than the mean
        # This ensures we return a "real" color combination that exists in the data
        return cluster_features[closest_idx]
        
    except Exception as e:
        print(f"Error calculating representative features: {e}")
        return cluster_features[0] if cluster_features else []