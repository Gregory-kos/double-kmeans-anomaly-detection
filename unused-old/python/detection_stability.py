import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv
from scipy.stats import chi2
import config

# =============================================================================
# STABILITY-BASED OUTLIER DETECTION
# =============================================================================

def sota_preprocess(df):
    """Preprocessing & Cleaning"""
    print("\n[PHASE 1] Preprocessing & Cleaning...")
    initial_rows = len(df)
    df_clean = df.copy()

    # 1. Conversion to Numeric
    cols = [0, 1]
    for c in cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')

    # 2. Handle Infinite Values
    try:
        temp_vals = df_clean.iloc[:, :2].astype(float)
        num_inf = np.isinf(temp_vals).values.sum()
        if num_inf > 0:
            print(f"   -> Detected {num_inf} infinite values. Replacing with NaN.")
            df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    except Exception as e:
        print(f"   [Warning] Infinity check failed: {e}")

    # 3. Drop NaN
    df_clean.dropna(inplace=True)

    # 4. Smart Deduplication
    temp_rounded = df_clean.round(6)
    duplicates_mask = temp_rounded.duplicated()
    num_dupes = duplicates_mask.sum()
    if num_dupes > 0:
        df_clean = df_clean[~duplicates_mask]
        print(f"   -> Removed {num_dupes} duplicate rows.")

    # Save original indices BEFORE reset_index
    df_clean['orig_index'] = df_clean.index
    df_clean.reset_index(drop=True, inplace=True)
    print(f"   -> Final dataset: {len(df_clean)} rows")

    # 5. Scaling (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean.iloc[:, :2].values)

    return df_clean, X_scaled


def run_stability_based_detection(X_scaled, n_clusters=5, n_runs=50, instability_threshold=0.4):
    """
    Stability-Based Outlier Detection

    Ιδέα:
    - Τρέχουμε k-means πολλές φορές με διαφορετικά seeds
    - Outlier = σημείο που αλλάζει cluster συχνά
    - instability(x) = #cluster_changes / runs
    - Αν instability > threshold → outlier

    Parameters:
    -----------
    X_scaled : array
        Scaled data
    n_clusters : int
        Number of clusters (default: 5)
    n_runs : int
        Number of k-means runs with different seeds (default: 50)
    instability_threshold : float
        Threshold for outlier detection (default: 0.4)
    """

    print(f"\n[PHASE 2] Stability-Based Outlier Detection")
    print(f"   -> Running k-means {n_runs} times with different seeds")
    print(f"   -> Number of clusters: {n_clusters}")
    print(f"   -> Instability threshold: {instability_threshold}")

    start_algo = time.time()

    n_points = len(X_scaled)

    # Store cluster assignments for each run
    cluster_assignments = np.zeros((n_runs, n_points), dtype=int)
    all_centers = []

    # Run k-means multiple times with different seeds
    for run_idx in range(n_runs):
        seed = run_idx * 42  # Different seed for each run
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        cluster_assignments[run_idx] = labels
        all_centers.append(kmeans.cluster_centers_)

        if (run_idx + 1) % 10 == 0:
            print(f"   -> Completed {run_idx + 1}/{n_runs} runs")

    # Calculate instability for each point
    print("\n[PHASE 3] Calculating Frequency-Based Instability Scores")

    instability_scores = np.zeros(n_points)

    for point_idx in range(n_points):
        # Get cluster assignments for this point across all runs
        point_clusters = cluster_assignments[:, point_idx]

        # FIX #1: FREQUENCY-BASED INSTABILITY (not unique count)
        # instability = 1 - (max_frequency / total_runs)
        unique, counts = np.unique(point_clusters, return_counts=True)
        max_freq = counts.max() / n_runs
        instability_scores[point_idx] = 1 - max_freq

    print(f"   -> Mean instability: {instability_scores.mean():.4f}")
    print(f"   -> Median instability: {np.median(instability_scores):.4f}")
    print(f"   -> Max instability: {instability_scores.max():.4f}")

    # Use the most common cluster assignment as the final cluster
    final_labels = np.zeros(n_points, dtype=int)
    for point_idx in range(n_points):
        # Find most common cluster for this point
        unique, counts = np.unique(cluster_assignments[:, point_idx], return_counts=True)
        final_labels[point_idx] = unique[np.argmax(counts)]

    # Calculate average centers across all runs
    avg_centers = np.mean(all_centers, axis=0)

    # Calculate distances to all cluster centers for each point
    print("\n[PHASE 4] Calculating Distances & Border Detection")

    distances_to_assigned = np.zeros(n_points)
    distance_ratios = np.zeros(n_points)

    for i in range(n_points):
        # Distance to all centers
        dists_to_all = np.linalg.norm(X_scaled[i] - avg_centers, axis=1)

        # Sort to get closest and second closest
        sorted_dists = np.sort(dists_to_all)
        d1 = sorted_dists[0]  # Closest
        d2 = sorted_dists[1] if len(sorted_dists) > 1 else d1  # Second closest

        distances_to_assigned[i] = dists_to_all[final_labels[i]]

        # FIX #4: DISTANCE RATIO FILTER (border detection)
        # If ratio > 0.8 → border point, not outlier
        if d2 > 0:
            distance_ratios[i] = d1 / d2
        else:
            distance_ratios[i] = 1.0

    # FIX #2: CLUSTER-AWARE THRESHOLDS (not global)
    print("\n[PHASE 5] Cluster-Aware Outlier Detection")

    is_outlier = np.zeros(n_points, dtype=int)
    cluster_stats = []

    for cluster_id in range(n_clusters):
        cluster_mask = (final_labels == cluster_id)
        n_in_cluster = cluster_mask.sum()

        if n_in_cluster == 0:
            continue

        cluster_points = X_scaled[cluster_mask]
        cluster_instability = instability_scores[cluster_mask]
        cluster_distances = distances_to_assigned[cluster_mask]
        cluster_ratios = distance_ratios[cluster_mask]

        # Calculate cluster statistics
        mean_dist = cluster_distances.mean()
        std_dist = cluster_distances.std()

        # FIX #3: ΔΙΦΑΣΙΚΗ ΑΠΟΦΑΣΗ (AND logic, not multiplication)
        # Outlier if: (instability > τ1) AND (distance > τ2) AND (not border)
        instability_threshold_cluster = 0.2  # 20% instability (more sensitive)
        distance_threshold_cluster = mean_dist + 1.5 * std_dist  # More lenient
        border_threshold = 0.85  # More lenient border detection

        # Identify candidates
        suspects_mask = (
            (cluster_instability > instability_threshold_cluster) &
            (cluster_distances > distance_threshold_cluster) &
            (cluster_ratios < border_threshold)  # NOT border
        )

        # Mark as outliers
        cluster_indices = np.where(cluster_mask)[0]
        suspect_indices = cluster_indices[suspects_mask]
        is_outlier[suspect_indices] = 1

        n_outliers = suspects_mask.sum()

        cluster_stats.append({
            'id': cluster_id,
            'size': n_in_cluster,
            'mean_dist': mean_dist,
            'std_dist': std_dist,
            'dist_threshold': distance_threshold_cluster,
            'outliers': n_outliers
        })

        print(f"   Cluster {cluster_id}: size={n_in_cluster}, "
              f"dist_thr={distance_threshold_cluster:.3f}, outliers={n_outliers}")

    # FIX #5: MAHALANOBIS VALIDATION (final filter)
    print("\n[PHASE 6] Mahalanobis Validation")

    from scipy.linalg import pinv
    from scipy.stats import chi2

    outlier_indices = np.where(is_outlier == 1)[0]
    validated_outliers = []

    chi2_threshold = chi2.ppf(0.99, df=2)  # 99% threshold for 2D = 9.21

    for cluster_id in range(n_clusters):
        cluster_mask = (final_labels == cluster_id)
        cluster_points = X_scaled[cluster_mask]

        if len(cluster_points) < 3:
            continue

        # Calculate covariance matrix
        cov = np.cov(cluster_points.T)
        cov += 1e-6 * np.eye(cov.shape[0])  # Regularization

        try:
            cov_inv = pinv(cov)
            center = avg_centers[cluster_id]

            # Check outlier candidates from this cluster
            cluster_outlier_mask = cluster_mask & (is_outlier == 1)
            cluster_outlier_indices = np.where(cluster_outlier_mask)[0]

            for idx in cluster_outlier_indices:
                # Mahalanobis distance squared
                diff = X_scaled[idx] - center
                mahal_dist_sq = diff @ cov_inv @ diff

                # Validate: if passes chi-square test, it's truly an outlier
                if mahal_dist_sq > chi2_threshold:
                    validated_outliers.append(idx)
        except:
            # If Mahalanobis fails, keep original outliers from this cluster
            validated_outliers.extend(np.where(cluster_outlier_mask)[0].tolist())

    # Update is_outlier with validated results
    is_outlier = np.zeros(n_points, dtype=int)
    is_outlier[validated_outliers] = 1

    print(f"   -> Before validation: {len(outlier_indices)} candidates")
    print(f"   -> After validation:  {len(validated_outliers)} confirmed outliers")
    print(f"   -> Chi-square threshold (99%): {chi2_threshold:.2f}")

    # Create results dataframe
    results_df = pd.DataFrame({
        'cluster': final_labels,
        'instability': instability_scores,
        'distance': distances_to_assigned,
        'distance_ratio': distance_ratios,
        'is_outlier': is_outlier,
        'scaled_x': X_scaled[:, 0],
        'scaled_y': X_scaled[:, 1]
    })

    end_algo = time.time()

    # Print Statistics
    print(f"\n[FINAL STATISTICS]")
    print(f"   -> Algorithm finished in {end_algo - start_algo:.4f} sec")
    print(f"   -> Total outliers detected: {is_outlier.sum()}")
    print(f"   -> Detection rate: {100 * is_outlier.sum() / n_points:.2f}%")

    print("\n[CLUSTER SUMMARY]")
    print(f"{'ID':<5} {'Size':<8} {'Outliers':<10} {'Rate%':<8}")
    print("-" * 40)
    for stats in cluster_stats:
        rate = 100 * stats['outliers'] / stats['size'] if stats['size'] > 0 else 0
        print(f"{stats['id']:<5} {stats['size']:<8} {stats['outliers']:<10} {rate:<8.2f}")

    # Print top outliers by instability
    outlier_mask = (is_outlier == 1)
    if outlier_mask.sum() > 0:
        print("\n[TOP 10 DETECTED OUTLIERS BY INSTABILITY]")
        outlier_indices = np.where(outlier_mask)[0]
        outlier_instabilities = instability_scores[outlier_indices]
        top_indices = outlier_indices[np.argsort(outlier_instabilities)[::-1][:10]]

        for idx in top_indices:
            print(f"   Point {idx}: instability={instability_scores[idx]:.4f}, "
                  f"distance={distances_to_assigned[idx]:.4f}, "
                  f"ratio={distance_ratios[idx]:.4f}")

    return results_df, avg_centers


def plot_stability_results(results_df, centers, filename):
    """Plot stability-based detection results"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Cluster assignments with outliers
    ax1 = axes[0]
    inliers = results_df[results_df['is_outlier'] == 0]
    outliers = results_df[results_df['is_outlier'] == 1]

    ax1.scatter(inliers['scaled_x'], inliers['scaled_y'],
                c=inliers['cluster'], cmap='viridis', s=20, alpha=0.6, label='Inliers')

    if not outliers.empty:
        ax1.scatter(outliers['scaled_x'], outliers['scaled_y'],
                    c='red', marker='x', s=80, linewidth=2, label='Outliers (Unstable)')

    ax1.scatter(centers[:, 0], centers[:, 1],
                c='black', s=300, marker='*', edgecolors='white', label='Centroids')

    ax1.set_title(f"Stability-Based Detection\nFile: {filename}")
    ax1.set_xlabel("Scaled X1")
    ax1.set_ylabel("Scaled X2")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Instability heatmap
    ax2 = axes[1]
    scatter = ax2.scatter(results_df['scaled_x'], results_df['scaled_y'],
                         c=results_df['instability'], cmap='YlOrRd', s=30, alpha=0.7)

    ax2.scatter(centers[:, 0], centers[:, 1],
                c='black', s=300, marker='*', edgecolors='white', label='Centroids')

    plt.colorbar(scatter, ax=ax2, label='Instability Score')
    ax2.set_title(f"Instability Heatmap\nFile: {filename}")
    ax2.set_xlabel("Scaled X1")
    ax2.set_ylabel("Scaled X2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    dataset_name = filename.replace('.txt', '').replace('.csv', '')
    output_dir = config.get_output_path('detection_stability', dataset_name, 'png')
    out_name = os.path.join(output_dir, f"plot_stability_{dataset_name}.png")
    plt.savefig(out_name, dpi=150)
    print(f"\n[VISUALIZATION] Plots saved as: {out_name}")
    plt.close()


def plot_raw_outliers(df_raw, df_clean, results_df, filename):
    """Plot outliers in original coordinate space"""
    plt.figure(figsize=(12, 8))

    # Get outlier indices
    outlier_indices = results_df.index[results_df['is_outlier'] == 1]
    orig_indices = df_clean.loc[outlier_indices, 'orig_index'].values

    # Create mask for inliers
    all_indices = df_raw.index
    inlier_mask = ~all_indices.isin(orig_indices)

    # Get numeric values
    inliers_x = pd.to_numeric(df_raw.loc[inlier_mask, 0], errors='coerce')
    inliers_y = pd.to_numeric(df_raw.loc[inlier_mask, 1], errors='coerce')

    outliers_x = pd.to_numeric(df_raw.iloc[orig_indices, 0], errors='coerce')
    outliers_y = pd.to_numeric(df_raw.iloc[orig_indices, 1], errors='coerce')

    # Plot
    plt.scatter(inliers_x, inliers_y, c='blue', s=20, alpha=0.6, label='Inliers')
    if len(orig_indices) > 0:
        plt.scatter(outliers_x, outliers_y, c='red', s=80, marker='x', linewidths=2,
                   label='Unstable Outliers')

    plt.title(f"Stability-Based Outliers in Original Coordinates\nFile: {filename}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    dataset_name = filename.replace('.txt', '').replace('.csv', '')
    output_dir = config.get_output_path('detection_stability', dataset_name, 'png')
    out_name = os.path.join(output_dir, f"plot_stability_raw_{dataset_name}.png")
    plt.savefig(out_name, dpi=150)
    print(f"[RAW VISUALIZATION] Plot saved as: {out_name}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Stability-Based Outlier Detection')
    parser.add_argument('filename', nargs='?', default=None,
                       help='Input data file')
    parser.add_argument('--n_runs', type=int, default=50,
                       help='Number of k-means runs (default: 50)')
    parser.add_argument('--threshold', type=float, default=0.4,
                       help='Instability threshold (default: 0.4)')
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of clusters (default: 5)')

    args = parser.parse_args()

    filename = args.filename
    if filename is None:
        filename = "synthetic_datasets/synth_concentric/ground_truth.csv"
        if not os.path.exists(filename):
            print("Usage: python detection_stability.py <filename>")
            sys.exit(1)

    # Create output directories
    config.create_output_directories()

    start_total = time.time()

    try:
        print(f"{'='*80}")
        print(f"🔬 STABILITY-BASED OUTLIER DETECTION")
        print(f"{'='*80}")
        print(f"Processing: {filename}")
        df_raw = pd.read_csv(filename, sep=',', header=None, on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # Preprocess
    df_clean, X_scaled = sota_preprocess(df_raw)

    if len(df_clean) < 10:
        print("Not enough data.")
        sys.exit(1)

    # Run stability-based detection
    results, centers = run_stability_based_detection(
        X_scaled,
        n_clusters=args.n_clusters,
        n_runs=args.n_runs,
        instability_threshold=args.threshold
    )

    # Results
    num_outliers = results['is_outlier'].sum()
    end_total = time.time()

    print("\n" + "="*80)
    print(f"FINAL SUMMARY (STABILITY-BASED DETECTION)")
    print("="*80)
    print(f"Total Time:       {end_total - start_total:.4f} sec")
    print(f"Outliers Found:   {num_outliers}")
    print(f"Parameters:")
    print(f"  - Number of runs:        {args.n_runs}")
    print(f"  - Instability threshold: {args.threshold}")
    print(f"  - Number of clusters:    {args.n_clusters}")

    if num_outliers > 0:
        print("\n[DETECTED OUTLIERS - ORIGINAL COORDINATES (Top 20)]")

        # Map back to original raw data using orig_index
        outlier_indices = results.index[results['is_outlier'] == 1]
        orig_indices = df_clean.loc[outlier_indices, 'orig_index'].values

        # Sort by instability score
        instability_sorted = results.loc[outlier_indices].sort_values('instability', ascending=False)
        top_indices = df_clean.loc[instability_sorted.index, 'orig_index'].values[:20]

        # Print with instability scores
        print(f"{'X':<20} {'Y':<20} {'Instability':<12}")
        print("-" * 52)
        for idx in top_indices:
            result_idx = df_clean[df_clean['orig_index'] == idx].index[0]
            inst_score = results.loc[result_idx, 'instability']
            x_val = df_raw.iloc[idx, 0]
            y_val = df_raw.iloc[idx, 1]
            print(f"{str(x_val):<20} {str(y_val):<20} {inst_score:<12.4f}")

        # Save outliers to CSV
        base_filename = os.path.basename(filename)
        if base_filename in ['data.txt', 'data.csv', 'ground_truth.csv']:
            parent_folder = os.path.basename(os.path.dirname(filename))
            dataset_name = parent_folder
        else:
            dataset_name = base_filename.replace('.txt', '').replace('.csv', '')

        output_dir = config.get_output_path('detection_stability', dataset_name, 'csv')
        csv_name = os.path.join(output_dir, f"stability_outliers_{dataset_name}.csv")

        # Add instability scores to output - only save x, y, instability
        outliers_data = df_raw.iloc[orig_indices, :2].copy()  # Only first 2 columns (x, y)
        outliers_data['instability'] = results.loc[outlier_indices, 'instability'].values
        outliers_data.to_csv(csv_name, index=False, header=False)
        print(f"\n-> Outliers saved to {csv_name}")

    # Generate plots
    base_filename = os.path.basename(filename)
    if base_filename in ['data.txt', 'data.csv', 'ground_truth.csv']:
        parent_folder = os.path.basename(os.path.dirname(filename))
        display_name = parent_folder
    else:
        display_name = base_filename

    # Visualization
    plot_stability_results(results, centers, display_name)
    plot_raw_outliers(df_raw, df_clean, results, display_name)

    print("\n" + "="*80)
    print("✅ Stability-based detection completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
