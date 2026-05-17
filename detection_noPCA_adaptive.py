import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from scipy import linalg
import config

# =============================================================================
# ADAPTIVE TWO-STAGE CLUSTERING WITHOUT PCA
# Βασισμένο σε πειραματική ανάλυση συνθετικών δεδομένων
# Χωρίς fine-tuning, χωρίς training - μόνο adaptive κανόνες
# MAHALANOBIS DISTANCE - λαμβάνει υπόψη τη συσχέτιση features
# =============================================================================

def calculate_mahalanobis_distances(points, center, cov_matrix=None, regularization=1e-6):
    """
    Υπολογίζει Mahalanobis distances από ένα σημείο (center) σε πολλά points

    Args:
        points: (N, D) array - τα σημεία
        center: (D,) array - το κέντρο
        cov_matrix: (D, D) array - covariance matrix (optional, υπολογίζεται αν None)
        regularization: float - regularization term για singular matrices

    Returns:
        distances: (N,) array - Mahalanobis distances
    """
    if len(points) < 2:
        # Fallback to Euclidean for very small clusters
        return np.linalg.norm(points - center, axis=1)

    # Υπολογισμός covariance matrix αν δεν δίνεται
    if cov_matrix is None:
        cov_matrix = np.cov(points.T)

    # Regularization για numerical stability (αποφυγή singular matrix)
    cov_matrix_reg = cov_matrix + regularization * np.eye(cov_matrix.shape[0])

    try:
        # Υπολογισμός inverse covariance matrix
        cov_inv = linalg.inv(cov_matrix_reg)

        # Υπολογισμός Mahalanobis distance για κάθε σημείο
        distances = np.array([
            mahalanobis(point, center, cov_inv)
            for point in points
        ])

        return distances

    except (linalg.LinAlgError, ValueError):
        # Fallback to Euclidean if Mahalanobis fails
        print("   ⚠️ Mahalanobis calculation failed, using Euclidean")
        return np.linalg.norm(points - center, axis=1)

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


def calculate_adaptive_safety_radius(X_used, final_labels, final_centers,
                                     min_radius=0.5, max_radius=5.0):
    """
    FULLY ADAPTIVE Safety Radius - NO fixed thresholds
    Percentile προσαρμόζεται στο RELATIVE cluster size (fraction του συνόλου)
    Όχι απόλυτοι αριθμοί (100, 500) - μόνο σχετικά κλάσματα
    MAHALANOBIS DISTANCE - λαμβάνει υπόψη τη συσχέτιση features
    """
    safety_radii = {}
    N_total = len(X_used)  # Συνολικός αριθμός σημείων

    print(f"\n[FULLY ADAPTIVE SAFETY RADIUS] Mahalanobis distance + relative cluster size")

    for cluster_id in np.unique(final_labels):
        cluster_mask = (final_labels == cluster_id)
        cluster_points = X_used[cluster_mask]
        cluster_size = len(cluster_points)

        center = final_centers[cluster_id]
        # MAHALANOBIS DISTANCE instead of Euclidean
        distances = calculate_mahalanobis_distances(cluster_points, center)

        # FULLY ADAPTIVE PERCENTILE - βασισμένο σε FRACTION του dataset
        # Αντί για απόλυτα μεγέθη (100, 500), χρησιμοποιούμε σχετικά κλάσματα
        fraction = cluster_size / N_total

        # Μικρά clusters (<10% του συνόλου) → πιο conservative (85%)
        # Μεσαία clusters → standard (90%)
        # Μεγάλα clusters (>33% του συνόλου) → πιο lenient (95%)
        if fraction < 0.10:
            percentile = 85
        elif fraction > 0.33:
            percentile = 95
        else:
            percentile = 90

        radius = np.percentile(distances, percentile)

        # Προσαρμογή για πολύ μικρά clusters (<5% του συνόλου)
        # Αντικαθιστά το "if cluster_size < 50" με relative threshold
        if fraction < 0.05:
            radius *= 0.8

        # Όρια ασφαλείας
        radius = np.clip(radius, min_radius, max_radius)

        safety_radii[cluster_id] = radius

        print(f"   Cluster {cluster_id}: size={cluster_size} ({fraction:.1%}), "
              f"percentile={percentile}, radius={radius:.4f}")

    return safety_radii


def run_two_stage_noPCA_adaptive(X_scaled, n_micro=None, n_macro=None):
    """
    Two-Stage Clustering με ADAPTIVE rules
    - Adaptive sigma (όχι fixed)
    - Adaptive safety radius percentile
    - Dynamic minimum micro-cluster size
    """

    # Use config parameters as defaults
    if n_micro is None:
        n_micro = config.DEFAULT_N_MICRO
    if n_macro is None:
        n_macro = 5

    # Validation check
    if n_macro != 5:
        print(f"⚠️ WARNING: Η εκφώνηση ζητάει 5 συστάδες, χρησιμοποιείτε {n_macro}")

    MIN_MICRO_SIZE = 3  # Hard minimum για ασφάλεια

    print(f"\n[PHASE 2] Two-Stage Clustering WITHOUT PCA (Fully Adaptive Strategy)")
    print(f"   -> MAHALANOBIS DISTANCE - accounts for feature correlation")
    print(f"   -> NO fixed sigma - adaptive per cluster")
    print(f"   -> NO fixed percentile - adaptive to cluster size (relative fractions)")

    start_algo = time.time()

    # --- A. NO PCA - Use Scaled Data Directly ---
    X_used = X_scaled

    # --- B. STAGE 1: Micro-Clustering ---
    kmeans_micro = KMeans(n_clusters=n_micro, random_state=config.KMEANS_RANDOM_STATE,
                         n_init=config.KMEANS_N_INIT_MICRO)
    micro_labels = kmeans_micro.fit_predict(X_used)
    micro_centers = kmeans_micro.cluster_centers_
    unique, counts = np.unique(micro_labels, return_counts=True)

    median_size = np.median(counts)
    print(f"   -> Dynamic Minimum Micro-Cluster Size: {MIN_MICRO_SIZE} (Median: {median_size})")

    # --- C. DENSITY FILTERING ---
    unique_micro, counts_micro = np.unique(micro_labels, return_counts=True)
    micro_counts_map = dict(zip(unique_micro, counts_micro))

    valid_micro_indices = []
    valid_weights = []
    for i in range(n_micro):
        if micro_counts_map.get(i, 0) >= MIN_MICRO_SIZE:
            valid_micro_indices.append(i)
            valid_weights.append(micro_counts_map.get(i, 0))

    if len(valid_micro_indices) < n_macro:
        valid_micro_centers = micro_centers
    else:
        valid_micro_centers = micro_centers[valid_micro_indices]

    # --- D. STAGE 2: Macro-Clustering ---
    kmeans_macro = KMeans(n_clusters=n_macro, random_state=config.KMEANS_RANDOM_STATE,
                         n_init=config.KMEANS_N_INIT_MACRO)
    kmeans_macro.fit(valid_micro_centers, sample_weight=valid_weights)
    final_centers = kmeans_macro.cluster_centers_

    # Ανάθεση labels
    macro_labels_for_all_micro = kmeans_macro.predict(micro_centers)
    final_labels = macro_labels_for_all_micro[micro_labels]

    # --- E. HYBRID OUTLIER DETECTION με ADAPTIVE RULES ---

    # 1. Υπολογισμός Micro-Distance με MAHALANOBIS
    print(f"   -> Computing Mahalanobis distances for micro-clusters...")
    micro_distances = np.zeros(len(X_used))
    for micro_id in range(n_micro):
        mask = (micro_labels == micro_id)
        if mask.sum() > 1:  # Need at least 2 points for covariance
            cluster_points = X_used[mask]
            center = micro_centers[micro_id]
            micro_distances[mask] = calculate_mahalanobis_distances(cluster_points, center)
        else:
            # Fallback to Euclidean for very small clusters
            center = micro_centers[micro_id]
            micro_distances[mask] = np.linalg.norm(X_used[mask] - center, axis=1)

    # 2. Υπολογισμός MACRO-Distance με MAHALANOBIS
    print(f"   -> Computing Mahalanobis distances for macro-clusters...")
    macro_distances = np.zeros(len(X_used))
    for macro_id in range(n_macro):
        mask = (final_labels == macro_id)
        if mask.sum() > 1:  # Need at least 2 points for covariance
            cluster_points = X_used[mask]
            center = final_centers[macro_id]
            macro_distances[mask] = calculate_mahalanobis_distances(cluster_points, center)
        else:
            # Fallback to Euclidean for very small clusters
            center = final_centers[macro_id]
            macro_distances[mask] = np.linalg.norm(X_used[mask] - center, axis=1)

    # 3. Density Rule
    is_isolated_mask = np.array([micro_counts_map[label] < MIN_MICRO_SIZE for label in micro_labels])

    analysis_df = pd.DataFrame({
        'cluster': final_labels,
        'micro_dist': micro_distances,
        'macro_dist': macro_distances,
        'is_isolated': is_isolated_mask
    })

    # 4. ADAPTIVE SAFETY RADIUS (percentile adapts to cluster size)
    safety_radii_dict = calculate_adaptive_safety_radius(
        X_used, final_labels, final_centers,
        min_radius=0.5,
        max_radius=5.0
    )

    analysis_df['safety_radius'] = analysis_df['cluster'].map(safety_radii_dict)

    # 5. ADAPTIVE SIGMA THRESHOLDS (NO fixed sigma)
    print("\n[ADAPTIVE SIGMA] Calculating per-cluster adaptive thresholds")

    cluster_stats = analysis_df.groupby('cluster')['micro_dist'].agg(['mean', 'std']).reset_index()
    cluster_stats['std'] = cluster_stats['std'].fillna(0)

    # ADAPTIVE SIGMA: cluster_ratio = std / mean
    # Υψηλό ratio → μεγάλη διακύμανση → χρειάζεται μεγαλύτερο sigma
    cluster_stats['cluster_ratio'] = cluster_stats['std'] / (cluster_stats['mean'] + 1e-6)
    cluster_stats['adaptive_sigma'] = np.clip(2.5 + cluster_stats['cluster_ratio'], 2.5, 4.5)

    analysis_df = analysis_df.merge(cluster_stats[['cluster', 'mean', 'std', 'adaptive_sigma']],
                                     on='cluster', how='left')

    # Threshold με adaptive sigma
    analysis_df['threshold'] = analysis_df['mean'] + (analysis_df['adaptive_sigma'] * analysis_df['std'])

    print(f"   -> Sigma range: [{cluster_stats['adaptive_sigma'].min():.2f}, "
          f"{cluster_stats['adaptive_sigma'].max():.2f}]")

    # 6. ΤΕΛΙΚΗ ΑΠΟΦΑΣΗ
    suspects = (analysis_df['micro_dist'] > analysis_df['threshold']) | (analysis_df['is_isolated'] == True)
    analysis_df['is_outlier'] = (suspects &
                                 (analysis_df['macro_dist'] > analysis_df['safety_radius'])).astype(int)

    # Coordinates for plotting
    analysis_df['scaled_x'] = X_used[:, 0]
    analysis_df['scaled_y'] = X_used[:, 1]

    end_algo = time.time()

    # Debug info
    n_saved = np.sum(suspects & (analysis_df['macro_dist'] <= analysis_df['safety_radius']))
    print(f"\n   -> Safety Shield: Saved {n_saved} points from false positives")
    print(f"   -> Algorithm finished in {end_algo - start_algo:.4f} sec")

    # Print Stats
    print("\n[CLUSTER STATISTICS]")
    print(f"{'ID':<5} {'Count':<8} {'MicroMean':<10} {'MicroStd':<10} {'AdaptΣ':<8} {'Threshold':<10} {'SafetyR':<10} {'Outliers':<8}")
    print("-" * 90)
    for i in range(n_macro):
        stats = analysis_df[analysis_df['cluster'] == i]
        if stats.empty: continue
        row = cluster_stats[cluster_stats['cluster'] == i].iloc[0]
        adaptive_sig = row['adaptive_sigma']
        t = row['mean'] + (adaptive_sig * row['std'])
        n_out = stats['is_outlier'].sum()
        radius = safety_radii_dict[i]
        print(f"{i:<5} {len(stats):<8} {row['mean']:<10.4f} {row['std']:<10.4f} "
              f"{adaptive_sig:<8.2f} {t:<10.4f} {radius:<10.4f} {n_out:<8}")

    return analysis_df, final_centers


def plot_scaled_space(results_df, centers, filename):
    """Plot in scaled space"""
    plt.figure(figsize=(12, 8))

    inliers = results_df[results_df['is_outlier'] == 0]
    outliers = results_df[results_df['is_outlier'] == 1]

    plt.scatter(inliers['scaled_x'], inliers['scaled_y'],
                c=inliers['cluster'], cmap='viridis', s=20, alpha=0.6, label='Inliers')

    if not outliers.empty:
        plt.scatter(outliers['scaled_x'], outliers['scaled_y'],
                    c='red', marker='x', s=80, linewidth=2, label='Outliers')

    plt.scatter(centers[:, 0], centers[:, 1],
                c='black', s=300, marker='*', edgecolors='white', label='Centroids')

    plt.title(f"Adaptive Two-Stage Clustering (NO PCA)\nFile: {filename}")
    plt.xlabel("Scaled X1")
    plt.ylabel("Scaled X2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    dataset_name = filename.replace('.txt', '').replace('.csv', '')
    output_dir = config.get_output_path('detection_noPCA_adaptive', dataset_name, 'png')
    out_name = os.path.join(output_dir, f"plot_adaptive_{dataset_name}.png")
    plt.savefig(out_name, dpi=150)
    print(f"\n[SCALED SPACE VISUALIZATION] Plot saved as: {out_name}")
    plt.close()


def plot_raw_outliers(df_raw, df_clean, results_df, filename):
    """Plot outliers in original coordinates"""
    plt.figure(figsize=(12, 8))

    outlier_indices = results_df.index[results_df['is_outlier'] == 1]
    orig_indices = df_clean.loc[outlier_indices, 'orig_index'].values

    all_indices = df_raw.index
    inlier_mask = ~all_indices.isin(orig_indices)

    inliers_x = pd.to_numeric(df_raw.loc[inlier_mask, 0], errors='coerce')
    inliers_y = pd.to_numeric(df_raw.loc[inlier_mask, 1], errors='coerce')

    outliers_x = pd.to_numeric(df_raw.iloc[orig_indices, 0], errors='coerce')
    outliers_y = pd.to_numeric(df_raw.iloc[orig_indices, 1], errors='coerce')

    plt.scatter(inliers_x, inliers_y, c='blue', s=20, alpha=0.6, label='Inliers')
    if len(orig_indices) > 0:
        plt.scatter(outliers_x, outliers_y, c='red', s=80, marker='x', linewidths=2, label='Outliers')

    plt.title(f"Adaptive Outlier Detection - Original Coordinates\nFile: {filename}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)

    dataset_name = filename.replace('.txt', '').replace('.csv', '')
    output_dir = config.get_output_path('detection_noPCA_adaptive', dataset_name, 'png')
    out_name = os.path.join(output_dir, f"plot_adaptive_raw_{dataset_name}.png")
    plt.savefig(out_name, dpi=150)
    print(f"[RAW VISUALIZATION] Plot saved as: {out_name}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Adaptive Two-Stage Outlier Detection')
    parser.add_argument('filename', nargs='?', default=None)

    args = parser.parse_args()

    filename = args.filename
    if filename is None:
        filename = "synthetic_datasets/synth_concentric/ground_truth.csv"
        if not os.path.exists(filename):
            print("Usage: python detection_noPCA_adaptive.py <filename>")
            sys.exit(1)

    # Create output directories
    config.create_output_directories()

    start_total = time.time()

    try:
        print(f"{'='*80}")
        print(f"ADAPTIVE TWO-STAGE OUTLIER DETECTION (NO PCA)")
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

    # Run adaptive algorithm
    results, centers = run_two_stage_noPCA_adaptive(X_scaled)

    # Results
    num_outliers = results['is_outlier'].sum()
    end_total = time.time()

    print("\n" + "="*80)
    print(f"FINAL SUMMARY (ADAPTIVE STRATEGY)")
    print("="*80)
    print(f"Total Time:       {end_total - start_total:.4f} sec")
    print(f"Outliers Found:   {num_outliers}")

    if num_outliers > 0:
        print("\n[DETECTED OUTLIERS - ORIGINAL COORDINATES (Top 20)]")

        outlier_indices = results.index[results['is_outlier'] == 1]
        orig_indices = df_clean.loc[outlier_indices, 'orig_index'].values

        print(df_raw.iloc[orig_indices].head(20).to_string(index=False, header=False))

        # Save outliers
        base_filename = os.path.basename(filename)
        if base_filename in ['data.txt', 'data.csv', 'ground_truth.csv']:
            parent_folder = os.path.basename(os.path.dirname(filename))
            dataset_name = parent_folder
        else:
            dataset_name = base_filename.replace('.txt', '').replace('.csv', '')

        output_dir = config.get_output_path('detection_noPCA_adaptive', dataset_name, 'csv')
        csv_name = os.path.join(output_dir, f"adaptive_outliers_{dataset_name}.csv")
        df_raw.iloc[orig_indices].to_csv(csv_name, index=False, header=False)
        print(f"\n-> Outliers saved to {csv_name}")

    # Generate plots
    base_filename = os.path.basename(filename)
    if base_filename in ['data.txt', 'data.csv', 'ground_truth.csv']:
        parent_folder = os.path.basename(os.path.dirname(filename))
        display_name = parent_folder
    else:
        display_name = base_filename

    plot_scaled_space(results, centers, display_name)
    plot_raw_outliers(df_raw, df_clean, results, display_name)

    print("\n" + "="*80)
    print("✅ Adaptive detection completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
