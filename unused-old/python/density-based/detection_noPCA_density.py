import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import config

# =============================================================================
# ADAPTIVE TWO-STAGE CLUSTERING WITHOUT PCA - DENSITY-BASED
# Χρησιμοποιεί density check για adaptive safety radius
# Euclidean distance + dynamic minimum size based on median
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


def calculate_adaptive_safety_radius_density(X_used, final_labels, final_centers,
                                             percentile=90, min_radius=0.5, max_radius=5.0):
    """
    Υβριδική με Density Check:
    - Percentile: Βασική ακτίνα
    - Size Check: Ποινή για λίγα δείγματα
    - Density Check: Ποινή για αραιά clusters (μεγάλη μέση απόσταση)
    """
    safety_radii = {}

    print(f"\n[ADAPTIVE RADIUS] Method: Hybrid + Density (percentile={percentile})")

    for cluster_id in np.unique(final_labels):
        cluster_mask = (final_labels == cluster_id)
        cluster_points = X_used[cluster_mask]
        cluster_size = len(cluster_points)
        center = final_centers[cluster_id]

        # Υπολογισμός αποστάσεων από το κέντρο
        distances = np.linalg.norm(cluster_points - center, axis=1)

        # 1. Βασική Ακτίνα (Percentile)
        radius = np.percentile(distances, percentile)

        # 2. Μέτρηση Πυκνότητας (Mean Distance)
        mean_dist = np.mean(distances)

        # Υπολογισμός Density Factor (Όσο μεγαλώνει το mean_dist, μικραίνει το factor)
        # Το 0.5 είναι ρυθμιστής ευαισθησίας (sensitivity)
        density_factor = 1.0 / (1.0 + 0.5 * mean_dist)

        # Εφαρμογή του Density Factor στην ακτίνα
        radius_before_density = radius
        radius *= density_factor

        # 3. Ποινή για πολύ μικρά clusters (Size Penalty)
        if cluster_size < 50:
            radius *= 0.80

        # 4. Όρια ασφαλείας (Hard Limits)
        radius = np.clip(radius, min_radius, max_radius)

        safety_radii[cluster_id] = radius

        print(f"   Cluster {cluster_id}: Size={cluster_size}")
        print(f"     -> Mean Dist (Sparsity): {mean_dist:.3f}, Density Factor: {density_factor:.3f}")
        print(f"     -> Radius: {radius_before_density:.3f} (Raw) -> {radius:.3f} (Final)")

    return safety_radii


def get_dynamic_min_size_smart(micro_labels):
    """
    Υπολογίζει dynamic minimum size βασισμένο στη διάμεσο των micro-clusters
    """
    # 1. Βρες πόσα σημεία έχει κάθε micro-cluster
    unique, counts = np.unique(micro_labels, return_counts=True)

    # 2. Υπολόγισε τη Διάμεσο (Median)
    median_size = np.median(counts)

    # 3. Ο Πορτιέρης: Το 20% της διαμέσου
    dynamic_limit = int(median_size * 0.20)

    # Πάντα βάζουμε ένα hard limit το 3 για ασφάλεια
    return max(3, dynamic_limit)


def run_two_stage_noPCA_density(X_scaled, n_micro=None, n_macro=None):
    """
    Two-Stage Clustering με Density-Based Adaptive Radius
    """

    # Use config parameters as defaults
    if n_micro is None:
        n_micro = config.DEFAULT_N_MICRO
    if n_macro is None:
        n_macro = 5

    # Validation check
    if n_macro != 5:
        print(f"⚠️ WARNING: Η εκφώνηση ζητάει 5 συστάδες, χρησιμοποιείτε {n_macro}")

    FIXED_SIGMA = config.FIXED_SIGMA

    print(f"\n[PHASE 2] Two-Stage Clustering WITHOUT PCA (Density-Based Adaptive Strategy)")
    print(f"   -> EUCLIDEAN DISTANCE with density-aware safety radius")
    print(f"   -> Dynamic minimum size based on median cluster size")

    start_algo = time.time()

    # --- A. NO PCA - Use Scaled Data Directly ---
    X_used = X_scaled

    # --- B. STAGE 1: Micro-Clustering ---
    kmeans_micro = KMeans(n_clusters=n_micro, random_state=config.KMEANS_RANDOM_STATE,
                         n_init=config.KMEANS_N_INIT_MICRO)
    micro_labels = kmeans_micro.fit_predict(X_used)
    micro_centers = kmeans_micro.cluster_centers_
    unique, counts = np.unique(micro_labels, return_counts=True)

    # Dynamic minimum size calculation
    median_size = np.median(counts)
    MIN_MICRO_SIZE = min(3, int(median_size * 0.2))
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

    # --- E. HYBRID OUTLIER DETECTION ---

    # 1. Υπολογισμός Micro-Distance
    my_micro_centers = micro_centers[micro_labels]
    micro_distances = np.linalg.norm(X_used - my_micro_centers, axis=1)

    # 2. Υπολογισμός MACRO-Distance
    my_macro_centers = final_centers[final_labels]
    macro_distances = np.linalg.norm(X_used - my_macro_centers, axis=1)

    # 3. Density Rule
    is_isolated_mask = np.array([micro_counts_map[label] < MIN_MICRO_SIZE for label in micro_labels])

    analysis_df = pd.DataFrame({
        'cluster': final_labels,
        'micro_dist': micro_distances,
        'macro_dist': macro_distances,
        'is_isolated': is_isolated_mask
    })

    # 4. ΥΠΟΛΟΓΙΣΜΟΣ ΔΥΝΑΜΙΚΗΣ ΑΣΠΙΔΑΣ με Density Check
    safety_radii_dict = calculate_adaptive_safety_radius_density(
        X_used, final_labels, final_centers,
        percentile=90,
        min_radius=0.5,
        max_radius=5.0
    )

    analysis_df['safety_radius'] = analysis_df['cluster'].map(safety_radii_dict)

    # 5. Υπολογισμός Thresholds
    cluster_stats = analysis_df.groupby('cluster')['micro_dist'].agg(['mean', 'std']).reset_index()
    cluster_stats['std'] = cluster_stats['std'].fillna(0)
    analysis_df = analysis_df.merge(cluster_stats, on='cluster', how='left')

    analysis_df['threshold'] = analysis_df['mean'] + (FIXED_SIGMA * analysis_df['std'])

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
    print(f"{'ID':<5} {'Count':<8} {'MicroMean':<10} {'MicroStd':<10} {'Threshold':<10} {'SafetyR':<10} {'Outliers':<8}")
    print("-" * 80)
    for i in range(n_macro):
        stats = analysis_df[analysis_df['cluster'] == i]
        if stats.empty: continue
        row = cluster_stats[cluster_stats['cluster'] == i].iloc[0]
        t = row['mean'] + (FIXED_SIGMA * row['std'])
        n_out = stats['is_outlier'].sum()
        radius = safety_radii_dict[i]
        print(f"{i:<5} {len(stats):<8} {row['mean']:<10.4f} {row['std']:<10.4f} {t:<10.4f} {radius:<10.4f} {n_out:<8}")

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

    plt.title(f"Two-Stage Clustering (Density-Based)\nFile: {filename}")
    plt.xlabel("Scaled X1")
    plt.ylabel("Scaled X2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    dataset_name = filename.replace('.txt', '').replace('.csv', '')
    output_dir = config.get_output_path('detection_noPCA_density', dataset_name, 'png')
    out_name = os.path.join(output_dir, f"plot_density_{dataset_name}.png")
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

    plt.title(f"Outliers Detection (Density-Based) - Original Coordinates\nFile: {filename}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)

    dataset_name = filename.replace('.txt', '').replace('.csv', '')
    output_dir = config.get_output_path('detection_noPCA_density', dataset_name, 'png')
    out_name = os.path.join(output_dir, f"plot_density_raw_{dataset_name}.png")
    plt.savefig(out_name, dpi=150)
    print(f"[RAW VISUALIZATION] Plot saved as: {out_name}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Density-Based Two-Stage Outlier Detection')
    parser.add_argument('filename', nargs='?', default=None)

    args = parser.parse_args()

    filename = args.filename
    if filename is None:
        filename = "202526files/data202526a_corrupted.txt"
        if not os.path.exists(filename):
            print("Usage: python detection_noPCA_density.py <filename>")
            sys.exit(1)

    # Create output directories
    config.create_output_directories()

    start_total = time.time()

    try:
        print(f"{'='*80}")
        print(f"DENSITY-BASED TWO-STAGE OUTLIER DETECTION (NO PCA)")
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

    # Run density-based algorithm
    results, centers = run_two_stage_noPCA_density(X_scaled)

    # Results
    num_outliers = results['is_outlier'].sum()
    end_total = time.time()

    print("\n" + "="*80)
    print(f"FINAL SUMMARY (DENSITY-BASED STRATEGY)")
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

        output_dir = config.get_output_path('detection_noPCA_density', dataset_name, 'csv')
        csv_name = os.path.join(output_dir, f"density_outliers_{dataset_name}.csv")
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
    print("✅ Density-based detection completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
