import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import config

# =============================================================================
# STEP 1: SOTA PREPROCESSING
# =============================================================================
def sota_preprocess(df):
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

    # ✅ FIX: Save original indices BEFORE reset_index
    df_clean['orig_index'] = df_clean.index
    df_clean.reset_index(drop=True, inplace=True)
    print(f"   -> Final dataset: {len(df_clean)} rows")

    # 5. Scaling (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean.iloc[:, :2].values)
    
    return df_clean, X_scaled

# =============================================================================
# STEP 2: TWO-STAGE CLUSTERING & ADAPTIVE OUTLIER DETECTION
# =============================================================================

def run_two_stage_adaptive(X_scaled):

    # --- USE CONFIG PARAMETERS (same as detection_noPCA for consistency) ---
    n_points = len(X_scaled)

    n_micro = config.DEFAULT_N_MICRO
    n_macro = config.DEFAULT_N_MACRO
    MIN_MICRO_SIZE = config.MIN_MICRO_SIZE

    print(f"\n[PHASE 2] Two-Stage Clustering with PCA")
    print(f"   -> Dataset size: {n_points} points")
    print(f"   -> n_micro: {n_micro}")
    print(f"   -> n_macro: {n_macro}")
    print(f"   -> MIN_MICRO_SIZE: {MIN_MICRO_SIZE}")

    start_algo = time.time()

    # --- A. ALWAYS APPLY PCA (even for 2D data) ---
    print(f"   -> Applying PCA (from {X_scaled.shape[1]}D to {config.PCA_N_COMPONENTS}D)")
    pca = PCA(n_components=config.PCA_N_COMPONENTS)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"   -> PCA explained variance: {explained_var:.2%}")
    
    # --- B. STAGE 1: Micro-Clustering ---
    kmeans_micro = KMeans(n_clusters=n_micro, random_state=config.KMEANS_RANDOM_STATE, n_init=config.KMEANS_N_INIT_MICRO)
    micro_labels = kmeans_micro.fit_predict(X_pca)
    micro_centers = kmeans_micro.cluster_centers_
    
    # --- C. DENSITY FILTERING ---
    unique_micro, counts_micro = np.unique(micro_labels, return_counts=True)
    micro_counts_map = dict(zip(unique_micro, counts_micro))
    
    valid_micro_indices = [i for i in range(n_micro) if micro_counts_map.get(i, 0) >= MIN_MICRO_SIZE]
    
    if len(valid_micro_indices) < n_macro:
        valid_micro_centers = micro_centers
    else:
        valid_micro_centers = micro_centers[valid_micro_indices]

    # --- D. STAGE 2: Macro-Clustering ---
    kmeans_macro = KMeans(n_clusters=n_macro, random_state=config.KMEANS_RANDOM_STATE, n_init=config.KMEANS_N_INIT_MACRO)
    kmeans_macro.fit(valid_micro_centers)
    final_centers_pca = kmeans_macro.cluster_centers_

    # Ανάθεση labels
    macro_labels_for_all_micro = kmeans_macro.predict(micro_centers)
    final_labels = macro_labels_for_all_micro[micro_labels]

    # Use fixed MACRO_SAFETY_RADIUS from config
    MACRO_SAFETY_RADIUS = config.MACRO_SAFETY_RADIUS
    print(f"   -> MACRO_SAFETY_RADIUS: {MACRO_SAFETY_RADIUS:.4f}")

    # --- E. HYBRID OUTLIER DETECTION ---
    
    # 1. Υπολογισμός Micro-Distance
    my_micro_centers = micro_centers[micro_labels]
    micro_distances = np.linalg.norm(X_pca - my_micro_centers, axis=1)

    # 2. Υπολογισμός MACRO-Distance
    my_macro_centers = final_centers_pca[final_labels]
    macro_distances = np.linalg.norm(X_pca - my_macro_centers, axis=1)

    # 3. Density Rule
    is_isolated_mask = np.array([micro_counts_map[label] < MIN_MICRO_SIZE for label in micro_labels])

    analysis_df = pd.DataFrame({
        'cluster': final_labels,
        'micro_dist': micro_distances,
        'macro_dist': macro_distances,
        'is_isolated': is_isolated_mask
    })

    # 4. Υπολογισμός Thresholds (mean + sigma approach)
    cluster_stats = analysis_df.groupby('cluster')['micro_dist'].agg(['mean', 'std']).reset_index()
    cluster_stats['std'] = cluster_stats['std'].fillna(0)
    analysis_df = analysis_df.merge(cluster_stats, on='cluster', how='left')

    # Calculate threshold using FIXED_SIGMA from config (same as detection_noPCA)
    FIXED_SIGMA = config.FIXED_SIGMA
    analysis_df['threshold'] = analysis_df['mean'] + (FIXED_SIGMA * analysis_df['std'])

    # 5. ΤΕΛΙΚΗ ΑΠΟΦΑΣΗ ΜΕ ΤΗΝ "ΑΣΠΙΔΑ"
    suspects = (analysis_df['micro_dist'] > analysis_df['threshold']) | (analysis_df['is_isolated'] == True)
    analysis_df['is_outlier'] = (suspects & (analysis_df['macro_dist'] > MACRO_SAFETY_RADIUS)).astype(int)
    
    # Coordinates for plotting
    analysis_df['pca_x'] = X_pca[:, 0]
    analysis_df['pca_y'] = X_pca[:, 1]
    
    end_algo = time.time()
    
    # Debug info
    n_saved = np.sum(suspects & (analysis_df['macro_dist'] <= MACRO_SAFETY_RADIUS))
    print(f"   -> Safety Shield: Saved {n_saved} points that were falsely flagged inside clusters.")
    print(f"   -> Algorithm finished in {end_algo - start_algo:.4f} sec")

    # Print Stats
    print("\n[CLUSTER STATISTICS]")
    print(f"{'ID':<5} {'Count':<8} {'MicroMean':<10} {'MicroStd':<10} {'Threshold':<10} {'Outliers':<8}")
    print("-" * 65)
    for i in range(n_macro):
        stats = analysis_df[analysis_df['cluster'] == i]
        if stats.empty: continue
        row = cluster_stats[cluster_stats['cluster'] == i].iloc[0]
        t = row['mean'] + (FIXED_SIGMA * row['std'])
        n_out = stats['is_outlier'].sum()
        print(f"{i:<5} {len(stats):<8} {row['mean']:<10.4f} {row['std']:<10.4f} {t:<10.4f} {n_out:<8}")

    return analysis_df, final_centers_pca


def plot_final(results_df, centers, filename):
    plt.figure(figsize=(12, 8))

    inliers = results_df[results_df['is_outlier'] == 0]
    outliers = results_df[results_df['is_outlier'] == 1]

    # Plot Inliers
    plt.scatter(inliers['pca_x'], inliers['pca_y'],
                c=inliers['cluster'], cmap='viridis', s=20, alpha=0.6, label='Inliers')

    # Plot Outliers
    if not outliers.empty:
        plt.scatter(outliers['pca_x'], outliers['pca_y'],
                    c='red', marker='x', s=80, linewidth=2, label='Outliers')

    # Plot Final Centers
    plt.scatter(centers[:, 0], centers[:, 1],
                c='black', s=300, marker='*', edgecolors='white', label='Centroids')

    plt.title(f"Two-Stage Clustering (Adaptive Detection)\nFile: {filename}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Get organized output path
    dataset_name = filename.replace('.txt', '').replace('.csv', '')
    output_dir = config.get_output_path('detection', dataset_name, 'png')
    out_name = os.path.join(output_dir, f"plot_detection_{dataset_name}.png")
    plt.savefig(out_name, dpi=150)
    print(f"\n[PCA VISUALIZATION] Plot saved as: {out_name}")


def plot_raw_outliers(df_raw, df_clean, results_df, filename):
    """
    Δείχνει τα outliers στον αρχικό χώρο των δεδομένων.
    df_raw      : Το αρχικό dataset (πριν οποιαδήποτε κανονικοποίηση)
    df_clean    : Το καθαρισμένο dataset που χρησιμοποιήθηκε για clustering
    results_df  : Το dataframe με τα αποτελέσματα του clustering
    """
    plt.figure(figsize=(12, 8))

    # Βρίσκουμε τα indices των outliers στο cleaned df
    outlier_indices = results_df.index[results_df['is_outlier'] == 1]
    orig_indices = df_clean.loc[outlier_indices, 'orig_index'].values

    # Δημιουργούμε mask για inliers
    all_indices = df_raw.index
    inlier_mask = ~all_indices.isin(orig_indices)

    # Εξασφαλίζουμε ότι παίρνουμε αριθμητικές τιμές
    inliers_x = pd.to_numeric(df_raw.loc[inlier_mask, 0], errors='coerce')
    inliers_y = pd.to_numeric(df_raw.loc[inlier_mask, 1], errors='coerce')

    outliers_x = pd.to_numeric(df_raw.iloc[orig_indices, 0], errors='coerce')
    outliers_y = pd.to_numeric(df_raw.iloc[orig_indices, 1], errors='coerce')

    # Σχεδίαση
    plt.scatter(inliers_x, inliers_y, c='blue', s=20, alpha=0.6, label='Inliers')
    if len(orig_indices) > 0:
        plt.scatter(outliers_x, outliers_y, c='red', s=80, marker='x', linewidths=2, label='Outliers')

    plt.title(f"Outliers in Original Coordinates\nFile: {filename}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Get organized output path
    dataset_name = filename.replace('.txt', '').replace('.csv', '')
    output_dir = config.get_output_path('detection', dataset_name, 'png')
    out_name = os.path.join(output_dir, f"plot_raw_outliers_{dataset_name}.png")
    plt.savefig(out_name, dpi=150)
    print(f"[RAW VISUALIZATION] Plot saved as: {out_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?', default=None)

    args = parser.parse_args()

    filename = args.filename
    if filename is None:
        filename = "202526files/data202526a_corrupted.txt"
        if not os.path.exists(filename):
            print("Usage: python script.py <filename>")
            sys.exit(1)

    # Create output directories
    config.create_output_directories()

    start_total = time.time()
    
    try:
        print(f"--- Processing: {filename} ---")
        df_raw = pd.read_csv(filename, sep=',', header=None, on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # Preprocess
    df_clean, X_scaled = sota_preprocess(df_raw)

    if len(df_clean) < 10:
        print("Not enough data.")
        sys.exit(1)

    # Algorithm (τώρα με δυναμικές παραμέτρους - όλα υπολογίζονται αυτόματα)
    results, centers = run_two_stage_adaptive(X_scaled)

    # Results
    num_outliers = results['is_outlier'].sum()
    end_total = time.time()

    print("\n" + "="*50)
    print(f"FINAL SUMMARY")
    print("="*50)
    print(f"Total Time:       {end_total - start_total:.4f} sec") 
    print(f"Outliers Found:   {num_outliers}")
    
    if num_outliers > 0:
        print("\n[DETECTED OUTLIERS - ORIGINAL COORDINATES (Top 20)]")

        # ✅ FIX: Map back to original raw data using orig_index
        outlier_indices = results.index[results['is_outlier'] == 1]
        orig_indices = df_clean.loc[outlier_indices, 'orig_index'].values

        # Print original raw coordinates
        print(df_raw.iloc[orig_indices].head(20).to_string(index=False, header=False))

        # Save original raw coordinates to CSV in organized directory
        # Generate unique dataset name
        base_filename = os.path.basename(filename)
        if base_filename in ['data.txt', 'data.csv']:
            # For synthetic datasets, use parent folder name
            parent_folder = os.path.basename(os.path.dirname(filename))
            dataset_name = parent_folder
        else:
            dataset_name = base_filename.replace('.txt', '').replace('.csv', '')

        output_dir = config.get_output_path('detection', dataset_name, 'csv')
        csv_name = os.path.join(output_dir, f"detection_outliers_{dataset_name}.csv")
        df_raw.iloc[orig_indices].to_csv(csv_name, index=False, header=False)
        print(f"\n-> Outliers saved to {csv_name}")

    # Generate unique dataset name for plots
    base_filename = os.path.basename(filename)
    if base_filename in ['data.txt', 'data.csv']:
        parent_folder = os.path.basename(os.path.dirname(filename))
        display_name = parent_folder
    else:
        display_name = base_filename

    # Visualization: PCA space (algorithm workspace)
    plot_final(results, centers, display_name)

    # Visualization: Original coordinates (verification)
    plot_raw_outliers(df_raw, df_clean, results, display_name)


if __name__ == "__main__":
    main()