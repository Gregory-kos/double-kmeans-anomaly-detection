import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =============================================================================
# STEP 1: PREPROCESSING
# =============================================================================
def sota_preprocess(df):
    print("\n[PHASE 1] Preprocessing & Cleaning...")
    df_clean = df.copy()

    # 1. Numeric Conversion
    for c in [0, 1]:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')

    # 2. Infinite Values
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 3. Drop NaN
    df_clean.dropna(inplace=True)

    # 4. Deduplication
    temp_rounded = df_clean.round(6)
    duplicates_mask = temp_rounded.duplicated()
    if duplicates_mask.sum() > 0:
        df_clean = df_clean[~duplicates_mask]
        print(f"   -> Removed {duplicates_mask.sum()} duplicate rows.")

    df_clean.reset_index(drop=True, inplace=True)
    
    # 5. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean.iloc[:, :2].values)
    
    return df_clean, X_scaled

# =============================================================================
# STEP 2: TWO-STAGE CLUSTERING (SINGLE LINKAGE) & STATISTICAL OUTLIER DETECTION
# =============================================================================

def run_two_stage_adaptive(df_original, X_scaled, n_micro=200, n_macro=5):
    
    # --- TUNING PARAMETERS ---
    FIXED_SIGMA = 3.5        
    MIN_MICRO_SIZE = 3       
    
    print(f"\n[PHASE 2] Two-Stage Clustering (Pure Statistical)")
    print(f"   -> Macro-Clustering: Agglomerative (Single Linkage)")
    print(f"   -> Threshold Logic: Mean + {FIXED_SIGMA} * StdDev (No Minimum Floor)")

    start_algo = time.time()

    # --- A. PCA ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # --- B. STAGE 1: Micro-Clustering (K-Means) ---
    kmeans_micro = KMeans(n_clusters=n_micro, random_state=42, n_init=10)
    micro_labels = kmeans_micro.fit_predict(X_pca)
    micro_centers = kmeans_micro.cluster_centers_
    
    # --- C. FILTERING SMALL MICRO-CLUSTERS ---
    unique_micro, counts_micro = np.unique(micro_labels, return_counts=True)
    micro_counts_map = dict(zip(unique_micro, counts_micro))
    
    valid_micro_indices = [i for i in range(n_micro) if micro_counts_map.get(i, 0) >= MIN_MICRO_SIZE]
    
    if len(valid_micro_indices) < n_macro:
        valid_micro_centers = micro_centers
        valid_indices_map = {i: i for i in range(n_micro)}
    else:
        valid_micro_centers = micro_centers[valid_micro_indices]
        valid_indices_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_micro_indices)}

    # --- D. STAGE 2: Macro-Clustering (AGGLOMERATIVE SINGLE LINKAGE) ---
    agglomerative = AgglomerativeClustering(
        n_clusters=n_macro,
        linkage='single'
    )
    
    # Fit on valid micro-centers
    macro_labels_for_valid_micro = agglomerative.fit_predict(valid_micro_centers)
    
    # Map back to original micro-centers
    final_micro_to_macro_map = np.zeros(n_micro, dtype=int)
    
    for original_idx, valid_idx in valid_indices_map.items():
        if original_idx in valid_micro_indices:
            final_micro_to_macro_map[original_idx] = macro_labels_for_valid_micro[valid_idx]
            
    # Handle leftovers (nearest neighbor assignment)
    if len(valid_micro_indices) < n_micro:
        from sklearn.metrics import pairwise_distances_argmin
        all_indices = np.arange(n_micro)
        invalid_indices = np.setdiff1d(all_indices, valid_micro_indices)
        
        if len(invalid_indices) > 0:
            nearest_valid_idx = pairwise_distances_argmin(
                micro_centers[invalid_indices], 
                micro_centers[valid_micro_indices]
            )
            for i, invalid_idx in enumerate(invalid_indices):
                mapped_valid = nearest_valid_idx[i]
                neighbor_macro = macro_labels_for_valid_micro[mapped_valid]
                final_micro_to_macro_map[invalid_idx] = neighbor_macro

    # Assign labels to points
    final_labels = final_micro_to_macro_map[micro_labels]

    # Calculate Centers for plotting
    final_centers_pca = np.zeros((n_macro, 2))
    for m in range(n_macro):
        mask = final_labels == m
        if np.any(mask):
            final_centers_pca[m] = X_pca[mask].mean(axis=0)

    # --- E. PURE STATISTICAL OUTLIER DETECTION ---
    
    # 1. Distances to Micro-Centers
    my_micro_centers = micro_centers[micro_labels]
    distances = np.linalg.norm(X_pca - my_micro_centers, axis=1)

    # 2. Is Isolated?
    is_isolated_mask = np.array([micro_counts_map[label] < MIN_MICRO_SIZE for label in micro_labels])

    analysis_df = pd.DataFrame({
        'cluster': final_labels,
        'distance': distances,
        'is_isolated': is_isolated_mask
    })

    # 3. Cluster Stats
    cluster_stats = analysis_df.groupby('cluster')['distance'].agg(['mean', 'std']).reset_index()
    cluster_stats['std'] = cluster_stats['std'].fillna(0) 
    analysis_df = analysis_df.merge(cluster_stats, on='cluster', how='left')
    
    # 4. THRESHOLD (No MIN_DIST_TOLERANCE anymore)
    analysis_df['final_threshold'] = analysis_df['mean'] + (FIXED_SIGMA * analysis_df['std'])

    # 5. DECISION
    analysis_df['is_outlier'] = (
        (analysis_df['distance'] > analysis_df['final_threshold']) | 
        (analysis_df['is_isolated'] == True)
    ).astype(int)
    
    # Plotting Coords
    analysis_df['pca_x'] = X_pca[:, 0]
    analysis_df['pca_y'] = X_pca[:, 1]
    
    end_algo = time.time()
    
    # --- DEBUG PRINT ---
    print("\n[DEBUG: THRESHOLD ANALYSIS]")
    print(f"{'Cluster':<8} {'Mean Dist':<10} {'Std Dev':<10} {'Final Thresh':<12}")
    print("-" * 50)
    
    for i in range(n_macro):
        stats = analysis_df[analysis_df['cluster'] == i]
        if stats.empty: continue
        
        row = cluster_stats.iloc[i]
        calc_t = row['mean'] + (FIXED_SIGMA * row['std'])
        
        print(f"{i:<8} {row['mean']:<10.4f} {row['std']:<10.4f} {calc_t:<12.4f}")

    return analysis_df, final_centers_pca

def plot_final(results_df, centers, filename):
    plt.figure(figsize=(12, 8))
    
    inliers = results_df[results_df['is_outlier'] == 0]
    outliers = results_df[results_df['is_outlier'] == 1]
    
    plt.scatter(inliers['pca_x'], inliers['pca_y'], 
                c=inliers['cluster'], cmap='viridis', s=20, alpha=0.6, label='Inliers')
    
    if not outliers.empty:
        plt.scatter(outliers['pca_x'], outliers['pca_y'], 
                    c='red', marker='x', s=80, linewidth=2, label='Outliers')
        
    plt.scatter(centers[:, 0], centers[:, 1], 
                c='black', s=300, marker='*', edgecolors='white', label='Centroids')

    plt.title(f"Two-Stage (Single Linkage) - Pure Statistical\nFile: {filename}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_name = f"plot_final_{filename.replace('.txt', '').replace('.csv', '')}.png"
    plt.savefig(out_name, dpi=150)
    print(f"\n[VISUALIZATION] Plot saved as: {out_name}")


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

    start_total = time.time()
    
    try:
        print(f"--- Processing: {filename} ---")
        df_raw = pd.read_csv(filename, sep=',', header=None, on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    df_clean, X_scaled = sota_preprocess(df_raw)

    if len(df_clean) < 10:
        print("Not enough data.")
        sys.exit(1)

    results, centers = run_two_stage_adaptive(df_clean, X_scaled, n_micro=200, n_macro=5)

    num_outliers = results['is_outlier'].sum()
    end_total = time.time()

    print("\n" + "="*50)
    print(f"FINAL SUMMARY")
    print("="*50)
    print(f"Total Time:       {end_total - start_total:.4f} sec") 
    print(f"Outliers Found:   {num_outliers}")
    
    if num_outliers > 0:
        outlier_indices = results.index[results['is_outlier'] == 1]
        
        csv_name = f"final_outliers_{os.path.basename(filename).replace('.txt','').replace('.csv','')}.csv"
        df_clean.iloc[outlier_indices].to_csv(csv_name, index=False, header=False)
        print(f"\n-> Outliers saved to {csv_name}")

    plot_final(results, centers, os.path.basename(filename))

if __name__ == "__main__":
    main()