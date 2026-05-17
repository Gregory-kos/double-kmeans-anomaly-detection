#!/usr/bin/env python3
"""
Script για δημιουργία presentation graphs για το density-based detection
Δημιουργεί 4 εικόνες που δείχνουν τα βήματα του αλγορίθμου
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import config

# Δημιουργία φακέλου
PRESENTATION_DIR = "presentation_graphs"
os.makedirs(PRESENTATION_DIR, exist_ok=True)

def preprocess_data(filename):
    """Φόρτωση και προεπεξεργασία δεδομένων"""
    print(f"\n[LOADING] {filename}")
    df_raw = pd.read_csv(filename, header=None, on_bad_lines='skip', engine='python')

    # Keep original for visualization
    df_original = df_raw.copy()
    initial_rows = len(df_raw)

    df_clean = df_raw.copy()

    # Conversion to numeric
    for c in [0, 1]:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')

    # Handle infinite values
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    rows_before_nan = len(df_clean)
    df_clean.dropna(inplace=True)
    rows_after_nan = len(df_clean)
    nan_removed = rows_before_nan - rows_after_nan

    # Deduplication
    df_before_dedup = df_clean.copy()
    temp_rounded = df_clean.round(6)
    duplicates_mask = temp_rounded.duplicated()
    num_dupes = duplicates_mask.sum()
    if num_dupes > 0:
        df_clean = df_clean[~duplicates_mask]
        print(f"   -> Removed {num_dupes} duplicates")

    df_clean.reset_index(drop=True, inplace=True)
    print(f"   -> Final dataset: {len(df_clean)} rows")

    # Scaling
    scaler = StandardScaler()
    X_before_scaling = df_clean.iloc[:, :2].values
    X_scaled = scaler.fit_transform(X_before_scaling)

    return df_clean, X_scaled, scaler, df_original, df_before_dedup, X_before_scaling, initial_rows, num_dupes


def plot_0a_original_data(df_original, initial_rows):
    """Εικόνα 0a: Αρχικά δεδομένα"""
    print("\n[PLOT 0a] Creating original data visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Convert to numeric
    x_vals = pd.to_numeric(df_original[0], errors='coerce')
    y_vals = pd.to_numeric(df_original[1], errors='coerce')

    # Remove NaN for plotting
    valid_mask = ~(x_vals.isna() | y_vals.isna())
    x_clean = x_vals[valid_mask]
    y_clean = y_vals[valid_mask]

    # Plot 1: Scatter plot
    ax1.scatter(x_clean, y_clean, c='steelblue', s=10, alpha=0.5)
    ax1.set_title(f'ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ 1: Αρχικά Δεδομένα\n(Σύνολο: {initial_rows:,} σημεία)',
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distribution histograms
    ax2.hist(x_clean, bins=50, alpha=0.6, color='blue', label='X values', edgecolor='black')
    ax2.hist(y_clean, bins=50, alpha=0.6, color='orange', label='Y values', edgecolor='black')
    ax2.set_title('Κατανομή Τιμών', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Τιμή', fontsize=12)
    ax2.set_ylabel('Συχνότητα', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filename = os.path.join(PRESENTATION_DIR, "0a_original_data.png")
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"   ✓ Saved: {filename}")
    plt.close()


def plot_0b_deduplication(df_before_dedup, df_clean, num_dupes):
    """Εικόνα 0b: Πριν και μετά την αφαίρεση duplicates"""
    print("\n[PLOT 0b] Creating deduplication visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Before deduplication
    x_before = df_before_dedup[0].values
    y_before = df_before_dedup[1].values
    ax1.scatter(x_before, y_before, c='coral', s=10, alpha=0.4)
    ax1.set_title(f'ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ 2a: Πριν την Αφαίρεση Duplicates\n(Σύνολο: {len(df_before_dedup):,} σημεία)',
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # After deduplication
    x_after = df_clean[0].values
    y_after = df_clean[1].values
    ax2.scatter(x_after, y_after, c='green', s=10, alpha=0.4)
    ax2.set_title(f'ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ 2b: Μετά την Αφαίρεση Duplicates\n(Σύνολο: {len(df_clean):,} σημεία, Αφαιρέθηκαν: {num_dupes:,})',
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = os.path.join(PRESENTATION_DIR, "0b_deduplication.png")
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"   ✓ Saved: {filename}")
    plt.close()


def plot_0c_scaling(X_before_scaling, X_scaled, df_clean):
    """Εικόνα 0c: Πριν και μετά την κανονικοποίηση (scaling)"""
    print("\n[PLOT 0c] Creating scaling visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Before scaling
    ax1.scatter(X_before_scaling[:, 0], X_before_scaling[:, 1],
               c='purple', s=10, alpha=0.4)
    ax1.set_title(f'ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ 3a: Πριν την Κανονικοποίηση\n(Original Scale)',
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (original)', fontsize=12)
    ax1.set_ylabel('Y (original)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Add statistics
    x_mean, x_std = X_before_scaling[:, 0].mean(), X_before_scaling[:, 0].std()
    y_mean, y_std = X_before_scaling[:, 1].mean(), X_before_scaling[:, 1].std()
    stats_text = f'X: μ={x_mean:.2f}, σ={x_std:.2f}\nY: μ={y_mean:.2f}, σ={y_std:.2f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # After scaling
    ax2.scatter(X_scaled[:, 0], X_scaled[:, 1],
               c='teal', s=10, alpha=0.4)
    ax2.set_title(f'ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ 3b: Μετά την Κανονικοποίηση\n(StandardScaler: μ=0, σ=1)',
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (scaled)', fontsize=12)
    ax2.set_ylabel('Y (scaled)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add statistics
    x_mean_scaled, x_std_scaled = X_scaled[:, 0].mean(), X_scaled[:, 0].std()
    y_mean_scaled, y_std_scaled = X_scaled[:, 1].mean(), X_scaled[:, 1].std()
    stats_text_scaled = f'X: μ={x_mean_scaled:.2e}, σ={x_std_scaled:.2f}\nY: μ={y_mean_scaled:.2e}, σ={y_std_scaled:.2f}'
    ax2.text(0.02, 0.98, stats_text_scaled, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    filename = os.path.join(PRESENTATION_DIR, "0c_scaling.png")
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"   ✓ Saved: {filename}")
    plt.close()


def plot_0d_preprocessing_summary(initial_rows, num_dupes, final_rows):
    """Εικόνα 0d: Συνολική ροή προεπεξεργασίας"""
    print("\n[PLOT 0d] Creating preprocessing summary...")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Data for the flow
    steps = ['Αρχικά\nΔεδομένα', 'Αφαίρεση\nNaN/Inf', 'Αφαίρεση\nDuplicates', 'Τελικά\nΔεδομένα']
    values = [initial_rows, initial_rows, initial_rows - num_dupes, final_rows]

    # Create bar chart
    bars = ax.bar(steps, values, color=['#3498db', '#e74c3c', '#f39c12', '#2ecc71'],
                  edgecolor='black', linewidth=2, alpha=0.7)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:,}', ha='center', va='bottom', fontsize=14, fontweight='bold')

        # Add change annotations
        if i > 0:
            change = values[i] - values[i-1]
            if change < 0:
                ax.annotate(f'{change:,}',
                           xy=(i-0.5, (values[i-1] + values[i])/2),
                           fontsize=12, color='red', fontweight='bold',
                           ha='center')

    ax.set_ylabel('Αριθμός Σημείων', fontsize=14, fontweight='bold')
    ax.set_title('ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ: Συνολική Ροή Δεδομένων',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, initial_rows * 1.2)

    # Add summary box - positioned lower to avoid overlap
    summary_text = f'ΣΥΝΟΨΗ:\n' \
                  f'• Αρχικά σημεία: {initial_rows:,}\n' \
                  f'• Duplicates αφαιρέθηκαν: {num_dupes:,}\n' \
                  f'• Τελικά σημεία: {final_rows:,}\n' \
                  f'• Ποσοστό διατήρησης: {(final_rows/initial_rows)*100:.2f}%'
    ax.text(0.98, 0.70, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=0.8, edgecolor='black', linewidth=1.5))

    plt.tight_layout()
    filename = os.path.join(PRESENTATION_DIR, "0d_preprocessing_summary.png")
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"   ✓ Saved: {filename}")
    plt.close()


def calculate_adaptive_safety_radius_density(X_used, final_labels, final_centers, percentile=90):
    """Υπολογισμός adaptive safety radius με density factor"""
    safety_radii = {}

    for cluster_id in np.unique(final_labels):
        cluster_mask = (final_labels == cluster_id)
        cluster_points = X_used[cluster_mask]
        cluster_size = len(cluster_points)
        center = final_centers[cluster_id]

        distances = np.linalg.norm(cluster_points - center, axis=1)
        radius = np.percentile(distances, percentile)

        # Density factor
        mean_dist = np.mean(distances)
        density_factor = 1.0 / (1.0 + 0.5 * mean_dist)
        radius *= density_factor

        # Size penalty
        if cluster_size < 50:
            radius *= 0.80

        # Bounds
        radius = np.clip(radius, 0.5, 5.0)
        safety_radii[cluster_id] = radius

    return safety_radii


def plot_1_microclusters(X_used, micro_labels, micro_centers):
    """Εικόνα 1: 200 Micro-clusters με κυκλάκια"""
    print("\n[PLOT 1] Creating micro-clusters visualization...")

    plt.figure(figsize=(14, 10))

    # Χρώματα για τα micro-clusters
    colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(micro_labels))))

    # Plot κάθε micro-cluster
    for i, cluster_id in enumerate(np.unique(micro_labels)):
        mask = micro_labels == cluster_id
        cluster_points = X_used[mask]

        # Plot points
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   c=[colors[i]], s=15, alpha=0.6)

        # Plot center με κυκλάκι
        center = micro_centers[cluster_id]
        circle = plt.Circle((center[0], center[1]), 0.1,
                           color=colors[i], fill=False, linewidth=2, alpha=0.8)
        plt.gca().add_patch(circle)
        plt.scatter(center[0], center[1], c='black', s=50, marker='o',
                   edgecolors='white', linewidth=1.5, zorder=5)

    plt.title(f"ΒΗΜΑ 1: Micro-Clustering (n={len(np.unique(micro_labels))} clusters)",
             fontsize=16, fontweight='bold')
    plt.xlabel("Scaled X", fontsize=12)
    plt.ylabel("Scaled Y", fontsize=12)
    plt.grid(True, alpha=0.3)

    filename = os.path.join(PRESENTATION_DIR, "1_microclusters.png")
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"   ✓ Saved: {filename}")
    plt.close()


def plot_2_macroclusters(X_used, final_labels, final_centers, safety_radii):
    """Εικόνα 2: 5 Τελικά Macro-clusters με safety radii"""
    print("\n[PLOT 2] Creating macro-clusters visualization...")

    plt.figure(figsize=(14, 10))

    # Χρώματα για macro-clusters
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for cluster_id in np.unique(final_labels):
        mask = final_labels == cluster_id
        cluster_points = X_used[mask]

        # Plot points
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   c=colors[cluster_id], s=20, alpha=0.5,
                   label=f'Cluster {cluster_id}')

        # Plot center
        center = final_centers[cluster_id]
        plt.scatter(center[0], center[1], c=colors[cluster_id],
                   s=400, marker='*', edgecolors='black', linewidth=2, zorder=5)

        # Plot safety radius
        radius = safety_radii[cluster_id]
        circle = plt.Circle((center[0], center[1]), radius,
                           color=colors[cluster_id], fill=False,
                           linewidth=3, linestyle='--', alpha=0.7)
        plt.gca().add_patch(circle)

    plt.title("ΒΗΜΑ 2: Macro-Clustering (n=5 clusters) με Safety Radii",
             fontsize=16, fontweight='bold')
    plt.xlabel("Scaled X", fontsize=12)
    plt.ylabel("Scaled Y", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    filename = os.path.join(PRESENTATION_DIR, "2_macroclusters.png")
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"   ✓ Saved: {filename}")
    plt.close()


def plot_3_candidates(X_used, final_labels, final_centers, analysis_df, safety_radii):
    """Εικόνα 3: Υποψήφια outliers (με βάση κανόνες micro-distance ή isolation)"""
    print("\n[PLOT 3] Creating candidate outliers visualization...")

    plt.figure(figsize=(14, 10))

    # Χρώματα
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Υπολογισμός suspects (με βάση τους κανόνες)
    suspects_mask = (analysis_df['micro_dist'] > analysis_df['threshold']) | (analysis_df['is_isolated'] == True)

    # Plot normal points
    normal_mask = ~suspects_mask
    for cluster_id in np.unique(final_labels):
        mask = (final_labels == cluster_id) & normal_mask
        if np.any(mask):
            plt.scatter(X_used[mask, 0], X_used[mask, 1],
                       c=colors[cluster_id], s=20, alpha=0.3)

    # Plot candidate outliers (suspects)
    candidates_X = X_used[suspects_mask]
    plt.scatter(candidates_X[:, 0], candidates_X[:, 1],
               c='orange', s=100, marker='o', edgecolors='black',
               linewidth=2, label=f'Υποψήφια Outliers ({np.sum(suspects_mask)})',
               zorder=4, alpha=0.8)

    # Plot centers with safety radii
    for cluster_id in np.unique(final_labels):
        center = final_centers[cluster_id]
        plt.scatter(center[0], center[1], c=colors[cluster_id],
                   s=400, marker='*', edgecolors='black', linewidth=2, zorder=5)

        radius = safety_radii[cluster_id]
        circle = plt.Circle((center[0], center[1]), radius,
                           color=colors[cluster_id], fill=False,
                           linewidth=2, linestyle='--', alpha=0.5)
        plt.gca().add_patch(circle)

    plt.title(f"ΒΗΜΑ 3: Υποψήφια Outliers (Κανόνες: micro-distance > threshold ΄Η isolation)",
             fontsize=16, fontweight='bold')
    plt.xlabel("Scaled X", fontsize=12)
    plt.ylabel("Scaled Y", fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    filename = os.path.join(PRESENTATION_DIR, "3_candidate_outliers.png")
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"   ✓ Saved: {filename}")
    plt.close()

    return suspects_mask


def plot_4_final_outliers(X_used, final_labels, final_centers, analysis_df,
                          suspects_mask, safety_radii):
    """Εικόνα 4: Τελικά outliers (μετά το safety shield)"""
    print("\n[PLOT 4] Creating final outliers visualization...")

    plt.figure(figsize=(14, 10))

    # Χρώματα
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Υπολογισμός τελικών outliers
    final_outliers_mask = analysis_df['is_outlier'] == 1

    # Υπολογισμός rejected (saved by safety shield)
    rejected_mask = suspects_mask & ~final_outliers_mask

    # Plot normal points
    normal_mask = ~suspects_mask
    for cluster_id in np.unique(final_labels):
        mask = (final_labels == cluster_id) & normal_mask
        if np.any(mask):
            plt.scatter(X_used[mask, 0], X_used[mask, 1],
                       c=colors[cluster_id], s=20, alpha=0.3)

    # Plot rejected candidates (saved by safety shield)
    if np.any(rejected_mask):
        rejected_X = X_used[rejected_mask]
        plt.scatter(rejected_X[:, 0], rejected_X[:, 1],
                   c='yellow', s=120, marker='s', edgecolors='green',
                   linewidth=2, label=f'Απορρίφθηκαν (Safety Shield: {np.sum(rejected_mask)})',
                   zorder=3, alpha=0.7)

    # Plot final outliers
    if np.any(final_outliers_mask):
        outliers_X = X_used[final_outliers_mask]
        plt.scatter(outliers_X[:, 0], outliers_X[:, 1],
                   c='red', s=150, marker='X', edgecolors='darkred',
                   linewidth=3, label=f'Τελικά Outliers ({np.sum(final_outliers_mask)})',
                   zorder=4)

    # Plot centers with safety radii
    for cluster_id in np.unique(final_labels):
        center = final_centers[cluster_id]
        plt.scatter(center[0], center[1], c=colors[cluster_id],
                   s=400, marker='*', edgecolors='black', linewidth=2, zorder=5)

        radius = safety_radii[cluster_id]
        circle = plt.Circle((center[0], center[1]), radius,
                           color=colors[cluster_id], fill=False,
                           linewidth=3, linestyle='--', alpha=0.7,
                           label=f'Safety Radius C{cluster_id}' if cluster_id == 0 else '')
        plt.gca().add_patch(circle)

    plt.title("ΒΗΜΑ 4: Τελικά Outliers (μετά το Safety Shield)",
             fontsize=16, fontweight='bold')
    plt.xlabel("Scaled X", fontsize=12)
    plt.ylabel("Scaled Y", fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    filename = os.path.join(PRESENTATION_DIR, "4_final_outliers.png")
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"   ✓ Saved: {filename}")
    plt.close()


def main():
    """Main function"""
    print("="*80)
    print("🎨 ΔΗΜΙΟΥΡΓΙΑ PRESENTATION GRAPHS - DENSITY-BASED DETECTION")
    print("="*80)

    filename = "202526files/data202526a_corrupted.txt"

    # 1. Preprocess
    df_clean, X_scaled, scaler, df_original, df_before_dedup, X_before_scaling, initial_rows, num_dupes = preprocess_data(filename)
    X_used = X_scaled
    final_rows = len(df_clean)

    # 2. Create preprocessing visualizations
    print("\n" + "="*80)
    print("📊 CREATING PREPROCESSING VISUALIZATIONS")
    print("="*80)

    plot_0a_original_data(df_original, initial_rows)
    plot_0b_deduplication(df_before_dedup, df_clean, num_dupes)
    plot_0c_scaling(X_before_scaling, X_scaled, df_clean)
    plot_0d_preprocessing_summary(initial_rows, num_dupes, final_rows)

    # 2. Micro-clustering (200 clusters)
    print("\n[MICRO-CLUSTERING] Creating 200 micro-clusters...")
    n_micro = 200
    kmeans_micro = KMeans(n_clusters=n_micro, random_state=42, n_init=10)
    micro_labels = kmeans_micro.fit_predict(X_used)
    micro_centers = kmeans_micro.cluster_centers_

    # Calculate dynamic MIN_MICRO_SIZE
    unique, counts = np.unique(micro_labels, return_counts=True)
    median_size = np.median(counts)
    MIN_MICRO_SIZE = max(3, int(median_size * 0.2))
    print(f"   -> Dynamic MIN_MICRO_SIZE: {MIN_MICRO_SIZE}")

    # 3. Density filtering
    print("\n[DENSITY FILTERING] Filtering micro-clusters...")
    micro_counts_map = dict(zip(*np.unique(micro_labels, return_counts=True)))

    valid_micro_indices = []
    valid_weights = []
    for i in range(n_micro):
        if micro_counts_map.get(i, 0) >= MIN_MICRO_SIZE:
            valid_micro_indices.append(i)
            valid_weights.append(micro_counts_map.get(i, 0))

    print(f"   -> Valid micro-clusters: {len(valid_micro_indices)}/{n_micro}")

    if len(valid_micro_indices) < 5:
        valid_micro_centers = micro_centers
    else:
        valid_micro_centers = micro_centers[valid_micro_indices]

    # 4. Macro-clustering (5 clusters)
    print("\n[MACRO-CLUSTERING] Creating 5 macro-clusters...")
    n_macro = 5
    kmeans_macro = KMeans(n_clusters=n_macro, random_state=42, n_init=30)
    kmeans_macro.fit(valid_micro_centers, sample_weight=valid_weights)
    final_centers = kmeans_macro.cluster_centers_

    macro_labels_for_all_micro = kmeans_macro.predict(micro_centers)
    final_labels = macro_labels_for_all_micro[micro_labels]

    # 5. Calculate distances and thresholds
    print("\n[OUTLIER DETECTION] Calculating distances and thresholds...")

    # Micro distances
    my_micro_centers = micro_centers[micro_labels]
    micro_distances = np.linalg.norm(X_used - my_micro_centers, axis=1)

    # Macro distances
    my_macro_centers = final_centers[final_labels]
    macro_distances = np.linalg.norm(X_used - my_macro_centers, axis=1)

    # Isolation check
    is_isolated_mask = np.array([micro_counts_map[label] < MIN_MICRO_SIZE for label in micro_labels])

    # Safety radii
    safety_radii = calculate_adaptive_safety_radius_density(X_used, final_labels, final_centers)

    # Analysis dataframe
    analysis_df = pd.DataFrame({
        'cluster': final_labels,
        'micro_dist': micro_distances,
        'macro_dist': macro_distances,
        'is_isolated': is_isolated_mask
    })

    # Calculate thresholds per cluster
    cluster_stats = analysis_df.groupby('cluster')['micro_dist'].agg(['mean', 'std']).reset_index()
    cluster_stats['std'] = cluster_stats['std'].fillna(0)
    analysis_df = analysis_df.merge(cluster_stats, on='cluster', how='left')

    FIXED_SIGMA = 3.5
    analysis_df['threshold'] = analysis_df['mean'] + (FIXED_SIGMA * analysis_df['std'])
    analysis_df['safety_radius'] = analysis_df['cluster'].map(safety_radii)

    # Final decision
    suspects = (analysis_df['micro_dist'] > analysis_df['threshold']) | (analysis_df['is_isolated'] == True)
    analysis_df['is_outlier'] = (suspects &
                                 (analysis_df['macro_dist'] > analysis_df['safety_radius'])).astype(int)

    # Statistics
    n_suspects = np.sum(suspects)
    n_saved = np.sum(suspects & (analysis_df['macro_dist'] <= analysis_df['safety_radius']))
    n_final = analysis_df['is_outlier'].sum()

    print(f"   -> Υποψήφια outliers: {n_suspects}")
    print(f"   -> Απορρίφθηκαν (safety shield): {n_saved}")
    print(f"   -> Τελικά outliers: {n_final}")

    # 6. Create algorithm plots
    print("\n" + "="*80)
    print("📊 CREATING ALGORITHM VISUALIZATIONS")
    print("="*80)

    plot_1_microclusters(X_used, micro_labels, micro_centers)
    plot_2_macroclusters(X_used, final_labels, final_centers, safety_radii)
    suspects_mask = plot_3_candidates(X_used, final_labels, final_centers, analysis_df, safety_radii)
    plot_4_final_outliers(X_used, final_labels, final_centers, analysis_df, suspects_mask, safety_radii)

    print("\n" + "="*80)
    print(f"✅ ΟΛΟΚΛΗΡΩΘΗΚΕ! Οι εικόνες αποθηκεύτηκαν στο φάκελο: {PRESENTATION_DIR}/")
    print("="*80)
    print("\n📋 ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ (4 εικόνες):")
    print("  0a. 0a_original_data.png         - Αρχικά δεδομένα & κατανομή")
    print("  0b. 0b_deduplication.png         - Πριν/μετά αφαίρεση duplicates")
    print("  0c. 0c_scaling.png               - Πριν/μετά κανονικοποίηση")
    print("  0d. 0d_preprocessing_summary.png - Συνολική ροή προεπεξεργασίας")
    print("\n📋 ΑΛΓΟΡΙΘΜΟΣ ΑΝΙΧΝΕΥΣΗΣ (4 εικόνες):")
    print("  1. 1_microclusters.png           - 200 micro-clusters με κυκλάκια")
    print("  2. 2_macroclusters.png           - 5 macro-clusters με safety radii")
    print("  3. 3_candidate_outliers.png      - Υποψήφια outliers")
    print("  4. 4_final_outliers.png          - Τελικά outliers (μετά safety shield)")
    print("\n📊 ΣΥΝΟΛΟ: 8 εικόνες presentation")
    print("="*80)


if __name__ == "__main__":
    main()
