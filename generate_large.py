#!/usr/bin/env python3
"""
Generate large dataset with 1 million samples for performance testing
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import os
import time

def generate_large_dataset(n_samples=1000000, n_clusters=5, outlier_fraction=0.01, random_state=42):
    """
    Generate a large dataset with 1M samples, 5 clusters, and outliers

    Parameters:
    -----------
    n_samples : int
        Total number of samples (default: 1,000,000)
    n_clusters : int
        Number of clusters (default: 5)
    outlier_fraction : float
        Fraction of outliers (default: 0.01 = 1%)
    random_state : int
        Random seed for reproducibility
    """

    print(f"Generating large dataset...")
    print(f"  Total samples: {n_samples:,}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Outlier fraction: {outlier_fraction * 100:.1f}%")

    start_time = time.time()

    # Calculate number of inliers and outliers
    n_outliers = int(n_samples * outlier_fraction)
    n_inliers = n_samples - n_outliers

    print(f"\n  Inliers: {n_inliers:,}")
    print(f"  Outliers: {n_outliers:,}")

    # Generate inliers using make_blobs
    print("\n[1/3] Generating inliers...")
    inliers, labels = make_blobs(
        n_samples=n_inliers,
        n_features=2,
        centers=n_clusters,
        cluster_std=[1.5, 2.0, 1.8, 2.2, 1.6],  # Different spreads
        center_box=(-10, 10),
        random_state=random_state
    )

    # Generate outliers (scattered randomly in a wider area)
    print("[2/3] Generating outliers...")
    np.random.seed(random_state)
    outliers = np.random.uniform(low=-20, high=20, size=(n_outliers, 2))

    # Combine inliers and outliers
    print("[3/3] Combining data...")
    X = np.vstack([inliers, outliers])

    # Create labels (0 = inlier, 1 = outlier)
    y = np.hstack([
        np.zeros(n_inliers, dtype=int),
        np.ones(n_outliers, dtype=int)
    ])

    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    generation_time = time.time() - start_time
    print(f"\n✅ Dataset generated in {generation_time:.2f} seconds")

    return X, y

def save_dataset(X, y, output_dir="synthetic_datasets/synth_large"):
    """Save dataset to CSV files"""

    print(f"\nSaving dataset to {output_dir}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save data (without labels - for algorithm input)
    data_path = os.path.join(output_dir, "data.csv")
    df_data = pd.DataFrame(X)
    df_data.to_csv(data_path, header=False, index=False)
    print(f"  ✓ Data saved: {data_path}")

    # Save ground truth (with labels - for evaluation)
    truth_path = os.path.join(output_dir, "ground_truth.csv")
    df_truth = pd.DataFrame(np.column_stack([X, y]))
    df_truth.to_csv(truth_path, header=False, index=False)
    print(f"  ✓ Ground truth saved: {truth_path}")

    # Save statistics
    stats_path = os.path.join(output_dir, "stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Dataset Statistics\n")
        f.write(f"==================\n\n")
        f.write(f"Total samples: {len(X):,}\n")
        f.write(f"Inliers: {np.sum(y == 0):,} ({np.sum(y == 0)/len(y)*100:.2f}%)\n")
        f.write(f"Outliers: {np.sum(y == 1):,} ({np.sum(y == 1)/len(y)*100:.2f}%)\n")
        f.write(f"\nFeature ranges:\n")
        f.write(f"  X1: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]\n")
        f.write(f"  X2: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]\n")
    print(f"  ✓ Statistics saved: {stats_path}")

    return data_path, truth_path

def main():
    print("="*60)
    print("LARGE DATASET GENERATOR (1 Million Samples)")
    print("="*60)

    # Generate dataset
    X, y = generate_large_dataset(
        n_samples=1_000_000,
        n_clusters=5,
        outlier_fraction=0.01,  # 1% outliers = 10,000 outliers
        random_state=42
    )

    # Save dataset
    data_path, truth_path = save_dataset(X, y)

    print("\n" + "="*60)
    print("COMPLETED!")
    print("="*60)
    print(f"Dataset ready at: synthetic_datasets/synth_large/")
    print(f"  • data.csv (input for algorithm)")
    print(f"  • ground_truth.csv (labels for evaluation)")
    print(f"  • stats.txt (dataset statistics)")
    print("\nNext step:")
    print("  python detection_noPCA_density.py synthetic_datasets/synth_large/ground_truth.csv")

if __name__ == "__main__":
    main()
