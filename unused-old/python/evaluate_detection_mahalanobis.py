#!/usr/bin/env python3
"""
Evaluation Script για detection_noPCA_mahalanobis.py
Υπολογίζει: TP, TN, FP, FN, Precision, Recall, F1-Score
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import glob


def load_ground_truth(dataset_folder):
    """Φορτώνει το ground truth"""
    gt_file = os.path.join(dataset_folder, "ground_truth.csv")
    if not os.path.exists(gt_file):
        return None
    df_gt = pd.read_csv(gt_file)
    outlier_indices = set(df_gt[df_gt['label'] == -1].index.tolist())
    return outlier_indices, len(df_gt)


def load_detected_outliers_mahalanobis(dataset_name):
    """Φορτώνει τα detected outliers από το detection_noPCA_mahalanobis.py output"""
    output_file = f"outputs/detection_noPCA_mahalanobis/{dataset_name}/detection_mahalanobis_outliers_{dataset_name}.csv"

    if not os.path.exists(output_file):
        print(f"   ⚠️  Warning: Output file not found: {output_file}")
        return None

    df_detected = pd.read_csv(output_file, header=None, names=['x', 'y'])
    return df_detected


def match_outliers_by_coordinates(ground_truth_df, detected_df, tolerance=1e-6):
    """Ταιριάζει outliers με βάση τις συντεταγμένες"""
    if detected_df is None or len(detected_df) == 0:
        return set()

    detected_indices = set()

    for _, det_row in detected_df.iterrows():
        det_x, det_y = det_row['x'], det_row['y']
        matches = ground_truth_df[
            (np.abs(ground_truth_df['x'] - det_x) < tolerance) &
            (np.abs(ground_truth_df['y'] - det_y) < tolerance)
        ]
        if len(matches) > 0:
            detected_indices.add(matches.index[0])

    return detected_indices


def calculate_metrics(gt_outliers, detected_indices, total_points):
    """Υπολογίζει TP, TN, FP, FN"""
    tp = len(gt_outliers & detected_indices)
    fp = len(detected_indices - gt_outliers)
    fn = len(gt_outliers - detected_indices)
    tn = total_points - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total_points if total_points > 0 else 0

    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Precision': precision, 'Recall': recall,
        'F1-Score': f1, 'Accuracy': accuracy,
        'Total_Points': total_points,
        'GT_Outliers': len(gt_outliers),
        'Detected_Outliers': len(detected_indices)
    }


def evaluate_dataset(dataset_name, dataset_folder):
    """Αξιολογεί ένα dataset"""
    print(f"\n{'='*80}")
    print(f"📊 Evaluating (MAHALANOBIS): {dataset_name}")
    print(f"{'='*80}")

    result = load_ground_truth(dataset_folder)
    if result is None:
        print("❌ No ground truth available (skipping)")
        return None

    gt_outliers, total_points = result
    print(f"   Ground Truth: {len(gt_outliers)} outliers out of {total_points} points")

    gt_file = os.path.join(dataset_folder, "ground_truth.csv")
    df_gt = pd.read_csv(gt_file)

    detected_df = load_detected_outliers_mahalanobis(dataset_name)
    if detected_df is None:
        print("❌ No detection results found (skipping)")
        return None

    print(f"   Detected: {len(detected_df)} outliers")

    detected_indices = match_outliers_by_coordinates(df_gt, detected_df)
    metrics = calculate_metrics(gt_outliers, detected_indices, total_points)

    print(f"\n   📈 METRICS:")
    print(f"   ├─ True Positives (TP):   {metrics['TP']:4d}")
    print(f"   ├─ True Negatives (TN):   {metrics['TN']:4d}")
    print(f"   ├─ False Positives (FP):  {metrics['FP']:4d}")
    print(f"   ├─ False Negatives (FN):  {metrics['FN']:4d}")
    print(f"   ├─ Precision:             {metrics['Precision']:.4f}")
    print(f"   ├─ Recall:                {metrics['Recall']:.4f}")
    print(f"   ├─ F1-Score:              {metrics['F1-Score']:.4f}")
    print(f"   └─ Accuracy:              {metrics['Accuracy']:.4f}")

    return metrics


def evaluate_basic_dataset(dataset_name, data_file, ground_truth_file):
    """Αξιολογεί βασικό dataset"""
    print(f"\n{'='*80}")
    print(f"📊 Evaluating (MAHALANOBIS): {dataset_name}")
    print(f"{'='*80}")

    if not os.path.exists(ground_truth_file):
        print(f"❌ No ground truth file found: {ground_truth_file}")
        return None

    df_gt = pd.read_csv(ground_truth_file)
    gt_outliers = set(df_gt[df_gt['label'] == -1].index.tolist())
    total_points = len(df_gt)

    print(f"   Ground Truth: {len(gt_outliers)} outliers out of {total_points} points")

    detected_df = load_detected_outliers_mahalanobis(dataset_name)
    if detected_df is None:
        print("❌ No detection results found (skipping)")
        return None

    print(f"   Detected: {len(detected_df)} outliers")

    detected_indices = match_outliers_by_coordinates(df_gt, detected_df, tolerance=1e-1)
    metrics = calculate_metrics(gt_outliers, detected_indices, total_points)

    print(f"\n   📈 METRICS:")
    print(f"   ├─ True Positives (TP):   {metrics['TP']:4d}")
    print(f"   ├─ True Negatives (TN):   {metrics['TN']:4d}")
    print(f"   ├─ False Positives (FP):  {metrics['FP']:4d}")
    print(f"   ├─ False Negatives (FN):  {metrics['FN']:4d}")
    print(f"   ├─ Precision:             {metrics['Precision']:.4f}")
    print(f"   ├─ Recall:                {metrics['Recall']:.4f}")
    print(f"   ├─ F1-Score:              {metrics['F1-Score']:.4f}")
    print(f"   └─ Accuracy:              {metrics['Accuracy']:.4f}")

    return metrics


def main():
    """Main function"""
    print("="*80)
    print("🎯 EVALUATION (MAHALANOBIS) - Comparing Detected Outliers with Ground Truth")
    print("="*80)

    all_metrics = []

    # 1. Αξιολόγηση βασικών datasets
    print("\n" + "="*80)
    print("📁 BASIC DATASETS (202526 files)")
    print("="*80)

    basic_datasets = [
        {
            'name': 'data202526a_corrupted',
            'data_file': '202526files/data202526a_corrupted.txt',
            'ground_truth': '202526files/data202526a_corrupted_ground_truth.csv'
        },
        {
            'name': 'data202526b_corrupted',
            'data_file': '202526files/data202526b_corrupted.txt',
            'ground_truth': '202526files/data202526b_corrupted_ground_truth.csv'
        }
    ]

    for ds in basic_datasets:
        metrics = evaluate_basic_dataset(ds['name'], ds['data_file'], ds['ground_truth'])
        if metrics:
            metrics['Dataset'] = ds['name']
            all_metrics.append(metrics)

    # 2. Αξιολόγηση synthetic datasets
    print("\n" + "="*80)
    print("📁 SYNTHETIC DATASETS")
    print("="*80)

    synthetic_dir = "synthetic_datasets"
    if os.path.exists(synthetic_dir):
        datasets = sorted([d for d in os.listdir(synthetic_dir)
                          if os.path.isdir(os.path.join(synthetic_dir, d))])

        print(f"\n📊 Found {len(datasets)} synthetic datasets to evaluate\n")

        for dataset_name in datasets:
            dataset_folder = os.path.join(synthetic_dir, dataset_name)
            metrics = evaluate_dataset(dataset_name, dataset_folder)

            if metrics:
                metrics['Dataset'] = dataset_name
                all_metrics.append(metrics)

    # Summary Table
    if all_metrics:
        print("\n\n" + "="*80)
        print("📋 SUMMARY TABLE (MAHALANOBIS)")
        print("="*80)

        df_summary = pd.DataFrame(all_metrics)
        df_summary = df_summary[['Dataset', 'TP', 'TN', 'FP', 'FN',
                                'Precision', 'Recall', 'F1-Score', 'Accuracy']]

        print("\n" + df_summary.to_string(index=False))

        output_file = "evaluation_results_mahalanobis.csv"
        df_summary.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")

        print("\n" + "="*80)
        print("📊 AVERAGE METRICS (MAHALANOBIS)")
        print("="*80)
        print(f"   Average Precision:  {df_summary['Precision'].mean():.4f}")
        print(f"   Average Recall:     {df_summary['Recall'].mean():.4f}")
        print(f"   Average F1-Score:   {df_summary['F1-Score'].mean():.4f}")
        print(f"   Average Accuracy:   {df_summary['Accuracy'].mean():.4f}")
        print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
