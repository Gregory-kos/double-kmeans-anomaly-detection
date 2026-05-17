#!/usr/bin/env python3
"""
Evaluation script για Density-Based Detection
Συγκρίνει τα detected outliers με το ground truth
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def load_ground_truth(dataset_path):
    """Load ground truth outliers"""
    gt_file = os.path.join(dataset_path, "outliers_only.csv")
    if not os.path.exists(gt_file):
        return None

    df = pd.read_csv(gt_file, header=None)
    return set(tuple(x) for x in df.values.tolist())

def load_detected_outliers(output_path):
    """Load detected outliers"""
    if not os.path.exists(output_path):
        # Return empty set if file doesn't exist (0 outliers detected)
        return set()

    df = pd.read_csv(output_path, header=None)
    if len(df) == 0:
        return set()
    return set(tuple(x) for x in df.values.tolist())

def calculate_metrics(ground_truth, detected, total_points):
    """Calculate TP, TN, FP, FN, Precision, Recall, F1, Accuracy"""

    # Διόρθωση: αφαιρούμε τα duplicates από το detected set
    detected = set(detected)
    ground_truth = set(ground_truth)

    TP = len(ground_truth & detected)  # True Positives
    FP = len(detected - ground_truth)   # False Positives
    FN = len(ground_truth - detected)   # False Negatives
    TN = total_points - TP - FP - FN    # True Negatives

    # Metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / total_points if total_points > 0 else 0

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Accuracy': accuracy
    }

def evaluate_dataset(dataset_name, dataset_path, output_base):
    """Evaluate a single dataset"""
    print(f"\n{'='*80}")
    print(f"📊 Evaluating (DENSITY-BASED): {dataset_name}")
    print(f"{'='*80}")

    # Load ground truth
    gt_outliers = load_ground_truth(dataset_path)
    if gt_outliers is None:
        print(f"   ⚠️ No ground truth found for {dataset_name}")
        return None

    # Load detected outliers
    detected_file = os.path.join(output_base, dataset_name,
                                 f"density_outliers_{dataset_name}.csv")
    detected_outliers = load_detected_outliers(detected_file)

    # Load data to get total points
    data_file = os.path.join(dataset_path, "data.txt")
    df_data = pd.read_csv(data_file, header=None)
    total_points = len(df_data)

    # Calculate metrics
    metrics = calculate_metrics(gt_outliers, detected_outliers, total_points)

    print(f"   Ground Truth: {len(gt_outliers)} outliers out of {total_points} points")
    print(f"   Detected: {len(detected_outliers)} outliers")
    print(f"\n   📈 METRICS:")
    print(f"   ├─ True Positives (TP):      {metrics['TP']}")
    print(f"   ├─ True Negatives (TN):   {metrics['TN']}")
    print(f"   ├─ False Positives (FP):     {metrics['FP']}")
    print(f"   ├─ False Negatives (FN):     {metrics['FN']}")
    print(f"   ├─ Precision:             {metrics['Precision']:.4f}")
    print(f"   ├─ Recall:                {metrics['Recall']:.4f}")
    print(f"   ├─ F1-Score:              {metrics['F1-Score']:.4f}")
    print(f"   └─ Accuracy:              {metrics['Accuracy']:.4f}")

    return metrics

def evaluate_basic_dataset(dataset_name, data_file, output_base):
    """Evaluate basic datasets (202526 files)"""
    print(f"\n{'='*80}")
    print(f"📊 Evaluating (DENSITY-BASED): {dataset_name}")
    print(f"{'='*80}")

    # Load ground truth from ground_truth.csv file
    gt_file = f"202526files/{dataset_name}_ground_truth.csv"
    if not os.path.exists(gt_file):
        print(f"   ⚠️ No ground truth file found: {gt_file}")
        return None

    df_gt = pd.read_csv(gt_file)
    gt_outliers = set()
    for _, row in df_gt[df_gt['label'] == -1].iterrows():
        gt_outliers.add((row['x'], row['y']))

    # Load detected outliers
    detected_file = os.path.join(output_base, dataset_name,
                                 f"density_outliers_{dataset_name}.csv")
    detected_outliers = load_detected_outliers(detected_file)

    # Load data to get total points
    df_data = pd.read_csv(data_file, header=None, sep=',', on_bad_lines='skip', engine='python')
    total_points = len(df_data)

    # Calculate metrics
    metrics = calculate_metrics(gt_outliers, detected_outliers, total_points)

    print(f"   Ground Truth: {len(gt_outliers)} outliers out of {total_points} points")
    print(f"   Detected: {len(detected_outliers)} outliers")
    print(f"\n   📈 METRICS:")
    print(f"   ├─ True Positives (TP):      {metrics['TP']}")
    print(f"   ├─ True Negatives (TN):   {metrics['TN']}")
    print(f"   ├─ False Positives (FP):     {metrics['FP']}")
    print(f"   ├─ False Negatives (FN):     {metrics['FN']}")
    print(f"   ├─ Precision:             {metrics['Precision']:.4f}")
    print(f"   ├─ Recall:                {metrics['Recall']:.4f}")
    print(f"   ├─ F1-Score:              {metrics['F1-Score']:.4f}")
    print(f"   └─ Accuracy:              {metrics['Accuracy']:.4f}")

    return metrics

def main():
    print(f"{'='*80}")
    print(f"🎯 EVALUATION (DENSITY-BASED) - Comparing Detected Outliers with Ground Truth")
    print(f"{'='*80}")

    output_base = "outputs/detection_noPCA_density"
    results = []

    # 1. Evaluate basic datasets
    print(f"\n{'='*80}")
    print(f"📁 BASIC DATASETS (202526 files)")
    print(f"{'='*80}")

    basic_datasets = [
        ("data202526a_corrupted", "202526files/data202526a_corrupted.txt"),
        ("data202526b_corrupted", "202526files/data202526b_corrupted.txt")
    ]

    for dataset_name, data_file in basic_datasets:
        if os.path.exists(data_file):
            metrics = evaluate_basic_dataset(dataset_name, data_file, output_base)
            if metrics:
                results.append({'Dataset': dataset_name, **metrics})

    # 2. Evaluate synthetic datasets
    print(f"\n{'='*80}")
    print(f"📁 SYNTHETIC DATASETS")
    print(f"{'='*80}")

    synthetic_dir = "synthetic_datasets"
    if os.path.exists(synthetic_dir):
        subdirs = sorted([d for d in os.listdir(synthetic_dir)
                         if os.path.isdir(os.path.join(synthetic_dir, d))])

        print(f"\n📊 Found {len(subdirs)} synthetic datasets to evaluate\n")

        for subdir in subdirs:
            dataset_path = os.path.join(synthetic_dir, subdir)
            metrics = evaluate_dataset(subdir, dataset_path, output_base)
            if metrics:
                results.append({'Dataset': subdir, **metrics})

    # Save results to CSV
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results[['Dataset', 'TP', 'TN', 'FP', 'FN',
                                 'Precision', 'Recall', 'F1-Score', 'Accuracy']]

        print(f"\n{'='*80}")
        print(f"📋 SUMMARY TABLE (DENSITY-BASED DETECTION)")
        print(f"{'='*80}\n")
        print(df_results.to_string(index=False))

        csv_file = "evaluation_results_density.csv"
        df_results.to_csv(csv_file, index=False)
        print(f"\n✅ Results saved to: {csv_file}")

        # Calculate averages
        print(f"\n{'='*80}")
        print(f"📊 AVERAGE METRICS (DENSITY-BASED)")
        print(f"{'='*80}")
        print(f"   Average Precision:  {df_results['Precision'].mean():.4f}")
        print(f"   Average Recall:     {df_results['Recall'].mean():.4f}")
        print(f"   Average F1-Score:   {df_results['F1-Score'].mean():.4f}")
        print(f"   Average Accuracy:   {df_results['Accuracy'].mean():.4f}")
        print(f"{'='*80}\n")
    else:
        print("\n⚠️ No results to display")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
