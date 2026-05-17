#!/usr/bin/env python3
"""
Comparison script for different outlier detection methods
"""

import pandas as pd
import numpy as np
import os

def load_results(filename):
    """Load evaluation results from CSV"""
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return None

def main():
    print("="*80)
    print("📊 COMPARISON OF OUTLIER DETECTION METHODS")
    print("="*80)

    # Load results from different methods
    methods = {
        'Stability-Based': 'evaluation_results_stability.csv',
    }

    # Try to load old results if they exist
    old_methods = {
        'With PCA': 'unused-old-python/old-outputs/evaluation_results.csv',
        'Without PCA (Euclidean)': 'unused-old-python/old-outputs/evaluation_results_noPCA.csv'
    }

    all_results = {}

    print("\n📁 Loading results...\n")

    for method_name, filename in methods.items():
        df = load_results(filename)
        if df is not None:
            all_results[method_name] = df
            print(f"✅ {method_name}: {filename}")
        else:
            print(f"❌ {method_name}: {filename} not found")

    for method_name, filename in old_methods.items():
        df = load_results(filename)
        if df is not None:
            all_results[method_name] = df
            print(f"✅ {method_name}: {filename}")

    if not all_results:
        print("\n❌ No evaluation results found!")
        return 1

    # Calculate average metrics for each method
    print("\n" + "="*80)
    print("📈 AVERAGE METRICS COMPARISON")
    print("="*80)
    print(f"\n{'Method':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12}")
    print("-" * 80)

    summary_data = []
    for method_name, df in all_results.items():
        avg_precision = df['Precision'].mean()
        avg_recall = df['Recall'].mean()
        avg_f1 = df['F1-Score'].mean()
        avg_accuracy = df['Accuracy'].mean()

        print(f"{method_name:<30} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f} {avg_accuracy:<12.4f}")

        summary_data.append({
            'Method': method_name,
            'Avg_Precision': avg_precision,
            'Avg_Recall': avg_recall,
            'Avg_F1': avg_f1,
            'Avg_Accuracy': avg_accuracy
        })

    # Find best method for each metric
    summary_df = pd.DataFrame(summary_data)

    print("\n" + "="*80)
    print("🏆 BEST METHODS PER METRIC")
    print("="*80)

    for metric in ['Avg_Precision', 'Avg_Recall', 'Avg_F1', 'Avg_Accuracy']:
        best_idx = summary_df[metric].idxmax()
        best_method = summary_df.loc[best_idx, 'Method']
        best_value = summary_df.loc[best_idx, metric]
        metric_name = metric.replace('Avg_', '')
        print(f"   {metric_name:<12}: {best_method:<30} ({best_value:.4f})")

    # Detailed comparison for each dataset
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("📋 DETAILED DATASET COMPARISON")
        print("="*80)

        # Get common datasets
        datasets = None
        for df in all_results.values():
            if datasets is None:
                datasets = set(df['Dataset'].values)
            else:
                datasets = datasets.intersection(set(df['Dataset'].values))

        if datasets:
            datasets = sorted(list(datasets))
            print(f"\n📊 Comparing {len(datasets)} common datasets\n")

            for dataset in datasets:
                print(f"\n{dataset}:")
                print(f"{'Method':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}")
                print("-" * 80)

                for method_name, df in all_results.items():
                    row = df[df['Dataset'] == dataset]
                    if not row.empty:
                        row = row.iloc[0]
                        print(f"{method_name:<30} {row['Precision']:<12.4f} {row['Recall']:<12.4f} "
                              f"{row['F1-Score']:<12.4f} {row['Accuracy']:<12.4f}")

    # Save comparison summary
    summary_df.to_csv('comparison_summary.csv', index=False)
    print("\n✅ Comparison summary saved to: comparison_summary.csv")

    print("\n" + "="*80)
    print("💡 KEY INSIGHTS:")
    print("="*80)
    print("   🔬 Stability-Based Detection:")
    print("      - Runs k-means multiple times with different seeds")
    print("      - Identifies points that frequently change clusters")
    print("      - Shows understanding of k-means initialization sensitivity")
    print("      - Doesn't blindly trust a single clustering result")
    print("="*80)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
