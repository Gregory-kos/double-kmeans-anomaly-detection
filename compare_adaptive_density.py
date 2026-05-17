#!/usr/bin/env python3
"""
Comparison script for Adaptive (Mahalanobis) vs Density-Based outlier detection
Reads evaluation results from both methods and creates side-by-side comparison
"""

import pandas as pd
import numpy as np

def main():
    print("="*100)
    print("📊 OUTLIER DETECTION METHOD COMPARISON")
    print("="*100)
    print("\nComparing two detection methods:")
    print("  1. ADAPTIVE (Mahalanobis): Uses Mahalanobis distance with adaptive sigma")
    print("  2. DENSITY-BASED: Uses Euclidean distance with density-aware safety radius")
    print("="*100)

    # Load evaluation results
    try:
        df_adaptive = pd.read_csv("evaluation_results_adaptive.csv")
        df_density = pd.read_csv("evaluation_results_density.csv")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please run evaluate_detection_adaptive.py and evaluate_detection_density.py first.")
        return 1

    # Merge datasets on Dataset column
    comparison = pd.merge(
        df_adaptive[['Dataset', 'Precision', 'Recall', 'F1-Score', 'Accuracy']],
        df_density[['Dataset', 'Precision', 'Recall', 'F1-Score', 'Accuracy']],
        on='Dataset',
        suffixes=('_Adaptive', '_Density')
    )

    # Calculate differences
    comparison['Precision_Diff'] = comparison['Precision_Adaptive'] - comparison['Precision_Density']
    comparison['Recall_Diff'] = comparison['Recall_Adaptive'] - comparison['Recall_Density']
    comparison['F1-Score_Diff'] = comparison['F1-Score_Adaptive'] - comparison['F1-Score_Density']
    comparison['Accuracy_Diff'] = comparison['Accuracy_Adaptive'] - comparison['Accuracy_Density']

    # Print detailed comparison
    print("\n" + "="*100)
    print("📋 DETAILED COMPARISON (PRECISION)")
    print("="*100)
    print(f"\n{'Dataset':<25} {'Adaptive':>12} {'Density':>12} {'Difference':>12}")
    print("-"*100)
    for _, row in comparison.iterrows():
        diff_symbol = "✓" if row['Precision_Diff'] > 0 else ("=" if row['Precision_Diff'] == 0 else "✗")
        print(f"{row['Dataset']:<25} {row['Precision_Adaptive']:>11.2%} {row['Precision_Density']:>11.2%} "
              f"{row['Precision_Diff']:>+11.2%} {diff_symbol}")

    print("\n" + "="*100)
    print("📋 DETAILED COMPARISON (RECALL)")
    print("="*100)
    print(f"\n{'Dataset':<25} {'Adaptive':>12} {'Density':>12} {'Difference':>12}")
    print("-"*100)
    for _, row in comparison.iterrows():
        diff_symbol = "✓" if row['Recall_Diff'] > 0 else ("=" if row['Recall_Diff'] == 0 else "✗")
        print(f"{row['Dataset']:<25} {row['Recall_Adaptive']:>11.2%} {row['Recall_Density']:>11.2%} "
              f"{row['Recall_Diff']:>+11.2%} {diff_symbol}")

    print("\n" + "="*100)
    print("📋 DETAILED COMPARISON (F1-SCORE)")
    print("="*100)
    print(f"\n{'Dataset':<25} {'Adaptive':>12} {'Density':>12} {'Difference':>12}")
    print("-"*100)
    for _, row in comparison.iterrows():
        diff_symbol = "✓" if row['F1-Score_Diff'] > 0 else ("=" if row['F1-Score_Diff'] == 0 else "✗")
        print(f"{row['Dataset']:<25} {row['F1-Score_Adaptive']:>11.2%} {row['F1-Score_Density']:>11.2%} "
              f"{row['F1-Score_Diff']:>+11.2%} {diff_symbol}")

    print("\n" + "="*100)
    print("📋 DETAILED COMPARISON (ACCURACY)")
    print("="*100)
    print(f"\n{'Dataset':<25} {'Adaptive':>12} {'Density':>12} {'Difference':>12}")
    print("-"*100)
    for _, row in comparison.iterrows():
        diff_symbol = "✓" if row['Accuracy_Diff'] > 0 else ("=" if row['Accuracy_Diff'] == 0 else "✗")
        print(f"{row['Dataset']:<25} {row['Accuracy_Adaptive']:>11.2%} {row['Accuracy_Density']:>11.2%} "
              f"{row['Accuracy_Diff']:>+11.2%} {diff_symbol}")

    # Average metrics comparison
    print("\n" + "="*100)
    print("📊 AVERAGE METRICS COMPARISON")
    print("="*100)

    avg_adaptive = {
        'Precision': comparison['Precision_Adaptive'].mean(),
        'Recall': comparison['Recall_Adaptive'].mean(),
        'F1-Score': comparison['F1-Score_Adaptive'].mean(),
        'Accuracy': comparison['Accuracy_Adaptive'].mean()
    }

    avg_density = {
        'Precision': comparison['Precision_Density'].mean(),
        'Recall': comparison['Recall_Density'].mean(),
        'F1-Score': comparison['F1-Score_Density'].mean(),
        'Accuracy': comparison['Accuracy_Density'].mean()
    }

    print(f"\n{'Metric':<20} {'Adaptive':>15} {'Density':>15} {'Difference':>15}")
    print("-"*100)
    for metric in ['Precision', 'Recall', 'F1-Score', 'Accuracy']:
        diff = avg_adaptive[metric] - avg_density[metric]
        diff_symbol = "✓✓✓" if diff > 0.1 else ("✓" if diff > 0 else ("=" if diff == 0 else "✗"))
        print(f"{metric:<20} {avg_adaptive[metric]:>14.2%} {avg_density[metric]:>14.2%} "
              f"{diff:>+14.2%} {diff_symbol}")

    # Win/Loss/Tie statistics
    print("\n" + "="*100)
    print("📈 WIN/LOSS/TIE STATISTICS")
    print("="*100)

    for metric in ['Precision', 'Recall', 'F1-Score', 'Accuracy']:
        diff_col = f"{metric}_Diff"
        wins = (comparison[diff_col] > 0).sum()
        losses = (comparison[diff_col] < 0).sum()
        ties = (comparison[diff_col] == 0).sum()

        print(f"\n{metric}:")
        print(f"  Adaptive wins:  {wins}/{len(comparison)} datasets")
        print(f"  Density wins:   {losses}/{len(comparison)} datasets")
        print(f"  Ties:           {ties}/{len(comparison)} datasets")

    # Summary by dataset category
    print("\n" + "="*100)
    print("📊 SUMMARY BY DATASET CATEGORY")
    print("="*100)

    basic_datasets = comparison[comparison['Dataset'].str.contains('data202526')]
    synthetic_datasets = comparison[~comparison['Dataset'].str.contains('data202526')]

    print("\n🔹 BASIC DATASETS (202526 files):")
    print(f"  Adaptive  - Avg Precision: {basic_datasets['Precision_Adaptive'].mean():.2%}, "
          f"Recall: {basic_datasets['Recall_Adaptive'].mean():.2%}, "
          f"F1: {basic_datasets['F1-Score_Adaptive'].mean():.2%}")
    print(f"  Density   - Avg Precision: {basic_datasets['Precision_Density'].mean():.2%}, "
          f"Recall: {basic_datasets['Recall_Density'].mean():.2%}, "
          f"F1: {basic_datasets['F1-Score_Density'].mean():.2%}")

    print("\n🔹 SYNTHETIC DATASETS:")
    print(f"  Adaptive  - Avg Precision: {synthetic_datasets['Precision_Adaptive'].mean():.2%}, "
          f"Recall: {synthetic_datasets['Recall_Adaptive'].mean():.2%}, "
          f"F1: {synthetic_datasets['F1-Score_Adaptive'].mean():.2%}")
    print(f"  Density   - Avg Precision: {synthetic_datasets['Precision_Density'].mean():.2%}, "
          f"Recall: {synthetic_datasets['Recall_Density'].mean():.2%}, "
          f"F1: {synthetic_datasets['F1-Score_Density'].mean():.2%}")

    # Key findings
    print("\n" + "="*100)
    print("🔍 KEY FINDINGS")
    print("="*100)

    print("\n1. Overall Performance:")
    if avg_adaptive['F1-Score'] > avg_density['F1-Score']:
        winner = "ADAPTIVE (Mahalanobis)"
        improvement = ((avg_adaptive['F1-Score'] - avg_density['F1-Score']) / avg_density['F1-Score'] * 100)
        print(f"   Winner: {winner}")
        print(f"   F1-Score improvement: {improvement:.1f}%")
    else:
        winner = "DENSITY-BASED"
        improvement = ((avg_density['F1-Score'] - avg_adaptive['F1-Score']) / avg_adaptive['F1-Score'] * 100)
        print(f"   Winner: {winner}")
        print(f"   F1-Score improvement: {improvement:.1f}%")

    print("\n2. Precision vs Recall Trade-off:")
    print(f"   Adaptive  - Higher Precision: {avg_adaptive['Precision']:.2%}, Lower Recall: {avg_adaptive['Recall']:.2%}")
    print(f"   Density   - Precision: {avg_density['Precision']:.2%}, Recall: {avg_density['Recall']:.2%}")

    print("\n3. Dataset-Specific Performance:")
    best_adaptive = comparison.nlargest(3, 'F1-Score_Diff')[['Dataset', 'F1-Score_Diff']]
    print("   Adaptive performs best on:")
    for _, row in best_adaptive.iterrows():
        print(f"     - {row['Dataset']}: +{row['F1-Score_Diff']:.2%}")

    if (comparison['F1-Score_Diff'] < 0).any():
        best_density = comparison.nsmallest(3, 'F1-Score_Diff')[['Dataset', 'F1-Score_Diff']]
        print("\n   Density performs best on:")
        for _, row in best_density.iterrows():
            print(f"     - {row['Dataset']}: {row['F1-Score_Diff']:.2%}")
    else:
        print("\n   Density does not outperform Adaptive on any dataset.")

    # Save comparison to CSV
    output_file = "comparison_adaptive_vs_density.csv"
    comparison.to_csv(output_file, index=False)
    print(f"\n✅ Detailed comparison saved to: {output_file}")

    print("\n" + "="*100)
    print("🏁 COMPARISON COMPLETE")
    print("="*100)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
