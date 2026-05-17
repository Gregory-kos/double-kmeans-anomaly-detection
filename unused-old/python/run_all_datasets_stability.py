#!/usr/bin/env python3
"""
Script to run detection_stability.py on all datasets
"""

import os
import sys
import subprocess
from pathlib import Path

def run_detection(file_path, n_runs=50, threshold=0.4):
    """Run stability detection script on a file"""
    print(f"\n{'='*80}")
    print(f"🔍 Processing: {file_path}")
    print(f"{'='*80}\n")

    cmd = [
        sys.executable, "detection_stability.py", file_path,
        "--n_runs", str(n_runs),
        "--threshold", str(threshold)
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"❌ Error processing {file_path}")
        return False
    return True

def main():
    """Main function"""
    print("="*80)
    print("🚀 BATCH PROCESSING - STABILITY-BASED DETECTION")
    print("="*80)
    print("\n💡 This method runs k-means multiple times with different seeds")
    print("   Outliers = points that frequently change clusters (unstable)")
    print("   Shows understanding of k-means sensitivity to initialization")
    print("="*80)

    datasets = []

    # 1. Basic datasets (202526 files)
    print("\n📁 Searching for 202526 datasets...")
    basic_files = [
        "202526files/data202526a_corrupted.txt",
        "202526files/data202526b_corrupted.txt"
    ]

    for f in basic_files:
        if os.path.exists(f):
            datasets.append(f)
            print(f"   ✓ Found: {f}")

    # 2. Synthetic datasets
    print("\n📁 Searching for synthetic datasets...")
    synthetic_dir = "synthetic_datasets"
    if os.path.exists(synthetic_dir):
        subdirs = sorted([d for d in os.listdir(synthetic_dir)
                         if os.path.isdir(os.path.join(synthetic_dir, d))])

        for subdir in subdirs:
            gt_path = os.path.join(synthetic_dir, subdir, "ground_truth.csv")
            if os.path.exists(gt_path):
                datasets.append(gt_path)
                print(f"   ✓ Found: {gt_path}")

    if not datasets:
        print("\n❌ No datasets found!")
        return 1

    print(f"\n{'='*80}")
    print(f"📊 Total datasets to process: {len(datasets)}")
    print(f"⚙️  Parameters:")
    print(f"   - Number of runs per dataset: 50")
    print(f"   - Instability threshold: 0.4")
    print(f"{'='*80}")

    # Process all datasets
    success_count = 0
    fail_count = 0

    for i, dataset in enumerate(datasets, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Processing [{i}/{len(datasets)}]: {dataset}")
        print(f"{'#'*80}")

        if run_detection(dataset, n_runs=50, threshold=0.4):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n\n" + "="*80)
    print("📋 BATCH PROCESSING SUMMARY (STABILITY-BASED)")
    print("="*80)
    print(f"   Total datasets:  {len(datasets)}")
    print(f"   ✅ Successful:   {success_count}")
    print(f"   ❌ Failed:       {fail_count}")
    print("="*80)

    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
