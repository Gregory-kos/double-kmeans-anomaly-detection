#!/usr/bin/env python3
"""
Script to run detection_noPCA_density.py on all datasets
"""

import os
import sys
import subprocess
from pathlib import Path

def run_detection(file_path):
    """Run density-based detection script on a file"""
    print(f"\n{'='*80}")
    print(f"🔍 Processing: {file_path}")
    print(f"{'='*80}\n")

    cmd = [sys.executable, "detection_noPCA_density.py", file_path]
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"❌ Error processing {file_path}")
        return False
    return True

def main():
    """Main function"""
    print("="*80)
    print("🚀 BATCH PROCESSING - DENSITY-BASED OUTLIER DETECTION")
    print("="*80)
    print("\n💡 This method uses DENSITY-AWARE adaptive parameters:")
    print("   - Density factor based on mean cluster distance")
    print("   - Dynamic minimum size (20% of median cluster size)")
    print("   - Euclidean distance with density-aware safety radius")
    print("   - NO fixed learned parameters - all rules emerge from data structure")
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
            data_path = os.path.join(synthetic_dir, subdir, "data.txt")
            if os.path.exists(data_path):
                datasets.append(data_path)
                print(f"   ✓ Found: {data_path}")

    if not datasets:
        print("\n❌ No datasets found!")
        return 1

    print(f"\n{'='*80}")
    print(f"📊 Total datasets to process: {len(datasets)}")
    print(f"{'='*80}")

    # Process all datasets
    success_count = 0
    fail_count = 0

    for i, dataset in enumerate(datasets, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Processing [{i}/{len(datasets)}]: {dataset}")
        print(f"{'#'*80}")

        if run_detection(dataset):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n\n" + "="*80)
    print("📋 BATCH PROCESSING SUMMARY (DENSITY-BASED)")
    print("="*80)
    print(f"   Total datasets:  {len(datasets)}")
    print(f"   ✅ Successful:   {success_count}")
    print(f"   ❌ Failed:       {fail_count}")
    print("="*80)

    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
