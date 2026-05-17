#!/usr/bin/env python3
"""
Main Runner Script - Τρέχει detection.py για όλα τα datasets
"""

import os
import sys
import time
import glob
import subprocess
from pathlib import Path

def find_datasets():
    """
    Βρίσκει όλα τα datasets:
    1. Τα 2 βασικά από το φάκελο 202526files/
    2. Τα 13 synthetic από το φάκελο synthetic_datasets/
    """
    datasets = []

    # 1. Βασικά datasets από 202526files
    basic_datasets_dir = "202526files"
    if os.path.exists(basic_datasets_dir):
        basic_files = glob.glob(os.path.join(basic_datasets_dir, "data202526*.txt"))
        datasets.extend(basic_files)

    # 2. Synthetic datasets από synthetic_datasets
    synthetic_datasets_dir = "synthetic_datasets"
    if os.path.exists(synthetic_datasets_dir):
        # Κάθε subfolder έχει ένα data.txt
        for subdir in sorted(os.listdir(synthetic_datasets_dir)):
            data_file = os.path.join(synthetic_datasets_dir, subdir, "data.txt")
            if os.path.exists(data_file):
                datasets.append(data_file)

    return datasets


def run_detection_on_dataset(dataset_path):
    """
    Τρέχει το detection.py για ένα dataset
    """
    print("\n" + "="*80)
    print(f"🔍 Processing: {dataset_path}")
    print("="*80)

    start_time = time.time()

    try:
        # Τρέχουμε το detection.py ως subprocess
        result = subprocess.run(
            [sys.executable, "detection.py", dataset_path],
            capture_output=False,
            text=True,
            check=False
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Completed successfully in {elapsed:.2f}s")
            return True
        else:
            print(f"❌ Failed with return code {result.returncode}")
            return False

    except Exception as e:
        print(f"❌ Error processing {dataset_path}: {e}")
        return False


def main():
    """
    Main function που τρέχει το detection για όλα τα datasets
    """
    print("="*80)
    print("🚀 BATCH DETECTION - Running detection.py on all datasets")
    print("="*80)

    # Βρίσκουμε όλα τα datasets
    datasets = find_datasets()

    if not datasets:
        print("❌ No datasets found!")
        sys.exit(1)

    print(f"\n📊 Found {len(datasets)} datasets to process:")
    for i, dataset in enumerate(datasets, 1):
        print(f"   {i}. {dataset}")

    # Αρχικοποίηση counters
    total = len(datasets)
    successful = 0
    failed = 0

    start_total = time.time()

    # Τρέχουμε detection για κάθε dataset
    for i, dataset in enumerate(datasets, 1):
        print(f"\n\n{'='*80}")
        print(f"📈 Progress: {i}/{total} ({i*100//total}%)")
        print(f"{'='*80}")

        if run_detection_on_dataset(dataset):
            successful += 1
        else:
            failed += 1

    # Final Summary
    elapsed_total = time.time() - start_total

    print("\n\n" + "="*80)
    print("🎯 BATCH DETECTION - FINAL SUMMARY")
    print("="*80)
    print(f"Total datasets:      {total}")
    print(f"✅ Successful:        {successful}")
    print(f"❌ Failed:            {failed}")
    print(f"⏱️  Total time:        {elapsed_total:.2f}s ({elapsed_total/60:.2f}m)")
    print(f"⏱️  Average per file:  {elapsed_total/total:.2f}s")
    print("="*80)

    if failed == 0:
        print("\n🎉 All datasets processed successfully!")
    else:
        print(f"\n⚠️  {failed} dataset(s) failed to process.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
