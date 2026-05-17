"""
Configuration file for outlier detection algorithms
Contains shared parameters and output directory settings
"""

import os

# =============================================================================
# CLUSTERING PARAMETERS
# =============================================================================

# Micro-clustering parameters
DEFAULT_N_MICRO = 200
DEFAULT_N_MACRO = 5

# Density filtering
MIN_MICRO_SIZE = 3

# Outlier detection parameters
FIXED_SIGMA = 3.5
OUTLIER_QUANTILE = 0.95  # 95% quantile for adaptive threshold

# Safety radius for macro-cluster protection
MACRO_SAFETY_RADIUS = 0.5

# PCA parameters
PCA_N_COMPONENTS = 2  # Target number of components
PCA_VARIANCE_THRESHOLD = 0.9  # Keep 90% variance

# KMeans parameters
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT_MICRO = 10
KMEANS_N_INIT_MACRO = 30

# =============================================================================
# OUTPUT DIRECTORY STRUCTURE
# =============================================================================

# Base output directory
OUTPUT_BASE_DIR = "outputs"

# Subdirectories for each script
OUTPUT_DIRS = {
    'detection': os.path.join(OUTPUT_BASE_DIR, 'detection_with_PCA'),
    'detection_noPCA': os.path.join(OUTPUT_BASE_DIR, 'detection_noPCA'),
    'detection_noPCA_mahalanobis': os.path.join(OUTPUT_BASE_DIR, 'detection_noPCA_mahalanobis'),
    'detection_noPCA_adaptive': os.path.join(OUTPUT_BASE_DIR, 'detection_noPCA_adaptive'),
    'detection_noPCA_adaptive_euclidean': os.path.join(OUTPUT_BASE_DIR, 'detection_noPCA_adaptive_euclidean'),
    'detection_noPCA_density': os.path.join(OUTPUT_BASE_DIR, 'detection_noPCA_density'),
    'detection_stability': os.path.join(OUTPUT_BASE_DIR, 'detection_stability'),
    'final': os.path.join(OUTPUT_BASE_DIR, 'final'),
    'ens': os.path.join(OUTPUT_BASE_DIR, 'ens')
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_output_path(script_name, dataset_name, file_type='png', subfolder=None):
    """
    Generate organized output path for a given script and dataset.

    Parameters:
    -----------
    script_name : str
        Name of the script ('detection', 'detection_noPCA', 'final', 'ens')
    dataset_name : str
        Name of the dataset (e.g., 'synth_easy', 'data202526a_corrupted')
    file_type : str
        File extension ('png' or 'csv')
    subfolder : str, optional
        Additional subfolder (e.g., 'plots', 'results')

    Returns:
    --------
    str : Full path to the output file
    """
    # Get base directory for this script
    base_dir = OUTPUT_DIRS.get(script_name, OUTPUT_BASE_DIR)

    # Create dataset-specific folder
    dataset_dir = os.path.join(base_dir, dataset_name)

    # Add subfolder if specified
    if subfolder:
        dataset_dir = os.path.join(dataset_dir, subfolder)

    # Create directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)

    return dataset_dir


def create_output_directories():
    """Create all output directories if they don't exist."""
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    for dir_path in OUTPUT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)
