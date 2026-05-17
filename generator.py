import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
import os

def make_spirals(n_samples=500, n_spirals=2, noise=0.1, random_state=None):
    """Δημιουργεί σπιράλ σχήματα"""
    np.random.seed(random_state)
    n_per_spiral = n_samples // n_spirals
    X = []
    y = []

    for i in range(n_spirals):
        theta = np.sqrt(np.random.rand(n_per_spiral)) * 2 * np.pi + (i * 2 * np.pi / n_spirals)
        r = theta
        x = r * np.cos(theta)
        y_coord = r * np.sin(theta)
        noise_x = np.random.randn(n_per_spiral) * noise
        noise_y = np.random.randn(n_per_spiral) * noise
        X.append(np.column_stack([x + noise_x, y_coord + noise_y]))
        y.append(np.full(n_per_spiral, i))

    return np.vstack(X), np.concatenate(y)

def make_concentric_circles(n_samples=500, n_circles=3, noise=0.05, random_state=None):
    """Δημιουργεί ομόκεντρους κύκλους"""
    np.random.seed(random_state)
    n_per_circle = n_samples // n_circles
    X = []
    y = []

    for i in range(n_circles):
        radius = (i + 1) * 2
        theta = np.random.rand(n_per_circle) * 2 * np.pi
        x = radius * np.cos(theta) + np.random.randn(n_per_circle) * noise
        y_coord = radius * np.sin(theta) + np.random.randn(n_per_circle) * noise
        X.append(np.column_stack([x, y_coord]))
        y.append(np.full(n_per_circle, i))

    return np.vstack(X), np.concatenate(y)

def make_grid(n_samples=500, n_rows=3, n_cols=2, noise=0.1, random_state=None):
    """Δημιουργεί πλέγμα από clusters"""
    np.random.seed(random_state)
    n_clusters = n_rows * n_cols
    n_per_cluster = n_samples // n_clusters
    X = []
    y = []

    cluster_id = 0
    for row in range(n_rows):
        for col in range(n_cols):
            center_x = col * 5
            center_y = row * 5
            x = np.random.randn(n_per_cluster) * noise + center_x
            y_coord = np.random.randn(n_per_cluster) * noise + center_y
            X.append(np.column_stack([x, y_coord]))
            y.append(np.full(n_per_cluster, cluster_id))
            cluster_id += 1

    return np.vstack(X), np.concatenate(y)

def make_diagonal_lines(n_samples=500, n_lines=5, noise=0.2, random_state=None):
    """Δημιουργεί διαγώνιες γραμμές"""
    np.random.seed(random_state)
    n_per_line = n_samples // n_lines
    X = []
    y = []

    for i in range(n_lines):
        t = np.random.rand(n_per_line) * 10
        offset = i * 3
        x = t + np.random.randn(n_per_line) * noise
        y_coord = t + offset + np.random.randn(n_per_line) * noise
        X.append(np.column_stack([x, y_coord]))
        y.append(np.full(n_per_line, i))

    return np.vstack(X), np.concatenate(y)

def make_ellipses(n_samples=500, n_ellipses=5, noise=0.1, random_state=None):
    """Δημιουργεί ελλείψεις"""
    np.random.seed(random_state)
    n_per_ellipse = n_samples // n_ellipses
    X = []
    y = []

    for i in range(n_ellipses):
        theta = np.random.rand(n_per_ellipse) * 2 * np.pi
        a = 2 + i * 0.5  # major axis
        b = 1 + i * 0.3  # minor axis
        x = a * np.cos(theta) + np.random.randn(n_per_ellipse) * noise
        y_coord = b * np.sin(theta) + np.random.randn(n_per_ellipse) * noise
        # Τοποθέτηση σε διαφορετικές θέσεις
        offset_x = (i % 3) * 6 - 6
        offset_y = (i // 3) * 5 - 5
        X.append(np.column_stack([x + offset_x, y_coord + offset_y]))
        y.append(np.full(n_per_ellipse, i))

    return np.vstack(X), np.concatenate(y)

def make_s_curves(n_samples=500, n_curves=5, noise=0.2, random_state=None):
    """Δημιουργεί S-καμπύλες"""
    np.random.seed(random_state)
    n_per_curve = n_samples // n_curves
    X = []
    y = []

    for i in range(n_curves):
        t = np.linspace(0, 4 * np.pi, n_per_curve)
        x = t + np.random.randn(n_per_curve) * noise
        y_coord = np.sin(t) * 3 + i * 4 + np.random.randn(n_per_curve) * noise
        X.append(np.column_stack([x, y_coord]))
        y.append(np.full(n_per_curve, i))

    return np.vstack(X), np.concatenate(y)

def generate_synthetic_dataset(
    filename="synthetic_data.txt",
    n_samples=1000,
    n_clusters=5,
    difficulty='easy',   # options: 'easy', 'hard', 'shapes', 'spirals', 'concentric',
                        # 'grid', 'diagonal', 'ellipses', 's_curves', 'mixed_hard', 'anisotropic'
    density='dense',     # options: 'dense', 'normal', 'sparse'
    outlier_ratio=0.05,  # 5% outliers
    random_state=42
):
    """
    Δημιουργεί συνθετικά δεδομένα 2D για clustering και outlier detection.

    Parameters:
    - difficulty:
        'easy': Καλά διαχωρισμένα clusters.
        'hard': Clusters που επικαλύπτονται (overlapping).
        'shapes': Μη γραμμικά σχήματα (π.χ. φεγγάρια) - δύσκολο για K-Means.
        'spirals': Σπιράλ σχήματα - πολύ δύσκολο
        'concentric': Ομόκεντροι κύκλοι - δύσκολο
        'grid': Πλέγμα από clusters - εύκολο/μέτριο
        'diagonal': Διαγώνιες γραμμές - μέτριο
        'ellipses': Ελλείψεις - μέτριο
        's_curves': S-καμπύλες - δύσκολο
        'mixed_hard': Μίξη διαφορετικών σχημάτων - πολύ δύσκολο
        'anisotropic': Clusters με διαφορετική διακύμανση - δύσκολο
    - density:
        'dense': Πολύ συγκεντρωμένα σημεία (μικρό std dev).
        'sparse': Απλωμένα σημεία (μεγάλο std dev).
    """
    
    np.random.seed(random_state)
    
    # 1. Ρύθμιση Πυκνότητας (Cluster Standard Deviation)
    if density == 'dense':
        cluster_std = 0.5
    elif density == 'sparse':
        cluster_std = 2.5
    else: # normal
        cluster_std = 1.0

    # 2. Δημιουργία Βασικών Clusters (Inliers)
    n_inliers = int(n_samples * (1 - outlier_ratio))

    if difficulty == 'shapes':
        # Δημιουργούμε 5 διαφορετικά σχήματα:
        # 1. Φεγγάρια (moons) - 2 σχήματα
        # 2. Κύκλοι (circles) - 2 σχήματα (εσωτερικός και εξωτερικός)
        # 3. Blob στο κέντρο - 1 σχήμα

        samples_per_shape = n_inliers // 5
        noise = 0.05 if density == 'dense' else 0.15

        # Shape 1 & 2: Moons (πάνω αριστερά)
        X_moons, y_moons = make_moons(n_samples=samples_per_shape * 2, noise=noise, random_state=random_state)
        X_moons = X_moons * 3 + np.array([-8, 8])

        # Shape 3 & 4: Circles (κάτω δεξιά)
        X_circles, y_circles = make_circles(n_samples=samples_per_shape * 2, noise=noise,
                                            factor=0.5, random_state=random_state + 1)
        X_circles = X_circles * 5 + np.array([8, -8])
        y_circles += 2  # Offset labels

        # Shape 5: Central blob (κέντρο)
        X_blob, y_blob = make_blobs(n_samples=samples_per_shape, centers=1,
                                    cluster_std=0.8, center_box=(0, 0),
                                    random_state=random_state + 2)
        y_blob += 4  # Offset label

        # Συνδυασμός όλων
        X_inliers = np.vstack([X_moons, X_circles, X_blob])
        y_inliers = np.concatenate([y_moons, y_circles, y_blob.flatten()])

    elif difficulty == 'spirals':
        # Σπιράλ - πολύ δύσκολο για K-Means
        noise = 0.3 if density == 'dense' else 0.8
        # Δημιουργούμε 5 σπιράλ ή 2-3 ανάλογα με το n_clusters
        n_spirals = min(n_clusters, 5)
        X_inliers, y_inliers = make_spirals(n_samples=n_inliers, n_spirals=n_spirals,
                                           noise=noise, random_state=random_state)

    elif difficulty == 'concentric':
        # Ομόκεντροι κύκλοι - δύσκολο
        noise = 0.1 if density == 'dense' else 0.3
        X_inliers, y_inliers = make_concentric_circles(n_samples=n_inliers, n_circles=n_clusters,
                                                      noise=noise, random_state=random_state)

    elif difficulty == 'grid':
        # Πλέγμα - εύκολο/μέτριο
        noise = 0.3 if density == 'dense' else 0.8
        # Για 5 clusters: 3x2 πλέγμα (6 clusters, παίρνουμε 5)
        if n_clusters == 5:
            n_rows, n_cols = 3, 2
        elif n_clusters == 6:
            n_rows, n_cols = 3, 2
        else:
            n_rows = int(np.sqrt(n_clusters))
            n_cols = int(np.ceil(n_clusters / n_rows))
        X_temp, y_temp = make_grid(n_samples=n_inliers, n_rows=n_rows, n_cols=n_cols,
                                   noise=noise, random_state=random_state)
        # Κρατάμε μόνο τα πρώτα n_clusters
        mask = y_temp < n_clusters
        X_inliers = X_temp[mask]
        y_inliers = y_temp[mask]

    elif difficulty == 'diagonal':
        # Διαγώνιες γραμμές - μέτριο
        noise = 0.3 if density == 'dense' else 0.7
        X_inliers, y_inliers = make_diagonal_lines(n_samples=n_inliers, n_lines=n_clusters,
                                                   noise=noise, random_state=random_state)

    elif difficulty == 'ellipses':
        # Ελλείψεις - μέτριο
        noise = 0.2 if density == 'dense' else 0.5
        X_inliers, y_inliers = make_ellipses(n_samples=n_inliers, n_ellipses=n_clusters,
                                            noise=noise, random_state=random_state)

    elif difficulty == 's_curves':
        # S-καμπύλες - δύσκολο
        noise = 0.3 if density == 'dense' else 0.8
        X_inliers, y_inliers = make_s_curves(n_samples=n_inliers, n_curves=n_clusters,
                                            noise=noise, random_state=random_state)

    elif difficulty == 'mixed_hard':
        # Μίξη διαφορετικών σχημάτων - πολύ δύσκολο
        samples_per_shape = n_inliers // 5
        noise = 0.1 if density == 'dense' else 0.3

        # Shape 1: Spiral
        X_spiral, y_spiral = make_spirals(n_samples=samples_per_shape, n_spirals=1,
                                         noise=noise, random_state=random_state)
        X_spiral = X_spiral * 0.5 + np.array([-10, 0])

        # Shape 2: Concentric circle
        X_circle, y_circle = make_concentric_circles(n_samples=samples_per_shape, n_circles=1,
                                                     noise=noise, random_state=random_state + 1)
        X_circle = X_circle * 0.8 + np.array([10, 0])
        y_circle += 1

        # Shape 3: S-curve
        X_s, y_s = make_s_curves(n_samples=samples_per_shape, n_curves=1,
                                noise=noise, random_state=random_state + 2)
        X_s = X_s * 0.4 + np.array([0, 10])
        y_s += 2

        # Shape 4: Ellipse
        X_ellipse, y_ellipse = make_ellipses(n_samples=samples_per_shape, n_ellipses=1,
                                            noise=noise, random_state=random_state + 3)
        X_ellipse = X_ellipse + np.array([0, -10])
        y_ellipse += 3

        # Shape 5: Moon
        X_moon, y_moon = make_moons(n_samples=samples_per_shape, noise=noise,
                                    random_state=random_state + 4)
        X_moon = X_moon * 4 + np.array([-5, -5])
        y_moon += 4

        X_inliers = np.vstack([X_spiral, X_circle, X_s, X_ellipse, X_moon])
        y_inliers = np.concatenate([y_spiral, y_circle, y_s, y_ellipse, y_moon])

    elif difficulty == 'anisotropic':
        # Clusters με διαφορετική διακύμανση - δύσκολο
        n_per_cluster = n_inliers // n_clusters
        X_list = []
        y_list = []

        for i in range(n_clusters):
            # Κάθε cluster έχει διαφορετική διακύμανση
            std_x = 0.5 + i * 0.4
            std_y = 2.0 - i * 0.3
            cov = [[std_x, 0], [0, std_y]]
            center = [i * 5 - 10, i * 3 - 6]
            X_cluster = np.random.multivariate_normal(center, cov, n_per_cluster)
            X_list.append(X_cluster)
            y_list.append(np.full(n_per_cluster, i))

        X_inliers = np.vstack(X_list)
        y_inliers = np.concatenate(y_list)

    elif difficulty == 'hard':
        # 5 κανονικά clusters πολύ κοντά μεταξύ τους με μεγάλο std για επικάλυψη
        # Δημιουργούμε centers πολύ κοντά για να είναι δύσκολο
        X_inliers, y_inliers = make_blobs(n_samples=n_inliers, centers=n_clusters,
                                          cluster_std=1.5,  # Μεγάλο std για πολλή επικάλυψη
                                          center_box=(-3.0, 3.0),  # Πολύ περιορισμένος χώρος
                                          random_state=random_state)
    else: # 'easy'
        # Clusters μακριά το ένα από το άλλο
        X_inliers, y_inliers = make_blobs(n_samples=n_inliers, centers=n_clusters,
                                          cluster_std=cluster_std,
                                          center_box=(-15.0, 15.0), # Μεγάλος χώρος
                                          random_state=random_state)

    # 3. Δημιουργία Outliers (Θόρυβος)
    n_outliers = n_samples - n_inliers
    if n_outliers > 0:
        # Βρίσκουμε τα όρια των δεδομένων για να ρίξουμε τα outliers τριγύρω
        x_min, x_max = X_inliers[:, 0].min(), X_inliers[:, 0].max()
        y_min, y_max = X_inliers[:, 1].min(), X_inliers[:, 1].max()
        
        # Προσθέτουμε λίγο "αέρα" (buffer) για να είναι όντως outliers
        buffer = 5.0
        outliers = np.random.uniform(low=[x_min - buffer, y_min - buffer], 
                                     high=[x_max + buffer, y_max + buffer], 
                                     size=(n_outliers, 2))
        
        # Ετικέτα -1 για τα outliers (για το plot μας μόνο)
        y_outliers = np.full(n_outliers, -1)
        
        # Ένωση
        X = np.vstack([X_inliers, outliers])
        y = np.concatenate([y_inliers, y_outliers])
    else:
        X = X_inliers
        y = y_inliers

    # 4. Δημιουργία φακέλου για το dataset
    base_name = filename.replace('.txt', '').replace('.csv', '')
    dataset_folder = os.path.join('synthetic_datasets', base_name)
    os.makedirs(dataset_folder, exist_ok=True)

    # 5. Αποθήκευση δεδομένων σε αρχείο
    data_filename = os.path.join(dataset_folder, 'data.txt')
    df = pd.DataFrame(X)
    df.to_csv(data_filename, header=False, index=False)

    # 6. Αποθήκευση Ground Truth Labels (με -1 για outliers)
    labels_filename = os.path.join(dataset_folder, 'ground_truth.csv')
    df_labels = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1], 'label': y})
    df_labels.to_csv(labels_filename, index=False)

    # 7. Αποθήκευση μόνο των Outliers
    if n_outliers > 0:
        outliers_filename = os.path.join(dataset_folder, 'outliers_only.csv')
        outliers_mask = y == -1
        df_outliers = pd.DataFrame({'x': X[outliers_mask, 0], 'y': X[outliers_mask, 1]})
        df_outliers.to_csv(outliers_filename, index=False)

    # 8. Αποθήκευση Plot
    plot_filename = os.path.join(dataset_folder, 'plot.png')
    plt.figure(figsize=(10, 6))

    # Plot Inliers
    if len(X[y != -1]) > 0:
        plt.scatter(X[y != -1, 0], X[y != -1, 1], c=y[y != -1], cmap='viridis',
                   s=20, label='Inliers', alpha=0.6)

    # Plot Outliers
    if -1 in y:
        plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='x',
                   s=100, label='True Outliers', linewidths=2)

    plt.title(f"{difficulty.upper()} | Density: {density.upper()} | "
             f"Samples: {n_samples} | Outliers: {n_outliers}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Generated dataset in: {dataset_folder}/")
    print(f"   -> Mode: {difficulty.upper()} | Density: {density.upper()}")
    print(f"   -> Samples: {n_samples} | Outliers: {n_outliers}")
    print(f"   -> Files: data.txt, ground_truth.csv, outliers_only.csv, plot.png")

    return X, y

# ==========================================================
# ΒΟΗΘΗΤΙΚΗ ΣΥΝΑΡΤΗΣΗ ΓΙΑ ΝΑ ΤΑ ΔΕΙΣ
# ==========================================================
def plot_ground_truth(X, y, title="Synthetic Data"):
    plt.figure(figsize=(10, 6))
    
    # Plot Inliers
    plt.scatter(X[y != -1, 0], X[y != -1, 1], c=y[y != -1], cmap='viridis', s=20, label='Inliers')
    
    # Plot Outliers
    if -1 in y:
        plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='x', s=50, label='True Outliers')
        
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================================================
# ΠΑΡΑΔΕΙΓΜΑΤΑ ΧΡΗΣΗΣ
# ==========================================================
if __name__ == "__main__":

    # ΠΑΡΑΔΕΙΓΜΑ 1: Εύκολο και Πυκνό
    print("\n1/13: Generating synth_easy...")
    generate_synthetic_dataset(filename="synth_easy.txt",
                              difficulty='easy',
                              density='dense',
                              outlier_ratio=0.02)

    # ΠΑΡΑΔΕΙΓΜΑ 2: Δύσκολο με πολύ θόρυβο (5 clusters πολύ κοντά)
    print("\n2/13: Generating synth_hard...")
    generate_synthetic_dataset(filename="synth_hard.txt",
                              difficulty='hard',
                              density='normal',
                              outlier_ratio=0.25) # 25% θόρυβος - πολύ δύσκολο!

    # ΠΑΡΑΔΕΙΓΜΑ 3: Σχήματα (5 διαφορετικά σχήματα) - Τεστ για μη-γραμμικά clusters
    print("\n3/13: Generating synth_shapes...")
    generate_synthetic_dataset(filename="synth_shapes.txt",
                              difficulty='shapes',
                              density='dense',
                              outlier_ratio=0.05)

    # ============ ΝΕΑ DATASETS ============

    # ΠΑΡΑΔΕΙΓΜΑ 4: Σπιράλ - Πολύ δύσκολο
    print("\n4/13: Generating synth_spirals...")
    generate_synthetic_dataset(filename="synth_spirals.txt",
                              difficulty='spirals',
                              density='dense',
                              outlier_ratio=0.05)

    # ΠΑΡΑΔΕΙΓΜΑ 5: Ομόκεντροι Κύκλοι - Δύσκολο
    print("\n5/13: Generating synth_concentric...")
    generate_synthetic_dataset(filename="synth_concentric.txt",
                              difficulty='concentric',
                              density='dense',
                              outlier_ratio=0.05)

    # ΠΑΡΑΔΕΙΓΜΑ 6: Πλέγμα - Εύκολο/Μέτριο
    print("\n6/13: Generating synth_grid...")
    generate_synthetic_dataset(filename="synth_grid.txt",
                              difficulty='grid',
                              density='dense',
                              outlier_ratio=0.05)

    # ΠΑΡΑΔΕΙΓΜΑ 7: Διαγώνιες Γραμμές - Μέτριο
    print("\n7/13: Generating synth_diagonal...")
    generate_synthetic_dataset(filename="synth_diagonal.txt",
                              difficulty='diagonal',
                              density='normal',
                              outlier_ratio=0.05)

    # ΠΑΡΑΔΕΙΓΜΑ 8: Ελλείψεις - Μέτριο
    print("\n8/13: Generating synth_ellipses...")
    generate_synthetic_dataset(filename="synth_ellipses.txt",
                              difficulty='ellipses',
                              density='dense',
                              outlier_ratio=0.05)

    # ΠΑΡΑΔΕΙΓΜΑ 9: S-Καμπύλες - Δύσκολο
    print("\n9/13: Generating synth_s_curves...")
    generate_synthetic_dataset(filename="synth_s_curves.txt",
                              difficulty='s_curves',
                              density='dense',
                              outlier_ratio=0.05)

    # ΠΑΡΑΔΕΙΓΜΑ 10: Μίξη Σχημάτων - Πολύ Δύσκολο
    print("\n10/13: Generating synth_mixed_hard...")
    generate_synthetic_dataset(filename="synth_mixed_hard.txt",
                              difficulty='mixed_hard',
                              density='dense',
                              outlier_ratio=0.08)

    # ΠΑΡΑΔΕΙΓΜΑ 11: Anisotropic Clusters - Δύσκολο
    print("\n11/13: Generating synth_anisotropic...")
    generate_synthetic_dataset(filename="synth_anisotropic.txt",
                              difficulty='anisotropic',
                              density='normal',
                              outlier_ratio=0.05)

    # ΠΑΡΑΔΕΙΓΜΑ 12: Σπιράλ με περισσότερο θόρυβο - Πολύ Δύσκολο
    print("\n12/13: Generating synth_spirals_noisy...")
    generate_synthetic_dataset(filename="synth_spirals_noisy.txt",
                              difficulty='spirals',
                              density='sparse',
                              outlier_ratio=0.15)

    # ΠΑΡΑΔΕΙΓΜΑ 13: Ομόκεντροι με πολύ θόρυβο - Δύσκολο
    print("\n13/13: Generating synth_concentric_noisy...")
    generate_synthetic_dataset(filename="synth_concentric_noisy.txt",
                              difficulty='concentric',
                              density='sparse',
                              outlier_ratio=0.12)

    print("\n✅ Δημιουργήθηκαν 13 datasets συνολικά!")
    print("   - 3 original (easy, hard, shapes)")
    print("   - 10 νέα (spirals, concentric, grid, diagonal, ellipses, s_curves, mixed_hard, anisotropic, + 2 noisy)")