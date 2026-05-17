import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time
import os
import argparse
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Ρυθμίσεις για να φαίνονται όλα τα δεδομένα στο print
pd.set_option('display.max_rows', None)

def load_data(file_path):
    """Φόρτωση δεδομένων με διαχείριση σφαλμάτων."""
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
        
    try:
        # Χρήση header=None γιατί τα αρχεία της άσκησης συνήθως δεν έχουν επικεφαλίδες
        data = pd.read_csv(file_path, sep=',', header=None, on_bad_lines='skip', engine='python')
        print(f"--> Data loaded successfully: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(df):
    """Καθαρισμός και μετατροπή δεδομένων."""
    df_clean = df.copy()
    
    # Μετατροπή σε numeric και αφαίρεση NaN
    df_clean[0] = pd.to_numeric(df_clean[0], errors='coerce')
    df_clean[1] = pd.to_numeric(df_clean[1], errors='coerce')
    
    initial_len = len(df_clean)
    df_clean.dropna(inplace=True)
    
    # Αφαίρεση διπλότυπων (προαιρετικό, ανάλογα με τη φύση των data)
    df_clean.drop_duplicates(inplace=True)
    
    if len(df_clean) < initial_len:
        print(f"--> Removed {initial_len - len(df_clean)} bad/duplicate rows.")
        
    return df_clean.reset_index(drop=True)

def ensemble_kmeans_outlier_detection(df, n_runs=10, k_clusters=150):
    """
    SOTA Προσέγγιση: Ensemble K-Means.
    Τρέχουμε τον K-Means πολλές φορές. Αν ένα σημείο είναι 'μακριά' 
    στις περισσότερες εκτελέσεις, τότε είναι σίγουρα outlier.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # PCA για μείωση διαστάσεων (βοηθάει τον K-Means να τρέξει πιο γρήγορα και σωστά)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    n_samples = X_pca.shape[0]
    # Πίνακας που μετράει πόσες φορές κάθε σημείο ψηφίστηκε ως outlier
    outlier_votes = np.zeros(n_samples)
    
    print(f"--> Running Ensemble K-Means ({n_runs} runs)...")
    
    for i in range(n_runs):
        # Τυχαίο state σε κάθε run για διαφορετική αρχικοποίηση
        kmeans = KMeans(n_clusters=k_clusters, init='k-means++', n_init=5, random_state=np.random.randint(0, 10000))
        kmeans.fit(X_pca)
        
        # Υπολογισμός αποστάσεων από το κέντρο του cluster
        distances = np.min(kmeans.transform(X_pca), axis=1)
        
        # --- SOTA: Dynamic Thresholding (IQR Method) ---
        # Αντί για σταθερό 99%, χρησιμοποιούμε στατιστική μέθοδο IQR
        Q1 = np.percentile(distances, 25)
        Q3 = np.percentile(distances, 75)
        IQR = Q3 - Q1
        # Όριο: Ότι είναι πάνω από Q3 + 1.5*IQR (ή 2.0 για πιο αυστηρά)
        threshold = Q3 + (2.0 * IQR)
        
        # Ψηφίζουμε τα outliers αυτής της εκτέλεσης
        outlier_votes[distances > threshold] += 1
    
    # Τελική Απόφαση: Αν ψηφίστηκε σε πάνω από το 70% των runs, είναι outlier
    consensus_threshold = n_runs * 0.7
    final_outlier_mask = outlier_votes >= consensus_threshold
    
    return final_outlier_mask, X_pca

def visualize_results(df_original, X_pca, outlier_mask, filename, execution_time):
    """
    Δημιουργία διαδραστικού γραφήματος (Plotly) και στατικού (Matplotlib).
    """
    # 1. Static Plot (Matplotlib) - Για την αναφορά/PDF
    plt.figure(figsize=(10, 6))
    plt.scatter(df_original.loc[~outlier_mask, 0], df_original.loc[~outlier_mask, 1], 
                c='blue', s=10, alpha=0.5, label='Normal')
    plt.scatter(df_original.loc[outlier_mask, 0], df_original.loc[outlier_mask, 1], 
                c='red', s=50, marker='x', label='Outliers')
    plt.title(f"Outlier Detection: {filename}\nTime: {execution_time:.4f}s | Outliers: {sum(outlier_mask)}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    static_name = f"result_{filename}.png"
    plt.savefig(static_name)
    print(f"--> Static plot saved: {static_name}")
    plt.close()

    # 2. Interactive Plot (Plotly) - SOTA visualization
    # Φτιάχνουμε DataFrame για το Plotly
    plot_df = df_original.copy()
    plot_df.columns = ['X', 'Y']
    plot_df['Status'] = np.where(outlier_mask, 'Outlier', 'Normal')
    
    fig = px.scatter(plot_df, x='X', y='Y', color='Status',
                     color_discrete_map={'Normal': 'dodgerblue', 'Outlier': 'red'},
                     symbol='Status',
                     title=f"Interactive Detection - {filename} (Ensemble K-Means)",
                     hover_data=['X', 'Y'])
    
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
    html_name = f"interactive_{filename}.html"
    fig.write_html(html_name)
    print(f"--> Interactive plot saved: {html_name}")

def main():
    # Parsing arguments από command line (όπως ζητάει η εκφώνηση "μοναδική παράμετρο")
    # Χρήση: python script.py my_data.csv
    parser = argparse.ArgumentParser(description='SOTA K-Means Outlier Detector')
    parser.add_argument('filename', type=str, nargs='?', help='Path to the input CSV file')
    args = parser.parse_args()

    # Fallback αν δεν δωθεί αρχείο (για testing μέσα από IDE)
    file_path = args.filename if args.filename else "202526files/data202526_corrupted.txt"
    
    print("="*60)
    print(f"PROCESSING FILE: {file_path}")
    print("="*60)
    
    # 1. Έναρξη Χρονομέτρησης 
    start_time = time.time()
    
    # 2. Φόρτωση & Καθαρισμός [cite: 4, 6]
    raw_data = load_data(file_path)
    clean_data = preprocess_data(raw_data)
    
    if clean_data.empty:
        print("Error: No valid data found after cleaning.")
        return

    # 3. Εφαρμογή Ensemble K-Means [cite: 7, 8]
    # Χρησιμοποιούμε k=50 (over-clustering) για να χαρτογραφήσουμε λεπτομερώς το χώρο
    is_outlier, _ = ensemble_kmeans_outlier_detection(clean_data, n_runs=10, k_clusters=100)
    
    # 4. Λήξη Χρονομέτρησης
    end_time = time.time()
    duration = end_time - start_time
    
    # 5. Εκτυπώσεις Αποτελεσμάτων 
    num_outliers = np.sum(is_outlier)
    print("\n" + "-"*30)
    print(f"RESULTS SUMMARY")
    print("-" * 30)
    print(f"Execution Time: {duration:.4f} seconds")
    print(f"Total Points:   {len(clean_data)}")
    print(f"Outliers Found: {num_outliers}")
    
    if num_outliers > 0:
        print("\n--> Outlier Coordinates (Original Values):")
        # Εκτύπωση των αρχικών, μη κανονικοποιημένων σημείων
        print(clean_data[is_outlier])
    else:
        print("\n--> No significant outliers detected.")
        
    # 6. Οπτικοποίηση
    filename_only = os.path.basename(file_path)
    visualize_results(clean_data, None, is_outlier, filename_only, duration)

if __name__ == "__main__":
    main()