import pandas as pd

# Διάβασμα του CSV αρχείου
df = pd.read_csv('evaluation_results_noPCA.csv')

# Υπολογισμός μέσων όρων για όλες τις αριθμητικές στήλες
averages = df[['TP', 'TN', 'FP', 'FN', 'Precision', 'Recall', 'F1-Score', 'Accuracy']].mean()

# Εμφάνιση αποτελεσμάτων
print("=" * 80)
print("📊 AVERAGE METRICS (NO PCA)")
print("=" * 80)
print(f"\n   Average TP:         {averages['TP']:.2f}")
print(f"   Average TN:         {averages['TN']:.2f}")
print(f"   Average FP:         {averages['FP']:.2f}")
print(f"   Average FN:         {averages['FN']:.2f}")
print(f"   Average Precision:  {averages['Precision']:.4f}")
print(f"   Average Recall:     {averages['Recall']:.4f}")
print(f"   Average F1-Score:   {averages['F1-Score']:.4f}")
print(f"   Average Accuracy:   {averages['Accuracy']:.4f}")
print("\n" + "=" * 80)
