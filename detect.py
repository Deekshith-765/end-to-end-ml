import pandas as pd
import joblib
import numpy as np

try:
    from evidently.dashboard import Dashboard
    from evidently.tabs import DataDriftTab, CatNumDriftTab
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("evidently not installed - using basic drift detection")

old_data = pd.read_csv("train.csv")
new_data = pd.read_csv("newdata.csv")

columns = ["age", "salary"]

old_mean = old_data[columns].mean()
new_mean = new_data[columns].mean()

difference = abs(old_mean - new_mean)

print("Mean Difference:\n", difference)

percentage_change = abs((old_mean - new_mean) / old_mean) * 100
print("\nPercentage Change:\n", percentage_change)

if percentage_change["salary"] > 20:
    print("\nData Drift detected!")
    data_drift = True
else:
    print("\nNo data drift detected")
    data_drift = False

def calculate_psi(reference, current, buckets=10):
    """Calculate Population Stability Index (PSI)"""
    breakpoints = np.linspace(0, 1, buckets + 1)
    ref_percentiles = np.percentile(reference, breakpoints * 100)
    current_percentiles = np.percentile(current, breakpoints * 100)
    
    ref_bins = np.digitize(reference, ref_percentiles[1:-1])
    current_bins = np.digitize(current, current_percentiles[1:-1])
    
    ref_proportions = np.bincount(ref_bins, minlength=buckets) / len(reference)
    current_proportions = np.bincount(current_bins, minlength=buckets) / len(current)
    
    ref_proportions = np.where(ref_proportions == 0, 0.0001, ref_proportions)
    current_proportions = np.where(current_proportions == 0, 0.0001, current_proportions)
    
    psi = np.sum((current_proportions - ref_proportions) * np.log(current_proportions / ref_proportions))
    return psi

def calculate_ks_test(reference, current):
    """Calculate Kolmogorov-Smirnov test statistic"""
    from scipy.stats import ks_2samp
    stat, p_value = ks_2samp(reference, current)
    return stat, p_value

def calculate_kl_divergence(reference, current, buckets=10):
    """Calculate KL Divergence"""
    ref_hist, _ = np.histogram(reference, bins=buckets, density=True)
    curr_hist, _ = np.histogram(current, bins=buckets, density=True)
    
    ref_hist = ref_hist / ref_hist.sum()
    curr_hist = curr_hist / curr_hist.sum()
    
    ref_hist = np.where(ref_hist == 0, 0.0001, ref_hist)
    curr_hist = np.where(curr_hist == 0, 0.0001, curr_hist)
    
    kl_div = np.sum(ref_hist * np.log(ref_hist / curr_hist))
    return kl_div

print("\n=== Advanced Drift Detection ===")

for col in columns:
    if col in old_data.columns and col in new_data.columns:
        psi = calculate_psi(old_data[col].values, new_data[col].values)
        ks_stat, ks_pvalue = calculate_ks_test(old_data[col].values, new_data[col].values)
        kl_div = calculate_kl_divergence(old_data[col].values, new_data[col].values)
        
        print(f"\n{col}:")
        print(f"  PSI: {psi:.4f} {'(drift)' if psi > 0.2 else ''}")
        print(f"  KS Test: {ks_stat:.4f} (p-value: {ks_pvalue:.4f}) {'(drift)' if ks_pvalue < 0.05 else ''}")
        print(f"  KL Divergence: {kl_div:.4f}")

if EVIDENTLY_AVAILABLE:
    print("\n=== Evidently Report ===")
    try:
        dashboard = Dashboard(tabs=[DataDriftTab()])
        dashboard.calculate(reference_data=old_data, current_data=new_data)
        dashboard.save("drift_report.html")
        print("Drift report saved to drift_report.html")
    except Exception as e:
        print(f"Evidently report generation failed: {e}")

try:
    model = joblib.load("model.pkl")

    X_test = new_data.drop("target", axis=1)
    y_test = new_data["target"]

    accuracy = model.score(X_test, y_test)
    print("\nModel Accuracy:", accuracy)

    if accuracy < 0.75:
        print("Concept Drift detected!")
        concept_drift = True
    else:
        print("No Concept Drift")
        concept_drift = False

except Exception as e:
    print("\nModel evaluation skipped:", e)
    concept_drift = None

with open("drift_status.txt", "w") as f:
    f.write(f"data_drift={data_drift}\n")
    f.write(f"concept_drift={concept_drift}\n")

print("\nDrift detection completed.")